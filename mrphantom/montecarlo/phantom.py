"""Monte Carlo DWI phantom synthesizer.

:class:`MCDWIPhantom` extends :class:`~mrphantom.dwi.phantom.DWIPhantom` with
a Monte Carlo outer loop: tissue parameters (S0, D, fractions) are sampled
from per-region distributions for each draw, and every sample is synthesized
with an independent noise seed.

Two execution modes are supported:

**Batch mode** — synthesize all N samples in memory at once, then save::

    from mrphantom.montecarlo import MCDWIPhantom
    from mrphantom.montecarlo.config import ParamDist, RegionMCConfig

    mc_regions = {
        "cortex": RegionMCConfig(
            name="cortex",
            S0=ParamDist(dist="normal", mean=180.0, std=10.0),
        )
    }
    phantom = MCDWIPhantom(n_samples=50, mc_regions=mc_regions, snr=30, seed=42)
    phantom.load_case(config).setup().synthesize().save("out.h5")

**Lazy mode** — write samples one at a time into a pre-allocated HDF5 file::

    phantom = MCDWIPhantom(n_samples=50, mc_regions=mc_regions, snr=30, seed=42)
    phantom.load_case(config).setup()
    with phantom.open_h5("out.h5") as h5:
        for i in range(phantom.n_samples):
            phantom.step(i)
            h5.flush(i)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import nibabel as nib
import numpy as np

from ..dwi.config import DWIConfig
from ..dwi.phantom import DWIPhantom
from ..io.mc_hdf5 import save_mc_hdf5
from .config import MCRunConfig, RegionMCConfig, load_montecarlo_config
from .sampler import SampledParams, _build_sample_config, sample_region_params

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# HDF5 incremental writer (private)
# ---------------------------------------------------------------------------


class _H5Writer:
    """Context manager for incremental per-sample HDF5 writing.

    Usage::

        phantom.load_case(config).setup()
        with phantom.open_h5("out.h5") as h5:
            for i in range(phantom.n_samples):
                phantom.step(i)
                h5.flush(i)

    On ``__enter__`` the HDF5 file is created and all datasets are
    pre-allocated with chunking ``(X, Y, Z, n_b, 1)`` so that each
    :meth:`flush` call writes exactly one chunk along the sample axis.

    On ``__exit__`` the file is closed (regardless of whether an exception
    was raised).
    """

    def __init__(self, phantom: "MCDWIPhantom", path: Path) -> None:
        self._phantom = phantom
        self._path = path
        self._file: h5py.File | None = None
        self._signal_ds: h5py.Dataset | None = None
        self._signal_gt_ds: h5py.Dataset | None = None

    def __enter__(self) -> "_H5Writer":
        cfg = self._phantom._config
        dims: list[int] = list(cfg.image_dims)
        n_b: int = len(cfg.bvalues)
        N: int = self._phantom.n_samples
        shape = tuple(dims) + (n_b, N)
        chunks = tuple(dims) + (n_b, 1)

        # Load segmentation and affine from the NIfTI
        seg_img: nib.Nifti1Image = nib.load(cfg.nifti_path)
        seg_raw: np.ndarray = np.round(seg_img.get_fdata()).astype(np.int32)
        seg_data: np.ndarray = seg_raw[..., 0] if seg_raw.ndim == 4 else seg_raw
        affine: np.ndarray = seg_img.affine

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self._path, "w")

        # Pre-allocate signal datasets with chunk-per-sample layout
        self._signal_ds = self._file.create_dataset(
            "signal",
            shape=shape,
            dtype=np.float64,
            chunks=chunks,
            compression="gzip",
        )
        self._signal_gt_ds = self._file.create_dataset(
            "signal_gt",
            shape=shape,
            dtype=np.float64,
            chunks=chunks,
            compression="gzip",
        )
        self._file.create_dataset(
            "volumes", data=np.asarray(cfg.bvalues, dtype=np.float64)
        )
        self._file.create_dataset("segmentation", data=seg_data, compression="gzip")
        self._file.create_dataset(
            "affine", data=np.asarray(affine, dtype=np.float64)
        )

        # Write sampled parameters (available from setup())
        sp: SampledParams = self._phantom._sampled_params  # type: ignore[assignment]
        grp = self._file.create_group("sampled_params")
        grp.create_dataset(
            "S0", data=sp.S0.astype(np.float64), compression="gzip"
        )
        grp.create_dataset(
            "D", data=sp.D.astype(np.float64), compression="gzip"
        )
        grp.create_dataset(
            "fractions", data=sp.fractions.astype(np.float64), compression="gzip"
        )
        vlen_str = h5py.string_dtype(encoding="utf-8")
        grp.create_dataset(
            "region_names",
            data=np.array(sp.region_names, dtype=object),
            dtype=vlen_str,
        )

        # Root-level metadata
        ph = self._phantom
        self._file.attrs["case_id"] = cfg.case_id
        self._file.attrs["name"] = cfg.name
        self._file.attrs["description"] = cfg.description
        self._file.attrs["n_samples"] = N
        if ph.snr is not None:
            self._file.attrs["snr"] = float(ph.snr)
        if ph.snr_region is not None:
            self._file.attrs["snr_region"] = ph.snr_region
        if ph.seed is not None:
            self._file.attrs["seed"] = int(ph.seed)

        return self

    def flush(self, i: int) -> None:
        """Write sample *i* from the phantom into the pre-allocated HDF5 file.

        :meth:`~MCDWIPhantom.step` must have been called with the same *i*
        before :meth:`flush`.

        Args:
            i: Sample index to write (0-indexed, must match the last
               :meth:`~MCDWIPhantom.step` call).

        Raises:
            ValueError: If the phantom's current sample index does not match
                *i* (i.e. :meth:`~MCDWIPhantom.step` was not called first).
        """
        if self._phantom._current_sample != i:
            raise ValueError(
                f"flush({i}) called but phantom._current_sample is "
                f"{self._phantom._current_sample!r}. "
                f"Call phantom.step({i}) first."
            )
        assert self._signal_ds is not None
        assert self._signal_gt_ds is not None
        self._signal_ds[..., i] = self._phantom._signal  # type: ignore[index]
        self._signal_gt_ds[..., i] = self._phantom._signal_gt  # type: ignore[index]

    def __exit__(self, *args: object) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


# ---------------------------------------------------------------------------
# MCDWIPhantom
# ---------------------------------------------------------------------------


class MCDWIPhantom(DWIPhantom):
    """Monte Carlo DWI phantom synthesizer.

    Extends :class:`~mrphantom.dwi.phantom.DWIPhantom` with a Monte Carlo
    outer loop.  Tissue parameters are sampled from per-region distributions
    for each of the *n_samples* draws, and every sample is synthesized with an
    independent noise seed.

    Lifecycle::

        phantom = MCDWIPhantom(n_samples=50, mc_regions=mc_regions, snr=30, seed=42)
        phantom.load_case(config)   # attach case
        phantom.setup()             # sample parameters, prepare RNG streams
        # --- batch mode ---
        phantom.synthesize()        # synthesize all N samples → 5-D arrays
        phantom.save("out.h5")      # HDF5 only (5-D cannot go to NIfTI)
        # --- lazy mode ---
        with phantom.open_h5("out.h5") as h5:
            for i in range(phantom.n_samples):
                phantom.step(i)     # synthesize sample i → 4-D arrays
                h5.flush(i)         # write to HDF5

    Attributes:
        n_samples:  Number of Monte Carlo draws.
        mc_regions: Per-region distribution specs, keyed by region name.
                    Regions absent from this dict use their base config values
                    unchanged across all draws.  Pass ``{}`` (the default) for
                    a pure noise-ensemble run with fixed tissue parameters.

    Examples:
        >>> phantom = MCDWIPhantom(n_samples=10, snr=30, seed=0)
        >>> phantom.load_case(config).setup().synthesize().save("mc_out.h5")
    """

    def __init__(
        self,
        *,
        n_samples: int,
        mc_regions: dict[str, RegionMCConfig] | None = None,
        snr: float | None = None,
        seed: int | None = None,
        snr_region: str | None = None,
    ) -> None:
        """Initialise with acquisition and Monte Carlo parameters.

        No case data is loaded at construction time.  Call :meth:`load_case`
        followed by :meth:`setup` before :meth:`synthesize` or :meth:`step`.

        Args:
            n_samples:  Number of Monte Carlo draws per case.
            mc_regions: Per-region distribution specs (by region name).
                        Regions absent from this dict are treated as fully
                        deterministic across all draws.  Defaults to ``{}``.
            snr:        Signal-to-noise ratio for Rician noise.  ``None``
                        (default) for noise-free output.
            seed:       Master random seed.  Derives independent streams for
                        parameter sampling and per-sample noise.
            snr_region: Name of the region whose S0 is used as the noise
                        reference.  ``None`` uses ``max(S0)`` across all
                        regions.
        """
        super().__init__(snr=snr, seed=seed, snr_region=snr_region)
        self.n_samples: int = n_samples
        self.mc_regions: dict[str, RegionMCConfig] = mc_regions if mc_regions is not None else {}

        self._setup_done: bool = False
        self._sampled_params: SampledParams | None = None
        self._noise_seeds: list | None = None
        self._current_sample: int | None = None
        self._batch_signal: np.ndarray | None = None
        self._batch_signal_gt: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Class-level factory
    # ------------------------------------------------------------------

    @classmethod
    def from_toml(cls, path: Path | str) -> "MCDWIPhantom":
        """Create an :class:`MCDWIPhantom` from a Monte Carlo TOML config.

        Reads the ``[run]``, ``[montecarlo]``, ``[export]``, and
        ``[[cases]]`` sections.  The returned phantom has its acquisition and
        MC parameters configured; call :meth:`load_case` and :meth:`setup`
        before synthesizing.

        The parsed :class:`~mrphantom.montecarlo.config.MCRunConfig` is
        accessible via :attr:`_mc_config` for downstream use (e.g. batch
        loops in the CLI).

        Args:
            path: Path to the ``.toml`` Monte Carlo config file.

        Returns:
            MCDWIPhantom: Instance with acquisition + MC parameters set.
        """
        cfg: MCRunConfig = load_montecarlo_config(path)
        phantom = cls(
            n_samples=cfg.n_samples,
            mc_regions=cfg.mc_regions,
            snr=cfg.snr,
            snr_region=cfg.snr_region,
            seed=cfg.seed,
        )
        phantom._mc_config = cfg
        return phantom

    # ------------------------------------------------------------------
    # MRPhantom interface — overrides
    # ------------------------------------------------------------------

    def load_case(self, config: DWIConfig) -> "MCDWIPhantom":
        """Attach a :class:`~mrphantom.dwi.config.DWIConfig` to this phantom.

        Clears any previously synthesized results and resets :meth:`setup`
        state so that stale sampled parameters are never reused with a
        different case.

        Args:
            config: Fully loaded :class:`~mrphantom.dwi.config.DWIConfig`.

        Returns:
            MCDWIPhantom: self, enables chaining:
            ``phantom.load_case(cfg).setup().synthesize()``.
        """
        super().load_case(config)
        # Reset MC-specific state so a new setup() must be called
        self._setup_done = False
        self._sampled_params = None
        self._noise_seeds = None
        self._current_sample = None
        self._batch_signal = None
        self._batch_signal_gt = None
        return self

    def setup(self) -> "MCDWIPhantom":
        """Sample tissue parameters and prepare per-sample RNG streams.

        Must be called after :meth:`load_case` and before :meth:`synthesize`,
        :meth:`step`, or :meth:`open_h5`.  Calling :meth:`load_case` on an
        already-set-up phantom invalidates the setup; call :meth:`setup` again
        before the next round of synthesis.

        Two independent RNG streams are derived from :attr:`seed` via
        :class:`numpy.random.SeedSequence`:

        - **param_rng** — used for tissue parameter sampling across all draws.
        - **noise_seeds** — one child seed per sample for Rician noise, so
          that changing *n_samples* does not perturb earlier samples' noise.

        Returns:
            MCDWIPhantom: self, enables chaining:
            ``phantom.load_case(cfg).setup().synthesize()``.

        Raises:
            RuntimeError: If :meth:`load_case` has not been called first.
        """
        if self._config is None:
            raise RuntimeError(
                f"{type(self).__name__}.setup() called before load_case(). "
                "Call load_case(config) first."
            )

        ss = np.random.SeedSequence(self.seed)
        param_ss, noise_ss = ss.spawn(2)
        param_rng = np.random.default_rng(param_ss)
        self._noise_seeds = noise_ss.spawn(self.n_samples)

        self._sampled_params = sample_region_params(
            self._config, self.mc_regions, self.n_samples, param_rng
        )
        self._setup_done = True
        return self

    def synthesize(self) -> "MCDWIPhantom":
        """Synthesize all *n_samples* draws in memory (batch mode).

        Iterates over all draws, calling :meth:`~DWIPhantom._run_synthesis`
        for each one with a per-sample config and noise seed.  Results are
        accumulated into 5-D arrays accessible via :attr:`signal` and
        :attr:`signal_gt`.

        Returns:
            MCDWIPhantom: self, enables chaining:
            ``phantom.setup().synthesize().save("out.h5")``.

        Raises:
            RuntimeError: If :meth:`setup` has not been called first.
        """
        if not self._setup_done:
            raise RuntimeError(
                f"{type(self).__name__}.synthesize() called before setup(). "
                "Call setup() first."
            )

        dims: list[int] = list(self._config.image_dims)  # type: ignore[union-attr]
        n_b: int = len(self._config.bvalues)  # type: ignore[union-attr]
        N: int = self.n_samples

        self._batch_signal = np.empty(tuple(dims) + (n_b, N), dtype=np.float64)
        self._batch_signal_gt = np.empty(tuple(dims) + (n_b, N), dtype=np.float64)

        assert self._sampled_params is not None
        assert self._noise_seeds is not None

        for i in range(N):
            sample_cfg = _build_sample_config(
                self._config, self._sampled_params, i  # type: ignore[arg-type]
            )
            sample_rng = (
                np.random.default_rng(self._noise_seeds[i])
                if self.snr is not None
                else None
            )
            self._run_synthesis(sample_rng, config=sample_cfg)
            self._batch_signal[..., i] = self._signal  # type: ignore[index]
            self._batch_signal_gt[..., i] = self._signal_gt  # type: ignore[index]

        return self

    def step(self, i: int) -> "MCDWIPhantom":
        """Synthesize sample *i* in lazy mode (single draw).

        Synthesizes sample *i* using the pre-sampled parameters and stores
        the result as a 4-D array in :attr:`signal` / :attr:`signal_gt` (the
        inherited single-sample slots).  This is the building block for the
        :meth:`open_h5` lazy writing workflow.

        Args:
            i: 0-indexed sample to synthesize.  Must be in
               ``[0, n_samples)``.

        Returns:
            MCDWIPhantom: self.

        Raises:
            RuntimeError: If :meth:`setup` has not been called first.
            RuntimeError: If *i* is outside ``[0, n_samples)``.
        """
        if not self._setup_done:
            raise RuntimeError(
                f"{type(self).__name__}.step() called before setup(). "
                "Call setup() first."
            )
        if i < 0 or i >= self.n_samples:
            raise RuntimeError(
                f"sample index {i} out of range [0, {self.n_samples})"
            )

        assert self._sampled_params is not None
        assert self._noise_seeds is not None

        sample_cfg = _build_sample_config(
            self._config, self._sampled_params, i  # type: ignore[arg-type]
        )
        sample_rng = (
            np.random.default_rng(self._noise_seeds[i])
            if self.snr is not None
            else None
        )
        self._run_synthesis(sample_rng, config=sample_cfg)
        self._current_sample = i
        return self

    def open_h5(self, path: Path | str) -> _H5Writer:
        """Open an HDF5 file for incremental lazy writing.

        The returned :class:`_H5Writer` context manager pre-allocates the
        full 5-D datasets on entry and provides a :meth:`~_H5Writer.flush`
        method for writing one sample at a time.

        Args:
            path: Destination HDF5 file path.

        Returns:
            _H5Writer: Context manager; use with the ``with`` statement.

        Raises:
            RuntimeError: If :meth:`setup` has not been called first.

        Examples:
            >>> phantom.load_case(config).setup()
            >>> with phantom.open_h5("out.h5") as h5:
            ...     for i in range(phantom.n_samples):
            ...         phantom.step(i)
            ...         h5.flush(i)
        """
        if not self._setup_done:
            raise RuntimeError(
                f"{type(self).__name__}.open_h5() called before setup(). "
                "Call setup() first."
            )
        return _H5Writer(self, Path(path))

    def save(self, *paths: Path | str) -> list[Path]:
        """Save the batch-synthesized phantom to one or more HDF5 files.

        Only ``.h5`` / ``.hdf5`` output is supported; 5-D signal arrays
        cannot be stored as NIfTI.

        Args:
            *paths: One or more output file paths (must use ``.h5`` or
                ``.hdf5`` extension).

        Returns:
            list[Path]: Resolved paths of every file that was written.

        Raises:
            RuntimeError: If :meth:`synthesize` has not been called first.
            ValueError: If no output paths are provided, if a ``.nii`` /
                ``.nii.gz`` path is given, or if an extension is not
                recognised.
        """
        if self._batch_signal is None:
            raise RuntimeError(
                f"{type(self).__name__}.save() called before synthesize(). "
                "Call synthesize() first."
            )
        if not paths:
            raise ValueError(
                f"{type(self).__name__}.save() requires at least one output path."
            )

        assert self._batch_signal_gt is not None
        assert self._sampled_params is not None
        assert self._config is not None
        assert self._signal_img is not None

        written: list[Path] = []
        for raw_path in paths:
            out_path = Path(raw_path)
            suffix = "".join(out_path.suffixes).lower()

            if suffix.endswith(".nii") or suffix.endswith(".nii.gz"):
                raise ValueError(
                    f"{type(self).__name__} cannot save 5-D signal to NIfTI "
                    f"({out_path.name!r}). Use .h5 / .hdf5."
                )
            elif suffix.endswith(".h5") or suffix.endswith(".hdf5"):
                written.append(
                    save_mc_hdf5(
                        signal=self._batch_signal,
                        signal_gt=self._batch_signal_gt,
                        volumes=self._config.bvalues,
                        segmentation=self._config.nifti_path,
                        affine=self._signal_img.affine,
                        S0_samples=self._sampled_params.S0,
                        D_samples=self._sampled_params.D,
                        fractions_samples=self._sampled_params.fractions,
                        region_names=self._sampled_params.region_names,
                        out_path=out_path,
                        case_id=self._config.case_id,
                        name=self._config.name,
                        description=self._config.description,
                        snr=self.snr,
                        snr_region=self.snr_region,
                        seed=self.seed,
                    )
                )
            else:
                raise ValueError(
                    f"Unrecognised output format for {out_path.name!r}. "
                    "Use .h5 / .hdf5."
                )

        return written

    # ------------------------------------------------------------------
    # Read-only properties — override / extend base
    # ------------------------------------------------------------------

    @property
    def signal(self) -> np.ndarray:
        """Synthesized signal array.

        Returns a 5-D array ``(X, Y, Z, n_b, N)`` after :meth:`synthesize`
        (batch mode) or a 4-D array ``(X, Y, Z, n_b)`` after :meth:`step`
        (lazy mode — current sample only).

        Raises:
            RuntimeError: If neither :meth:`synthesize` nor :meth:`step` has
                been called.
        """
        if self._batch_signal is not None:
            return self._batch_signal
        if self._signal is not None:
            return self._signal
        raise RuntimeError(
            f"{type(self).__name__}.signal accessed before synthesize() or step()."
        )

    @property
    def signal_gt(self) -> np.ndarray:
        """Noise-free ground-truth signal array.

        Returns a 5-D array ``(X, Y, Z, n_b, N)`` after :meth:`synthesize`
        (batch mode) or a 4-D array ``(X, Y, Z, n_b)`` after :meth:`step`
        (lazy mode — current sample only).

        Raises:
            RuntimeError: If neither :meth:`synthesize` nor :meth:`step` has
                been called.
        """
        if self._batch_signal_gt is not None:
            return self._batch_signal_gt
        if self._signal_gt is not None:
            return self._signal_gt
        raise RuntimeError(
            f"{type(self).__name__}.signal_gt accessed before synthesize() or step()."
        )

    @property
    def sampled_params(self) -> SampledParams:
        """Pre-sampled tissue parameters, available after :meth:`setup`.

        Raises:
            RuntimeError: If :meth:`setup` has not been called.
        """
        if self._sampled_params is None:
            raise RuntimeError(
                f"{type(self).__name__}.sampled_params accessed before setup()."
            )
        return self._sampled_params

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        """Return constructor parameters as a dictionary.

        Extends the base-class implementation to include :attr:`n_samples`
        and :attr:`mc_regions`.

        Returns:
            dict: ``{"snr": ..., "seed": ..., "snr_region": ...,
            "n_samples": ..., "mc_regions": ...}``.
        """
        params = super().get_params()
        params["n_samples"] = self.n_samples
        params["mc_regions"] = self.mc_regions
        return params
