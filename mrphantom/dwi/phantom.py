"""DWI phantom synthesizer.

:class:`DWIPhantom` implements the multi-exponential diffusion signal model::

    S(b) = S0 * Σ_i ( f_i * exp(-b * D_i) )

and optionally adds Rician noise at a given SNR.

Typical usage
-------------
**Programmatic (single case)**::

    from pathlib import Path
    from mrphantom import DWIPhantom

    phantom = DWIPhantom(snr=30, seed=42)
    phantom.load_case(config).synthesize().save("out.nii.gz", "out.h5")

    signal    = phantom.signal        # 4-D float64 array (X, Y, Z, n_bvalues)
    signal_gt = phantom.signal_gt     # noise-free ground-truth, same shape
    img       = phantom.signal_img    # nibabel.Nifti1Image

**Convenience factory**::

    phantom = DWIPhantom.from_json("phantom.json", "bvals.txt", snr=30)
    phantom.synthesize().save("out.nii.gz")

**Batch loop**::

    phantom = DWIPhantom(snr=30, seed=42)
    for entry in run_config.cases:
        cfg = load_dwi_config(entry.config, entry.bvals)
        phantom.load_case(cfg).synthesize().save(f"output/{cfg.case_id}.h5")
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from ..base import MRPhantom
from ..io import save_hdf5, save_nifti
from ..motion import MotionConfig
from .config import DWIConfig, load_dwi_config
from .synthesis import synthesize_dwi


class DWIPhantom(MRPhantom):
    """Diffusion-weighted MR phantom synthesizer.

    Attributes:
        snr: Signal-to-noise ratio for Rician noise. ``None`` (default)
            produces a noise-free phantom.
        seed: Integer random seed for reproducible noise. Ignored when ``snr``
            is ``None``.
        snr_region: Name of the region whose S0 is used as the noise reference
            when computing σ.  ``None`` (default) uses ``max(S0)`` across all
            regions.  Ignored when ``snr`` is ``None``.
        motion: Optional :class:`~mrphantom.motion.MotionConfig`.  When given,
            every volume is rigidly moved before noise is added; see
            :attr:`motion_matrices` and :attr:`signal_moved_gt`.

    Examples:
        >>> phantom = DWIPhantom(snr=30, seed=42)
        >>> phantom.load_case(config).synthesize().save("out.nii.gz")

        >>> # Use cortex S0 as the SNR reference instead of max(S0)
        >>> phantom = DWIPhantom(snr=30, snr_region="cortex", seed=42)
        >>> phantom.load_case(config).synthesize().save("out.nii.gz")
    """

    def __init__(
        self,
        *,
        snr: float | None = None,
        seed: int | None = None,
        snr_region: str | None = None,
        motion: MotionConfig | None = None,
    ) -> None:
        """Initialise with acquisition parameters only; no case data loaded yet."""
        super().__init__(snr=snr, seed=seed)
        self.snr_region = snr_region
        self.motion = motion
        self._motion_matrices: np.ndarray | None = None
        self._signal_moved_gt: np.ndarray | None = None
        self._config: DWIConfig | None = None
        self._signal: np.ndarray | None = None
        self._signal_gt: np.ndarray | None = None
        self._signal_img: nib.Nifti1Image | None = None

    # ------------------------------------------------------------------
    # Class-level factory
    # ------------------------------------------------------------------

    @classmethod
    def from_json(
        cls,
        json_path: Path | str,
        bval_path: Path | str,
        *,
        snr: float | None = None,
        seed: int | None = None,
        snr_region: str | None = None,
    ) -> "DWIPhantom":
        """Create a :class:`DWIPhantom` and load a case in one step.

        Equivalent to::

            DWIPhantom(snr=snr, seed=seed, snr_region=snr_region).load_case(
                load_dwi_config(json_path, bval_path)
            )

        Args:
            json_path:  Path to the phantom JSON file.
            bval_path:  Path to the b-values text file.
            snr:        Signal-to-noise ratio. ``None`` for noise-free.
            seed:       Random seed for reproducible noise.
            snr_region: Name of the region whose S0 is used as the noise
                        reference.  ``None`` uses ``max(S0)``.

        Returns:
            DWIPhantom: Instance with the case already loaded, ready to call
            :meth:`synthesize`.
        """
        config = load_dwi_config(Path(json_path), Path(bval_path))
        return cls(snr=snr, seed=seed, snr_region=snr_region).load_case(config)

    # ------------------------------------------------------------------
    # MRPhantom interface
    # ------------------------------------------------------------------

    def load_case(self, config: DWIConfig) -> "DWIPhantom":
        """Attach a :class:`~mrphantom.dwi.config.DWIConfig` to this phantom.

        Clears any previously synthesized signal so that stale results are
        never accessible after a case is swapped.

        Args:
            config: Fully loaded :class:`~mrphantom.dwi.config.DWIConfig` (use
                :func:`~mrphantom.dwi.config.load_dwi_config` to create one).

        Returns:
            DWIPhantom: self, enables chaining:
            ``phantom.load_case(cfg).synthesize()``.
        """
        self._config = config
        self._signal = None
        self._signal_gt = None
        self._signal_img = None
        self._motion_matrices = None
        self._signal_moved_gt = None
        return self

    def synthesize(self) -> "DWIPhantom":
        """Synthesize the DWI signal from the loaded case.

        Evaluates the multi-exponential model for every segmented region and
        optionally adds Rician noise. Results are cached on the instance and
        exposed via :attr:`signal`, :attr:`signal_gt`, and :attr:`signal_img`.

        Returns:
            DWIPhantom: self, enables chaining:
            ``phantom.synthesize().save("out.nii.gz")``.

        Raises:
            RuntimeError: If :meth:`load_case` has not been called first.
        """
        if self._config is None:
            raise RuntimeError(
                f"{type(self).__name__}.synthesize() called before load_case(). "
                "Call load_case(config) first."
            )

        rng: np.random.Generator | None = None
        if self.snr is not None:
            rng = np.random.default_rng(self.seed)

        self._run_synthesis(rng)
        return self

    def _run_synthesis(
        self,
        rng: "np.random.Generator | None",
        config: "DWIConfig | None" = None,
    ) -> None:
        """Run synthesis and store signal, GT, and image on the instance.

        This internal helper is extracted so that subclasses (e.g.
        :class:`~mrphantom.montecarlo.phantom.MCDWIPhantom`) can reuse the
        synthesis logic with a pre-seeded per-sample RNG and a per-sample
        config without duplicating the call site.

        Args:
            rng:    Pre-seeded :class:`numpy.random.Generator` for Rician
                    noise, or ``None`` for a noise-free run.  Passed directly
                    to :func:`~mrphantom.dwi.synthesis.synthesize_phantom`.
            config: :class:`~mrphantom.dwi.config.DWIConfig` to synthesize.
                    Defaults to :attr:`_config` when ``None``.
        """
        if config is None:
            config = self._config
        res = synthesize_dwi(
            config,
            snr=self.snr,
            rng=rng,
            snr_region=self.snr_region,
            motion=self.motion,
        )
        self._signal, self._signal_gt, self._signal_img = (
            res.signal,
            res.signal_gt,
            res.signal_img,
        )
        self._motion_matrices = res.motion_matrices
        self._signal_moved_gt = res.signal_moved_gt

    # ------------------------------------------------------------------
    # Introspection — extend base with snr_region
    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        """Return constructor parameters as a dictionary.

        Extends the base-class implementation to include :attr:`snr_region`.

        Returns:
            dict: ``{"snr": ..., "seed": ..., "snr_region": ...}``.

        Examples:
            >>> DWIPhantom(snr=30, seed=42, snr_region="cortex").get_params()
            {'snr': 30, 'seed': 42, 'snr_region': 'cortex'}
        """
        params = super().get_params()
        params["snr_region"] = self.snr_region
        return params

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save(self, *paths: Path | str) -> list[Path]:
        """Save the synthesized phantom to one or more files.

        The output format is inferred from the file extension:

        - ``.nii`` / ``.nii.gz`` → NIfTI
        - ``.h5`` / ``.hdf5``    → HDF5

        Args:
            *paths: One or more output file paths. At least one must be
                provided.

        Returns:
            list[Path]: Resolved paths of every file that was written.

        Raises:
            RuntimeError: If :meth:`synthesize` has not been called first.
            ValueError: If no output paths are provided, or if an extension
                is not recognised.
        """
        if self._signal is None or self._signal_img is None:
            raise RuntimeError(
                f"{type(self).__name__}.save() called before synthesize(). "
                "Call synthesize() first."
            )
        if not paths:
            raise ValueError(
                f"{type(self).__name__}.save() requires at least one output path."
            )

        written: list[Path] = []
        for raw_path in paths:
            out_path = Path(raw_path)
            suffix = "".join(out_path.suffixes).lower()

            if suffix.endswith(".nii") or suffix.endswith(".nii.gz"):
                nii_path = save_nifti(self._signal_img, out_path)
                written.append(nii_path)
                if self._config.bvecs is not None:
                    written.append(self._write_bvec(nii_path))
            elif suffix.endswith(".npy"):
                if self._motion_matrices is None:
                    raise ValueError(
                        "No motion matrices to save: the phantom has no motion."
                    )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(out_path, self._motion_matrices)
                written.append(out_path)
            elif suffix.endswith(".h5") or suffix.endswith(".hdf5"):
                written.append(
                    save_hdf5(
                        self._signal,
                        self._signal_img,
                        self._config.bvalues,
                        self._config.nifti_path,
                        out_path,
                        case_id=self._config.case_id,
                        name=self._config.name,
                        description=self._config.description,
                        snr=self.snr,
                        snr_region=self.snr_region,
                        signal_gt=self._signal_gt,
                        motion_matrices=self._motion_matrices,
                        signal_moved_gt=self._signal_moved_gt,
                        bvecs=self._config.bvecs,
                    )
                )
            else:
                raise ValueError(
                    f"Unrecognised output format for {out_path.name!r}. "
                    "Use .nii / .nii.gz (NIfTI), .h5 / .hdf5 (HDF5) or .npy (motion matrices)."
                )

        return written

    def _write_bvec(self, nii_path: Path) -> Path:
        """Write the gradient directions as an FSL ``3 x N`` ``.bvec`` next to *nii_path*."""
        name = nii_path.name
        stem = name[: -len(".nii.gz")] if name.endswith(".nii.gz") else name[: -len(".nii")]
        bvec_path = nii_path.with_name(stem + ".bvec")
        np.savetxt(bvec_path, self._config.bvecs.T, fmt="%.8f")
        return bvec_path

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> DWIConfig:
        """The loaded :class:`~mrphantom.dwi.config.DWIConfig`.

        Raises:
            RuntimeError: If :meth:`load_case` has not been called.
        """
        if self._config is None:
            raise RuntimeError(
                f"{type(self).__name__}.config accessed before load_case()."
            )
        return self._config

    @property
    def signal(self) -> np.ndarray:
        """Synthesized 4-D signal array, shape ``(X, Y, Z, n_bvalues)``.

        Raises:
            RuntimeError: If :meth:`synthesize` has not been called.
        """
        if self._signal is None:
            raise RuntimeError(
                f"{type(self).__name__}.signal accessed before synthesize()."
            )
        return self._signal

    @property
    def signal_gt(self) -> np.ndarray:
        """Noise-free ground-truth signal, shape ``(X, Y, Z, n_bvalues)``.

        Always identical to :attr:`signal` when the phantom was created with
        ``snr=None`` (noise-free).  When noise was added, this array holds the
        clean model output before Rician corruption.

        Raises:
            RuntimeError: If :meth:`synthesize` has not been called.
        """
        if self._signal_gt is None:
            raise RuntimeError(
                f"{type(self).__name__}.signal_gt accessed before synthesize()."
            )
        return self._signal_gt

    @property
    def motion_matrices(self) -> np.ndarray | None:
        """``(n_volumes, 4, 4)`` pull matrices, or ``None`` without motion.

        ``volume(x) = object(M x)`` in the spacing-only mm frame.

        Raises:
            RuntimeError: If :meth:`synthesize` has not been called.
        """
        if self._signal is None:
            raise RuntimeError(
                f"{type(self).__name__}.motion_matrices accessed before synthesize()."
            )
        return self._motion_matrices

    @property
    def signal_moved_gt(self) -> np.ndarray | None:
        """Noise-free moved signal, or ``None`` without motion.

        Raises:
            RuntimeError: If :meth:`synthesize` has not been called.
        """
        if self._signal is None:
            raise RuntimeError(
                f"{type(self).__name__}.signal_moved_gt accessed before synthesize()."
            )
        return self._signal_moved_gt

    @property
    def signal_img(self) -> nib.Nifti1Image:
        """Synthesized signal as a :class:`nibabel.Nifti1Image`.

        Raises:
            RuntimeError: If :meth:`synthesize` has not been called.
        """
        if self._signal_img is None:
            raise RuntimeError(
                f"{type(self).__name__}.signal_img accessed before synthesize()."
            )
        return self._signal_img
