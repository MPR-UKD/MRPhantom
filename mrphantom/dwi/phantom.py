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
from .config import DWIConfig, load_dwi_config
from .synthesis import synthesize_phantom


class DWIPhantom(MRPhantom):
    """Diffusion-weighted MR phantom synthesizer.

    Attributes:
        snr: Signal-to-noise ratio for Rician noise. ``None`` (default)
            produces a noise-free phantom.
        seed: Integer random seed for reproducible noise. Ignored when ``snr``
            is ``None``.

    Examples:
        >>> phantom = DWIPhantom(snr=30, seed=42)
        >>> phantom.load_case(config).synthesize().save("out.nii.gz")
    """

    def __init__(self, *, snr: float | None = None, seed: int | None = None) -> None:
        """Initialise with acquisition parameters only; no case data loaded yet."""
        super().__init__(snr=snr, seed=seed)
        self._config: DWIConfig | None = None
        self._signal: np.ndarray | None = None
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
    ) -> "DWIPhantom":
        """Create a :class:`DWIPhantom` and load a case in one step.

        Equivalent to::

            DWIPhantom(snr=snr, seed=seed).load_case(
                load_dwi_config(json_path, bval_path)
            )

        Args:
            json_path: Path to the phantom JSON file.
            bval_path: Path to the b-values text file.
            snr: Signal-to-noise ratio. ``None`` for noise-free.
            seed: Random seed for reproducible noise.

        Returns:
            DWIPhantom: Instance with the case already loaded, ready to call
            :meth:`synthesize`.
        """
        config = load_dwi_config(Path(json_path), Path(bval_path))
        return cls(snr=snr, seed=seed).load_case(config)

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
        self._signal_img = None
        return self

    def synthesize(self) -> "DWIPhantom":
        """Synthesize the DWI signal from the loaded case.

        Evaluates the multi-exponential model for every segmented region and
        optionally adds Rician noise. Results are cached on the instance and
        exposed via :attr:`signal` and :attr:`signal_img`.

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

        self._signal, self._signal_img = synthesize_phantom(
            self._config, snr=self.snr, rng=rng
        )
        return self

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
                written.append(save_nifti(self._signal_img, out_path))
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
                    )
                )
            else:
                raise ValueError(
                    f"Unrecognised output format for {out_path.name!r}. "
                    "Use .nii / .nii.gz (NIfTI) or .h5 / .hdf5 (HDF5)."
                )

        return written

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
