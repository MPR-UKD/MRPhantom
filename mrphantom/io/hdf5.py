"""HDF5 export and import for synthesized phantom data.

Schema
------
The HDF5 file written by :func:`save_hdf5` has the following layout::

    /signal          float64  (X, Y, Z, n_bvalues)  — gzip-compressed
    /bvalues         float64  (n_bvalues,)
    /segmentation    int32    (X, Y, Z)              — gzip-compressed
    /affine          float64  (4, 4)

    Attributes on the root group:
        case_id      str
        name         str
        description  str
        snr          float  (absent when noise-free)

:func:`load_hdf5` reads the file back and returns a :class:`PhantomData`
dataclass that bundles all arrays and metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np

from ..config import PhantomConfig


# ---------------------------------------------------------------------------
# Return type for load_hdf5
# ---------------------------------------------------------------------------


@dataclass
class PhantomData:
    """All arrays and metadata loaded from a saved phantom HDF5 file.

    Attributes:
        signal:       4-D signal array, shape ``(X, Y, Z, n_bvalues)``.
        bvalues:      1-D b-value array, shape ``(n_bvalues,)``.
        segmentation: 3-D integer segmentation array, shape ``(X, Y, Z)``.
        affine:       4×4 NIfTI affine matrix.
        case_id:      Case identifier string.
        name:         Short phantom name.
        description:  Human-readable description.
        snr:          SNR used during synthesis, or ``None`` if noise-free.
    """

    signal: np.ndarray
    bvalues: np.ndarray
    segmentation: np.ndarray
    affine: np.ndarray
    case_id: str
    name: str
    description: str
    snr: float | None

    def as_nifti(self) -> nib.Nifti1Image:
        """Reconstruct a :class:`nibabel.Nifti1Image` from the stored arrays.

        Returns:
            NIfTI image wrapping :attr:`signal` with :attr:`affine`.
        """
        return nib.Nifti1Image(self.signal, self.affine)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def save_hdf5(
    signal: np.ndarray,
    signal_img: nib.Nifti1Image,
    config: PhantomConfig,
    out_path: Path | str,
    snr: float | None = None,
    compression: str | None = "gzip",
) -> Path:
    """Save a synthesized phantom to an HDF5 file.

    The segmentation is re-loaded from *config.nifti_path* so that the integer
    label array is always stored alongside the signal — even when the NIfTI
    image passed in is already the signal (not the segmentation).

    Args:
        signal:      4-D ``float64`` signal array ``(X, Y, Z, n_bvalues)``
                     from :func:`~mrphantom.synthesis_dwi.synthesize_phantom`.
        signal_img:  Corresponding :class:`nibabel.Nifti1Image` (used for the
                     affine matrix).
        config:      :class:`~mrphantom.config.PhantomConfig` used to synthesize
                     *signal* (provides metadata and segmentation path).
        out_path:    Destination file path.  The suffix ``.h5`` is appended
                     automatically if the path does not end in ``.h5`` or
                     ``.hdf5``.
        snr:         SNR value stored as metadata attribute.  Pass ``None``
                     for noise-free phantoms.
        compression: HDF5 compression filter for large datasets (default
                     ``"gzip"``).  Pass ``None`` to disable.

    Returns:
        The resolved :class:`Path` of the written file.

    Raises:
        FileNotFoundError: If *config.nifti_path* does not exist.
    """
    if not config.nifti_path.is_file():
        raise FileNotFoundError(f"Segmentation NIfTI not found: {config.nifti_path}")

    out_path = Path(out_path)
    suffixes = "".join(out_path.suffixes).lower()
    if not (suffixes.endswith(".h5") or suffixes.endswith(".hdf5")):
        out_path = out_path.with_suffix(".h5")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load segmentation for storage
    seg_img: nib.Nifti1Image = nib.load(config.nifti_path)
    seg_raw = np.round(seg_img.get_fdata()).astype(np.int32)
    seg_data: np.ndarray = seg_raw[..., 0] if seg_raw.ndim == 4 else seg_raw

    with h5py.File(out_path, "w") as f:
        f.create_dataset(
            "signal", data=signal.astype(np.float64), compression=compression
        )
        f.create_dataset("bvalues", data=config.bvalues.astype(np.float64))
        f.create_dataset("segmentation", data=seg_data, compression=compression)
        f.create_dataset("affine", data=signal_img.affine.astype(np.float64))

        # Root-level metadata
        f.attrs["case_id"] = config.case_id
        f.attrs["name"] = config.name
        f.attrs["description"] = config.description
        if snr is not None:
            f.attrs["snr"] = float(snr)

    return out_path


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def load_hdf5(path: Path | str) -> PhantomData:
    """Load a phantom HDF5 file written by :func:`save_hdf5`.

    Args:
        path: Path to the ``.h5`` / ``.hdf5`` file.

    Returns:
        :class:`PhantomData` with all arrays and metadata populated.

    Raises:
        FileNotFoundError: If *path* does not exist.
        KeyError: If a required dataset or attribute is missing.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {path}")

    with h5py.File(path, "r") as f:
        signal: np.ndarray = f["signal"][()]  # type: ignore[index]
        bvalues: np.ndarray = f["bvalues"][()]  # type: ignore[index]
        segmentation: np.ndarray = f["segmentation"][()]  # type: ignore[index]
        affine: np.ndarray = f["affine"][()]  # type: ignore[index]

        case_id: str = str(f.attrs["case_id"])
        name: str = str(f.attrs["name"])
        description: str = str(f.attrs["description"])
        snr: float | None = float(f.attrs["snr"]) if "snr" in f.attrs else None

    return PhantomData(
        signal=signal,
        bvalues=bvalues,
        segmentation=segmentation,
        affine=affine,
        case_id=case_id,
        name=name,
        description=description,
        snr=snr,
    )
