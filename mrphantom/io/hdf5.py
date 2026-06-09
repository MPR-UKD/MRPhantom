"""HDF5 export and import for synthesized phantom data.

Schema
------
The HDF5 file written by :func:`save_hdf5` has the following layout::

    /signal          float64  (X, Y, Z, n_volumes)  — gzip-compressed
    /signal_gt       float64  (X, Y, Z, n_volumes)  — gzip-compressed
    /volumes         float64  (n_volumes,)
    /segmentation    int32    (X, Y, Z)              — gzip-compressed
    /affine          float64  (4, 4)

    Attributes on the root group:
        case_id      str
        name         str
        description  str
        snr          float   (absent when noise-free)
        snr_region   str     (absent when using max(S0) as reference)

``/signal_gt`` stores the noise-free ground-truth signal.  It is identical to
``/signal`` for noise-free phantoms and is absent in files written by older
versions of the library (``load_hdf5`` will set :attr:`PhantomData.signal_gt`
to ``None`` in that case).

``/volumes`` stores the acquisition-parameter axis (e.g. b-values for DWI,
inversion times for T1, echo times for T2).

:func:`load_hdf5` reads the file back and returns a :class:`PhantomData`
dataclass that bundles all arrays and metadata.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np


# ---------------------------------------------------------------------------
# Return type for load_hdf5
# ---------------------------------------------------------------------------


@dataclass
class PhantomData:
    """All arrays and metadata loaded from a saved phantom HDF5 file.

    Attributes:
        signal:       4-D signal array, shape ``(X, Y, Z, n_volumes)``.
        signal_gt:    4-D noise-free ground-truth array, same shape as
                      :attr:`signal`.  ``None`` when loading a file written
                      by an older version of the library that did not store
                      the ground truth.
        volumes:      1-D acquisition-parameter array, shape ``(n_volumes,)``.
                      Stores b-values for DWI, inversion times for T1, etc.
        segmentation: 3-D integer segmentation array, shape ``(X, Y, Z)``.
        affine:       4×4 NIfTI affine matrix.
        case_id:      Case identifier string.
        name:         Short phantom name.
        description:  Human-readable description.
        snr:          SNR used during synthesis, or ``None`` if noise-free.
        snr_region:   Name of the region whose S0 was used as the SNR noise
                      reference, or ``None`` if ``max(S0)`` was used.
    """

    signal: np.ndarray
    signal_gt: np.ndarray | None
    volumes: np.ndarray
    segmentation: np.ndarray
    affine: np.ndarray
    case_id: str
    name: str
    description: str
    snr: float | None
    snr_region: str | None = None

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
    volumes: np.ndarray,
    segmentation: np.ndarray | Path,
    out_path: Path | str,
    *,
    case_id: str = "",
    name: str = "",
    description: str = "",
    snr: float | None = None,
    snr_region: str | None = None,
    signal_gt: np.ndarray | None = None,
    compression: str | None = "gzip",
) -> Path:
    """Save a synthesized phantom to an HDF5 file.

    Args:
        signal:      4-D ``float64`` signal array ``(X, Y, Z, n_volumes)``.
        signal_img:  Corresponding :class:`nibabel.Nifti1Image` (used for the
                     affine matrix).
        volumes:     1-D acquisition-parameter array ``(n_volumes,)`` — e.g.
                     b-values for DWI or inversion times for T1.
        segmentation: Either a 3-D integer ``ndarray`` of segment labels, or a
                     :class:`~pathlib.Path` to a NIfTI segmentation file that
                     will be loaded and stored alongside the signal.
        out_path:    Destination file path.  The suffix ``.h5`` is appended
                     automatically if the path does not end in ``.h5`` or
                     ``.hdf5``.
        case_id:     Case identifier stored as a root attribute.
        name:        Short phantom name stored as a root attribute.
        description: Human-readable description stored as a root attribute.
        snr:         SNR value stored as a root attribute.  Pass ``None``
                     for noise-free phantoms.
        snr_region:  Name of the region whose S0 was used as the noise
                     reference.  Pass ``None`` when ``max(S0)`` was used.
                     Stored as a root attribute only when not ``None``.
        signal_gt:   4-D ``float64`` noise-free ground-truth array, same shape
                     as *signal*.  Written as ``/signal_gt`` when provided.
                     Pass ``None`` to omit (e.g. when saving noise-free phantoms
                     where signal and ground truth are identical).
        compression: HDF5 compression filter for large datasets (default
                     ``"gzip"``).  Pass ``None`` to disable.

    Returns:
        The resolved :class:`Path` of the written file.

    Raises:
        FileNotFoundError: If *segmentation* is a path and does not exist.
    """
    out_path = Path(out_path)
    suffixes = "".join(out_path.suffixes).lower()
    if not (suffixes.endswith(".h5") or suffixes.endswith(".hdf5")):
        out_path = out_path.with_suffix(".h5")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        warnings.warn(f"Overwriting existing file: {out_path}")

    # Resolve segmentation — accept either an array or a path to a NIfTI file.
    if isinstance(segmentation, (str, Path)):
        seg_path = Path(segmentation)
        if not seg_path.is_file():
            raise FileNotFoundError(f"Segmentation NIfTI not found: {seg_path}")
        seg_img: nib.Nifti1Image = nib.load(seg_path)
        seg_raw = np.round(seg_img.get_fdata()).astype(np.int32)
        seg_data: np.ndarray = seg_raw[..., 0] if seg_raw.ndim == 4 else seg_raw
    else:
        seg_data = np.asarray(segmentation, dtype=np.int32)

    with h5py.File(out_path, "w") as f:
        f.create_dataset(
            "signal", data=signal.astype(np.float64), compression=compression
        )
        if signal_gt is not None:
            f.create_dataset(
                "signal_gt", data=signal_gt.astype(np.float64), compression=compression
            )
        f.create_dataset("volumes", data=np.asarray(volumes, dtype=np.float64))
        f.create_dataset("segmentation", data=seg_data, compression=compression)
        f.create_dataset("affine", data=signal_img.affine.astype(np.float64))

        # Root-level metadata
        f.attrs["case_id"] = case_id
        f.attrs["name"] = name
        f.attrs["description"] = description
        if snr is not None:
            f.attrs["snr"] = float(snr)
        if snr_region is not None:
            f.attrs["snr_region"] = snr_region

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
        signal_gt: np.ndarray | None = (
            f["signal_gt"][()] if "signal_gt" in f else None  # type: ignore[index]
        )
        volumes: np.ndarray = f["volumes"][()]  # type: ignore[index]
        segmentation: np.ndarray = f["segmentation"][()]  # type: ignore[index]
        affine: np.ndarray = f["affine"][()]  # type: ignore[index]

        case_id: str = str(f.attrs["case_id"])
        name: str = str(f.attrs["name"])
        description: str = str(f.attrs["description"])
        snr: float | None = float(f.attrs["snr"]) if "snr" in f.attrs else None
        snr_region: str | None = (
            str(f.attrs["snr_region"]) if "snr_region" in f.attrs else None
        )

    return PhantomData(
        signal=signal,
        signal_gt=signal_gt,
        volumes=volumes,
        segmentation=segmentation,
        affine=affine,
        case_id=case_id,
        name=name,
        description=description,
        snr=snr,
        snr_region=snr_region,
    )
