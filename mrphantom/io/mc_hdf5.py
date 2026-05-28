"""HDF5 export and import for Monte Carlo simulation results.

Schema
------
The HDF5 file written by :func:`save_mc_hdf5` has the following layout::

    /signal          float64  (X, Y, Z, n_b, N)              — gzip
    /signal_gt       float64  (X, Y, Z, n_b, N)              — gzip
    /volumes         float64  (n_b,)
    /segmentation    int32    (X, Y, Z)                       — gzip
    /affine          float64  (4, 4)

    /sampled_params/
        S0           float64  (n_regions, N)                  — gzip
        D            float64  (n_regions, max_n_components, N) — gzip
        fractions    float64  (n_regions, max_n_components, N) — gzip
        region_names variable-length UTF-8 string (n_regions,)

    Root attributes:
        case_id      str
        name         str
        description  str
        n_samples    int
        snr          float   (absent when noise-free)
        snr_region   str     (absent when None)
        seed         int     (absent when None)

:func:`load_mc_hdf5` reads the file back into an :class:`MCPhantomData`
dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np


# ---------------------------------------------------------------------------
# Return type for load_mc_hdf5
# ---------------------------------------------------------------------------


@dataclass
class MCPhantomData:
    """All arrays and metadata loaded from a Monte Carlo HDF5 file.

    Attributes:
        signal:             5-D noisy signal, shape ``(X, Y, Z, n_b, N)``.
        signal_gt:          5-D noise-free GT signal, same shape as
                            :attr:`signal`.
        volumes:            1-D acquisition-parameter array ``(n_b,)``
                            (e.g. b-values for DWI).
        segmentation:       3-D integer segmentation array ``(X, Y, Z)``.
        affine:             4×4 NIfTI affine matrix.
        case_id:            Case identifier string.
        name:               Short phantom name.
        description:        Human-readable description.
        n_samples:          Number of Monte Carlo draws stored.
        snr:                SNR used during synthesis, or ``None``.
        snr_region:         Name of the SNR reference region, or ``None``.
        seed:               Master seed used, or ``None``.
        S0_samples:         Sampled S0 values, shape ``(n_regions, N)``.
        D_samples:          Sampled diffusivities,
                            shape ``(n_regions, max_comp, N)``.
        fractions_samples:  Sampled fractions,
                            shape ``(n_regions, max_comp, N)``.
        region_names:       Ordered list of region names, length
                            ``n_regions``.
    """

    signal: np.ndarray              # (X, Y, Z, n_b, N)
    signal_gt: np.ndarray           # (X, Y, Z, n_b, N)
    volumes: np.ndarray             # (n_b,)
    segmentation: np.ndarray        # (X, Y, Z)
    affine: np.ndarray              # (4, 4)
    case_id: str
    name: str
    description: str
    n_samples: int
    snr: float | None
    snr_region: str | None
    seed: int | None
    S0_samples: np.ndarray          # (n_regions, N)
    D_samples: np.ndarray           # (n_regions, max_comp, N)
    fractions_samples: np.ndarray   # (n_regions, max_comp, N)
    region_names: list[str]         # (n_regions,)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def save_mc_hdf5(
    signal: np.ndarray,
    signal_gt: np.ndarray,
    volumes: np.ndarray,
    segmentation: np.ndarray | Path,
    affine: np.ndarray,
    S0_samples: np.ndarray,
    D_samples: np.ndarray,
    fractions_samples: np.ndarray,
    region_names: list[str],
    out_path: Path | str,
    *,
    case_id: str = "",
    name: str = "",
    description: str = "",
    snr: float | None = None,
    snr_region: str | None = None,
    seed: int | None = None,
    compression: str | None = "gzip",
) -> Path:
    """Save Monte Carlo simulation results to an HDF5 file.

    Args:
        signal:             5-D noisy signal ``(X, Y, Z, n_b, N)``.
        signal_gt:          5-D noise-free GT signal, same shape as
                            *signal*.
        volumes:            1-D acquisition-parameter array ``(n_b,)``.
        segmentation:       3-D integer segmentation ``ndarray``, or a
                            :class:`~pathlib.Path` to a NIfTI segmentation
                            file that will be loaded.
        affine:             4×4 NIfTI affine matrix.
        S0_samples:         Sampled S0 values, shape ``(n_regions, N)``.
        D_samples:          Sampled diffusivities,
                            shape ``(n_regions, max_comp, N)``.
        fractions_samples:  Sampled fractions,
                            shape ``(n_regions, max_comp, N)``.
        region_names:       Ordered region names, length ``n_regions``.
        out_path:           Destination file path.  The suffix ``.h5`` is
                            appended automatically if absent.
        case_id:            Case identifier stored as a root attribute.
        name:               Short phantom name.
        description:        Human-readable description.
        snr:                SNR value (root attr).  ``None`` → omitted.
        snr_region:         SNR reference region name.  ``None`` → omitted.
        seed:               Master seed (root attr).  ``None`` → omitted.
        compression:        HDF5 compression filter (default ``"gzip"``).
                            Pass ``None`` to disable.

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

    # Resolve segmentation
    if isinstance(segmentation, (str, Path)):
        seg_path = Path(segmentation)
        if not seg_path.is_file():
            raise FileNotFoundError(f"Segmentation NIfTI not found: {seg_path}")
        seg_img: nib.Nifti1Image = nib.load(seg_path)
        seg_raw = np.round(seg_img.get_fdata()).astype(np.int32)
        seg_data: np.ndarray = seg_raw[..., 0] if seg_raw.ndim == 4 else seg_raw
    else:
        seg_data = np.asarray(segmentation, dtype=np.int32)

    n_samples = signal.shape[-1]

    with h5py.File(out_path, "w") as f:
        f.create_dataset(
            "signal", data=signal.astype(np.float64), compression=compression
        )
        f.create_dataset(
            "signal_gt", data=signal_gt.astype(np.float64), compression=compression
        )
        f.create_dataset("volumes", data=np.asarray(volumes, dtype=np.float64))
        f.create_dataset("segmentation", data=seg_data, compression=compression)
        f.create_dataset("affine", data=np.asarray(affine, dtype=np.float64))

        # Sampled parameters group
        grp = f.create_group("sampled_params")
        grp.create_dataset(
            "S0", data=S0_samples.astype(np.float64), compression=compression
        )
        grp.create_dataset(
            "D", data=D_samples.astype(np.float64), compression=compression
        )
        grp.create_dataset(
            "fractions",
            data=fractions_samples.astype(np.float64),
            compression=compression,
        )
        # Variable-length UTF-8 strings for region names
        vlen_str = h5py.string_dtype(encoding="utf-8")
        grp.create_dataset(
            "region_names",
            data=np.array(region_names, dtype=object),
            dtype=vlen_str,
        )

        # Root-level metadata
        f.attrs["case_id"] = case_id
        f.attrs["name"] = name
        f.attrs["description"] = description
        f.attrs["n_samples"] = n_samples
        if snr is not None:
            f.attrs["snr"] = float(snr)
        if snr_region is not None:
            f.attrs["snr_region"] = snr_region
        if seed is not None:
            f.attrs["seed"] = int(seed)

    return out_path


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def load_mc_hdf5(path: Path | str) -> MCPhantomData:
    """Load a Monte Carlo HDF5 file written by :func:`save_mc_hdf5`.

    Args:
        path: Path to the ``.h5`` / ``.hdf5`` file.

    Returns:
        :class:`MCPhantomData` with all arrays and metadata.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"MC HDF5 file not found: {path}")

    with h5py.File(path, "r") as f:
        signal: np.ndarray = f["signal"][()]  # type: ignore[index]
        signal_gt: np.ndarray = f["signal_gt"][()]  # type: ignore[index]
        volumes: np.ndarray = f["volumes"][()]  # type: ignore[index]
        segmentation: np.ndarray = f["segmentation"][()]  # type: ignore[index]
        affine: np.ndarray = f["affine"][()]  # type: ignore[index]

        grp = f["sampled_params"]
        S0_samples: np.ndarray = grp["S0"][()]  # type: ignore[index]
        D_samples: np.ndarray = grp["D"][()]  # type: ignore[index]
        fractions_samples: np.ndarray = grp["fractions"][()]  # type: ignore[index]
        region_names: list[str] = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in grp["region_names"][()]  # type: ignore[index]
        ]

        case_id: str = str(f.attrs["case_id"])
        name: str = str(f.attrs["name"])
        description: str = str(f.attrs["description"])
        n_samples: int = int(f.attrs["n_samples"])
        snr: float | None = float(f.attrs["snr"]) if "snr" in f.attrs else None
        snr_region: str | None = (
            str(f.attrs["snr_region"]) if "snr_region" in f.attrs else None
        )
        seed: int | None = int(f.attrs["seed"]) if "seed" in f.attrs else None

    return MCPhantomData(
        signal=signal,
        signal_gt=signal_gt,
        volumes=volumes,
        segmentation=segmentation,
        affine=affine,
        case_id=case_id,
        name=name,
        description=description,
        n_samples=n_samples,
        snr=snr,
        snr_region=snr_region,
        seed=seed,
        S0_samples=S0_samples,
        D_samples=D_samples,
        fractions_samples=fractions_samples,
        region_names=region_names,
    )
