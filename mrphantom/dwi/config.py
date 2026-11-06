"""DWI phantom data model and JSON config loader.

Parses the phantom JSON file (one region per NIfTI segment label) and loads the
shared b-values file into a :class:`DWIConfig` ready for signal synthesis.

Null regions — those with ``S0: null`` or empty ``D`` / ``fractions`` lists —
automatically inherit all diffusion parameters from the region named
``"remaining_tissue"``.

A region may instead carry a single-compartment diffusion *tensor*
(``diffusivity.tensor = {"evals": [...], "evecs": [[...], ...]}``).  Tensor
regions need gradient directions (``bvecs``) to be synthesized.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Internal b-value loader — no external dependencies
# ---------------------------------------------------------------------------

def _load_bvalues(bval_path: Path) -> np.ndarray:
    """Load b-values from a whitespace- or newline-separated text file.

    Args:
        bval_path: Path to the b-values file.

    Returns:
        1-D ``float64`` array of b-values in s/mm².

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains no numeric values or negative values.
    """
    values = np.loadtxt(bval_path, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError(f"B-values file is empty: {bval_path}")
    if np.any(values < 0):
        raise ValueError(
            f"B-values must be non-negative, got negative values in {bval_path}"
        )
    return values


def _validate_bvecs(bvecs: np.ndarray) -> np.ndarray:
    """Check that every gradient direction is a unit vector or zero."""
    arr = np.asarray(bvecs, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"bvecs must have shape (n, 3), got {arr.shape}.")
    norms = np.linalg.norm(arr, axis=1)
    ok = np.isclose(norms, 1.0, atol=1e-3) | np.isclose(norms, 0.0, atol=1e-8)
    if not np.all(ok):
        raise ValueError(
            "Every b-vector must be a unit vector (or zero for b=0); "
            f"offending norms: {norms[~ok].tolist()}."
        )
    return arr


def _load_bvecs(bvec_path: Path, n_volumes: int) -> np.ndarray:
    """Load gradient directions from a text file (FSL ``3 x N`` or ``N x 3``).

    Args:
        bvec_path:  Path to the bvec file.
        n_volumes:  Expected number of volumes (length of the b-values).

    Returns:
        ``(n_volumes, 3)`` ``float64`` array.  Each row is a unit vector, or the
        zero vector for a b=0 volume.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the shape does not match *n_volumes* or a row is neither
            a unit vector nor zero.
    """
    if not bvec_path.is_file():
        raise FileNotFoundError(f"B-vectors file not found: {bvec_path}")
    arr = np.atleast_2d(np.loadtxt(bvec_path, dtype=np.float64))
    # FSL stores 3 rows x N columns; an N x 3 file is accepted too.
    if arr.shape == (3, n_volumes):
        arr = arr.T
    elif arr.shape != (n_volumes, 3):
        raise ValueError(
            f"B-vectors in {bvec_path} have shape {arr.shape}; expected "
            f"(3, {n_volumes}) or ({n_volumes}, 3)."
        )
    return _validate_bvecs(arr)


def _tensor_from_spec(spec: dict, region_name: str) -> np.ndarray:
    """Build the symmetric 3x3 diffusion tensor from ``{"evals", "evecs"}``.

    ``evecs[i]`` is the eigenvector belonging to ``evals[i]`` (one per row).

    Raises:
        ValueError: If the shapes are wrong, an eigenvalue is negative or the
            eigenvectors are not orthonormal.
    """
    evals = np.asarray(spec["evals"], dtype=np.float64)
    evecs = np.asarray(spec["evecs"], dtype=np.float64)
    if evals.shape != (3,) or evecs.shape != (3, 3):
        raise ValueError(
            f"Region {region_name!r}: tensor needs 3 evals and a 3x3 evecs "
            f"matrix, got {evals.shape} and {evecs.shape}."
        )
    if np.any(evals < 0):
        raise ValueError(f"Region {region_name!r} has negative tensor eigenvalues.")
    if not np.allclose(evecs @ evecs.T, np.eye(3), atol=1e-3):
        raise ValueError(f"Region {region_name!r}: tensor evecs must be orthonormal.")
    return (evecs.T * evals) @ evecs


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RegionConfig:
    """Diffusion parameters for one segmented region.

    Attributes:
        name:         Human-readable region name (e.g. ``"cortex"``).
        index:        Integer label in the NIfTI segmentation array.
        n_components: Number of diffusion compartments.
        S0:           Baseline signal intensity.
        fractions:    Compartment signal fractions; must sum to 1.
        D:            Compartment diffusivities in mm²/s; length == n_components.
        tensor:       Optional symmetric ``(3, 3)`` diffusion tensor in mm²/s.
                      When set the region is a single compartment
                      (``n_components == 1``, ``fractions == [1.0]``) and ``D``
                      holds its mean diffusivity.  Requires
                      :attr:`DWIConfig.bvecs` for synthesis.
    """

    name: str
    index: int
    n_components: int
    S0: float
    fractions: list[float]
    D: list[float]
    tensor: np.ndarray | None = None


@dataclass
class DWIConfig:
    """Complete description of a synthetic DWI phantom case.

    Attributes:
        case_id:     Unique case identifier (from JSON ``case_id`` field).
        name:        Short name (from JSON ``name`` field).
        description: Human-readable description.
        nifti_path:  Absolute path to the segmentation NIfTI file.
        image_dims:  Spatial dimensions ``[X, Y, Z]`` reported in the JSON.
        bvalues:     1-D array of b-values in s/mm².
        regions:     Map from NIfTI integer label → :class:`RegionConfig`.
        bvecs:       Optional ``(n_bvalues, 3)`` gradient directions (unit
                     vectors, zero for b=0).  ``None`` for direction-free cases.
    """

    case_id: str
    name: str
    description: str
    nifti_path: Path
    image_dims: list[int]
    bvalues: np.ndarray
    regions: dict[int, RegionConfig] = field(default_factory=dict)
    bvecs: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_dwi_config(
    json_path: Path, bval_path: Path, bvec_path: Path | None = None
) -> DWIConfig:
    """Parse a phantom JSON file and load b-values into a :class:`DWIConfig`.

    Regions whose ``S0`` is ``null`` or whose ``D`` / ``fractions`` lists are
    empty inherit all diffusion parameters from the region named
    ``"remaining_tissue"``.

    Args:
        json_path: Path to the phantom JSON file.
        bval_path: Path to the b-values text file (space- or line-separated).
        bvec_path: Optional path to a bvec file (FSL ``3 x N`` or ``N x 3``).
            Required to synthesize regions that define a ``tensor``.

    Returns:
        Fully resolved :class:`DWIConfig`.

    Raises:
        FileNotFoundError: If either input file does not exist.
        KeyError: If the JSON is missing required fields.
        ValueError: If the JSON contains no ``"remaining_tissue"`` region but
            null regions are present, or if the b-values file is empty.
    """
    if not json_path.is_file():
        raise FileNotFoundError(f"Phantom JSON not found: {json_path}")
    if not bval_path.is_file():
        raise FileNotFoundError(f"B-values file not found: {bval_path}")

    with open(json_path, encoding="utf-8") as fh:
        data: dict = json.load(fh)

    bvalues: np.ndarray = _load_bvalues(bval_path)
    bvecs: np.ndarray | None = None
    if bvec_path is not None:
        bvecs = _load_bvecs(Path(bvec_path), bvalues.size)

    nifti_path = json_path.parent / data["file"]

    # ------------------------------------------------------------------
    # First pass: build all RegionConfig objects (null regions deferred)
    # ------------------------------------------------------------------
    raw_regions: list[RegionConfig] = []
    null_indices: list[int] = []

    for region_data in data["regions"].values():
        diff = region_data["diffusivity"]
        s0_raw = diff.get("S0")
        d_raw: list = diff.get("D", [])
        fractions_raw: list = diff.get("fractions", [])
        n_comp_raw = diff.get("n_components")

        tensor_raw = diff.get("tensor")

        is_null = (s0_raw is None) or (len(d_raw) == 0 and tensor_raw is None)

        if is_null:
            region = RegionConfig(
                name=region_data["name"],
                index=region_data["index"],
                n_components=0,
                S0=0.0,
                fractions=[],
                D=[],
            )
            null_indices.append(region_data["index"])
        elif tensor_raw is not None:
            if len(d_raw) > 0:
                raise ValueError(
                    f"Region {region_data['name']!r} defines both 'tensor' and "
                    "'D'; use one of them."
                )
            tensor = _tensor_from_spec(tensor_raw, region_data["name"])
            region = RegionConfig(
                name=region_data["name"],
                index=region_data["index"],
                n_components=1,
                S0=float(s0_raw),
                fractions=[1.0],
                D=[float(np.trace(tensor) / 3.0)],
                tensor=tensor,
            )
        else:
            region = RegionConfig(
                name=region_data["name"],
                index=region_data["index"],
                n_components=int(n_comp_raw),
                S0=float(s0_raw),
                fractions=[float(f) for f in fractions_raw],
                D=[float(d) for d in d_raw],
            )

        raw_regions.append(region)

    # ------------------------------------------------------------------
    # Find remaining_tissue for null-region fallback
    # ------------------------------------------------------------------
    remaining: RegionConfig | None = next(
        (r for r in raw_regions if r.name == "remaining_tissue"), None
    )

    if null_indices and remaining is None:
        raise ValueError(
            "Phantom JSON has null regions but no 'remaining_tissue' region to "
            "inherit from."
        )

    # ------------------------------------------------------------------
    # Second pass: resolve null regions
    # ------------------------------------------------------------------
    regions: dict[int, RegionConfig] = {}
    seen_names: set[str] = set()
    for region in raw_regions:
        if region.index in null_indices and remaining is not None:
            region = RegionConfig(
                name=region.name,
                index=region.index,
                n_components=remaining.n_components,
                S0=remaining.S0,
                fractions=list(remaining.fractions),
                D=list(remaining.D),
                tensor=None if remaining.tensor is None else remaining.tensor.copy(),
            )

        if region.S0 < 0:
            raise ValueError(
                f"Region {region.name!r} has negative S0 ({region.S0})."
            )
        if any(d < 0 for d in region.D):
            raise ValueError(
                f"Region {region.name!r} has negative diffusivity values."
            )
        if not np.isclose(sum(region.fractions), 1.0):
            raise ValueError(
                f"Region {region.name!r} fractions sum to "
                f"{sum(region.fractions):.4f}, expected 1.0."
            )
        if region.name in seen_names:
            raise ValueError(
                f"Duplicate region name {region.name!r}. "
                "Region names must be unique."
            )
        seen_names.add(region.name)
        regions[region.index] = region

    if not regions:
        raise ValueError(
            "Phantom JSON must contain at least one region "
            "(non-null after null-region resolution)."
        )

    return DWIConfig(
        case_id=data["case_id"],
        name=data["name"],
        description=data["description"],
        nifti_path=nifti_path,
        image_dims=list(data["image"]["dimensions"]),
        bvalues=bvalues,
        regions=regions,
        bvecs=bvecs,
    )
