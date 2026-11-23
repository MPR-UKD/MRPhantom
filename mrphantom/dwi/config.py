"""DWI phantom data model and JSON config loader.

Parses the phantom JSON file (one region per NIfTI segment label) and loads the
shared b-values file into a :class:`DWIConfig` ready for signal synthesis.

Null regions — those with ``S0: null`` or empty ``D`` / ``fractions`` lists —
automatically inherit all diffusion parameters from the region named
``"remaining_tissue"``.
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
        probability_file: Optional NIfTI with this region's per-voxel
                      probability (0–1, partial volume).  ``None`` uses the
                      hard label map.
        S0_cv:        Coefficient of variation of the smooth random texture
                      applied to S0 (``0`` = off).
        D_cv:         Same for the diffusivities (``0`` = off).
    """

    name: str
    index: int
    n_components: int
    S0: float
    fractions: list[float]
    D: list[float]
    probability_file: Path | None = None
    S0_cv: float = 0.0
    D_cv: float = 0.0


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
        texture_sigma: Smoothing width (voxels of the synthesised grid) of the
                     texture fields used when a region sets ``S0_cv`` / ``D_cv``.
        voxel_size:  Optional target voxel size in mm (scalar or per axis).
                     Label / probability maps are block-averaged to it before
                     synthesis; ``None`` keeps the native grid.
    """

    case_id: str
    name: str
    description: str
    nifti_path: Path
    image_dims: list[int]
    bvalues: np.ndarray
    regions: dict[int, RegionConfig] = field(default_factory=dict)
    texture_sigma: float = 2.0
    voxel_size: float | tuple[float, ...] | None = None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_dwi_config(json_path: Path, bval_path: Path) -> DWIConfig:
    """Parse a phantom JSON file and load b-values into a :class:`DWIConfig`.

    Regions whose ``S0`` is ``null`` or whose ``D`` / ``fractions`` lists are
    empty inherit all diffusion parameters from the region named
    ``"remaining_tissue"``.

    Args:
        json_path: Path to the phantom JSON file.
        bval_path: Path to the b-values text file (space- or line-separated).

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

        is_null = (s0_raw is None) or (len(d_raw) == 0)

        prob_raw = region_data.get("probability")
        prob_file = (json_path.parent / prob_raw) if prob_raw else None
        s0_cv = float(diff.get("S0_cv", 0.0) or 0.0)
        d_cv = float(diff.get("D_cv", 0.0) or 0.0)

        if is_null:
            region = RegionConfig(
                name=region_data["name"],
                index=region_data["index"],
                n_components=0,
                S0=0.0,
                fractions=[],
                D=[],
                probability_file=prob_file,
                S0_cv=s0_cv,
                D_cv=d_cv,
            )
            null_indices.append(region_data["index"])
        else:
            region = RegionConfig(
                name=region_data["name"],
                index=region_data["index"],
                n_components=int(n_comp_raw),
                S0=float(s0_raw),
                fractions=[float(f) for f in fractions_raw],
                D=[float(d) for d in d_raw],
                probability_file=prob_file,
                S0_cv=s0_cv,
                D_cv=d_cv,
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
                probability_file=region.probability_file,
                S0_cv=region.S0_cv or remaining.S0_cv,
                D_cv=region.D_cv or remaining.D_cv,
            )

        if region.S0_cv < 0 or region.D_cv < 0:
            raise ValueError(
                f"Region {region.name!r} has a negative S0_cv / D_cv."
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
        texture_sigma=float(data.get("texture", {}).get("sigma", 2.0)),
        voxel_size=(
            tuple(vs) if isinstance(vs := data.get("resample", {}).get("voxel_size"), list) else vs
        ),
    )
