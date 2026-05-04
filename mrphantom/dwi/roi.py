"""Kidney-specific ROI constants for DWI phantom evaluation.

Defines the three evaluation regions used for quantitative accuracy assessment
in the kidney DWI phantom:

- **hilum**   — renal hilum
- **cortex**  — renal cortex
- **medulla** — renal medulla

These are matched by name rather than by hard-coded index so the functions
remain correct even when phantom JSONs with different label numbering are used.
"""

from __future__ import annotations

from .config import DWIConfig

# Names of regions selected for quantitative accuracy evaluation.
EVALUATION_REGION_NAMES: frozenset[str] = frozenset({"hilum", "cortex", "medulla"})


def get_evaluation_segments(config: DWIConfig) -> list[int]:
    """Return NIfTI label indices of the kidney evaluation focus regions.

    Regions are matched by name (see :data:`EVALUATION_REGION_NAMES`).  This is
    name-driven rather than index-hardcoded so the function remains correct even
    if the user provides a phantom JSON with different label numbering.

    Args:
        config: Loaded :class:`~mrphantom.dwi.config.DWIConfig`.

    Returns:
        Sorted list of integer segment indices present in *config* whose name
        appears in :data:`EVALUATION_REGION_NAMES`.
    """
    return sorted(
        region.index
        for region in config.regions.values()
        if region.name in EVALUATION_REGION_NAMES
    )
