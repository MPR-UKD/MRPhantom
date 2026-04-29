"""Region-of-interest selection for phantom evaluation and fitting.

Evaluation focus (kidney-only):
    - **hilum** (index 1)
    - **cortex** (index 10)
    - **medulla** (index 11)

All fitters receive a reduced mask covering only the evaluation segments.
Liver and other non-kidney regions are excluded from both fitting and
evaluation so that off-target tissue does not influence accuracy metrics.
"""

from __future__ import annotations

import numpy as np

from .config import PhantomConfig

# Names of regions selected for quantitative accuracy evaluation.
EVALUATION_REGION_NAMES: frozenset[str] = frozenset({"hilum", "cortex", "medulla"})


def get_evaluation_segments(config: PhantomConfig) -> list[int]:
    """Return NIfTI label indices of the evaluation focus regions.

    Regions are matched by name (see :data:`EVALUATION_REGION_NAMES`).  This is
    name-driven rather than index-hardcoded so the function remains correct even
    if the user provides a phantom JSON with different label numbering.

    Args:
        config: Loaded :class:`~mrsim.config.PhantomConfig`.

    Returns:
        Sorted list of integer segment indices present in *config* whose name
        appears in :data:`EVALUATION_REGION_NAMES`.
    """
    return sorted(
        region.index
        for region in config.regions.values()
        if region.name in EVALUATION_REGION_NAMES
    )


def get_fitting_mask(
    segmentation: np.ndarray,
    segment_indices: list[int],
) -> np.ndarray:
    """Build a boolean mask selecting the given segment labels.

    Args:
        segmentation:    3-D integer array of NIfTI segment labels, shape
                         ``(X, Y, Z)``.
        segment_indices: Label values to include in the mask.

    Returns:
        Boolean array of the same shape as *segmentation*; ``True`` where the
        voxel belongs to one of the requested segments.
    """
    mask = np.zeros(segmentation.shape, dtype=bool)
    for idx in segment_indices:
        mask |= segmentation == idx
    return mask
