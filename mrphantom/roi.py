"""General region-of-interest mask utilities.

Provides :func:`get_fitting_mask`, a modality-agnostic helper that builds a
boolean 3-D voxel mask from a list of integer NIfTI segment labels.

For DWI-specific kidney evaluation regions (hilum, cortex, medulla) see
:mod:`mrphantom.dwi.roi`.
"""

from __future__ import annotations

import numpy as np


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
