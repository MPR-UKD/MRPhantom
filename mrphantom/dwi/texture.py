"""Smooth random fields for intra-region texture."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def smooth_field(
    shape: tuple[int, ...],
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Zero-mean, unit-variance Gaussian-filtered noise field.

    Args:
        shape: Output shape.
        sigma: Gaussian kernel width in voxels (``0`` gives white noise).
        rng: Seeded generator.

    Returns:
        ``float32`` array of *shape*.
    """
    field = rng.standard_normal(shape).astype(np.float32)
    if sigma > 0:
        field = gaussian_filter(field, sigma=sigma, mode="reflect")
    field -= field.mean()
    std = field.std()
    if std > 0:
        field /= std
    return field


def scale_field(shape: tuple[int, ...], cv: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Multiplicative texture ``max(1 + cv * g, 0)`` with ``g`` a unit smooth field.

    Args:
        shape: Output shape.
        cv: Coefficient of variation (relative std) of the texture.
        sigma: Smoothing width in voxels.
        rng: Seeded generator.

    Returns:
        Non-negative ``float32`` array of *shape* with mean ≈ 1.
    """
    return np.clip(1.0 + cv * smooth_field(shape, sigma, rng), 0.0, None)
