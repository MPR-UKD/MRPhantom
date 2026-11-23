"""Block-average downsampling of label and probability maps.

Used to bring a high-resolution segmentation onto a coarser voxel grid
*before* signal synthesis.  Probabilities are averaged over integer blocks of
voxels, so a coarse voxel on a tissue boundary receives fractional
probabilities (partial volume) instead of a hard label.
"""

from __future__ import annotations

import numpy as np


def block_factors(
    zooms: tuple[float, ...] | np.ndarray,
    voxel_size: float | tuple[float, ...] | np.ndarray,
    tol: float = 0.05,
) -> np.ndarray:
    """Integer downsampling factors from the source to the target voxel size.

    Args:
        zooms: Source voxel size per axis in mm (only the first three are used).
        voxel_size: Target voxel size in mm, either a scalar (isotropic) or one
            value per axis.
        tol: Maximum allowed deviation of ``voxel_size / zooms`` from an
            integer.

    Returns:
        ``int`` array of shape ``(3,)`` with factors ``>= 1``.

    Raises:
        ValueError: If a target size is smaller than the source (upsampling is
            not supported) or is not an integer multiple of the source size.
    """
    src = np.asarray(zooms, dtype=np.float64)[:3]
    target = np.broadcast_to(np.asarray(voxel_size, dtype=np.float64), (3,))
    ratio = target / src
    factors = np.rint(ratio).astype(int)
    if np.any(factors < 1):
        raise ValueError(
            f"Target voxel size {target.tolist()} is smaller than the source "
            f"voxel size {src.tolist()}; upsampling is not supported."
        )
    if np.any(np.abs(ratio - factors) > tol):
        raise ValueError(
            f"Target voxel size {target.tolist()} is not an integer multiple of "
            f"the source voxel size {src.tolist()}."
        )
    return factors


def block_average(volume: np.ndarray, factors: np.ndarray | tuple[int, ...]) -> np.ndarray:
    """Average a 3-D volume over non-overlapping blocks.

    The volume is zero-padded at the far edges to a multiple of *factors*, so
    no voxels are dropped (zero is background for probabilities).

    Args:
        volume: 3-D array.
        factors: Integer block size per axis.

    Returns:
        ``float32`` array of shape ``ceil(shape / factors)``.
    """
    fx, fy, fz = (int(f) for f in factors)
    if volume.ndim != 3:
        raise ValueError(f"block_average expects a 3-D array, got {volume.ndim}-D.")
    pad = [(0, (-s) % f) for s, f in zip(volume.shape, (fx, fy, fz))]
    vol = np.pad(volume.astype(np.float32, copy=False), pad)
    nx, ny, nz = (s // f for s, f in zip(vol.shape, (fx, fy, fz)))
    blocks = vol.reshape(nx, fx, ny, fy, nz, fz)
    return blocks.mean(axis=(1, 3, 5), dtype=np.float32)


def resample_affine(affine: np.ndarray, factors: np.ndarray | tuple[int, ...]) -> np.ndarray:
    """Affine of the block-averaged grid.

    New voxel centres sit at the centres of the source blocks.

    Args:
        affine: Source 4×4 voxel→world affine.
        factors: Integer block size per axis.

    Returns:
        New 4×4 affine.
    """
    f = np.asarray(factors, dtype=np.float64)
    out = np.array(affine, dtype=np.float64, copy=True)
    out[:3, :3] = affine[:3, :3] * f[None, :]
    out[:3, 3] = affine[:3, 3] + affine[:3, :3] @ ((f - 1.0) / 2.0)
    return out
