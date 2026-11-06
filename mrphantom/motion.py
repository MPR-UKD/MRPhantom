"""Rigid motion for phantom volumes: sampling of pull matrices and resampling.

Convention (identical to dwipipe): a motion is a 4x4 **pull matrix** ``M`` in the spacing-only
mm frame, where voxel index ``(i, j, k)`` is the point ``(i*sx, j*sy, k*sz)``. The moved volume
is ``moved(x) = volume(M @ x)``. Resampling needs scipy (optional extra ``motion``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = ["MotionConfig", "sample_motion", "apply_motion", "rotation_matrix"]


@dataclass(frozen=True)
class MotionConfig:
    """Rigid motion limits.

    Attributes:
        max_shift_mm: Maximum absolute shift per axis in mm (uniform in ``[-m, m]``).
        max_rotation_deg: Maximum absolute rotation per axis in degrees. ``0`` gives pure shifts.
        reference_volume: Index of the volume that stays at identity.
        seed: Seed of the random generator.
    """

    max_shift_mm: float = 0.0
    max_rotation_deg: float = 0.0
    reference_volume: int = 0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_shift_mm < 0 or self.max_rotation_deg < 0:
            raise ValueError("max_shift_mm and max_rotation_deg must be >= 0")


def rotation_matrix(rotvec_rad: Sequence[float]) -> np.ndarray:
    """Rotation matrix (3x3) from a rotation vector in radians (Rodrigues)."""
    v = np.asarray(rotvec_rad, dtype=float)
    theta = np.linalg.norm(v)
    if theta < 1e-12:
        return np.eye(3)
    k = v / theta
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * kx + (1 - np.cos(theta)) * (kx @ kx)


def sample_motion(
    n_volumes: int,
    cfg: MotionConfig,
    voxel_size: Sequence[float],
    center_mm: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Draw one rigid pull matrix per volume.

    Args:
        n_volumes: Number of volumes.
        cfg: Motion limits and seed.
        voxel_size: Voxel spacing in mm (length 3); validated, the matrices are in mm.
        center_mm: Point (mm) the rotation is about; default is the origin of the mm frame.

    Returns:
        Array ``(n_volumes, 4, 4)``; the reference volume is the identity.

    Raises:
        ValueError: If ``voxel_size`` is invalid or ``reference_volume`` is out of range.
    """
    if len(voxel_size) != 3 or min(voxel_size) <= 0:
        raise ValueError("voxel_size must be three positive values")
    if not 0 <= cfg.reference_volume < n_volumes:
        raise ValueError("reference_volume out of range")
    rng = np.random.default_rng(cfg.seed)
    center = np.zeros(3) if center_mm is None else np.asarray(center_mm, dtype=float)
    out = np.tile(np.eye(4), (n_volumes, 1, 1))
    for n in range(n_volumes):
        # draw for every volume so the stream does not depend on reference_volume
        shift = rng.uniform(-1, 1, 3) * cfg.max_shift_mm
        rot = rotation_matrix(np.deg2rad(rng.uniform(-1, 1, 3) * cfg.max_rotation_deg))
        if n == cfg.reference_volume:
            continue
        out[n, :3, :3] = rot
        out[n, :3, 3] = center - rot @ center + shift
    return out


def apply_motion(
    volume: np.ndarray,
    matrix: np.ndarray,
    voxel_size: Sequence[float],
    order: int = 1,
) -> np.ndarray:
    """Resample a 3D volume so that ``moved(x) = volume(matrix @ x)``; zero outside the view.

    Raises:
        ImportError: If scipy is not installed.
        ValueError: If ``volume`` is not 3D or ``matrix`` is not 4x4.
    """
    try:
        from scipy.ndimage import affine_transform
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "mrphantom motion needs scipy; install with `pip install mrphantom[motion]`"
        ) from exc
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError("volume must be 3D")
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError("matrix must be 4x4")
    s = np.asarray(voxel_size, dtype=float)
    # index space: i_in = S^-1 (R S i + t)
    a = matrix[:3, :3] * s[None, :] / s[:, None]
    offset = matrix[:3, 3] / s
    return affine_transform(volume, a, offset=offset, order=order, mode="constant", cval=0.0)
