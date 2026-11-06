"""Tests for mrphantom.motion (matrix convention, anisotropic voxels, reproducibility)."""

import numpy as np
import pytest

pytest.importorskip("scipy")

from mrphantom.motion import (  # noqa: E402
    MotionConfig,
    apply_motion,
    rotation_matrix,
    sample_motion,
)

VOXEL = (1.0, 2.0, 3.0)
SHAPE = (24, 20, 16)
CENTER = (np.asarray(SHAPE) - 1) / 2 * np.asarray(VOXEL)


def blob_at(points_mm):
    """Analytic anisotropic Gaussian, off-center so rotations and shifts both change it."""
    d = points_mm - (CENTER + [2.0, -3.0, 1.0])
    return np.exp(-((d / [9.0, 6.0, 12.0]) ** 2).sum(-1) / 2)


INNER = (slice(4, -4),) * 3  # away from the zero-filled border


def grid_mm():
    return np.stack(np.indices(SHAPE), -1).astype(float) * np.asarray(VOXEL)


def blob():
    return blob_at(grid_mm())


def analytic(matrix):
    """moved(x) = object(M x) evaluated on the grid, from the analytic blob."""
    return blob_at(grid_mm() @ matrix[:3, :3].T + matrix[:3, 3])


class TestSampleMotion:
    def test_reference_is_identity(self):
        m = sample_motion(5, MotionConfig(2.0, 5.0, reference_volume=2, seed=1), VOXEL)
        assert np.array_equal(m[2], np.eye(4))
        assert not np.allclose(m[0], np.eye(4))

    def test_seeded_and_reproducible(self):
        cfg = MotionConfig(2.0, 5.0, seed=7)
        assert np.array_equal(sample_motion(4, cfg, VOXEL), sample_motion(4, cfg, VOXEL))
        other = sample_motion(4, MotionConfig(2.0, 5.0, seed=8), VOXEL)
        assert not np.array_equal(sample_motion(4, cfg, VOXEL), other)

    def test_zero_rotation_is_pure_translation(self):
        m = sample_motion(6, MotionConfig(2.0, 0.0, seed=3), VOXEL, center_mm=(5, 5, 5))
        assert np.allclose(m[:, :3, :3], np.eye(3))
        assert (np.abs(m[:, :3, 3]) <= 2.0).all()

    def test_rotation_is_orthonormal_about_center(self):
        center = np.array([10.0, 8.0, 6.0])
        m = sample_motion(4, MotionConfig(0.0, 10.0, seed=4), VOXEL, center_mm=center)
        for mat in m:
            assert np.allclose(mat[:3, :3] @ mat[:3, :3].T, np.eye(3))
            assert np.allclose(mat[:3, :3] @ center + mat[:3, 3], center)

    def test_validation(self):
        with pytest.raises(ValueError):
            MotionConfig(-1.0)
        with pytest.raises(ValueError):
            sample_motion(2, MotionConfig(reference_volume=5), VOXEL)
        with pytest.raises(ValueError):
            sample_motion(2, MotionConfig(), (1.0, 0.0, 1.0))


class TestRotationMatrix:
    def test_hand_computed_90deg_about_z(self):
        r = rotation_matrix([0, 0, np.pi / 2])
        assert np.allclose(r @ [1, 0, 0], [0, 1, 0])


class TestApplyMotion:
    def test_identity(self):
        v = blob()
        assert np.allclose(apply_motion(v, np.eye(4), VOXEL), v)

    def test_integer_shift_exact_and_hand_computed(self):
        """t = (2, 4, 3) mm = (2, 2, 1) voxels, so moved[i] = v[i + (2, 2, 1)]."""
        v = np.random.default_rng(0).random((12, 10, 8))
        m = np.eye(4)
        m[:3, 3] = [2.0, 4.0, 3.0]
        out = apply_motion(v, m, VOXEL)
        assert np.allclose(out[:-2, :-2, :-1], v[2:, 2:, 1:])
        assert (out[-2:] == 0).all()

    def test_subvoxel_shift_matches_analytic(self):
        m = np.eye(4)
        m[:3, 3] = [0.4, -1.1, 1.2]  # different voxel fractions per axis
        out = apply_motion(blob(), m, VOXEL)
        assert np.abs(out - analytic(m))[INNER].max() < 0.03
        assert np.abs(out - blob())[INNER].max() > 0.03  # the shift is visible
        inv = np.eye(4)
        inv[:3, 3] = -m[:3, 3]  # push instead of pull must not match
        assert np.abs(out - analytic(inv))[INNER].max() > 0.03

    def test_rotation_matches_analytic(self):
        rot = rotation_matrix(np.deg2rad([5.0, -8.0, 12.0]))
        m = np.eye(4)
        m[:3, :3] = rot
        m[:3, 3] = CENTER - rot @ CENTER + [0.5, 0.0, -1.0]
        out = apply_motion(blob(), m, VOXEL)
        assert np.abs(out - analytic(m))[INNER].max() < 0.03
        assert np.abs(out - blob())[INNER].max() > 0.03
        assert np.abs(out - analytic(np.linalg.inv(m)))[INNER].max() > 0.03

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            apply_motion(np.zeros((4, 4)), np.eye(4), VOXEL)
        with pytest.raises(ValueError):
            apply_motion(np.zeros((4, 4, 4)), np.eye(3), VOXEL)
