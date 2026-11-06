"""Tests for motion in DWI synthesis (Phase 3): ordering, rotation consistency, export."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

pytest.importorskip("scipy")

from mrphantom import DWIPhantom, MCDWIPhantom, MotionConfig  # noqa: E402
from mrphantom.dwi.config import DWIConfig, RegionConfig  # noqa: E402
from mrphantom.dwi.synthesis import synthesize_dwi, synthesize_phantom  # noqa: E402
from mrphantom.io import load_hdf5  # noqa: E402

ZOOMS = (1.0, 1.0, 2.0)
SHAPE = (48, 48, 24)
EVALS = (0.0020, 0.0005, 0.0005)
S0 = 100.0
G_OBLIQUE = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)


def _rod_config(
    tmp_path: Path, bvalues, bvecs, tensor: bool = True, name: str = "seg.nii.gz"
) -> DWIConfig:
    """Rod (long along object x, mm) with a fiber tensor along x; background is label 0."""
    ii = np.indices(SHAPE).astype(float) * np.asarray(ZOOMS)[:, None, None, None]
    c = (np.asarray(SHAPE) - 1) / 2 * np.asarray(ZOOMS)
    d = (ii - c[:, None, None, None]) / np.array([14.0, 5.0, 5.0])[:, None, None, None]
    seg = ((d**2).sum(0) <= 1.0).astype(np.int32)
    affine = np.diag([*ZOOMS, 1.0])
    path = tmp_path / name
    nib.save(nib.Nifti1Image(seg, affine), path)
    region = RegionConfig(
        name="rod",
        index=1,
        n_components=1,
        S0=S0,
        fractions=[1.0],
        D=[float(np.mean(EVALS))],
        tensor=np.diag(EVALS) if tensor else None,
    )
    return DWIConfig(
        case_id="rod",
        name="rod",
        description="rod",
        nifti_path=path,
        image_dims=list(SHAPE),
        bvalues=np.asarray(bvalues, dtype=float),
        regions={1: region},
        bvecs=None if bvecs is None else np.asarray(bvecs, dtype=float),
    )


def _principal_axis(volume: np.ndarray) -> np.ndarray:
    """Long axis (mm frame) of the object in a moved b=0 volume."""
    idx = np.argwhere(volume > S0 / 2) * np.asarray(ZOOMS)
    cov = np.cov((idx - idx.mean(0)).T)
    w, v = np.linalg.eigh(cov)
    return v[:, -1]


@pytest.fixture
def rod_dirs(tmp_path):
    bvals = [0.0, 1000.0, 1000.0, 1000.0]
    g = [[0, 0, 0], G_OBLIQUE, G_OBLIQUE, G_OBLIQUE]
    return bvals, g


# ---------------------------------------------------------------------------
# Old path unchanged
# ---------------------------------------------------------------------------


class TestNoMotion:
    def test_three_tuple_unchanged(self, tmp_path, rod_dirs):
        cfg = _rod_config(tmp_path, *rod_dirs)
        out = synthesize_phantom(cfg, snr=20, rng=np.random.default_rng(3))
        assert len(out) == 3
        res = synthesize_dwi(cfg, snr=20, rng=np.random.default_rng(3))
        assert np.array_equal(out[0], res.signal)
        assert np.array_equal(out[1], res.signal_gt)
        assert res.motion_matrices is None and res.signal_moved_gt is None

    def test_phantom_without_motion_has_no_matrices(self, tmp_path, rod_dirs):
        p = DWIPhantom().load_case(_rod_config(tmp_path, *rod_dirs)).synthesize()
        assert p.motion_matrices is None and p.signal_moved_gt is None
        assert np.array_equal(p.signal, p.signal_gt)


# ---------------------------------------------------------------------------
# Motion
# ---------------------------------------------------------------------------


class TestMotion:
    def test_reference_volume_unchanged_signal_gt_is_reference_frame(self, tmp_path, rod_dirs):
        cfg = _rod_config(tmp_path, *rod_dirs)
        res = synthesize_dwi(cfg, motion=MotionConfig(2.0, 10.0, seed=1))
        assert np.array_equal(res.motion_matrices[0], np.eye(4))
        assert np.array_equal(res.signal_moved_gt[..., 0], res.signal_gt[..., 0])
        assert not np.allclose(res.signal_moved_gt[..., 1], res.signal_gt[..., 1])
        # without noise the signal is the moved ground truth
        assert np.array_equal(res.signal, res.signal_moved_gt)
        assert res.motion_matrices.shape == (4, 4, 4)

    def test_signal_gt_independent_of_motion(self, tmp_path, rod_dirs):
        cfg = _rod_config(tmp_path, *rod_dirs)
        plain = synthesize_dwi(cfg)
        moved = synthesize_dwi(cfg, motion=MotionConfig(2.0, 10.0, seed=1))
        assert np.array_equal(plain.signal_gt, moved.signal_gt)

    def test_noise_applied_after_motion(self, tmp_path):
        """Sub-voxel shifted volumes keep unsmoothed Rician noise: E[s^2] = 2 sigma^2 in air."""
        cfg = _rod_config(tmp_path, [0.0, 0.0], None, tensor=False)
        snr = 5.0
        sigma = S0 / snr
        res = synthesize_dwi(
            cfg,
            snr=snr,
            rng=np.random.default_rng(0),
            motion=MotionConfig(1.3, 0.0, seed=2),
        )
        air = np.zeros(SHAPE, dtype=bool)
        air[:6] = True  # far from the rod, inside the zero-filled and the real background
        assert (res.signal_moved_gt[air][:, 1] == 0).all()
        for n in (0, 1):  # reference and moved volume share the same noise level
            ms = np.mean(res.signal[..., n][air] ** 2)
            assert ms == pytest.approx(2 * sigma**2, rel=0.05)

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_gradient_rotation_follows_object(self, tmp_path, seed):
        """Measured rod axis (from a b=0 run) predicts the tensor signal of the moved run.

        The object axis is measured geometrically, independent of the gradient code, so this
        fixes the sign of the gradient rotation (``g_obj = R g``) for oblique gradients.
        """
        bvals = [0.0, 1000.0, 1000.0, 1000.0]
        g = [[0, 0, 0], G_OBLIQUE, G_OBLIQUE, G_OBLIQUE]
        motion = MotionConfig(0.0, 30.0, seed=seed)
        geo = synthesize_dwi(_rod_config(tmp_path, [0.0] * 4, g), motion=motion)
        run = synthesize_dwi(_rod_config(tmp_path, bvals, g, name="b.nii.gz"), motion=motion)
        assert np.array_equal(geo.motion_matrices, run.motion_matrices)
        for n in (1, 2, 3):
            axis = _principal_axis(geo.signal[..., n])
            interior = geo.signal[..., n] > 0.999 * S0
            assert interior.sum() > 20
            adc = EVALS[1] + (EVALS[0] - EVALS[1]) * float(axis @ G_OBLIQUE) ** 2
            expected = S0 * np.exp(-1000.0 * adc)
            np.testing.assert_allclose(run.signal[..., n][interior], expected, rtol=0.02)
            # the sign convention matters: the opposite rotation would predict a different value
            rot = run.motion_matrices[n, :3, :3]
            wrong_axis = rot @ np.array([1.0, 0.0, 0.0])
            wrong_adc = EVALS[1] + (EVALS[0] - EVALS[1]) * float(wrong_axis @ G_OBLIQUE) ** 2
            assert abs(S0 * np.exp(-1000.0 * wrong_adc) - expected) > 0.02 * expected

    def test_tensor_without_bvecs_still_raises_with_motion(self, tmp_path):
        cfg = _rod_config(tmp_path, [0.0, 1000.0], None)
        with pytest.raises(ValueError, match="bvec"):
            synthesize_dwi(cfg, motion=MotionConfig(1.0, 5.0, seed=0))


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExport:
    def _phantom(self, tmp_path, rod_dirs, motion=True):
        cfg = _rod_config(tmp_path, *rod_dirs)
        m = MotionConfig(1.0, 5.0, seed=4) if motion else None
        return DWIPhantom(motion=m).load_case(cfg).synthesize()

    def test_nifti_writes_bvec_next_to_it(self, tmp_path, rod_dirs):
        p = self._phantom(tmp_path, rod_dirs)
        written = p.save(tmp_path / "out" / "case.nii.gz")
        bvec = tmp_path / "out" / "case.bvec"
        assert bvec in written
        loaded = np.loadtxt(bvec)
        assert loaded.shape == (3, 4)
        np.testing.assert_allclose(loaded.T, p.config.bvecs, atol=1e-7)

    def test_nifti_without_bvecs_writes_no_bvec(self, tmp_path):
        cfg = _rod_config(tmp_path, [0.0, 500.0], None, tensor=False)
        written = DWIPhantom().load_case(cfg).synthesize().save(tmp_path / "o.nii.gz")
        assert [w.name for w in written] == ["o.nii.gz"]

    def test_npy_matrices(self, tmp_path, rod_dirs):
        p = self._phantom(tmp_path, rod_dirs)
        (path,) = p.save(tmp_path / "motion.npy")
        np.testing.assert_array_equal(np.load(path), p.motion_matrices)

    def test_npy_without_motion_raises(self, tmp_path, rod_dirs):
        p = self._phantom(tmp_path, rod_dirs, motion=False)
        with pytest.raises(ValueError, match="motion"):
            p.save(tmp_path / "motion.npy")

    def test_hdf5_roundtrip(self, tmp_path, rod_dirs):
        p = self._phantom(tmp_path, rod_dirs)
        (path,) = p.save(tmp_path / "case.h5")
        data = load_hdf5(path)
        np.testing.assert_array_equal(data.motion_matrices, p.motion_matrices)
        np.testing.assert_array_equal(data.signal_moved_gt, p.signal_moved_gt)
        np.testing.assert_array_equal(data.bvecs, p.config.bvecs)

    def test_hdf5_without_motion_has_no_motion_fields(self, tmp_path, rod_dirs):
        p = self._phantom(tmp_path, rod_dirs, motion=False)
        (path,) = p.save(tmp_path / "case.h5")
        data = load_hdf5(path)
        assert data.motion_matrices is None and data.signal_moved_gt is None


# ---------------------------------------------------------------------------
# Monte Carlo still works
# ---------------------------------------------------------------------------


def test_mc_phantom_smoke(mini_config):
    p = MCDWIPhantom(n_samples=2, snr=30, seed=1)
    p.load_case(mini_config).setup().synthesize()
    assert p.signal.shape[-2:] == (len(mini_config.bvalues), 2)
    assert p.motion is None
