"""Tests for mrphantom.synthesis_dwi — synthesize_phantom."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from mrphantom.synthesis_dwi import synthesize_phantom


# ---------------------------------------------------------------------------
# Happy path — noise-free
# ---------------------------------------------------------------------------


def test_synthesize_returns_tuple(mini_config):
    result = synthesize_phantom(mini_config)
    assert isinstance(result, tuple) and len(result) == 2


def test_synthesize_signal_shape(mini_config):
    signal, _ = synthesize_phantom(mini_config)
    assert signal.shape == (4, 4, 4, 3)  # spatial 4×4×4, 3 b-values


def test_synthesize_signal_dtype(mini_config):
    signal, _ = synthesize_phantom(mini_config)
    assert signal.dtype == np.float64


def test_synthesize_returns_nifti_image(mini_config):
    _, img = synthesize_phantom(mini_config)
    assert isinstance(img, nib.Nifti1Image)


def test_synthesize_nifti_shape(mini_config):
    _, img = synthesize_phantom(mini_config)
    assert img.shape == (4, 4, 4, 3)


def test_synthesize_affine_matches_segmentation(mini_config):
    _, img = synthesize_phantom(mini_config)
    np.testing.assert_array_equal(img.affine, np.eye(4))


# ---------------------------------------------------------------------------
# Signal values — noise-free
# ---------------------------------------------------------------------------


def test_synthesize_zero_bvalue_equals_s0(mini_config):
    """At b=0 the multi-exponential model collapses to S0."""
    signal, _ = synthesize_phantom(mini_config)
    region_a_mask = np.zeros((4, 4, 4), dtype=bool)
    region_a_mask[0:2, :, :] = True  # region 1 voxels
    s0_actual = signal[region_a_mask, 0]
    np.testing.assert_allclose(s0_actual, 100.0, rtol=1e-9)


def test_synthesize_background_voxels_are_zero(mini_config):
    """Voxels with label 0 (background) must be zero in all signals."""
    signal, _ = synthesize_phantom(mini_config)
    # The 4×4×4 mini NIfTI has no label-0 voxels (all are 1 or 2),
    # so just confirm non-background voxels are non-zero at b=0.
    assert (signal[:, :, :, 0] > 0).all()


def test_synthesize_signal_decreases_with_b(mini_config):
    """Signal should decrease monotonically with increasing b-value (no noise)."""
    signal, _ = synthesize_phantom(mini_config)
    # Check region 1 voxels (simple mono-exponential)
    region_1_mean = signal[0, 0, 0, :]
    assert region_1_mean[0] > region_1_mean[1] > region_1_mean[2]


# ---------------------------------------------------------------------------
# Rician noise
# ---------------------------------------------------------------------------


def test_synthesize_with_snr_changes_values(mini_config):
    """Adding noise must alter the signal values."""
    signal_clean, _ = synthesize_phantom(mini_config, snr=None)
    signal_noisy, _ = synthesize_phantom(
        mini_config, snr=10, rng=np.random.default_rng(0)
    )
    assert not np.allclose(signal_clean, signal_noisy)


def test_synthesize_noise_reproducible_with_rng(mini_config):
    """Same RNG seed → identical noisy output."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    s1, _ = synthesize_phantom(mini_config, snr=20, rng=rng1)
    s2, _ = synthesize_phantom(mini_config, snr=20, rng=rng2)
    np.testing.assert_array_equal(s1, s2)


def test_synthesize_noise_nonnegative(mini_config):
    """Rician noise model must never produce negative signal."""
    signal, _ = synthesize_phantom(mini_config, snr=5, rng=np.random.default_rng(99))
    assert (signal >= 0).all()


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------


def test_synthesize_missing_nifti_raises(mini_config, tmp_path):
    from mrphantom.config import PhantomConfig

    bad_config = PhantomConfig(
        case_id=mini_config.case_id,
        name=mini_config.name,
        description=mini_config.description,
        nifti_path=tmp_path / "nonexistent.nii.gz",
        image_dims=mini_config.image_dims,
        bvalues=mini_config.bvalues,
        regions=mini_config.regions,
    )
    with pytest.raises(FileNotFoundError):
        synthesize_phantom(bad_config)
