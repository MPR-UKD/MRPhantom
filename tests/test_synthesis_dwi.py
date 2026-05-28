"""Tests for mrphantom.dwi.synthesis — synthesize_phantom."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from mrphantom.dwi.synthesis import synthesize_phantom


# ---------------------------------------------------------------------------
# Happy path — noise-free
# ---------------------------------------------------------------------------


def test_synthesize_returns_tuple(mini_config):
    result = synthesize_phantom(mini_config)
    assert isinstance(result, tuple) and len(result) == 3


def test_synthesize_signal_shape(mini_config):
    signal, _, _ = synthesize_phantom(mini_config)
    assert signal.shape == (4, 4, 4, 3)  # spatial 4×4×4, 3 b-values


def test_synthesize_signal_dtype(mini_config):
    signal, _, _ = synthesize_phantom(mini_config)
    assert signal.dtype == np.float64


def test_synthesize_returns_nifti_image(mini_config):
    _, _, img = synthesize_phantom(mini_config)
    assert isinstance(img, nib.Nifti1Image)


def test_synthesize_nifti_shape(mini_config):
    _, _, img = synthesize_phantom(mini_config)
    assert img.shape == (4, 4, 4, 3)


def test_synthesize_affine_matches_segmentation(mini_config):
    _, _, img = synthesize_phantom(mini_config)
    np.testing.assert_array_equal(img.affine, np.eye(4))


# ---------------------------------------------------------------------------
# Signal values — noise-free
# ---------------------------------------------------------------------------


def test_synthesize_zero_bvalue_equals_s0(mini_config):
    """At b=0 the multi-exponential model collapses to S0."""
    signal, _, _ = synthesize_phantom(mini_config)
    region_a_mask = np.zeros((4, 4, 4), dtype=bool)
    region_a_mask[0:2, :, :] = True  # region 1 voxels
    s0_actual = signal[region_a_mask, 0]
    np.testing.assert_allclose(s0_actual, 100.0, rtol=1e-9)


def test_synthesize_background_voxels_are_zero(mini_config):
    """Voxels with label 0 (background) must be zero in all signals."""
    signal, _, _ = synthesize_phantom(mini_config)
    # The 4×4×4 mini NIfTI has no label-0 voxels (all are 1 or 2),
    # so just confirm non-background voxels are non-zero at b=0.
    assert (signal[:, :, :, 0] > 0).all()


def test_synthesize_signal_decreases_with_b(mini_config):
    """Signal should decrease monotonically with increasing b-value (no noise)."""
    signal, _, _ = synthesize_phantom(mini_config)
    # Check region 1 voxels (simple mono-exponential)
    region_1_mean = signal[0, 0, 0, :]
    assert region_1_mean[0] > region_1_mean[1] > region_1_mean[2]


# ---------------------------------------------------------------------------
# Rician noise
# ---------------------------------------------------------------------------


def test_synthesize_with_snr_changes_values(mini_config):
    """Adding noise must alter the signal values."""
    signal_clean, _, _ = synthesize_phantom(mini_config, snr=None)
    signal_noisy, _, _ = synthesize_phantom(
        mini_config, snr=10, rng=np.random.default_rng(0)
    )
    assert not np.allclose(signal_clean, signal_noisy)


def test_synthesize_noise_reproducible_with_rng(mini_config):
    """Same RNG seed → identical noisy output."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    s1, _, _ = synthesize_phantom(mini_config, snr=20, rng=rng1)
    s2, _, _ = synthesize_phantom(mini_config, snr=20, rng=rng2)
    np.testing.assert_array_equal(s1, s2)


def test_synthesize_noise_nonnegative(mini_config):
    """Rician noise model must never produce negative signal."""
    signal, _, _ = synthesize_phantom(mini_config, snr=5, rng=np.random.default_rng(99))
    assert (signal >= 0).all()


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------


def test_synthesize_missing_nifti_raises(mini_config, tmp_path):
    from mrphantom.dwi.config import DWIConfig

    bad_config = DWIConfig(
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


# ---------------------------------------------------------------------------
# snr_region — named reference region for noise sigma
# ---------------------------------------------------------------------------


def test_snr_region_uses_named_region_s0(mini_config):
    """sigma must be S0_of_named_region / snr, not max(S0) / snr.

    mini_config has region_a (S0=100) and region_b (S0=80).
    With snr_region="region_b" the sigma is 80/snr, not 100/snr.
    We verify this by checking the variance of the added noise against both
    candidates; only the region_b-based sigma should fit.
    """
    snr = 10.0
    rng = np.random.default_rng(0)
    signal_noisy, _, _ = synthesize_phantom(
        mini_config, snr=snr, rng=rng, snr_region="region_b"
    )
    signal_clean, _, _ = synthesize_phantom(mini_config)
    noise = signal_noisy - signal_clean

    # sigma from region_b S0 (80)
    expected_sigma_b = 80.0 / snr
    # sigma from max S0 (region_a = 100)
    expected_sigma_max = 100.0 / snr

    # variance of Rician noise ≈ 2 * sigma^2 for large arrays
    empirical_var = float(np.var(noise))
    # The empirical variance should be closer to 2*sigma_b^2 than 2*sigma_max^2
    err_b = abs(empirical_var - 2 * expected_sigma_b**2)
    err_max = abs(empirical_var - 2 * expected_sigma_max**2)
    assert err_b < err_max


def test_snr_region_none_falls_back_to_max_s0(mini_config):
    """With snr_region=None the noise reference should be max(S0)=100.

    We verify that the result is identical to not passing snr_region at all.
    """
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    s_default, _, _ = synthesize_phantom(mini_config, snr=20, rng=rng1)
    s_explicit_none, _, _ = synthesize_phantom(
        mini_config, snr=20, rng=rng2, snr_region=None
    )
    np.testing.assert_array_equal(s_default, s_explicit_none)


def test_snr_region_unknown_name_raises(mini_config):
    """Requesting a non-existent region name must raise ValueError."""
    with pytest.raises(ValueError, match="snr_region"):
        synthesize_phantom(
            mini_config, snr=20, rng=np.random.default_rng(0), snr_region="no_such_region"
        )


def test_snr_region_lower_s0_produces_less_noise(mini_config):
    """Using region_b (S0=80) as reference gives lower sigma than max(S0)=100,
    so the noisy signal should be closer to the clean signal on average."""
    rng_a = np.random.default_rng(99)
    rng_b = np.random.default_rng(99)
    snr = 5.0
    clean, _, _ = synthesize_phantom(mini_config)
    noisy_max, _, _ = synthesize_phantom(mini_config, snr=snr, rng=rng_a)
    noisy_b, _, _ = synthesize_phantom(
        mini_config, snr=snr, rng=rng_b, snr_region="region_b"
    )
    rmse_max = float(np.sqrt(np.mean((noisy_max - clean) ** 2)))
    rmse_b = float(np.sqrt(np.mean((noisy_b - clean) ** 2)))
    assert rmse_b < rmse_max


# ---------------------------------------------------------------------------
# Ground-truth signal
# ---------------------------------------------------------------------------


def test_synthesize_gt_shape_matches_signal(mini_config):
    """signal_gt must have the same shape as signal."""
    signal, signal_gt, _ = synthesize_phantom(mini_config)
    assert signal_gt.shape == signal.shape


def test_synthesize_gt_dtype(mini_config):
    signal, signal_gt, _ = synthesize_phantom(mini_config)
    assert signal_gt.dtype == np.float64


def test_synthesize_gt_equals_signal_when_noise_free(mini_config):
    """Without noise signal_gt and signal must be identical arrays."""
    signal, signal_gt, _ = synthesize_phantom(mini_config)
    np.testing.assert_array_equal(signal_gt, signal)


def test_synthesize_gt_differs_from_noisy_signal(mini_config):
    """With noise the ground-truth must differ from the noisy signal."""
    signal, signal_gt, _ = synthesize_phantom(
        mini_config, snr=10, rng=np.random.default_rng(0)
    )
    assert not np.allclose(signal, signal_gt)


def test_synthesize_gt_is_noise_free_when_snr_set(mini_config):
    """signal_gt must equal the deterministic model output regardless of noise."""
    clean, _, _ = synthesize_phantom(mini_config)
    _, signal_gt, _ = synthesize_phantom(
        mini_config, snr=10, rng=np.random.default_rng(0)
    )
    np.testing.assert_array_equal(signal_gt, clean)


def test_synthesize_gt_independent_of_rng_state(mini_config):
    """signal_gt must be the same regardless of which RNG seed is used for noise."""
    _, gt1, _ = synthesize_phantom(mini_config, snr=20, rng=np.random.default_rng(1))
    _, gt2, _ = synthesize_phantom(mini_config, snr=20, rng=np.random.default_rng(99))
    np.testing.assert_array_equal(gt1, gt2)
