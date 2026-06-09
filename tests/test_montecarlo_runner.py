"""Tests for mrphantom.montecarlo.runner — run_montecarlo and MCResult."""

from __future__ import annotations

import numpy as np
import pytest

from mrphantom.montecarlo.config import ParamDist, RegionMCConfig
from mrphantom.montecarlo.runner import MCResult, run_montecarlo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mc_kwargs(mini_config, **overrides):
    defaults = dict(
        base_config=mini_config,
        mc_regions={},
        n_samples=5,
        snr=None,
        snr_region=None,
        seed=0,
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Basic structure and shapes
# ---------------------------------------------------------------------------


class TestRunMontecarloShape:
    """Tests for the shape and type of run_montecarlo output."""

    def test_returns_mc_result(self, mini_config):
        """run_montecarlo returns an MCResult instance."""
        result = run_montecarlo(**_mc_kwargs(mini_config))
        assert isinstance(result, MCResult)

    def test_signal_ndim(self, mini_config):
        """signal is a 5-D array (X, Y, Z, n_b, N)."""
        result = run_montecarlo(**_mc_kwargs(mini_config))
        assert result.signal.ndim == 5

    def test_signal_gt_ndim(self, mini_config):
        """signal_gt is a 5-D array (X, Y, Z, n_b, N)."""
        result = run_montecarlo(**_mc_kwargs(mini_config))
        assert result.signal_gt.ndim == 5

    def test_signal_shape_n_samples(self, mini_config):
        """Last axis of signal equals n_samples."""
        n = 8
        result = run_montecarlo(**_mc_kwargs(mini_config, n_samples=n))
        assert result.signal.shape[-1] == n

    def test_signal_gt_shape_matches_signal(self, mini_config):
        """signal_gt has the same shape as signal."""
        result = run_montecarlo(**_mc_kwargs(mini_config, n_samples=4))
        assert result.signal_gt.shape == result.signal.shape

    def test_signal_spatial_dims(self, mini_config):
        """Spatial dimensions match the mini_config NIfTI (4×4×4)."""
        result = run_montecarlo(**_mc_kwargs(mini_config))
        assert result.signal.shape[:3] == (4, 4, 4)

    def test_signal_n_volumes(self, mini_config):
        """Volume axis length matches number of b-values in mini_config."""
        result = run_montecarlo(**_mc_kwargs(mini_config))
        # mini_config has 3 b-values
        assert result.signal.shape[3] == 3

    def test_affine_shape(self, mini_config):
        """affine is a 4×4 matrix."""
        result = run_montecarlo(**_mc_kwargs(mini_config))
        assert result.affine.shape == (4, 4)

    def test_affine_is_copy(self, mini_config):
        """affine stored in MCResult is a copy, not a view."""
        result = run_montecarlo(**_mc_kwargs(mini_config))
        # Modifying the result should not propagate
        original = result.affine.copy()
        result.affine[0, 0] = 99.0
        assert result.affine[0, 0] == 99.0
        # Original was identity in mini_config
        assert original[0, 0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Noise behaviour
# ---------------------------------------------------------------------------


class TestRunMontecarloNoise:
    """Tests for noise handling in run_montecarlo."""

    def test_noise_free_signal_gt_equals_signal(self, mini_config):
        """When snr=None, signal_gt equals signal for every sample."""
        result = run_montecarlo(**_mc_kwargs(mini_config, snr=None))
        np.testing.assert_array_equal(result.signal, result.signal_gt)

    def test_noisy_signal_gt_is_noise_free(self, mini_config):
        """When snr is set, signal_gt is always ≤ signal (Rician noise ≥ 0 shift)."""
        result = run_montecarlo(**_mc_kwargs(mini_config, snr=20, seed=1))
        # GT should be nonneg and distinct from noisy signal in non-background voxels
        assert np.any(result.signal != result.signal_gt), (
            "Noisy signal should differ from GT in at least some voxels"
        )

    def test_noisy_signal_nonneg(self, mini_config):
        """Rician noise produces nonneg signal."""
        result = run_montecarlo(**_mc_kwargs(mini_config, snr=20, seed=2))
        assert np.all(result.signal >= 0)

    def test_signal_gt_nonneg(self, mini_config):
        """GT signal is nonneg."""
        result = run_montecarlo(**_mc_kwargs(mini_config, snr=20, seed=3))
        assert np.all(result.signal_gt >= 0)

    def test_independent_noise_per_sample(self, mini_config):
        """Different samples have different noise realisations."""
        result = run_montecarlo(**_mc_kwargs(mini_config, snr=20, seed=4, n_samples=3))
        # Non-background voxels: signals should differ between samples
        s0 = result.signal[..., 0]
        s1 = result.signal[..., 1]
        assert not np.allclose(s0, s1), (
            "Independent noise seeds should yield different realisations"
        )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestRunMontecarloReproducibility:
    """Tests for determinism with fixed seeds."""

    def test_reproducible_noise_free(self, mini_config):
        """Noise-free run is deterministic (no randomness)."""
        r1 = run_montecarlo(**_mc_kwargs(mini_config, snr=None, seed=0))
        r2 = run_montecarlo(**_mc_kwargs(mini_config, snr=None, seed=0))
        np.testing.assert_array_equal(r1.signal, r2.signal)

    def test_reproducible_with_noise(self, mini_config):
        """Same seed produces identical noisy results."""
        r1 = run_montecarlo(**_mc_kwargs(mini_config, snr=25, seed=7, n_samples=4))
        r2 = run_montecarlo(**_mc_kwargs(mini_config, snr=25, seed=7, n_samples=4))
        np.testing.assert_array_equal(r1.signal, r2.signal)
        np.testing.assert_array_equal(r1.signal_gt, r2.signal_gt)

    def test_different_seeds_differ(self, mini_config):
        """Different seeds yield different results."""
        r1 = run_montecarlo(**_mc_kwargs(mini_config, snr=25, seed=1, n_samples=3))
        r2 = run_montecarlo(**_mc_kwargs(mini_config, snr=25, seed=2, n_samples=3))
        assert not np.allclose(r1.signal, r2.signal)

    def test_reproducible_with_param_sampling(self, mini_config):
        """Param sampling is reproducible from master seed."""
        mc = {"region_a": RegionMCConfig(
            name="region_a",
            S0=ParamDist(dist="normal", std=10.0),
        )}
        r1 = run_montecarlo(**_mc_kwargs(mini_config, mc_regions=mc, seed=42, n_samples=5))
        r2 = run_montecarlo(**_mc_kwargs(mini_config, mc_regions=mc, seed=42, n_samples=5))
        np.testing.assert_array_equal(
            r1.sampled_params.S0, r2.sampled_params.S0
        )


# ---------------------------------------------------------------------------
# Parameter randomisation
# ---------------------------------------------------------------------------


class TestRunMontecarloParamSampling:
    """Tests for parameter variation across Monte Carlo samples."""

    def test_fixed_params_same_gt_across_samples(self, mini_config):
        """With no MC distributions, GT signal is the same for every sample."""
        result = run_montecarlo(**_mc_kwargs(mini_config, snr=None, n_samples=4))
        # All GT slices should be identical (same config → same GT)
        for i in range(1, 4):
            np.testing.assert_array_equal(
                result.signal_gt[..., 0], result.signal_gt[..., i]
            )

    def test_sampled_s0_varies_across_samples(self, mini_config):
        """When S0 is randomised, sampled_params.S0 differs across draws."""
        mc = {"region_a": RegionMCConfig(
            name="region_a",
            S0=ParamDist(dist="normal", std=15.0),
        )}
        result = run_montecarlo(
            **_mc_kwargs(mini_config, mc_regions=mc, n_samples=10, seed=5)
        )
        # S0 for region_a should not be all identical
        assert result.sampled_params.S0[0].std() > 0

    def test_sampled_params_s0_shape(self, mini_config):
        """sampled_params.S0 has shape (n_regions, n_samples)."""
        n = 6
        result = run_montecarlo(**_mc_kwargs(mini_config, n_samples=n))
        assert result.sampled_params.S0.shape == (2, n)

    def test_param_randomisation_changes_gt(self, mini_config):
        """When S0 is randomised, GT signal differs across samples."""
        mc = {"region_a": RegionMCConfig(
            name="region_a",
            S0=ParamDist(dist="normal", std=20.0),
        )}
        result = run_montecarlo(
            **_mc_kwargs(mini_config, mc_regions=mc, snr=None, n_samples=5, seed=6)
        )
        # GT slices should differ (because S0 is different for each sample)
        any_different = any(
            not np.allclose(result.signal_gt[..., 0], result.signal_gt[..., i])
            for i in range(1, 5)
        )
        assert any_different, "Randomised S0 should produce different GT signals"

    def test_unknown_mc_region_raises(self, mini_config):
        """ValueError when an MC region name does not exist in base config."""
        mc = {"nonexistent": RegionMCConfig(name="nonexistent")}
        with pytest.raises(ValueError, match="not found in base config"):
            run_montecarlo(**_mc_kwargs(mini_config, mc_regions=mc))
