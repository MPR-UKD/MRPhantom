"""Tests for mrphantom.montecarlo.sampler — sample_region_params."""

from __future__ import annotations

import numpy as np
import pytest

from mrphantom.montecarlo.config import ParamDist, RegionMCConfig
from mrphantom.montecarlo.sampler import (
    SampledParams,
    _sample_dirichlet,
    _sample_scalar,
    sample_region_params,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# _sample_scalar
# ---------------------------------------------------------------------------


class TestSampleScalar:
    """Unit tests for the internal _sample_scalar helper."""

    def test_normal_returns_correct_shape(self):
        """Normal sampling returns an array of the requested length."""
        dist = ParamDist(dist="normal", mean=5.0, std=1.0)
        out = _sample_scalar(dist, 20, _make_rng())
        assert out.shape == (20,)
        assert out.dtype == np.float64

    def test_normal_mean_approx(self):
        """Normal samples have approximately the specified mean."""
        dist = ParamDist(dist="normal", mean=100.0, std=5.0)
        out = _sample_scalar(dist, 10_000, _make_rng(1))
        np.testing.assert_allclose(out.mean(), 100.0, atol=1.0)

    def test_lognormal_returns_positive(self):
        """Lognormal samples are always positive."""
        dist = ParamDist(dist="lognormal", mean=0.003, sigma=0.1)
        out = _sample_scalar(dist, 100, _make_rng(2))
        assert np.all(out > 0)

    def test_lognormal_geometric_mean_approx(self):
        """Lognormal geometric mean is approximately dist.mean."""
        dist = ParamDist(dist="lognormal", mean=0.003, sigma=0.1)
        out = _sample_scalar(dist, 50_000, _make_rng(3))
        geom_mean = np.exp(np.log(out).mean())
        np.testing.assert_allclose(geom_mean, 0.003, rtol=0.02)

    def test_unknown_dist_raises(self):
        """A non-scalar distribution raises ValueError."""
        dist = ParamDist(dist="dirichlet", alpha=[1.0, 1.0])
        with pytest.raises(ValueError, match="scalar distribution"):
            _sample_scalar(dist, 10, _make_rng())


# ---------------------------------------------------------------------------
# _sample_dirichlet
# ---------------------------------------------------------------------------


class TestSampleDirichlet:
    """Unit tests for the internal _sample_dirichlet helper."""

    def test_shape(self):
        """Dirichlet samples have shape (n, k)."""
        out = _sample_dirichlet(kappa=10.0, base_fractions=[0.5, 0.3, 0.2], n=30, rng=_make_rng())
        assert out.shape == (30, 3)

    def test_sums_to_one(self):
        """Each Dirichlet draw sums to 1."""
        out = _sample_dirichlet(kappa=10.0, base_fractions=[0.5, 0.3, 0.2], n=50, rng=_make_rng(4))
        np.testing.assert_allclose(out.sum(axis=1), np.ones(50), atol=1e-12)

    def test_mean_anchored_to_base_fractions(self):
        """Dirichlet mean is approximately the base fractions."""
        base = [0.7, 0.2, 0.1]
        out = _sample_dirichlet(kappa=50.0, base_fractions=base, n=20_000, rng=_make_rng(5))
        np.testing.assert_allclose(out.mean(axis=0), base, atol=0.01)

    def test_kappa_non_positive_raises(self):
        """kappa <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="kappa must be positive"):
            _sample_dirichlet(kappa=0.0, base_fractions=[0.5, 0.5], n=10, rng=_make_rng())


# ---------------------------------------------------------------------------
# sample_region_params
# ---------------------------------------------------------------------------


class TestSampleRegionParams:
    """Tests for sample_region_params()."""

    def test_returns_sampled_params(self, mini_config):
        """Function returns a SampledParams instance."""
        result = sample_region_params(mini_config, {}, n_samples=5, rng=_make_rng())
        assert isinstance(result, SampledParams)

    def test_region_names(self, mini_config):
        """region_names matches base config order (ascending NIfTI label)."""
        result = sample_region_params(mini_config, {}, n_samples=5, rng=_make_rng())
        # mini_config has labels 1 and 2 → region_a, region_b
        assert result.region_names == ["region_a", "region_b"]

    def test_S0_shape(self, mini_config):
        """S0 array has shape (n_regions, n_samples)."""
        n_samples = 7
        result = sample_region_params(mini_config, {}, n_samples=n_samples, rng=_make_rng())
        assert result.S0.shape == (2, n_samples)

    def test_D_shape(self, mini_config):
        """D array has shape (n_regions, max_n_components, n_samples)."""
        n_samples = 7
        result = sample_region_params(mini_config, {}, n_samples=n_samples, rng=_make_rng())
        # region_a has 1 comp, region_b has 2 → max = 2
        assert result.D.shape == (2, 2, n_samples)

    def test_fractions_shape(self, mini_config):
        """fractions array has shape (n_regions, max_n_components, n_samples)."""
        n_samples = 7
        result = sample_region_params(mini_config, {}, n_samples=n_samples, rng=_make_rng())
        assert result.fractions.shape == (2, 2, n_samples)

    def test_no_mc_regions_uses_base_values(self, mini_config):
        """Without MC distributions, base config values are broadcast."""
        result = sample_region_params(mini_config, {}, n_samples=10, rng=_make_rng())
        # region_a: S0=100, index 0
        np.testing.assert_array_equal(result.S0[0], 100.0)
        # region_b: S0=80, index 1
        np.testing.assert_array_equal(result.S0[1], 80.0)

    def test_no_mc_regions_base_D_broadcast(self, mini_config):
        """Without MC distributions, base D values are broadcast."""
        result = sample_region_params(mini_config, {}, n_samples=5, rng=_make_rng())
        # region_a: D=[0.001], 1 component
        np.testing.assert_array_equal(result.D[0, 0], 0.001)
        # Padding component for region_a should remain zero
        np.testing.assert_array_equal(result.D[0, 1], 0.0)
        # region_b: D=[0.001, 0.01], 2 components
        np.testing.assert_array_equal(result.D[1, 0], 0.001)
        np.testing.assert_array_equal(result.D[1, 1], 0.01)

    def test_no_mc_regions_base_fractions_broadcast(self, mini_config):
        """Without MC distributions, base fractions are broadcast."""
        result = sample_region_params(mini_config, {}, n_samples=5, rng=_make_rng())
        # region_a: fractions=[1.0]
        np.testing.assert_array_equal(result.fractions[0, 0], 1.0)
        # region_b: fractions=[0.7, 0.3]
        np.testing.assert_array_equal(result.fractions[1, 0], 0.7)
        np.testing.assert_array_equal(result.fractions[1, 1], 0.3)

    def test_s0_sampled_when_dist_provided(self, mini_config):
        """S0 values differ across samples when a distribution is specified."""
        mc = {"region_a": RegionMCConfig(
            name="region_a",
            S0=ParamDist(dist="normal", mean=100.0, std=10.0),
        )}
        result = sample_region_params(mini_config, mc, n_samples=50, rng=_make_rng(7))
        # Values should not all be 100 (they're sampled)
        assert not np.all(result.S0[0] == 100.0)

    def test_s0_not_sampled_region_stays_fixed(self, mini_config):
        """Regions without MC spec have fixed S0 even when others are sampled."""
        mc = {"region_a": RegionMCConfig(
            name="region_a",
            S0=ParamDist(dist="normal", mean=100.0, std=10.0),
        )}
        result = sample_region_params(mini_config, mc, n_samples=20, rng=_make_rng(8))
        # region_b (index 1) has no MC spec → fixed S0=80
        np.testing.assert_array_equal(result.S0[1], 80.0)

    def test_d_sampled_when_dist_provided(self, mini_config):
        """D values differ across samples when lognormal distributions are given."""
        mc = {"region_b": RegionMCConfig(
            name="region_b",
            D=[
                ParamDist(dist="lognormal", mean=0.001, sigma=0.1),
                ParamDist(dist="lognormal", mean=0.01, sigma=0.1),
            ],
        )}
        result = sample_region_params(mini_config, mc, n_samples=20, rng=_make_rng(9))
        assert not np.all(result.D[1, 0] == 0.001)
        assert not np.all(result.D[1, 1] == 0.01)

    def test_fractions_sampled_sum_to_one(self, mini_config):
        """Sampled fractions for multi-component regions sum to 1."""
        mc = {"region_b": RegionMCConfig(
            name="region_b",
            fractions=ParamDist(dist="dirichlet", kappa=10.0),
        )}
        result = sample_region_params(mini_config, mc, n_samples=30, rng=_make_rng(10))
        # fractions_arr[1, :2, :] should sum to 1 for each sample
        frac_sums = result.fractions[1, :2, :].sum(axis=0)
        np.testing.assert_allclose(frac_sums, np.ones(30), atol=1e-12)

    def test_fractions_mean_anchored_to_base(self, mini_config):
        """Dirichlet fraction mean is anchored to base config values."""
        mc = {"region_b": RegionMCConfig(
            name="region_b",
            fractions=ParamDist(dist="dirichlet", kappa=50.0),
        )}
        result = sample_region_params(mini_config, mc, n_samples=20_000, rng=_make_rng(11))
        # region_b base fractions are [0.7, 0.3]
        np.testing.assert_allclose(
            result.fractions[1, 0].mean(), 0.7, atol=0.01
        )
        np.testing.assert_allclose(
            result.fractions[1, 1].mean(), 0.3, atol=0.01
        )

    def test_unknown_region_raises(self, mini_config):
        """ValueError when an MC region name is not in base config."""
        mc = {"does_not_exist": RegionMCConfig(name="does_not_exist")}
        with pytest.raises(ValueError, match="not found in base config"):
            sample_region_params(mini_config, mc, n_samples=5, rng=_make_rng())

    def test_wrong_d_length_raises(self, mini_config):
        """ValueError when D distribution list length != n_components."""
        mc = {"region_a": RegionMCConfig(
            name="region_a",
            D=[
                ParamDist(dist="lognormal", mean=0.001, sigma=0.1),
                ParamDist(dist="lognormal", mean=0.002, sigma=0.1),  # too many
            ],
        )}
        with pytest.raises(ValueError, match="component"):
            sample_region_params(mini_config, mc, n_samples=5, rng=_make_rng())

    def test_fractions_dirichlet_kappa_non_positive_raises(self, mini_config):
        """ValueError when Dirichlet kappa <= 0."""
        mc = {"region_b": RegionMCConfig(
            name="region_b",
            fractions=ParamDist(dist="dirichlet", kappa=0.0),
        )}
        with pytest.raises(ValueError, match="kappa"):
            sample_region_params(mini_config, mc, n_samples=5, rng=_make_rng())

    def test_non_dirichlet_fractions_raises(self, mini_config):
        """ValueError when fractions distribution is not 'dirichlet'."""
        mc = {"region_a": RegionMCConfig(
            name="region_a",
            fractions=ParamDist(dist="normal", mean=0.5, std=0.1),
        )}
        with pytest.raises(ValueError, match="dirichlet"):
            sample_region_params(mini_config, mc, n_samples=5, rng=_make_rng())

    def test_reproducible_with_same_seed(self, mini_config):
        """Two calls with the same seed return identical results."""
        mc = {"region_a": RegionMCConfig(
            name="region_a",
            S0=ParamDist(dist="normal", mean=100.0, std=5.0),
        )}
        r1 = sample_region_params(mini_config, mc, n_samples=10, rng=_make_rng(42))
        r2 = sample_region_params(mini_config, mc, n_samples=10, rng=_make_rng(42))
        np.testing.assert_array_equal(r1.S0, r2.S0)

    def test_different_seeds_differ(self, mini_config):
        """Two calls with different seeds produce different results."""
        mc = {"region_a": RegionMCConfig(
            name="region_a",
            S0=ParamDist(dist="normal", mean=100.0, std=10.0),
        )}
        r1 = sample_region_params(mini_config, mc, n_samples=20, rng=_make_rng(1))
        r2 = sample_region_params(mini_config, mc, n_samples=20, rng=_make_rng(2))
        assert not np.allclose(r1.S0, r2.S0)
