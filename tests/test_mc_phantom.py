"""Tests for mrphantom.montecarlo.phantom — MCDWIPhantom and _H5Writer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mrphantom import MCDWIPhantom
from mrphantom.io.mc_hdf5 import load_mc_hdf5
from mrphantom.montecarlo.config import ParamDist, RegionMCConfig
from mrphantom.montecarlo.sampler import SampledParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mc_regions_a() -> dict[str, RegionMCConfig]:
    """MC distributions for region_a only (S0 normal, D lognormal)."""
    return {
        "region_a": RegionMCConfig(
            name="region_a",
            S0=ParamDist(dist="normal", std=10.0),
            D=[ParamDist(dist="lognormal", sigma=0.1)],
        )
    }


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestMCDWIPhantomInit:
    """Tests for MCDWIPhantom constructor and default attribute values."""

    def test_n_samples_stored(self):
        """n_samples is stored on the instance."""
        p = MCDWIPhantom(n_samples=20)
        assert p.n_samples == 20

    def test_mc_regions_default_empty(self):
        """mc_regions defaults to empty dict when not provided."""
        p = MCDWIPhantom(n_samples=5)
        assert p.mc_regions == {}

    def test_mc_regions_stored(self):
        """mc_regions passed in constructor is stored."""
        mc = _mc_regions_a()
        p = MCDWIPhantom(n_samples=5, mc_regions=mc)
        assert p.mc_regions is mc

    def test_snr_default_none(self):
        """snr defaults to None (noise-free)."""
        p = MCDWIPhantom(n_samples=5)
        assert p.snr is None

    def test_snr_stored(self):
        """snr is stored on the instance."""
        p = MCDWIPhantom(n_samples=5, snr=20.0)
        assert p.snr == 20.0

    def test_seed_stored(self):
        """seed is stored on the instance."""
        p = MCDWIPhantom(n_samples=5, seed=42)
        assert p.seed == 42

    def test_snr_region_stored(self):
        """snr_region is stored on the instance."""
        p = MCDWIPhantom(n_samples=5, snr=30.0, snr_region="region_a")
        assert p.snr_region == "region_a"

    def test_setup_not_done_initially(self):
        """setup flag is False on fresh instance."""
        p = MCDWIPhantom(n_samples=5)
        assert not p._setup_done

    def test_is_instance_of_dwiphantom(self):
        """MCDWIPhantom inherits from DWIPhantom."""
        from mrphantom import DWIPhantom
        p = MCDWIPhantom(n_samples=5)
        assert isinstance(p, DWIPhantom)


# ---------------------------------------------------------------------------
# load_case
# ---------------------------------------------------------------------------


class TestMCDWIPhantomLoadCase:
    """Tests for load_case on MCDWIPhantom."""

    def test_returns_self(self, mini_config):
        """load_case returns self for chaining."""
        p = MCDWIPhantom(n_samples=5)
        assert p.load_case(mini_config) is p

    def test_config_accessible_after_load(self, mini_config):
        """Config is accessible after load_case."""
        p = MCDWIPhantom(n_samples=5)
        p.load_case(mini_config)
        assert p.config is mini_config

    def test_load_case_resets_setup(self, mini_config):
        """Calling load_case after setup() resets the setup state."""
        p = MCDWIPhantom(n_samples=5)
        p.load_case(mini_config).setup()
        assert p._setup_done
        p.load_case(mini_config)
        assert not p._setup_done

    def test_load_case_resets_sampled_params(self, mini_config):
        """Calling load_case clears sampled params."""
        p = MCDWIPhantom(n_samples=5)
        p.load_case(mini_config).setup()
        assert p._sampled_params is not None
        p.load_case(mini_config)
        assert p._sampled_params is None

    def test_load_case_resets_batch_signal(self, mini_config):
        """Calling load_case clears any previously synthesized batch signal."""
        p = MCDWIPhantom(n_samples=3)
        p.load_case(mini_config).setup().synthesize()
        assert p._batch_signal is not None
        p.load_case(mini_config)
        assert p._batch_signal is None


# ---------------------------------------------------------------------------
# setup()
# ---------------------------------------------------------------------------


class TestMCDWIPhantomSetup:
    """Tests for the setup() method."""

    def test_setup_before_load_case_raises(self):
        """setup() before load_case() raises RuntimeError."""
        p = MCDWIPhantom(n_samples=5)
        with pytest.raises(RuntimeError, match="load_case"):
            p.setup()

    def test_setup_returns_self(self, mini_config):
        """setup() returns self for chaining."""
        p = MCDWIPhantom(n_samples=5)
        assert p.load_case(mini_config).setup() is p

    def test_setup_marks_done(self, mini_config):
        """setup() sets the _setup_done flag."""
        p = MCDWIPhantom(n_samples=5)
        p.load_case(mini_config).setup()
        assert p._setup_done

    def test_setup_produces_sampled_params(self, mini_config):
        """setup() populates _sampled_params as a SampledParams instance."""
        p = MCDWIPhantom(n_samples=5)
        p.load_case(mini_config).setup()
        assert isinstance(p._sampled_params, SampledParams)

    def test_setup_sampled_params_shape(self, mini_config):
        """Sampled S0 has shape (n_regions, n_samples)."""
        n = 7
        p = MCDWIPhantom(n_samples=n)
        p.load_case(mini_config).setup()
        assert p._sampled_params.S0.shape == (2, n)

    def test_setup_produces_noise_seeds(self, mini_config):
        """setup() populates _noise_seeds with one entry per sample."""
        n = 6
        p = MCDWIPhantom(n_samples=n, snr=20.0)
        p.load_case(mini_config).setup()
        assert p._noise_seeds is not None
        assert len(p._noise_seeds) == n

    def test_sampled_params_property_accessible_after_setup(self, mini_config):
        """sampled_params property is accessible after setup()."""
        p = MCDWIPhantom(n_samples=5)
        p.load_case(mini_config).setup()
        sp = p.sampled_params
        assert isinstance(sp, SampledParams)

    def test_sampled_params_property_before_setup_raises(self):
        """sampled_params property before setup() raises RuntimeError."""
        p = MCDWIPhantom(n_samples=5)
        with pytest.raises(RuntimeError, match="setup"):
            _ = p.sampled_params

    def test_setup_reproducible_with_seed(self, mini_config):
        """Same seed produces identical sampled params."""
        p1 = MCDWIPhantom(n_samples=5, seed=0)
        p1.load_case(mini_config).setup()
        p2 = MCDWIPhantom(n_samples=5, seed=0)
        p2.load_case(mini_config).setup()
        np.testing.assert_array_equal(
            p1.sampled_params.S0, p2.sampled_params.S0
        )


# ---------------------------------------------------------------------------
# synthesize() — batch mode
# ---------------------------------------------------------------------------


class TestMCDWIPhantomSynthesize:
    """Tests for batch synthesis via synthesize()."""

    def test_synthesize_before_setup_raises(self, mini_config):
        """synthesize() before setup() raises RuntimeError."""
        p = MCDWIPhantom(n_samples=5)
        p.load_case(mini_config)
        with pytest.raises(RuntimeError, match="setup"):
            p.synthesize()

    def test_synthesize_returns_self(self, mini_config):
        """synthesize() returns self for chaining."""
        p = MCDWIPhantom(n_samples=3)
        assert p.load_case(mini_config).setup().synthesize() is p

    def test_signal_is_5d_after_synthesize(self, mini_config):
        """signal is 5-D (X, Y, Z, n_b, N) after batch synthesize()."""
        N = 4
        p = MCDWIPhantom(n_samples=N)
        p.load_case(mini_config).setup().synthesize()
        assert p.signal.ndim == 5

    def test_signal_shape_after_synthesize(self, mini_config):
        """signal shape is (4, 4, 4, 3, N) for the mini_config fixture."""
        N = 4
        p = MCDWIPhantom(n_samples=N)
        p.load_case(mini_config).setup().synthesize()
        assert p.signal.shape == (4, 4, 4, 3, N)

    def test_signal_gt_shape_matches_signal(self, mini_config):
        """signal_gt has the same shape as signal after synthesize()."""
        p = MCDWIPhantom(n_samples=3)
        p.load_case(mini_config).setup().synthesize()
        assert p.signal_gt.shape == p.signal.shape

    def test_signal_before_synthesize_raises(self, mini_config):
        """signal property before synthesize() or step() raises RuntimeError."""
        p = MCDWIPhantom(n_samples=3)
        p.load_case(mini_config).setup()
        with pytest.raises(RuntimeError):
            _ = p.signal

    def test_signal_gt_before_synthesize_raises(self, mini_config):
        """signal_gt property before synthesize() or step() raises RuntimeError."""
        p = MCDWIPhantom(n_samples=3)
        p.load_case(mini_config).setup()
        with pytest.raises(RuntimeError):
            _ = p.signal_gt

    def test_noise_free_signal_gt_equals_signal(self, mini_config):
        """Without noise, signal_gt equals signal for every sample."""
        p = MCDWIPhantom(n_samples=4)
        p.load_case(mini_config).setup().synthesize()
        np.testing.assert_array_equal(p.signal, p.signal_gt)

    def test_noisy_signal_differs_from_gt(self, mini_config):
        """With noise, signal differs from signal_gt in at least some voxels."""
        p = MCDWIPhantom(n_samples=4, snr=20.0, seed=0)
        p.load_case(mini_config).setup().synthesize()
        assert not np.allclose(p.signal, p.signal_gt)

    def test_signal_nonneg(self, mini_config):
        """All signal values are non-negative (Rician noise property)."""
        p = MCDWIPhantom(n_samples=4, snr=20.0, seed=1)
        p.load_case(mini_config).setup().synthesize()
        assert np.all(p.signal >= 0)

    def test_fixed_params_identical_gt_across_samples(self, mini_config):
        """With no MC distributions, GT is identical across all samples."""
        p = MCDWIPhantom(n_samples=5)
        p.load_case(mini_config).setup().synthesize()
        for i in range(1, 5):
            np.testing.assert_array_equal(
                p.signal_gt[..., 0], p.signal_gt[..., i]
            )

    def test_param_randomisation_changes_gt(self, mini_config):
        """Randomising S0 produces different GT signals across samples."""
        mc = _mc_regions_a()
        p = MCDWIPhantom(n_samples=5, mc_regions=mc, seed=7)
        p.load_case(mini_config).setup().synthesize()
        any_different = any(
            not np.allclose(p.signal_gt[..., 0], p.signal_gt[..., i])
            for i in range(1, 5)
        )
        assert any_different, "Randomised S0 must produce different GT signals"

    def test_reproducible_with_seed(self, mini_config):
        """Same seed gives identical batch results."""
        p1 = MCDWIPhantom(n_samples=3, snr=25.0, seed=42)
        p1.load_case(mini_config).setup().synthesize()
        p2 = MCDWIPhantom(n_samples=3, snr=25.0, seed=42)
        p2.load_case(mini_config).setup().synthesize()
        np.testing.assert_array_equal(p1.signal, p2.signal)
        np.testing.assert_array_equal(p1.signal_gt, p2.signal_gt)

    def test_different_seeds_differ(self, mini_config):
        """Different seeds produce different results."""
        p1 = MCDWIPhantom(n_samples=3, snr=25.0, seed=1)
        p1.load_case(mini_config).setup().synthesize()
        p2 = MCDWIPhantom(n_samples=3, snr=25.0, seed=2)
        p2.load_case(mini_config).setup().synthesize()
        assert not np.allclose(p1.signal, p2.signal)


# ---------------------------------------------------------------------------
# step() — lazy mode
# ---------------------------------------------------------------------------


class TestMCDWIPhantomStep:
    """Tests for lazy synthesis via step()."""

    def test_step_before_setup_raises(self, mini_config):
        """step() before setup() raises RuntimeError."""
        p = MCDWIPhantom(n_samples=5)
        p.load_case(mini_config)
        with pytest.raises(RuntimeError, match="setup"):
            p.step(0)

    def test_step_negative_index_raises(self, mini_config):
        """step(-1) raises RuntimeError."""
        p = MCDWIPhantom(n_samples=5)
        p.load_case(mini_config).setup()
        with pytest.raises(RuntimeError, match="out of range"):
            p.step(-1)

    def test_step_out_of_range_raises(self, mini_config):
        """step(n_samples) raises RuntimeError."""
        p = MCDWIPhantom(n_samples=5)
        p.load_case(mini_config).setup()
        with pytest.raises(RuntimeError, match="out of range"):
            p.step(5)

    def test_step_returns_self(self, mini_config):
        """step(i) returns self for chaining."""
        p = MCDWIPhantom(n_samples=5)
        p.load_case(mini_config).setup()
        assert p.step(0) is p

    def test_signal_is_4d_after_step(self, mini_config):
        """signal is 4-D (X, Y, Z, n_b) after step()."""
        p = MCDWIPhantom(n_samples=5)
        p.load_case(mini_config).setup().step(0)
        assert p.signal.ndim == 4

    def test_signal_shape_after_step(self, mini_config):
        """signal shape is (4, 4, 4, 3) after step() for mini_config."""
        p = MCDWIPhantom(n_samples=5)
        p.load_case(mini_config).setup().step(0)
        assert p.signal.shape == (4, 4, 4, 3)

    def test_step_sets_current_sample(self, mini_config):
        """step(i) updates _current_sample to i."""
        p = MCDWIPhantom(n_samples=5)
        p.load_case(mini_config).setup()
        p.step(2)
        assert p._current_sample == 2

    def test_step_all_samples(self, mini_config):
        """All sample indices can be stepped through without error."""
        N = 4
        p = MCDWIPhantom(n_samples=N)
        p.load_case(mini_config).setup()
        for i in range(N):
            p.step(i)
            assert p.signal.shape == (4, 4, 4, 3)

    def test_step_noise_free_signal_gt_equals_signal(self, mini_config):
        """Without noise, signal_gt equals signal after step()."""
        p = MCDWIPhantom(n_samples=3)
        p.load_case(mini_config).setup().step(0)
        np.testing.assert_array_equal(p.signal, p.signal_gt)

    def test_step_same_as_batch_sample(self, mini_config):
        """Lazy step(i) produces the same result as the batch synthesize()[...,i]."""
        N = 4
        seed = 5

        batch = MCDWIPhantom(n_samples=N, seed=seed)
        batch.load_case(mini_config).setup().synthesize()

        lazy = MCDWIPhantom(n_samples=N, seed=seed)
        lazy.load_case(mini_config).setup()
        for i in range(N):
            lazy.step(i)
            np.testing.assert_array_equal(
                lazy.signal, batch.signal[..., i],
                err_msg=f"Mismatch at sample {i}",
            )

    def test_step_with_noise_same_as_batch(self, mini_config):
        """Noisy lazy step(i) matches noisy batch synthesize()[...,i]."""
        N = 3
        seed = 11

        batch = MCDWIPhantom(n_samples=N, snr=20.0, seed=seed)
        batch.load_case(mini_config).setup().synthesize()

        lazy = MCDWIPhantom(n_samples=N, snr=20.0, seed=seed)
        lazy.load_case(mini_config).setup()
        for i in range(N):
            lazy.step(i)
            np.testing.assert_array_equal(
                lazy.signal, batch.signal[..., i],
                err_msg=f"Noisy mismatch at sample {i}",
            )


# ---------------------------------------------------------------------------
# save() — batch HDF5 export
# ---------------------------------------------------------------------------


class TestMCDWIPhantomSave:
    """Tests for save() in batch mode."""

    def test_save_before_synthesize_raises(self, mini_config, tmp_path):
        """save() before synthesize() raises RuntimeError."""
        p = MCDWIPhantom(n_samples=3)
        p.load_case(mini_config).setup()
        with pytest.raises(RuntimeError, match="synthesize"):
            p.save(tmp_path / "out.h5")

    def test_save_no_paths_raises(self, mini_config):
        """save() with no arguments raises ValueError."""
        p = MCDWIPhantom(n_samples=3)
        p.load_case(mini_config).setup().synthesize()
        with pytest.raises(ValueError, match="at least one"):
            p.save()

    def test_save_nifti_raises(self, mini_config, tmp_path):
        """Saving to a .nii.gz path raises ValueError (5-D cannot be NIfTI)."""
        p = MCDWIPhantom(n_samples=3)
        p.load_case(mini_config).setup().synthesize()
        with pytest.raises(ValueError, match="NIfTI"):
            p.save(tmp_path / "out.nii.gz")

    def test_save_unknown_extension_raises(self, mini_config, tmp_path):
        """Saving to an unknown extension raises ValueError."""
        p = MCDWIPhantom(n_samples=3)
        p.load_case(mini_config).setup().synthesize()
        with pytest.raises(ValueError, match="Unrecognised"):
            p.save(tmp_path / "out.xyz")

    def test_save_hdf5_creates_file(self, mini_config, tmp_path):
        """save() creates the HDF5 file."""
        out = tmp_path / "mc.h5"
        p = MCDWIPhantom(n_samples=3)
        p.load_case(mini_config).setup().synthesize().save(out)
        assert out.is_file()

    def test_save_returns_path_list(self, mini_config, tmp_path):
        """save() returns a list containing the written path."""
        out = tmp_path / "mc.h5"
        p = MCDWIPhantom(n_samples=3)
        written = p.load_case(mini_config).setup().synthesize().save(out)
        assert isinstance(written, list)
        assert len(written) == 1
        assert written[0] == out

    def test_saved_hdf5_signal_shape(self, mini_config, tmp_path):
        """Loaded HDF5 has signal shape (4, 4, 4, 3, N)."""
        N = 4
        out = tmp_path / "mc.h5"
        MCDWIPhantom(n_samples=N).load_case(mini_config).setup().synthesize().save(out)
        data = load_mc_hdf5(out)
        assert data.signal.shape == (4, 4, 4, 3, N)

    def test_saved_hdf5_signal_gt_shape(self, mini_config, tmp_path):
        """Loaded HDF5 has signal_gt shape matching signal."""
        N = 3
        out = tmp_path / "mc.h5"
        MCDWIPhantom(n_samples=N).load_case(mini_config).setup().synthesize().save(out)
        data = load_mc_hdf5(out)
        assert data.signal_gt.shape == data.signal.shape

    def test_saved_hdf5_metadata(self, mini_config, tmp_path):
        """Metadata (case_id, n_samples, snr, seed) is stored in HDF5."""
        N = 3
        out = tmp_path / "mc.h5"
        p = MCDWIPhantom(n_samples=N, snr=25.0, seed=99)
        p.load_case(mini_config).setup().synthesize().save(out)
        data = load_mc_hdf5(out)
        assert data.case_id == "mini_001"
        assert data.n_samples == N
        assert data.snr == pytest.approx(25.0)
        assert data.seed == 99

    def test_saved_hdf5_region_names(self, mini_config, tmp_path):
        """region_names in HDF5 match the config regions."""
        out = tmp_path / "mc.h5"
        p = MCDWIPhantom(n_samples=2)
        p.load_case(mini_config).setup().synthesize().save(out)
        data = load_mc_hdf5(out)
        assert set(data.region_names) == {"region_a", "region_b"}

    def test_saved_hdf5_roundtrip_signal_values(self, mini_config, tmp_path):
        """Saved and reloaded signal values match the in-memory batch signal."""
        out = tmp_path / "mc.h5"
        p = MCDWIPhantom(n_samples=3, seed=0)
        p.load_case(mini_config).setup().synthesize().save(out)
        data = load_mc_hdf5(out)
        np.testing.assert_allclose(data.signal, p.signal)
        np.testing.assert_allclose(data.signal_gt, p.signal_gt)


# ---------------------------------------------------------------------------
# open_h5 / _H5Writer — lazy mode
# ---------------------------------------------------------------------------


class TestMCDWIPhantomOpenH5:
    """Tests for lazy HDF5 writing via open_h5() and _H5Writer."""

    def test_open_h5_before_setup_raises(self, mini_config, tmp_path):
        """open_h5() before setup() raises RuntimeError."""
        p = MCDWIPhantom(n_samples=5)
        p.load_case(mini_config)
        with pytest.raises(RuntimeError, match="setup"):
            p.open_h5(tmp_path / "out.h5")

    def test_open_h5_returns_context_manager(self, mini_config, tmp_path):
        """open_h5() returns an object usable as a context manager."""
        p = MCDWIPhantom(n_samples=3)
        p.load_case(mini_config).setup()
        writer = p.open_h5(tmp_path / "out.h5")
        assert hasattr(writer, "__enter__")
        assert hasattr(writer, "__exit__")
        # Clean up without entering to avoid file creation side effects
        # (just checks the attribute, not the full protocol)

    def test_lazy_write_creates_file(self, mini_config, tmp_path):
        """open_h5 context manager creates the HDF5 file."""
        N = 3
        out = tmp_path / "lazy.h5"
        p = MCDWIPhantom(n_samples=N)
        p.load_case(mini_config).setup()
        with p.open_h5(out) as h5:
            for i in range(N):
                p.step(i)
                h5.flush(i)
        assert out.is_file()

    def test_lazy_write_signal_shape(self, mini_config, tmp_path):
        """Lazy-written HDF5 has signal shape (4, 4, 4, 3, N)."""
        N = 4
        out = tmp_path / "lazy.h5"
        p = MCDWIPhantom(n_samples=N)
        p.load_case(mini_config).setup()
        with p.open_h5(out) as h5:
            for i in range(N):
                p.step(i)
                h5.flush(i)
        data = load_mc_hdf5(out)
        assert data.signal.shape == (4, 4, 4, 3, N)

    def test_lazy_write_matches_batch_write(self, mini_config, tmp_path):
        """Lazy and batch workflows produce identical signal arrays."""
        N = 4
        seed = 13

        # Batch
        batch_out = tmp_path / "batch.h5"
        batch = MCDWIPhantom(n_samples=N, seed=seed)
        batch.load_case(mini_config).setup().synthesize().save(batch_out)

        # Lazy
        lazy_out = tmp_path / "lazy.h5"
        lazy = MCDWIPhantom(n_samples=N, seed=seed)
        lazy.load_case(mini_config).setup()
        with lazy.open_h5(lazy_out) as h5:
            for i in range(N):
                lazy.step(i)
                h5.flush(i)

        batch_data = load_mc_hdf5(batch_out)
        lazy_data = load_mc_hdf5(lazy_out)
        np.testing.assert_array_equal(batch_data.signal, lazy_data.signal)
        np.testing.assert_array_equal(batch_data.signal_gt, lazy_data.signal_gt)

    def test_flush_wrong_index_raises(self, mini_config, tmp_path):
        """flush(i) without calling step(i) first raises ValueError."""
        N = 3
        out = tmp_path / "wrong.h5"
        p = MCDWIPhantom(n_samples=N)
        p.load_case(mini_config).setup()
        with p.open_h5(out) as h5:
            p.step(0)
            with pytest.raises(ValueError, match="step"):
                h5.flush(1)  # wrong index — step(1) was not called

    def test_lazy_write_metadata(self, mini_config, tmp_path):
        """Metadata stored in lazy HDF5 matches phantom parameters."""
        N = 2
        out = tmp_path / "meta.h5"
        p = MCDWIPhantom(n_samples=N, snr=15.0, seed=7)
        p.load_case(mini_config).setup()
        with p.open_h5(out) as h5:
            for i in range(N):
                p.step(i)
                h5.flush(i)
        data = load_mc_hdf5(out)
        assert data.case_id == "mini_001"
        assert data.n_samples == N
        assert data.snr == pytest.approx(15.0)
        assert data.seed == 7

    def test_file_closed_after_context_exit(self, mini_config, tmp_path):
        """HDF5 file is closed after exiting the context manager."""
        import h5py
        N = 2
        out = tmp_path / "close.h5"
        p = MCDWIPhantom(n_samples=N)
        p.load_case(mini_config).setup()
        writer = p.open_h5(out)
        with writer:
            for i in range(N):
                p.step(i)
                writer.flush(i)
        assert writer._file is None


# ---------------------------------------------------------------------------
# get_params
# ---------------------------------------------------------------------------


class TestMCDWIPhantomGetParams:
    """Tests for get_params() and set_params() integration."""

    def test_get_params_includes_n_samples(self):
        """get_params() includes n_samples."""
        p = MCDWIPhantom(n_samples=10)
        assert p.get_params()["n_samples"] == 10

    def test_get_params_includes_mc_regions(self):
        """get_params() includes mc_regions."""
        mc = _mc_regions_a()
        p = MCDWIPhantom(n_samples=5, mc_regions=mc)
        assert p.get_params()["mc_regions"] is mc

    def test_get_params_includes_snr_region(self):
        """get_params() includes snr_region from DWIPhantom."""
        p = MCDWIPhantom(n_samples=5, snr=30.0, snr_region="cortex")
        assert p.get_params()["snr_region"] == "cortex"

    def test_set_params_snr(self):
        """set_params(snr=...) updates snr."""
        p = MCDWIPhantom(n_samples=5)
        p.set_params(snr=30.0)
        assert p.snr == 30.0

    def test_set_params_unknown_raises(self):
        """set_params() with unknown key raises ValueError."""
        p = MCDWIPhantom(n_samples=5)
        with pytest.raises(ValueError, match="unknown parameter"):
            p.set_params(nonexistent=42)


# ---------------------------------------------------------------------------
# from_toml classmethod
# ---------------------------------------------------------------------------


class TestMCDWIPhantomFromToml:
    """Tests for the from_toml() factory classmethod."""

    def test_from_toml_returns_mc_phantom(self, assets_dir):
        """from_toml returns an MCDWIPhantom instance."""
        toml_path = assets_dir / "kidney" / "mc.toml"
        pytest.importorskip("mrphantom.montecarlo.phantom")
        if not toml_path.is_file():
            pytest.skip("mc.toml not found in assets")
        p = MCDWIPhantom.from_toml(toml_path)
        assert isinstance(p, MCDWIPhantom)

    def test_from_toml_stores_mc_config(self, assets_dir):
        """from_toml attaches the MCRunConfig as _mc_config."""
        toml_path = assets_dir / "kidney" / "mc.toml"
        if not toml_path.is_file():
            pytest.skip("mc.toml not found in assets")
        p = MCDWIPhantom.from_toml(toml_path)
        assert hasattr(p, "_mc_config")

    def test_from_toml_n_samples(self, assets_dir):
        """from_toml sets n_samples from TOML."""
        toml_path = assets_dir / "kidney" / "mc.toml"
        if not toml_path.is_file():
            pytest.skip("mc.toml not found in assets")
        p = MCDWIPhantom.from_toml(toml_path)
        assert p.n_samples == p._mc_config.n_samples


# ---------------------------------------------------------------------------
# Full lifecycle chains
# ---------------------------------------------------------------------------


class TestMCDWIPhantomFullChain:
    """Integration tests covering complete MCDWIPhantom lifecycles."""

    def test_batch_chain_noise_free(self, mini_config, tmp_path):
        """Full batch chain without noise completes and writes HDF5."""
        out = tmp_path / "nf.h5"
        MCDWIPhantom(n_samples=3).load_case(mini_config).setup().synthesize().save(out)
        assert out.is_file()

    def test_batch_chain_with_noise(self, mini_config, tmp_path):
        """Full batch chain with noise completes and writes HDF5."""
        out = tmp_path / "noisy.h5"
        MCDWIPhantom(n_samples=3, snr=20.0, seed=0).load_case(
            mini_config
        ).setup().synthesize().save(out)
        assert out.is_file()

    def test_lazy_chain_noise_free(self, mini_config, tmp_path):
        """Full lazy chain without noise completes and writes HDF5."""
        N = 3
        out = tmp_path / "lazy_nf.h5"
        p = MCDWIPhantom(n_samples=N)
        p.load_case(mini_config).setup()
        with p.open_h5(out) as h5:
            for i in range(N):
                p.step(i)
                h5.flush(i)
        assert out.is_file()

    def test_lazy_chain_with_noise(self, mini_config, tmp_path):
        """Full lazy chain with noise completes and writes HDF5."""
        N = 3
        out = tmp_path / "lazy_noisy.h5"
        p = MCDWIPhantom(n_samples=N, snr=25.0, seed=1)
        p.load_case(mini_config).setup()
        with p.open_h5(out) as h5:
            for i in range(N):
                p.step(i)
                h5.flush(i)
        assert out.is_file()

    def test_batch_reuse_across_cases(self, mini_config, tmp_path):
        """A single MCDWIPhantom can be reused across multiple cases."""
        p = MCDWIPhantom(n_samples=3)
        for j in range(2):
            out = tmp_path / f"case_{j}.h5"
            p.load_case(mini_config).setup().synthesize().save(out)
            assert out.is_file()
