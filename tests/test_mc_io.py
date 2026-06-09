"""Tests for mrphantom.io.mc_hdf5 — save_mc_hdf5 and load_mc_hdf5."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mrphantom.io.mc_hdf5 import MCPhantomData, load_mc_hdf5, save_mc_hdf5
from mrphantom.montecarlo.runner import run_montecarlo


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mc_result(mini_config):
    """A small MCResult (5 samples, noise-free) for I/O tests."""
    return run_montecarlo(
        base_config=mini_config,
        mc_regions={},
        n_samples=5,
        snr=None,
        snr_region=None,
        seed=0,
    )


@pytest.fixture
def mc_result_noisy(mini_config):
    """A small MCResult (4 samples, snr=20) for noise-related I/O tests."""
    return run_montecarlo(
        base_config=mini_config,
        mc_regions={},
        n_samples=4,
        snr=20.0,
        snr_region=None,
        seed=1,
    )


def _save(mc_result, mini_config, tmp_path, **kwargs) -> Path:
    """Helper: save mc_result to a temp HDF5 and return the path."""
    out = tmp_path / "mc.h5"
    defaults = dict(
        signal=mc_result.signal,
        signal_gt=mc_result.signal_gt,
        volumes=mini_config.bvalues,
        segmentation=mini_config.nifti_path,
        affine=mc_result.affine,
        S0_samples=mc_result.sampled_params.S0,
        D_samples=mc_result.sampled_params.D,
        fractions_samples=mc_result.sampled_params.fractions,
        region_names=mc_result.sampled_params.region_names,
        out_path=out,
        case_id="mini_001",
        name="mini",
        description="test",
        snr=None,
        snr_region=None,
        seed=0,
    )
    defaults.update(kwargs)
    return save_mc_hdf5(**defaults)


# ---------------------------------------------------------------------------
# save_mc_hdf5
# ---------------------------------------------------------------------------


class TestSaveMcHdf5:
    """Tests for save_mc_hdf5()."""

    def test_creates_file(self, mc_result, mini_config, tmp_path):
        """save_mc_hdf5 creates an HDF5 file on disk."""
        path = _save(mc_result, mini_config, tmp_path)
        assert path.is_file()

    def test_returns_path(self, mc_result, mini_config, tmp_path):
        """Return value is the resolved path of the written file."""
        path = _save(mc_result, mini_config, tmp_path)
        assert isinstance(path, Path)

    def test_appends_h5_extension(self, mc_result, mini_config, tmp_path):
        """'.h5' suffix is appended when missing."""
        out = tmp_path / "mc_output"
        path = save_mc_hdf5(
            signal=mc_result.signal,
            signal_gt=mc_result.signal_gt,
            volumes=mini_config.bvalues,
            segmentation=mini_config.nifti_path,
            affine=mc_result.affine,
            S0_samples=mc_result.sampled_params.S0,
            D_samples=mc_result.sampled_params.D,
            fractions_samples=mc_result.sampled_params.fractions,
            region_names=mc_result.sampled_params.region_names,
            out_path=out,
        )
        assert path.suffix == ".h5"
        assert path.is_file()

    def test_missing_segmentation_nifti_raises(self, mc_result, mini_config, tmp_path):
        """FileNotFoundError is raised when segmentation path doesn't exist."""
        with pytest.raises(FileNotFoundError):
            save_mc_hdf5(
                signal=mc_result.signal,
                signal_gt=mc_result.signal_gt,
                volumes=mini_config.bvalues,
                segmentation=tmp_path / "nonexistent.nii.gz",
                affine=mc_result.affine,
                S0_samples=mc_result.sampled_params.S0,
                D_samples=mc_result.sampled_params.D,
                fractions_samples=mc_result.sampled_params.fractions,
                region_names=mc_result.sampled_params.region_names,
                out_path=tmp_path / "out.h5",
            )


# ---------------------------------------------------------------------------
# load_mc_hdf5
# ---------------------------------------------------------------------------


class TestLoadMcHdf5:
    """Tests for load_mc_hdf5()."""

    def test_returns_mc_phantom_data(self, mc_result, mini_config, tmp_path):
        """load_mc_hdf5 returns an MCPhantomData instance."""
        path = _save(mc_result, mini_config, tmp_path)
        data = load_mc_hdf5(path)
        assert isinstance(data, MCPhantomData)

    def test_missing_file_raises(self, tmp_path):
        """FileNotFoundError is raised for a non-existent file."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_mc_hdf5(tmp_path / "missing.h5")

    def test_signal_roundtrip(self, mc_result, mini_config, tmp_path):
        """Loaded signal matches saved signal."""
        path = _save(mc_result, mini_config, tmp_path)
        data = load_mc_hdf5(path)
        np.testing.assert_array_equal(data.signal, mc_result.signal)

    def test_signal_gt_roundtrip(self, mc_result, mini_config, tmp_path):
        """Loaded signal_gt matches saved signal_gt."""
        path = _save(mc_result, mini_config, tmp_path)
        data = load_mc_hdf5(path)
        np.testing.assert_array_equal(data.signal_gt, mc_result.signal_gt)

    def test_volumes_roundtrip(self, mc_result, mini_config, tmp_path):
        """Loaded volumes (b-values) match original."""
        path = _save(mc_result, mini_config, tmp_path)
        data = load_mc_hdf5(path)
        np.testing.assert_array_equal(data.volumes, mini_config.bvalues)

    def test_n_samples_roundtrip(self, mc_result, mini_config, tmp_path):
        """n_samples attribute is preserved."""
        path = _save(mc_result, mini_config, tmp_path)
        data = load_mc_hdf5(path)
        assert data.n_samples == 5

    def test_case_id_roundtrip(self, mc_result, mini_config, tmp_path):
        """case_id metadata is preserved."""
        path = _save(mc_result, mini_config, tmp_path, case_id="test_case")
        data = load_mc_hdf5(path)
        assert data.case_id == "test_case"

    def test_snr_roundtrip(self, mc_result_noisy, mini_config, tmp_path):
        """snr attribute is preserved."""
        path = _save(mc_result_noisy, mini_config, tmp_path, snr=20.0)
        data = load_mc_hdf5(path)
        assert data.snr == pytest.approx(20.0)

    def test_snr_none_when_not_stored(self, mc_result, mini_config, tmp_path):
        """snr is None when not stored (noise-free phantom)."""
        path = _save(mc_result, mini_config, tmp_path, snr=None)
        data = load_mc_hdf5(path)
        assert data.snr is None

    def test_snr_region_roundtrip(self, mc_result, mini_config, tmp_path):
        """snr_region is preserved when set."""
        path = _save(mc_result, mini_config, tmp_path, snr_region="region_a")
        data = load_mc_hdf5(path)
        assert data.snr_region == "region_a"

    def test_snr_region_none_when_absent(self, mc_result, mini_config, tmp_path):
        """snr_region is None when not stored."""
        path = _save(mc_result, mini_config, tmp_path, snr_region=None)
        data = load_mc_hdf5(path)
        assert data.snr_region is None

    def test_seed_roundtrip(self, mc_result, mini_config, tmp_path):
        """seed attribute is preserved."""
        path = _save(mc_result, mini_config, tmp_path, seed=42)
        data = load_mc_hdf5(path)
        assert data.seed == 42

    def test_seed_none_when_absent(self, mc_result, mini_config, tmp_path):
        """seed is None when not stored."""
        path = _save(mc_result, mini_config, tmp_path, seed=None)
        data = load_mc_hdf5(path)
        assert data.seed is None

    def test_S0_samples_roundtrip(self, mc_result, mini_config, tmp_path):
        """S0_samples array is preserved."""
        path = _save(mc_result, mini_config, tmp_path)
        data = load_mc_hdf5(path)
        np.testing.assert_array_equal(data.S0_samples, mc_result.sampled_params.S0)

    def test_D_samples_roundtrip(self, mc_result, mini_config, tmp_path):
        """D_samples array is preserved."""
        path = _save(mc_result, mini_config, tmp_path)
        data = load_mc_hdf5(path)
        np.testing.assert_array_equal(data.D_samples, mc_result.sampled_params.D)

    def test_fractions_samples_roundtrip(self, mc_result, mini_config, tmp_path):
        """fractions_samples array is preserved."""
        path = _save(mc_result, mini_config, tmp_path)
        data = load_mc_hdf5(path)
        np.testing.assert_array_equal(
            data.fractions_samples, mc_result.sampled_params.fractions
        )

    def test_region_names_roundtrip(self, mc_result, mini_config, tmp_path):
        """region_names list is preserved."""
        path = _save(mc_result, mini_config, tmp_path)
        data = load_mc_hdf5(path)
        assert data.region_names == mc_result.sampled_params.region_names

    def test_signal_shape(self, mc_result, mini_config, tmp_path):
        """Loaded signal has expected 5-D shape."""
        path = _save(mc_result, mini_config, tmp_path)
        data = load_mc_hdf5(path)
        assert data.signal.ndim == 5
        assert data.signal.shape[-1] == 5  # n_samples

    def test_segmentation_shape(self, mc_result, mini_config, tmp_path):
        """Loaded segmentation has expected 3-D shape."""
        path = _save(mc_result, mini_config, tmp_path)
        data = load_mc_hdf5(path)
        assert data.segmentation.ndim == 3
        assert data.segmentation.shape == (4, 4, 4)

    def test_affine_roundtrip(self, mc_result, mini_config, tmp_path):
        """Affine matrix is preserved."""
        path = _save(mc_result, mini_config, tmp_path)
        data = load_mc_hdf5(path)
        np.testing.assert_array_equal(data.affine, mc_result.affine)
