"""Tests for mrphantom.config — PhantomConfig, RegionConfig, load_phantom_config."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mrphantom.config import PhantomConfig, RegionConfig, load_phantom_config


# ---------------------------------------------------------------------------
# load_phantom_config — file-not-found guards
# ---------------------------------------------------------------------------


def test_load_phantom_config_missing_json(tmp_path, bval_path):
    with pytest.raises(FileNotFoundError, match="Phantom JSON not found"):
        load_phantom_config(tmp_path / "nonexistent.json", bval_path)


def test_load_phantom_config_missing_bvals(tmp_path, json_path):
    with pytest.raises(FileNotFoundError, match="B-values file not found"):
        load_phantom_config(json_path, tmp_path / "nonexistent.txt")


# ---------------------------------------------------------------------------
# load_phantom_config — happy path (real kidney case)
# ---------------------------------------------------------------------------


def test_load_phantom_config_returns_phantom_config(phantom_config):
    assert isinstance(phantom_config, PhantomConfig)


def test_phantom_config_case_id(phantom_config):
    assert phantom_config.case_id == "kidney_001"


def test_phantom_config_bvalues_shape(phantom_config):
    assert phantom_config.bvalues.ndim == 1
    assert phantom_config.bvalues.size == 6  # 0 50 100 200 400 800


def test_phantom_config_bvalues_values(phantom_config):
    expected = np.array([0.0, 50.0, 100.0, 200.0, 400.0, 800.0])
    np.testing.assert_array_equal(phantom_config.bvalues, expected)


def test_phantom_config_nifti_path_exists(phantom_config):
    assert phantom_config.nifti_path.is_file()


def test_phantom_config_regions_count(phantom_config):
    # The JSON defines 11 regions
    assert len(phantom_config.regions) == 11


def test_phantom_config_all_regions_are_region_config(phantom_config):
    for region in phantom_config.regions.values():
        assert isinstance(region, RegionConfig)


def test_phantom_config_null_region_inherits_remaining_tissue(phantom_config):
    # Region 4 ("bones") has S0: null — should inherit from remaining_tissue (index 7)
    bones = phantom_config.regions[4]
    remaining = phantom_config.regions[7]
    assert bones.S0 == remaining.S0
    assert bones.D == remaining.D
    assert bones.fractions == remaining.fractions


def test_phantom_config_non_null_region_keeps_own_params(phantom_config):
    cortex = phantom_config.regions[10]
    assert cortex.S0 == 180.0
    assert cortex.n_components == 3


# ---------------------------------------------------------------------------
# load_phantom_config — b-values file edge cases
# ---------------------------------------------------------------------------


def test_load_phantom_config_empty_bvals_raises(tmp_path, json_path):
    empty_bval = tmp_path / "empty.txt"
    empty_bval.write_text("")
    with pytest.raises(ValueError, match="B-values file is empty"):
        load_phantom_config(json_path, empty_bval)


# ---------------------------------------------------------------------------
# mini_config fixture sanity
# ---------------------------------------------------------------------------


def test_mini_config_fixture(mini_config):
    assert mini_config.case_id == "mini_001"
    assert len(mini_config.regions) == 2
    assert mini_config.bvalues.shape == (3,)
    assert mini_config.nifti_path.is_file()
