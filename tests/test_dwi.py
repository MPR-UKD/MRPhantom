"""Tests for mrphantom.dwi — DWIPhantom."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from mrphantom import DWIPhantom, load_dwi_config
from mrphantom.dwi.config import DWIConfig


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


def test_dwiphantom_defaults():
    p = DWIPhantom()
    assert p.snr is None
    assert p.seed is None


def test_dwiphantom_with_params():
    p = DWIPhantom(snr=30.0, seed=42)
    assert p.snr == 30.0
    assert p.seed == 42


# ---------------------------------------------------------------------------
# from_json classmethod
# ---------------------------------------------------------------------------


def test_from_json_returns_dwiphantom(json_path, bval_path):
    p = DWIPhantom.from_json(json_path, bval_path)
    assert isinstance(p, DWIPhantom)


def test_from_json_config_is_loaded(json_path, bval_path):
    p = DWIPhantom.from_json(json_path, bval_path)
    assert isinstance(p.config, DWIConfig)
    assert p.config.case_id == "kidney_000"


def test_from_json_passes_snr_seed(json_path, bval_path):
    p = DWIPhantom.from_json(json_path, bval_path, snr=20.0, seed=1)
    assert p.snr == 20.0
    assert p.seed == 1


# ---------------------------------------------------------------------------
# load_case
# ---------------------------------------------------------------------------


def test_load_case_returns_self(mini_config):
    p = DWIPhantom()
    assert p.load_case(mini_config) is p


def test_load_case_makes_config_accessible(mini_config):
    p = DWIPhantom()
    p.load_case(mini_config)
    assert p.config is mini_config


def test_load_case_clears_stale_signal(mini_config):
    """After load_case on a used phantom, signal and signal_gt must be inaccessible."""
    p = DWIPhantom()
    p.load_case(mini_config).synthesize()
    # Now reload a case — stale signal should be cleared
    p.load_case(mini_config)
    with pytest.raises(RuntimeError):
        _ = p.signal
    with pytest.raises(RuntimeError):
        _ = p.signal_gt


# ---------------------------------------------------------------------------
# synthesize
# ---------------------------------------------------------------------------


def test_synthesize_before_load_raises():
    p = DWIPhantom()
    with pytest.raises(RuntimeError, match="load_case"):
        p.synthesize()


def test_synthesize_returns_self(mini_config):
    p = DWIPhantom()
    assert p.load_case(mini_config).synthesize() is p


def test_synthesize_signal_accessible(mini_config):
    p = DWIPhantom()
    p.load_case(mini_config).synthesize()
    assert isinstance(p.signal, np.ndarray)
    assert p.signal.shape == (4, 4, 4, 3)


def test_synthesize_signal_gt_accessible(mini_config):
    p = DWIPhantom()
    p.load_case(mini_config).synthesize()
    assert isinstance(p.signal_gt, np.ndarray)
    assert p.signal_gt.shape == (4, 4, 4, 3)


def test_synthesize_signal_img_accessible(mini_config):
    p = DWIPhantom()
    p.load_case(mini_config).synthesize()
    assert isinstance(p.signal_img, nib.Nifti1Image)


# ---------------------------------------------------------------------------
# Properties — error before synthesis
# ---------------------------------------------------------------------------


def test_config_before_load_raises():
    p = DWIPhantom()
    with pytest.raises(RuntimeError, match="load_case"):
        _ = p.config


def test_signal_before_synthesize_raises(mini_config):
    p = DWIPhantom()
    p.load_case(mini_config)
    with pytest.raises(RuntimeError, match="synthesize"):
        _ = p.signal


def test_signal_gt_before_synthesize_raises(mini_config):
    p = DWIPhantom()
    p.load_case(mini_config)
    with pytest.raises(RuntimeError, match="synthesize"):
        _ = p.signal_gt


def test_signal_img_before_synthesize_raises(mini_config):
    p = DWIPhantom()
    p.load_case(mini_config)
    with pytest.raises(RuntimeError, match="synthesize"):
        _ = p.signal_img


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------


def test_save_nifti(mini_config, tmp_path):
    out = tmp_path / "out.nii.gz"
    DWIPhantom().load_case(mini_config).synthesize().save(str(out))
    assert out.is_file()


def test_save_hdf5(mini_config, tmp_path):
    out = tmp_path / "out.h5"
    DWIPhantom().load_case(mini_config).synthesize().save(str(out))
    assert out.is_file()


def test_save_both_formats(mini_config, tmp_path):
    nii = tmp_path / "out.nii.gz"
    h5 = tmp_path / "out.h5"
    written = DWIPhantom().load_case(mini_config).synthesize().save(nii, h5)
    assert len(written) == 2
    assert nii.is_file()
    assert h5.is_file()


def test_save_before_synthesize_raises(mini_config, tmp_path):
    p = DWIPhantom()
    p.load_case(mini_config)
    with pytest.raises(RuntimeError, match="synthesize"):
        p.save(tmp_path / "out.nii.gz")


def test_save_no_paths_raises(mini_config):
    p = DWIPhantom().load_case(mini_config).synthesize()
    with pytest.raises(ValueError, match="at least one"):
        p.save()


def test_save_unknown_extension_raises(mini_config, tmp_path):
    p = DWIPhantom().load_case(mini_config).synthesize()
    with pytest.raises(ValueError, match="Unrecognised"):
        p.save(tmp_path / "out.xyz")


# ---------------------------------------------------------------------------
# get_params / set_params (inherited)
# ---------------------------------------------------------------------------


def test_get_params(mini_config):
    p = DWIPhantom(snr=30.0, seed=5)
    assert p.get_params() == {"snr": 30.0, "seed": 5, "snr_region": None}


def test_set_params_returns_self():
    p = DWIPhantom()
    assert p.set_params(snr=10.0) is p
    assert p.snr == 10.0


# ---------------------------------------------------------------------------
# Full chain — noise-free and noisy
# ---------------------------------------------------------------------------


def test_full_chain_noise_free(mini_config, tmp_path):
    out = tmp_path / "chain.nii.gz"
    DWIPhantom().load_case(mini_config).synthesize().save(str(out))
    assert out.is_file()


def test_full_chain_with_noise(mini_config, tmp_path):
    out = tmp_path / "noisy.h5"
    DWIPhantom(snr=20.0, seed=0).load_case(mini_config).synthesize().save(str(out))
    assert out.is_file()


def test_batch_reuse(mini_config):
    """Reusing one DWIPhantom instance across multiple load_case calls."""
    p = DWIPhantom(snr=30.0, seed=42)
    results = []
    for _ in range(3):
        p.load_case(mini_config).synthesize()
        results.append(p.signal.copy())
    # All three runs should be identical (same config + same seed)
    np.testing.assert_array_equal(results[0], results[1])
    np.testing.assert_array_equal(results[1], results[2])


# ---------------------------------------------------------------------------
# snr_region — DWIPhantom integration
# ---------------------------------------------------------------------------


def test_dwiphantom_snr_region_stored():
    p = DWIPhantom(snr=20.0, snr_region="cortex")
    assert p.snr_region == "cortex"


def test_dwiphantom_snr_region_default_none():
    p = DWIPhantom(snr=20.0)
    assert p.snr_region is None


def test_get_params_includes_snr_region():
    p = DWIPhantom(snr=30.0, seed=5, snr_region="medulla")
    params = p.get_params()
    assert "snr_region" in params
    assert params["snr_region"] == "medulla"


def test_get_params_snr_region_none_by_default():
    p = DWIPhantom(snr=30.0, seed=5)
    assert p.get_params()["snr_region"] is None


def test_set_params_snr_region():
    p = DWIPhantom()
    p.set_params(snr_region="region_a")
    assert p.snr_region == "region_a"


def test_from_json_passes_snr_region(json_path, bval_path):
    p = DWIPhantom.from_json(json_path, bval_path, snr=20.0, snr_region="cortex")
    assert p.snr_region == "cortex"


def test_synthesize_with_snr_region(mini_config):
    """DWIPhantom with snr_region should produce a valid signal array."""
    p = DWIPhantom(snr=20.0, seed=0, snr_region="region_a")
    p.load_case(mini_config).synthesize()
    assert p.signal.shape == (4, 4, 4, 3)
    assert (p.signal >= 0).all()


def test_synthesize_invalid_snr_region_raises(mini_config):
    """DWIPhantom should propagate ValueError for unknown snr_region."""
    p = DWIPhantom(snr=20.0, seed=0, snr_region="no_such_region")
    p.load_case(mini_config)
    with pytest.raises(ValueError, match="snr_region"):
        p.synthesize()


# ---------------------------------------------------------------------------
# signal_gt — noise-free ground-truth property
# ---------------------------------------------------------------------------


def test_signal_gt_equals_signal_when_noise_free(mini_config):
    """Without noise, signal_gt and signal must be identical."""
    p = DWIPhantom()
    p.load_case(mini_config).synthesize()
    np.testing.assert_array_equal(p.signal_gt, p.signal)


def test_signal_gt_differs_from_noisy_signal(mini_config):
    """With noise applied, signal_gt must differ from the noisy signal."""
    p = DWIPhantom(snr=10.0, seed=0)
    p.load_case(mini_config).synthesize()
    assert not np.allclose(p.signal, p.signal_gt)


def test_signal_gt_is_noise_free(mini_config):
    """signal_gt must equal the clean model output independent of the noise seed."""
    clean = DWIPhantom().load_case(mini_config).synthesize().signal_gt
    noisy_phantom = DWIPhantom(snr=10.0, seed=0)
    noisy_phantom.load_case(mini_config).synthesize()
    np.testing.assert_array_equal(noisy_phantom.signal_gt, clean)
