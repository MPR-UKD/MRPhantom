"""Tests for mrphantom.io — save_nifti, save_hdf5, load_hdf5, PhantomData."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from mrphantom.io import PhantomData, load_hdf5, save_hdf5, save_nifti
from mrphantom.synthesis import synthesize_phantom


# ---------------------------------------------------------------------------
# save_nifti
# ---------------------------------------------------------------------------


def test_save_nifti_creates_file(mini_nifti, tmp_path):
    out = tmp_path / "out.nii.gz"
    result = save_nifti(mini_nifti, out)
    assert result.is_file()


def test_save_nifti_returns_path(mini_nifti, tmp_path):
    out = tmp_path / "out.nii.gz"
    result = save_nifti(mini_nifti, out)
    assert isinstance(result, Path)
    assert result == out


def test_save_nifti_appends_extension_if_missing(mini_nifti, tmp_path):
    out = tmp_path / "out_no_ext"
    result = save_nifti(mini_nifti, out)
    assert result.name.endswith(".nii.gz")
    assert result.is_file()


def test_save_nifti_creates_parent_dirs(mini_nifti, tmp_path):
    out = tmp_path / "deep" / "nested" / "out.nii.gz"
    save_nifti(mini_nifti, out)
    assert out.is_file()


def test_save_nifti_not_a_nifti_raises(tmp_path):
    with pytest.raises(TypeError, match="nibabel.Nifti1Image"):
        save_nifti("not_an_image", tmp_path / "out.nii.gz")  # type: ignore[arg-type]


def test_save_nifti_roundtrip(mini_nifti, tmp_path):
    out = tmp_path / "round.nii.gz"
    save_nifti(mini_nifti, out)
    loaded = nib.load(out)
    np.testing.assert_array_equal(loaded.get_fdata(), mini_nifti.get_fdata())


# ---------------------------------------------------------------------------
# save_hdf5 / load_hdf5
# ---------------------------------------------------------------------------


@pytest.fixture
def synthesized(mini_config):
    signal, signal_img = synthesize_phantom(mini_config)
    return signal, signal_img, mini_config


def test_save_hdf5_creates_file(synthesized, tmp_path):
    signal, signal_img, config = synthesized
    out = tmp_path / "out.h5"
    result = save_hdf5(signal, signal_img, config, out)
    assert result.is_file()


def test_save_hdf5_returns_path(synthesized, tmp_path):
    signal, signal_img, config = synthesized
    out = tmp_path / "out.h5"
    result = save_hdf5(signal, signal_img, config, out)
    assert isinstance(result, Path)


def test_save_hdf5_appends_extension_if_missing(synthesized, tmp_path):
    signal, signal_img, config = synthesized
    out = tmp_path / "no_ext"
    result = save_hdf5(signal, signal_img, config, out)
    assert result.suffix == ".h5"
    assert result.is_file()


def test_load_hdf5_returns_phantom_data(synthesized, tmp_path):
    signal, signal_img, config = synthesized
    out = tmp_path / "out.h5"
    save_hdf5(signal, signal_img, config, out)
    data = load_hdf5(out)
    assert isinstance(data, PhantomData)


def test_load_hdf5_signal_roundtrip(synthesized, tmp_path):
    signal, signal_img, config = synthesized
    out = tmp_path / "out.h5"
    save_hdf5(signal, signal_img, config, out)
    data = load_hdf5(out)
    np.testing.assert_allclose(data.signal, signal, rtol=1e-10)


def test_load_hdf5_bvalues_roundtrip(synthesized, tmp_path):
    signal, signal_img, config = synthesized
    out = tmp_path / "out.h5"
    save_hdf5(signal, signal_img, config, out)
    data = load_hdf5(out)
    np.testing.assert_array_equal(data.bvalues, config.bvalues)


def test_load_hdf5_metadata(synthesized, tmp_path):
    signal, signal_img, config = synthesized
    out = tmp_path / "out.h5"
    save_hdf5(signal, signal_img, config, out, snr=25.0)
    data = load_hdf5(out)
    assert data.case_id == config.case_id
    assert data.name == config.name
    assert data.snr == pytest.approx(25.0)


def test_load_hdf5_snr_none_when_not_stored(synthesized, tmp_path):
    signal, signal_img, config = synthesized
    out = tmp_path / "out.h5"
    save_hdf5(signal, signal_img, config, out, snr=None)
    data = load_hdf5(out)
    assert data.snr is None


def test_load_hdf5_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_hdf5(tmp_path / "nonexistent.h5")


def test_phantom_data_as_nifti(synthesized, tmp_path):
    signal, signal_img, config = synthesized
    out = tmp_path / "out.h5"
    save_hdf5(signal, signal_img, config, out)
    data = load_hdf5(out)
    img = data.as_nifti()
    assert isinstance(img, nib.Nifti1Image)
    assert img.shape == signal.shape


def test_save_hdf5_missing_segmentation_raises(synthesized, tmp_path):
    signal, signal_img, config = synthesized
    from mrphantom.config import PhantomConfig

    bad_config = PhantomConfig(
        case_id=config.case_id,
        name=config.name,
        description=config.description,
        nifti_path=tmp_path / "ghost.nii.gz",
        image_dims=config.image_dims,
        bvalues=config.bvalues,
        regions=config.regions,
    )
    with pytest.raises(FileNotFoundError):
        save_hdf5(signal, signal_img, bad_config, tmp_path / "out.h5")
