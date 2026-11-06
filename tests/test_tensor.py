"""Tests for tensor regions and gradient directions (config + synthesis)."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from mrphantom.dwi.config import DWIConfig, RegionConfig, load_dwi_config
from mrphantom.dwi.synthesis import synthesize_phantom

BVALS = np.array([0.0, 500.0, 500.0, 500.0, 1000.0])
BVECS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0] / np.sqrt(2.0),
    ]
)
EVALS = [0.0020, 0.0005, 0.0005]
EVECS = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def _write_case(tmp_path: Path, tensor_region: dict | None = None) -> tuple[Path, Path]:
    """Write a 2-region case (label 1: tensor, label 2: remaining_tissue)."""
    seg = np.zeros((4, 4, 4), dtype=np.int32)
    seg[:2] = 1
    seg[2:] = 2
    nib.save(nib.Nifti1Image(seg, np.eye(4)), tmp_path / "seg.nii.gz")
    tensor_region = tensor_region or {
        "S0": 100.0,
        "tensor": {"evals": EVALS, "evecs": EVECS},
    }
    case = {
        "case_id": "t",
        "name": "t",
        "description": "t",
        "file": "seg.nii.gz",
        "image": {"dimensions": [4, 4, 4]},
        "regions": {
            "a": {"name": "organ", "index": 1, "diffusivity": tensor_region},
            "b": {
                "name": "remaining_tissue",
                "index": 2,
                "diffusivity": {
                    "n_components": 1,
                    "S0": 50.0,
                    "fractions": [1.0],
                    "D": [0.001],
                },
            },
        },
    }
    json_path = tmp_path / "case.json"
    json_path.write_text(json.dumps(case), encoding="utf-8")
    bval_path = tmp_path / "case.bval"
    np.savetxt(bval_path, BVALS[None, :])
    return json_path, bval_path


def _write_bvec(tmp_path: Path, bvecs: np.ndarray = BVECS, fsl: bool = True) -> Path:
    path = tmp_path / "case.bvec"
    np.savetxt(path, bvecs.T if fsl else bvecs)
    return path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_tensor_region_is_loaded(tmp_path):
    json_path, bval_path = _write_case(tmp_path)
    cfg = load_dwi_config(json_path, bval_path, _write_bvec(tmp_path))
    organ = cfg.regions[1]
    np.testing.assert_allclose(organ.tensor, np.diag(EVALS))
    assert organ.n_components == 1
    assert organ.fractions == [1.0]
    assert organ.D == pytest.approx([np.mean(EVALS)])
    assert cfg.regions[2].tensor is None


def test_tensor_from_rotated_evecs(tmp_path):
    # Principal axis along y: evecs row 0 is the eigenvector of evals[0].
    evecs = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
    json_path, bval_path = _write_case(
        tmp_path, {"S0": 100.0, "tensor": {"evals": EVALS, "evecs": evecs}}
    )
    cfg = load_dwi_config(json_path, bval_path)
    np.testing.assert_allclose(cfg.regions[1].tensor, np.diag([0.0005, 0.0020, 0.0005]))


def test_bvecs_both_orientations(tmp_path):
    json_path, bval_path = _write_case(tmp_path)
    fsl = load_dwi_config(json_path, bval_path, _write_bvec(tmp_path, fsl=True))
    rows = load_dwi_config(json_path, bval_path, _write_bvec(tmp_path, fsl=False))
    np.testing.assert_allclose(fsl.bvecs, BVECS)
    np.testing.assert_allclose(rows.bvecs, BVECS)


def test_no_bvec_path_gives_none(tmp_path):
    json_path, bval_path = _write_case(tmp_path)
    assert load_dwi_config(json_path, bval_path).bvecs is None


def test_bvecs_wrong_length_raises(tmp_path):
    json_path, bval_path = _write_case(tmp_path)
    bvec = _write_bvec(tmp_path, BVECS[:3])
    with pytest.raises(ValueError, match="shape"):
        load_dwi_config(json_path, bval_path, bvec)


def test_bvecs_not_unit_raises(tmp_path):
    json_path, bval_path = _write_case(tmp_path)
    bad = BVECS.copy()
    bad[1] = [2.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="unit vector"):
        load_dwi_config(json_path, bval_path, _write_bvec(tmp_path, bad))


def test_bvec_file_missing_raises(tmp_path):
    json_path, bval_path = _write_case(tmp_path)
    with pytest.raises(FileNotFoundError, match="B-vectors file not found"):
        load_dwi_config(json_path, bval_path, tmp_path / "nope.bvec")


@pytest.mark.parametrize(
    "tensor, match",
    [
        ({"evals": [1e-3, 1e-3], "evecs": EVECS}, "3 evals"),
        ({"evals": [-1e-3, 1e-3, 1e-3], "evecs": EVECS}, "negative"),
        ({"evals": EVALS, "evecs": [[1, 0, 0], [1, 0, 0], [0, 0, 1]]}, "orthonormal"),
    ],
)
def test_invalid_tensor_raises(tmp_path, tensor, match):
    json_path, bval_path = _write_case(tmp_path, {"S0": 100.0, "tensor": tensor})
    with pytest.raises(ValueError, match=match):
        load_dwi_config(json_path, bval_path)


def test_tensor_and_D_together_raise(tmp_path):
    json_path, bval_path = _write_case(
        tmp_path,
        {
            "S0": 100.0,
            "n_components": 1,
            "fractions": [1.0],
            "D": [0.001],
            "tensor": {"evals": EVALS, "evecs": EVECS},
        },
    )
    with pytest.raises(ValueError, match="both 'tensor' and 'D'"):
        load_dwi_config(json_path, bval_path)


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


def test_tensor_signal_matches_analytic(tmp_path):
    json_path, bval_path = _write_case(tmp_path)
    cfg = load_dwi_config(json_path, bval_path, _write_bvec(tmp_path))
    signal, gt, _ = synthesize_phantom(cfg)
    adc = np.array([0.0, 0.0020, 0.0005, 0.0005, 0.5 * (0.0020 + 0.0005)])
    np.testing.assert_allclose(signal[0, 0, 0], 100.0 * np.exp(-BVALS * adc))
    np.testing.assert_array_equal(signal, gt)
    # scalar region unaffected
    np.testing.assert_allclose(signal[3, 0, 0], 50.0 * np.exp(-BVALS * 0.001))


def test_isotropic_tensor_equals_scalar_D(tmp_path):
    iso = {"S0": 100.0, "tensor": {"evals": [1e-3] * 3, "evecs": EVECS}}
    json_path, bval_path = _write_case(tmp_path, iso)
    cfg = load_dwi_config(json_path, bval_path, _write_bvec(tmp_path))
    signal, _, _ = synthesize_phantom(cfg)
    np.testing.assert_allclose(signal[0, 0, 0], 100.0 * np.exp(-BVALS * 1e-3))


def test_tensor_without_bvecs_raises(tmp_path):
    json_path, bval_path = _write_case(tmp_path)
    cfg = load_dwi_config(json_path, bval_path)
    with pytest.raises(ValueError, match="no bvecs"):
        synthesize_phantom(cfg)


def test_bvecs_ignored_without_tensor_regions(mini_config):
    """Scalar-D phantoms are byte-identical with or without bvecs."""
    base, _, _ = synthesize_phantom(mini_config)
    mini_config.bvecs = np.tile([1.0, 0.0, 0.0], (3, 1))
    with_bvecs, _, _ = synthesize_phantom(mini_config)
    np.testing.assert_array_equal(base, with_bvecs)


def test_scalar_path_matches_v041_formula(phantom_config):
    """Regression: no tensors -> exactly the v0.4.1 multi-exponential result."""
    signal, _, _ = synthesize_phantom(phantom_config)
    seg = np.round(nib.load(phantom_config.nifti_path).get_fdata()).astype(np.int32)
    seg = seg[..., 0] if seg.ndim == 4 else seg
    expected = np.zeros((*seg.shape, phantom_config.bvalues.size))
    for region in phantom_config.regions.values():
        mask = seg == region.index
        if np.any(mask):
            expected[mask] = region.S0 * sum(
                f * np.exp(-phantom_config.bvalues * d)
                for f, d in zip(region.fractions, region.D)
            )
    np.testing.assert_array_equal(signal, expected)


def test_remaining_tissue_tensor_is_inherited(tmp_path):
    """A null region inherits the tensor of remaining_tissue."""
    seg = np.zeros((2, 2, 2), dtype=np.int32)
    seg[0] = 1
    seg[1] = 2
    nib.save(nib.Nifti1Image(seg, np.eye(4)), tmp_path / "seg.nii.gz")
    case = {
        "case_id": "t",
        "name": "t",
        "description": "t",
        "file": "seg.nii.gz",
        "image": {"dimensions": [2, 2, 2]},
        "regions": {
            "a": {"name": "remaining_tissue", "index": 1, "diffusivity": {
                "S0": 10.0, "tensor": {"evals": EVALS, "evecs": EVECS}}},
            "b": {"name": "other", "index": 2, "diffusivity": {"S0": None}},
        },
    }
    json_path = tmp_path / "case.json"
    json_path.write_text(json.dumps(case), encoding="utf-8")
    bval_path = tmp_path / "case.bval"
    np.savetxt(bval_path, BVALS[None, :])
    cfg = load_dwi_config(json_path, bval_path)
    np.testing.assert_allclose(cfg.regions[2].tensor, np.diag(EVALS))
    assert cfg.regions[2].tensor is not cfg.regions[1].tensor


def test_montecarlo_rejects_tensor_regions(mini_config):
    from mrphantom.montecarlo.sampler import _build_sample_config

    mini_config.regions[1].tensor = np.eye(3) * 1e-3
    with pytest.raises(NotImplementedError, match="tensor"):
        _build_sample_config(mini_config, None, 0)
