"""Tests for partial volume, texture, resampling and ground-truth parameter maps."""

from __future__ import annotations

import dataclasses

import nibabel as nib
import numpy as np
import pytest

from mrphantom import DWIPhantom
from mrphantom.dwi.config import DWIConfig, RegionConfig, load_dwi_config
from mrphantom.dwi.resample import block_average, block_factors, resample_affine
from mrphantom.dwi.synthesis import synthesize_phantom, synthesize_with_truth
from mrphantom.dwi.texture import scale_field, smooth_field
from mrphantom.io import load_hdf5

BVALUES = np.array([0.0, 100.0, 500.0])


def _make_config(tmp_path, seg, zooms=(1.0, 1.0, 1.0), probs=None, **region_kw):
    """Two-region config on *seg*; *probs* maps region index -> probability array."""
    affine = np.diag([*zooms, 1.0])
    nib.save(nib.Nifti1Image(seg.astype(np.int32), affine), tmp_path / "seg.nii.gz")
    prob_files = {}
    for idx, arr in (probs or {}).items():
        path = tmp_path / f"p{idx}.nii.gz"
        nib.save(nib.Nifti1Image(arr.astype(np.float32), affine), path)
        prob_files[idx] = path
    regions = {
        1: RegionConfig("a", 1, 1, 100.0, [1.0], [0.001],
                        probability_file=prob_files.get(1), **region_kw),
        2: RegionConfig("b", 2, 1, 50.0, [1.0], [0.002],
                        probability_file=prob_files.get(2)),
    }
    return DWIConfig(
        case_id="pv", name="pv", description="", nifti_path=tmp_path / "seg.nii.gz",
        image_dims=list(seg.shape), bvalues=BVALUES, regions=regions,
    )


def _signal(s0, d):
    return s0 * np.exp(-BVALUES * d)


@pytest.fixture
def halves_seg():
    """4x4x4 label map: region 1 in x<2, region 2 in x>=2."""
    seg = np.zeros((4, 4, 4), dtype=np.int32)
    seg[:2] = 1
    seg[2:] = 2
    return seg


class TestPartialVolume:
    """Signal is the probability-weighted sum of region signals."""

    def test_signal_is_probability_weighted_sum(self, tmp_path, halves_seg):
        """A 30/70 voxel mixes the two region signals linearly."""
        p1 = np.zeros((4, 4, 4)); p1[0, 0, 0] = 0.3
        p2 = np.zeros((4, 4, 4)); p2[0, 0, 0] = 0.7
        cfg = _make_config(tmp_path, halves_seg, probs={1: p1, 2: p2})
        signal, _, _ = synthesize_phantom(cfg)
        expected = 0.3 * _signal(100, 0.001) + 0.7 * _signal(50, 0.002)
        np.testing.assert_allclose(signal[0, 0, 0], expected, rtol=1e-6)

    def test_voxels_without_probability_are_background(self, tmp_path, halves_seg):
        """When both regions use probability maps, zero probability means zero signal."""
        p = np.zeros((4, 4, 4)); p[0, 0, 0] = 1.0
        cfg = _make_config(tmp_path, halves_seg, probs={1: p, 2: np.zeros((4, 4, 4))})
        signal, _, _ = synthesize_phantom(cfg)
        assert np.count_nonzero(signal[..., 0]) == 1

    def test_mixed_label_and_probability_regions(self, tmp_path, halves_seg):
        """A region without a probability file still uses its hard label mask."""
        p1 = np.full((4, 4, 4), 0.5)
        cfg = _make_config(tmp_path, halves_seg, probs={1: p1})
        signal, _, _ = synthesize_phantom(cfg)
        np.testing.assert_allclose(signal[0, 0, 0], 0.5 * _signal(100, 0.001) , rtol=1e-6)
        np.testing.assert_allclose(
            signal[3, 0, 0], 0.5 * _signal(100, 0.001) + _signal(50, 0.002), rtol=1e-6
        )

    def test_probability_shape_mismatch_raises(self, tmp_path, halves_seg):
        """A probability map of the wrong shape is rejected."""
        cfg = _make_config(tmp_path, halves_seg, probs={1: np.zeros((3, 3, 3))})
        with pytest.raises(ValueError, match="expected"):
            synthesize_phantom(cfg)

    def test_missing_probability_file_raises(self, tmp_path, halves_seg):
        """A missing probability file raises FileNotFoundError."""
        cfg = _make_config(tmp_path, halves_seg)
        cfg.regions[1].probability_file = tmp_path / "nope.nii.gz"
        with pytest.raises(FileNotFoundError):
            synthesize_phantom(cfg)

    def test_label_only_case_matches_hard_labels(self, tmp_path, halves_seg):
        """Without opt-in the signal equals the plain label-wise model."""
        cfg = _make_config(tmp_path, halves_seg)
        signal, _, _ = synthesize_phantom(cfg)
        np.testing.assert_array_equal(signal[0, 0, 0], _signal(100, 0.001))
        np.testing.assert_array_equal(signal[3, 0, 0], _signal(50, 0.002))

    def test_json_probability_field_is_parsed(self, tmp_path, halves_seg):
        """`probability` and `*_cv` keys in the JSON reach RegionConfig."""
        import json

        _make_config(tmp_path, halves_seg)
        spec = {
            "case_id": "c", "name": "n", "description": "d", "file": "seg.nii.gz",
            "image": {"dimensions": [4, 4, 4]},
            "texture": {"sigma": 1.5}, "resample": {"voxel_size": [2, 2, 2]},
            "regions": {
                "1": {"name": "a", "index": 1, "probability": "p1.nii.gz",
                      "diffusivity": {"n_components": 1, "S0": 1, "fractions": [1],
                                      "D": [0.001], "S0_cv": 0.1, "D_cv": 0.2}},
            },
        }
        (tmp_path / "c.json").write_text(json.dumps(spec))
        (tmp_path / "b.txt").write_text("0 100")
        cfg = load_dwi_config(tmp_path / "c.json", tmp_path / "b.txt")
        assert cfg.regions[1].probability_file == tmp_path / "p1.nii.gz"
        assert (cfg.regions[1].S0_cv, cfg.regions[1].D_cv) == (0.1, 0.2)
        assert cfg.texture_sigma == 1.5
        assert cfg.voxel_size == (2, 2, 2)


class TestTexture:
    """Smooth, seeded S0 / D modulation."""

    def test_smooth_field_is_unit_variance_zero_mean(self):
        """The raw field is normalised."""
        field = smooth_field((24, 24, 24), 2.0, np.random.default_rng(0))
        assert abs(field.mean()) < 1e-5
        assert field.std() == pytest.approx(1.0, abs=1e-4)

    def test_smooth_field_is_spatially_correlated(self):
        """Smoothing makes neighbouring voxels correlated, white noise is not."""
        rng = np.random.default_rng(0)
        smooth = smooth_field((32, 32, 32), 2.0, rng)
        white = smooth_field((32, 32, 32), 0.0, rng)
        corr = lambda f: np.corrcoef(f[:-1].ravel(), f[1:].ravel())[0, 1]
        assert corr(smooth) > 0.5 > abs(corr(white))

    def test_scale_field_is_non_negative(self):
        """Large CV is clipped at zero."""
        assert scale_field((10, 10, 10), 5.0, 1.0, np.random.default_rng(1)).min() >= 0

    def test_texture_off_by_default(self, tmp_path, halves_seg):
        """With cv = 0 the region signal is constant."""
        signal, _, _ = synthesize_phantom(_make_config(tmp_path, halves_seg))
        assert np.ptp(signal[:2, ..., 1]) == 0

    def test_texture_varies_within_region(self, tmp_path, halves_seg):
        """S0_cv > 0 makes the clean signal vary inside a region."""
        cfg = _make_config(tmp_path, halves_seg, S0_cv=0.2)
        _, gt, _ = synthesize_phantom(cfg, texture_seed=1)
        assert np.std(gt[:2, ..., 0]) > 1.0
        assert np.ptp(gt[2:, ..., 0]) == 0, "untextured region must stay constant"

    def test_d_texture_leaves_b0_unchanged(self, tmp_path, halves_seg):
        """D_cv only changes the signal at b > 0."""
        cfg = _make_config(tmp_path, halves_seg, D_cv=0.3)
        _, gt, _ = synthesize_phantom(cfg, texture_seed=1)
        assert np.ptp(gt[:2, ..., 0]) == 0
        assert np.ptp(gt[:2, ..., 2]) > 0

    def test_same_seed_reproduces_and_other_seed_differs(self, tmp_path, halves_seg):
        """Texture is deterministic per seed."""
        cfg = _make_config(tmp_path, halves_seg, S0_cv=0.2)
        a = synthesize_phantom(cfg, texture_seed=5)[1]
        b = synthesize_phantom(cfg, texture_seed=5)[1]
        c = synthesize_phantom(cfg, texture_seed=6)[1]
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_texture_does_not_change_noise_stream(self, tmp_path, halves_seg):
        """Noise draws are independent of texture seed handling."""
        cfg = _make_config(tmp_path, halves_seg)
        a = synthesize_phantom(cfg, snr=20, rng=np.random.default_rng(3))[0]
        b = synthesize_phantom(cfg, snr=20, rng=np.random.default_rng(3), texture_seed=9)[0]
        np.testing.assert_array_equal(a, b)


class TestResampling:
    """Block-average downsampling."""

    def test_block_factors_integer_ratio(self):
        """2 mm target from 1 mm source gives factor 2."""
        np.testing.assert_array_equal(block_factors((1, 1, 1), 2.0), [2, 2, 2])

    def test_block_factors_anisotropic(self):
        """Per-axis targets are honoured."""
        np.testing.assert_array_equal(block_factors((1, 1, 2), (2, 2, 2)), [2, 2, 1])

    @pytest.mark.parametrize("target", [0.5, 1.5], ids=["upsample", "non-integer"])
    def test_block_factors_invalid_raises(self, target):
        """Upsampling and non-integer ratios are rejected."""
        with pytest.raises(ValueError):
            block_factors((1, 1, 1), target)

    def test_block_average_values(self):
        """Mean over 2x2x2 blocks."""
        vol = np.zeros((4, 4, 4)); vol[:2, :2, :2] = 1.0; vol[0, 0, 0] = 0.0
        out = block_average(vol, (2, 2, 2))
        assert out.shape == (2, 2, 2)
        assert out[0, 0, 0] == pytest.approx(7 / 8)
        assert out[1, 1, 1] == 0

    def test_block_average_pads_odd_shapes(self):
        """Odd sizes are zero padded, not cropped."""
        out = block_average(np.ones((5, 4, 4)), (2, 2, 2))
        assert out.shape == (3, 2, 2)
        assert out[2, 0, 0] == pytest.approx(0.5)

    def test_resample_affine_scales_and_centres(self):
        """Voxel size doubles and origin shifts to the first block centre."""
        out = resample_affine(np.eye(4), (2, 2, 2))
        np.testing.assert_allclose(np.diag(out)[:3], [2, 2, 2])
        np.testing.assert_allclose(out[:3, 3], [0.5, 0.5, 0.5])

    def test_resampled_signal_has_partial_volume_on_boundary(self, tmp_path):
        """A block straddling two labels mixes their signals."""
        seg = np.zeros((4, 4, 4), dtype=np.int32)
        seg[:3] = 1
        seg[3] = 2
        cfg = dataclasses.replace(_make_config(tmp_path, seg), voxel_size=2.0)
        res = synthesize_with_truth(cfg)
        assert res.signal.shape == (2, 2, 2, 3)
        np.testing.assert_allclose(res.signal[0, 0, 0], _signal(100, 0.001), rtol=1e-6)
        expected = 0.5 * _signal(100, 0.001) + 0.5 * _signal(50, 0.002)
        np.testing.assert_allclose(res.signal[1, 0, 0], expected, rtol=1e-6)

    def test_resampled_image_header_and_labels(self, tmp_path):
        """Output affine/zooms follow the new grid and labels are the argmax."""
        seg = np.zeros((4, 4, 4), dtype=np.int32)
        seg[:3] = 1
        seg[3] = 2
        cfg = dataclasses.replace(_make_config(tmp_path, seg), voxel_size=2.0)
        res = synthesize_with_truth(cfg)
        assert res.signal_img.header.get_zooms()[:3] == (2.0, 2.0, 2.0)
        np.testing.assert_allclose(np.diag(res.signal_img.affine)[:3], [2, 2, 2])
        assert res.segmentation.shape == (2, 2, 2)
        assert res.segmentation[0, 0, 0] == 1

    def test_native_voxel_size_is_noop(self, tmp_path, halves_seg):
        """Requesting the native size changes nothing."""
        cfg = _make_config(tmp_path, halves_seg)
        base = synthesize_phantom(cfg)[0]
        same = synthesize_phantom(dataclasses.replace(cfg, voxel_size=1.0))[0]
        np.testing.assert_array_equal(base, same)


class TestGroundTruth:
    """Clean signal and parameter maps are available and stored in HDF5."""

    def test_params_contents_without_texture(self, tmp_path, halves_seg):
        """Params hold region metadata and probabilities, no scale fields."""
        res = synthesize_with_truth(_make_config(tmp_path, halves_seg))
        assert list(res.params["region_index"]) == [1, 2]
        np.testing.assert_array_equal(res.params["S0"], [100.0, 50.0])
        np.testing.assert_array_equal(res.params["D"], [[0.001], [0.002]])
        assert res.params["probability"].shape == (2, 4, 4, 4)
        assert "S0_scale" not in res.params

    def test_params_include_scales_with_texture(self, tmp_path, halves_seg):
        """Texture adds S0_scale / D_scale maps consistent with the clean signal."""
        cfg = _make_config(tmp_path, halves_seg, S0_cv=0.2)
        res = synthesize_with_truth(cfg, texture_seed=2)
        s0_map = 100.0 * res.params["S0_scale"][0]
        np.testing.assert_allclose(res.signal_gt[:2, ..., 0], s0_map[:2], rtol=1e-5)

    def test_hdf5_stores_signal_gt_and_params(self, tmp_path, halves_seg):
        """save() writes /signal_gt and /params, load_hdf5 reads them back."""
        cfg = _make_config(tmp_path, halves_seg, S0_cv=0.2)
        phantom = DWIPhantom(snr=20, seed=4).load_case(cfg).synthesize()
        out = phantom.save(tmp_path / "out.h5")[0]
        data = load_hdf5(out)
        np.testing.assert_allclose(data.signal_gt, phantom.signal_gt)
        assert data.params is not None
        assert {"probability", "S0_scale", "D_scale", "S0", "D"} <= set(data.params)
        np.testing.assert_array_equal(data.segmentation, phantom.segmentation)

    def test_hdf5_without_params_loads_none(self, tmp_path, halves_seg):
        """Files lacking /params (older versions) load with params=None."""
        from mrphantom.io import save_hdf5

        cfg = _make_config(tmp_path, halves_seg)
        signal, gt, img = synthesize_phantom(cfg)
        out = save_hdf5(signal, img, BVALUES, halves_seg, tmp_path / "old.h5", signal_gt=gt)
        assert load_hdf5(out).params is None

    def test_phantom_segmentation_matches_resampled_grid(self, tmp_path, halves_seg):
        """DWIPhantom.segmentation follows the synthesis grid."""
        cfg = dataclasses.replace(_make_config(tmp_path, halves_seg), voxel_size=2.0)
        phantom = DWIPhantom().load_case(cfg).synthesize()
        assert phantom.segmentation.shape == phantom.signal.shape[:3]
