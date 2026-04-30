"""Tests for mrphantom.roi — get_evaluation_segments, get_fitting_mask."""

from __future__ import annotations

import numpy as np
import pytest

from mrphantom.roi import (
    EVALUATION_REGION_NAMES,
    get_evaluation_segments,
    get_fitting_mask,
)


# ---------------------------------------------------------------------------
# EVALUATION_REGION_NAMES constant
# ---------------------------------------------------------------------------


def test_evaluation_region_names_is_frozenset():
    assert isinstance(EVALUATION_REGION_NAMES, frozenset)


def test_evaluation_region_names_contains_kidney_regions():
    assert "hilum" in EVALUATION_REGION_NAMES
    assert "cortex" in EVALUATION_REGION_NAMES
    assert "medulla" in EVALUATION_REGION_NAMES


# ---------------------------------------------------------------------------
# get_evaluation_segments
# ---------------------------------------------------------------------------


def test_get_evaluation_segments_returns_list(phantom_config):
    result = get_evaluation_segments(phantom_config)
    assert isinstance(result, list)


def test_get_evaluation_segments_sorted(phantom_config):
    result = get_evaluation_segments(phantom_config)
    assert result == sorted(result)


def test_get_evaluation_segments_correct_indices(phantom_config):
    # hilum=1, cortex=10, medulla=11 in the kidney JSON
    result = get_evaluation_segments(phantom_config)
    assert set(result) == {1, 10, 11}


def test_get_evaluation_segments_no_matches_returns_empty(mini_config):
    # mini_config has regions named "region_a" and "region_b" — not in evaluation set
    result = get_evaluation_segments(mini_config)
    assert result == []


# ---------------------------------------------------------------------------
# get_fitting_mask
# ---------------------------------------------------------------------------


def test_get_fitting_mask_shape():
    seg = np.zeros((4, 4, 4), dtype=np.int32)
    seg[0, :, :] = 1
    mask = get_fitting_mask(seg, [1])
    assert mask.shape == seg.shape


def test_get_fitting_mask_dtype():
    seg = np.zeros((4, 4, 4), dtype=np.int32)
    mask = get_fitting_mask(seg, [1])
    assert mask.dtype == bool


def test_get_fitting_mask_selects_correct_voxels():
    seg = np.zeros((4, 4, 4), dtype=np.int32)
    seg[0:2, :, :] = 1
    seg[2:4, :, :] = 2
    mask = get_fitting_mask(seg, [1])
    assert mask[0, 0, 0] is np.bool_(True)
    assert mask[2, 0, 0] is np.bool_(False)


def test_get_fitting_mask_multiple_indices():
    seg = np.zeros((4, 4, 4), dtype=np.int32)
    seg[0, :, :] = 10
    seg[1, :, :] = 11
    seg[2, :, :] = 99
    mask = get_fitting_mask(seg, [10, 11])
    assert mask[0, 0, 0]  # label 10
    assert mask[1, 0, 0]  # label 11
    assert not mask[2, 0, 0]  # label 99
    assert not mask[3, 0, 0]  # label 0


def test_get_fitting_mask_empty_indices_returns_all_false():
    seg = np.ones((3, 3, 3), dtype=np.int32)
    mask = get_fitting_mask(seg, [])
    assert not mask.any()


def test_get_fitting_mask_integration(phantom_config):
    """Integration: build a mask for the real kidney phantom segmentation."""
    import nibabel as nib

    seg_img = nib.load(phantom_config.nifti_path)
    seg = np.round(seg_img.get_fdata()).astype(np.int32)
    if seg.ndim == 4:
        seg = seg[..., 0]

    indices = get_evaluation_segments(phantom_config)
    mask = get_fitting_mask(seg, indices)

    assert mask.shape == seg.shape[:3]
    assert mask.any()  # At least some voxels selected
    # Non-evaluation labels must not appear in the mask
    for label in [2, 3, 4, 5, 6, 7, 8, 9]:
        assert not mask[seg == label].any()
