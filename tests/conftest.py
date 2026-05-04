"""Shared pytest fixtures for mrphantom tests.

Provides:
- ``assets_dir``  — Path to the bundled assets directory.
- ``case_dir``    — Path to the kidney case_001 directory.
- ``json_path``   — Path to kid_001_cor_ang.json.
- ``bval_path``   — Path to bvals.txt.
- ``phantom_config`` — Pre-loaded DWIConfig for kidney_001.
- ``mini_config`` — A minimal in-memory DWIConfig backed by a tiny
                    synthetic NIfTI (avoids loading the real 176×3×176 image
                    in fast unit tests).
- ``mini_nifti``  — The nibabel.Nifti1Image corresponding to mini_config.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from mrphantom.dwi.config import DWIConfig, RegionConfig, load_dwi_config

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
ASSETS_DIR = REPO_ROOT / "assets"
CASE_DIR = ASSETS_DIR / "kidney" / "case_001"


@pytest.fixture(scope="session")
def assets_dir() -> Path:
    return ASSETS_DIR


@pytest.fixture(scope="session")
def case_dir() -> Path:
    return CASE_DIR


@pytest.fixture(scope="session")
def json_path() -> Path:
    return CASE_DIR / "kid_001_cor_ang.json"


@pytest.fixture(scope="session")
def bval_path() -> Path:
    return CASE_DIR / "kidney.bval"


@pytest.fixture(scope="session")
def phantom_config(json_path, bval_path) -> DWIConfig:
    """Full kidney_001 config (session-scoped — loaded once)."""
    return load_dwi_config(json_path, bval_path)


# ---------------------------------------------------------------------------
# Minimal in-memory fixtures (fast, no real NIfTI I/O in unit tests)
# ---------------------------------------------------------------------------


def _make_mini_nifti(tmp_path: Path) -> tuple[Path, nib.Nifti1Image]:
    """Create a 4×4×4 segmentation NIfTI with three regions (0, 1, 2)."""
    seg = np.zeros((4, 4, 4), dtype=np.int32)
    seg[0:2, :, :] = 1  # region 1
    seg[2:4, :, :] = 2  # region 2
    affine = np.eye(4)
    img = nib.Nifti1Image(seg, affine)
    nifti_path = tmp_path / "mini_seg.nii.gz"
    nib.save(img, nifti_path)
    return nifti_path, img


@pytest.fixture
def mini_nifti(tmp_path):
    """4×4×4 segmentation image with labels 1 and 2."""
    _, img = _make_mini_nifti(tmp_path)
    return img


@pytest.fixture
def mini_config(tmp_path) -> DWIConfig:
    """Minimal DWIConfig with two regions backed by a tiny NIfTI."""
    nifti_path, _ = _make_mini_nifti(tmp_path)
    bvalues = np.array([0.0, 100.0, 500.0])
    regions = {
        1: RegionConfig(
            name="region_a",
            index=1,
            n_components=1,
            S0=100.0,
            fractions=[1.0],
            D=[0.001],
        ),
        2: RegionConfig(
            name="region_b",
            index=2,
            n_components=2,
            S0=80.0,
            fractions=[0.7, 0.3],
            D=[0.001, 0.01],
        ),
    }
    return DWIConfig(
        case_id="mini_001",
        name="mini",
        description="Minimal test phantom",
        nifti_path=nifti_path,
        image_dims=[4, 4, 4],
        bvalues=bvalues,
        regions=regions,
    )
