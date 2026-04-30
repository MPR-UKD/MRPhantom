"""Tests for mrphantom.run_config — RunConfig, CaseEntry, ExportConfig, load_run_config."""

from __future__ import annotations

from pathlib import Path

import pytest

from mrphantom.run_config import (
    CaseEntry,
    ExportConfig,
    RunConfig,
    load_run_config,
)


# ---------------------------------------------------------------------------
# Helper: write a minimal TOML file
# ---------------------------------------------------------------------------


def _write_toml(tmp_path: Path, content: str, name: str = "run.toml") -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# ExportConfig.resolve_paths
# ---------------------------------------------------------------------------


def test_resolve_paths_nifti():
    ec = ExportConfig(output_dir=Path("/out"), formats=["nifti"])
    paths = ec.resolve_paths("case_001")
    assert paths == [Path("/out/case_001.nii.gz")]


def test_resolve_paths_hdf5():
    ec = ExportConfig(output_dir=Path("/out"), formats=["hdf5"])
    paths = ec.resolve_paths("case_001")
    assert paths == [Path("/out/case_001.h5")]


def test_resolve_paths_both_formats():
    ec = ExportConfig(output_dir=Path("/out"), formats=["nifti", "hdf5"])
    paths = ec.resolve_paths("case_001")
    assert len(paths) == 2
    assert any(str(p).endswith(".nii.gz") for p in paths)
    assert any(str(p).endswith(".h5") for p in paths)


def test_resolve_paths_template_interpolation():
    ec = ExportConfig(
        output_dir=Path("/out"),
        filename_template="{case_id}_snr30",
        formats=["nifti"],
    )
    paths = ec.resolve_paths("kid_001")
    assert paths[0].name == "kid_001_snr30.nii.gz"


def test_resolve_paths_unknown_format_raises():
    ec = ExportConfig(output_dir=Path("/out"), formats=["jpg"])
    with pytest.raises(ValueError, match="Unknown export format"):
        ec.resolve_paths("x")


# ---------------------------------------------------------------------------
# load_run_config — file-not-found
# ---------------------------------------------------------------------------


def test_load_run_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run-config TOML not found"):
        load_run_config(tmp_path / "nonexistent.toml")


# ---------------------------------------------------------------------------
# load_run_config — minimal valid TOML
# ---------------------------------------------------------------------------

MINIMAL_TOML = """\
[export]
output_dir = "output/"

[[cases]]
config = "case_001/phantom.json"
bvals  = "case_001/bvals.txt"
"""


def test_load_run_config_returns_run_config(tmp_path):
    toml = _write_toml(tmp_path, MINIMAL_TOML)
    rc = load_run_config(toml)
    assert isinstance(rc, RunConfig)


def test_load_run_config_defaults(tmp_path):
    toml = _write_toml(tmp_path, MINIMAL_TOML)
    rc = load_run_config(toml)
    assert rc.snr is None
    assert rc.seed is None


def test_load_run_config_one_case(tmp_path):
    toml = _write_toml(tmp_path, MINIMAL_TOML)
    rc = load_run_config(toml)
    assert len(rc.cases) == 1
    assert isinstance(rc.cases[0], CaseEntry)


def test_load_run_config_paths_are_absolute(tmp_path):
    toml = _write_toml(tmp_path, MINIMAL_TOML)
    rc = load_run_config(toml)
    assert rc.cases[0].config.is_absolute()
    assert rc.cases[0].bvals.is_absolute()


def test_load_run_config_paths_relative_to_toml(tmp_path):
    toml = _write_toml(tmp_path, MINIMAL_TOML)
    rc = load_run_config(toml)
    expected_config = (tmp_path / "case_001" / "phantom.json").resolve()
    assert rc.cases[0].config == expected_config


# ---------------------------------------------------------------------------
# load_run_config — with [run] section
# ---------------------------------------------------------------------------

FULL_TOML = """\
[run]
snr  = 30
seed = 42

[export]
output_dir        = "results/"
filename_template = "{case_id}_snr30"
formats           = ["nifti", "hdf5"]

[[cases]]
config = "case_001/phantom.json"
bvals  = "case_001/bvals.txt"

[[cases]]
config = "case_002/phantom.json"
bvals  = "case_002/bvals.txt"
"""


def test_load_run_config_snr_seed(tmp_path):
    toml = _write_toml(tmp_path, FULL_TOML)
    rc = load_run_config(toml)
    assert rc.snr == pytest.approx(30.0)
    assert rc.seed == 42


def test_load_run_config_two_cases(tmp_path):
    toml = _write_toml(tmp_path, FULL_TOML)
    rc = load_run_config(toml)
    assert len(rc.cases) == 2


def test_load_run_config_export_config(tmp_path):
    toml = _write_toml(tmp_path, FULL_TOML)
    rc = load_run_config(toml)
    assert isinstance(rc.export, ExportConfig)
    assert rc.export.filename_template == "{case_id}_snr30"
    assert set(rc.export.formats) == {"nifti", "hdf5"}


# ---------------------------------------------------------------------------
# load_run_config — empty [[cases]] raises
# ---------------------------------------------------------------------------

EMPTY_CASES_TOML = """\
[export]
output_dir = "output/"
"""


def test_load_run_config_empty_cases_raises(tmp_path):
    toml = _write_toml(tmp_path, EMPTY_CASES_TOML)
    with pytest.raises(ValueError, match="no \\[\\[cases\\]\\]"):
        load_run_config(toml)


# ---------------------------------------------------------------------------
# load_run_config — real kidney run.toml
# ---------------------------------------------------------------------------


def test_real_kidney_run_toml(assets_dir):
    toml = assets_dir / "kidney" / "run.toml"
    rc = load_run_config(toml)
    assert rc.snr == pytest.approx(30.0)
    assert rc.seed == 42
    assert len(rc.cases) == 1
    # Paths must be resolved and point to real files
    assert rc.cases[0].config.is_file()
    assert rc.cases[0].bvals.is_file()
