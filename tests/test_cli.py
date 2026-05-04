"""Tests for mrphantom CLI — synth, run, info, inspect commands."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from mrphantom.cli import main


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# Top-level group
# ---------------------------------------------------------------------------


def test_main_help(runner):
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "mrphantom" in result.output.lower()


def test_main_version(runner):
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "mrphantom" in result.output.lower() or result.output.strip()


# ---------------------------------------------------------------------------
# mrphantom synth
# ---------------------------------------------------------------------------


def test_synth_noise_free_nifti(runner, json_path, bval_path, tmp_path):
    out = tmp_path / "out.nii.gz"
    result = runner.invoke(
        main,
        [
            "synth",
            str(json_path),
            str(bval_path),
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.is_file()


def test_synth_with_snr(runner, json_path, bval_path, tmp_path):
    out = tmp_path / "out.nii.gz"
    result = runner.invoke(
        main,
        [
            "synth",
            str(json_path),
            str(bval_path),
            "--snr",
            "30",
            "--seed",
            "42",
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.is_file()


def test_synth_hdf5_output(runner, json_path, bval_path, tmp_path):
    out = tmp_path / "out.h5"
    result = runner.invoke(
        main,
        [
            "synth",
            str(json_path),
            str(bval_path),
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.is_file()


def test_synth_multiple_outputs(runner, json_path, bval_path, tmp_path):
    nii = tmp_path / "out.nii.gz"
    h5 = tmp_path / "out.h5"
    result = runner.invoke(
        main,
        [
            "synth",
            str(json_path),
            str(bval_path),
            "-o",
            str(nii),
            "-o",
            str(h5),
        ],
    )
    assert result.exit_code == 0, result.output
    assert nii.is_file()
    assert h5.is_file()


def test_synth_missing_json_fails(runner, bval_path, tmp_path):
    out = tmp_path / "out.nii.gz"
    result = runner.invoke(
        main,
        [
            "synth",
            "/nonexistent/phantom.json",
            str(bval_path),
            "-o",
            str(out),
        ],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# mrphantom run
# ---------------------------------------------------------------------------


def test_run_command(runner, assets_dir, tmp_path):
    # Use the real kidney run.toml — but point output to tmp_path
    import shutil

    # Copy the TOML to tmp_path and patch the output_dir to tmp_path.
    # Use POSIX (forward-slash) paths so they are valid inside TOML strings on
    # all platforms (backslashes are escape characters in TOML).
    toml_content = (assets_dir / "kidney" / "run.toml").read_text()
    patched = toml_content.replace(
        'output_dir        = "output/"',
        f'output_dir = "{tmp_path.as_posix()}/"',
    )
    patched_toml = tmp_path / "run.toml"
    patched_toml.write_text(patched)

    # Also replicate the case paths relative to patched_toml location → need
    # absolute paths so we patch those too.
    case_dir = assets_dir / "kidney" / "case_000"
    json_abs = case_dir / "kid_000_synthetic.json"
    bval_abs = case_dir / "kidney.bval"
    patched2 = patched_toml.read_text()
    patched2 = patched2.replace(
        'config = "case_000/kid_000_synthetic.json"',
        f'config = "{json_abs.as_posix()}"',
    ).replace(
        'bvals  = "case_000/kidney.bval"',
        f'bvals  = "{bval_abs.as_posix()}"',
    )
    patched_toml.write_text(patched2)

    result = runner.invoke(main, ["run", str(patched_toml)])
    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# mrphantom info
# ---------------------------------------------------------------------------


def test_info_command(runner, json_path, bval_path):
    result = runner.invoke(main, ["info", str(json_path), str(bval_path)])
    assert result.exit_code == 0, result.output
    assert "kidney_000" in result.output


# ---------------------------------------------------------------------------
# mrphantom inspect
# ---------------------------------------------------------------------------


def test_inspect_command(runner, json_path, bval_path, tmp_path):
    # First synthesize an HDF5 file to inspect
    h5 = tmp_path / "test.h5"
    runner.invoke(
        main,
        [
            "synth",
            str(json_path),
            str(bval_path),
            "-o",
            str(h5),
        ],
    )
    result = runner.invoke(main, ["inspect", str(h5)])
    assert result.exit_code == 0, result.output
    assert "kidney_000" in result.output


def test_inspect_missing_file(runner, tmp_path):
    result = runner.invoke(main, ["inspect", str(tmp_path / "ghost.h5")])
    assert result.exit_code != 0
