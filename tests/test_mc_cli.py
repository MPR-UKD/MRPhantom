"""Tests for the mrphantom montecarlo CLI command."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from mrphantom.cli import main
from mrphantom.io.mc_hdf5 import load_mc_hdf5


# ---------------------------------------------------------------------------
# Paths to real assets
# ---------------------------------------------------------------------------

_CASE_DIR = Path(__file__).parent.parent / "assets" / "kidney" / "case_000"
_JSON_PATH = _CASE_DIR / "kid_000_synthetic.json"
_BVAL_PATH = _CASE_DIR / "kidney.bval"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_toml(tmp_path: Path, n_samples: int = 3, snr: float | None = None) -> Path:
    """Write a minimal MC TOML config to *tmp_path*."""
    snr_line = f"snr = {snr}" if snr is not None else ""
    toml_path = tmp_path / "mc.toml"
    toml_path.write_text(
        f"""
[run]
seed = 1
{snr_line}

[montecarlo]
n_samples = {n_samples}

[export]
output_dir        = "{(tmp_path / 'output').as_posix()}"
filename_template = "{{case_id}}"
formats           = ["hdf5"]

[[cases]]
config = "{_JSON_PATH.as_posix()}"
bvals  = "{_BVAL_PATH.as_posix()}"
""",
        encoding="utf-8",
    )
    return toml_path


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestMontecarloCommand:
    """Tests for the `mrphantom montecarlo` CLI command."""

    def test_help(self):
        """montecarlo --help exits with code 0."""
        runner = CliRunner()
        result = runner.invoke(main, ["montecarlo", "--help"])
        assert result.exit_code == 0
        assert "Monte Carlo" in result.output

    def test_creates_hdf5_output(self, tmp_path):
        """montecarlo command creates an HDF5 file for each case."""
        toml_path = _write_toml(tmp_path, n_samples=3)
        runner = CliRunner()
        result = runner.invoke(main, ["montecarlo", str(toml_path)])
        assert result.exit_code == 0, result.output
        out_files = list((tmp_path / "output").glob("*.h5"))
        assert len(out_files) == 1

    def test_output_is_valid_mc_hdf5(self, tmp_path):
        """The written HDF5 is readable by load_mc_hdf5."""
        toml_path = _write_toml(tmp_path, n_samples=3)
        runner = CliRunner()
        result = runner.invoke(main, ["montecarlo", str(toml_path)])
        assert result.exit_code == 0, result.output
        out_file = next((tmp_path / "output").glob("*.h5"))
        data = load_mc_hdf5(out_file)
        assert data.n_samples == 3

    def test_signal_shape_is_5d(self, tmp_path):
        """Signal in the output file has 5 dimensions."""
        toml_path = _write_toml(tmp_path, n_samples=2)
        runner = CliRunner()
        runner.invoke(main, ["montecarlo", str(toml_path)])
        out_file = next((tmp_path / "output").glob("*.h5"))
        data = load_mc_hdf5(out_file)
        assert data.signal.ndim == 5

    def test_snr_stored_when_set(self, tmp_path):
        """SNR is stored in the HDF5 when specified in TOML."""
        toml_path = _write_toml(tmp_path, n_samples=2, snr=30.0)
        runner = CliRunner()
        runner.invoke(main, ["montecarlo", str(toml_path)])
        out_file = next((tmp_path / "output").glob("*.h5"))
        data = load_mc_hdf5(out_file)
        assert data.snr == pytest.approx(30.0)

    def test_missing_toml_fails(self, tmp_path):
        """Non-existent TOML causes a non-zero exit code."""
        runner = CliRunner()
        result = runner.invoke(main, ["montecarlo", str(tmp_path / "nope.toml")])
        assert result.exit_code != 0

    def test_done_in_output(self, tmp_path):
        """'Done.' appears in stdout on success."""
        toml_path = _write_toml(tmp_path, n_samples=2)
        runner = CliRunner()
        result = runner.invoke(main, ["montecarlo", str(toml_path)])
        assert "Done." in result.output
