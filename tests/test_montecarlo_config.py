"""Tests for mrphantom.montecarlo.config — MCRunConfig and TOML loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from mrphantom.montecarlo.config import (
    MCRunConfig,
    ParamDist,
    RegionMCConfig,
    _parse_param_dist,
    load_montecarlo_config,
)
from mrphantom.run_config import CaseEntry, ExportConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CASE_DIR = Path(__file__).parent.parent / "assets" / "kidney" / "case_000"
_JSON_PATH = _CASE_DIR / "kid_000_synthetic.json"
_BVAL_PATH = _CASE_DIR / "kidney.bval"


def _write_mc_toml(tmp_path: Path, extra: str = "") -> Path:
    """Write a minimal valid MC TOML to *tmp_path* and return its path."""
    toml_path = tmp_path / "mc.toml"
    toml_path.write_text(
        f"""
[run]
snr  = 30.0
seed = 42

[montecarlo]
n_samples = 10

[export]
output_dir = "output/"
filename_template = "{{case_id}}"
formats = ["hdf5"]

[[cases]]
config = "{_JSON_PATH}"
bvals  = "{_BVAL_PATH}"

{extra}
""",
        encoding="utf-8",
    )
    return toml_path


# ---------------------------------------------------------------------------
# _parse_param_dist
# ---------------------------------------------------------------------------


class TestParseParamDist:
    """Unit tests for the internal _parse_param_dist helper."""

    def test_normal_returns_param_dist(self):
        """Normal distribution is parsed into a ParamDist with correct fields."""
        raw = {"dist": "normal", "mean": 180.0, "std": 10.0}
        result = _parse_param_dist(raw)
        assert result.dist == "normal"
        assert result.mean == 180.0
        assert result.std == 10.0

    def test_normal_defaults(self):
        """Normal distribution uses default mean=0 and std=1 when absent."""
        result = _parse_param_dist({"dist": "normal"})
        assert result.mean == 0.0
        assert result.std == 1.0

    def test_lognormal_returns_param_dist(self):
        """Lognormal distribution is parsed into a ParamDist with correct fields."""
        raw = {"dist": "lognormal", "mean": 0.003, "sigma": 0.1}
        result = _parse_param_dist(raw)
        assert result.dist == "lognormal"
        assert result.mean == 0.003
        assert result.sigma == 0.1

    def test_lognormal_defaults(self):
        """Lognormal distribution uses default mean=0 and sigma=1 when absent."""
        result = _parse_param_dist({"dist": "lognormal"})
        assert result.mean == 0.0
        assert result.sigma == 1.0

    def test_dirichlet_returns_param_dist(self):
        """Dirichlet distribution is parsed into a ParamDist with kappa."""
        raw = {"dist": "dirichlet", "kappa": 10.0}
        result = _parse_param_dist(raw)
        assert result.dist == "dirichlet"
        assert result.kappa == pytest.approx(10.0)

    def test_unknown_dist_raises(self):
        """An unrecognised distribution name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown distribution"):
            _parse_param_dist({"dist": "uniform", "low": 0.0, "high": 1.0})

    def test_missing_dist_key_raises(self):
        """Missing 'dist' key raises KeyError."""
        with pytest.raises(KeyError):
            _parse_param_dist({"mean": 1.0})


# ---------------------------------------------------------------------------
# ParamDist dataclass
# ---------------------------------------------------------------------------


class TestParamDist:
    """Unit tests for the ParamDist dataclass."""

    def test_defaults(self):
        """ParamDist has sensible defaults."""
        pd = ParamDist(dist="normal")
        assert pd.mean == 0.0
        assert pd.std == 1.0
        assert pd.sigma == 1.0
        assert pd.alpha == []
        assert pd.kappa == 0.0

    def test_dirichlet_kappa(self):
        """ParamDist stores dirichlet kappa correctly."""
        pd = ParamDist(dist="dirichlet", kappa=10.0)
        assert pd.kappa == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# RegionMCConfig dataclass
# ---------------------------------------------------------------------------


class TestRegionMCConfig:
    """Unit tests for the RegionMCConfig dataclass."""

    def test_defaults_are_none(self):
        """All distribution fields default to None."""
        rmc = RegionMCConfig(name="cortex")
        assert rmc.S0 is None
        assert rmc.D is None
        assert rmc.fractions is None

    def test_with_s0_dist(self):
        """S0 distribution is stored correctly."""
        dist = ParamDist(dist="normal", std=10.0)
        rmc = RegionMCConfig(name="cortex", S0=dist)
        assert rmc.S0 is dist


# ---------------------------------------------------------------------------
# load_montecarlo_config — basic structure
# ---------------------------------------------------------------------------


class TestLoadMontecarloConfig:
    """Tests for load_montecarlo_config()."""

    def test_missing_file_raises(self, tmp_path):
        """FileNotFoundError is raised for a non-existent TOML."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_montecarlo_config(tmp_path / "missing.toml")

    def test_returns_mc_run_config(self, tmp_path):
        """A valid TOML returns an MCRunConfig instance."""
        toml_path = _write_mc_toml(tmp_path)
        result = load_montecarlo_config(toml_path)
        assert isinstance(result, MCRunConfig)

    def test_n_samples(self, tmp_path):
        """n_samples is read from [montecarlo] section."""
        toml_path = _write_mc_toml(tmp_path)
        result = load_montecarlo_config(toml_path)
        assert result.n_samples == 10

    def test_snr(self, tmp_path):
        """SNR is read from [run] section as float."""
        toml_path = _write_mc_toml(tmp_path)
        result = load_montecarlo_config(toml_path)
        assert result.snr == pytest.approx(30.0)

    def test_seed(self, tmp_path):
        """Seed is read from [run] section as int."""
        toml_path = _write_mc_toml(tmp_path)
        result = load_montecarlo_config(toml_path)
        assert result.seed == 42

    def test_snr_none_by_default(self, tmp_path):
        """SNR defaults to None when not specified."""
        toml_path = tmp_path / "mc.toml"
        toml_path.write_text(
            f"""
[montecarlo]
n_samples = 5

[[cases]]
config = "{_JSON_PATH}"
bvals  = "{_BVAL_PATH}"
""",
            encoding="utf-8",
        )
        result = load_montecarlo_config(toml_path)
        assert result.snr is None

    def test_snr_region(self, tmp_path):
        """snr_region is parsed from [run] section."""
        toml_path = tmp_path / "mc.toml"
        toml_path.write_text(
            f"""
[run]
snr        = 30.0
snr_region = "cortex"
seed       = 1

[montecarlo]
n_samples = 5

[[cases]]
config = "{_JSON_PATH}"
bvals  = "{_BVAL_PATH}"
""",
            encoding="utf-8",
        )
        result = load_montecarlo_config(toml_path)
        assert result.snr_region == "cortex"

    def test_snr_region_defaults_none(self, tmp_path):
        """snr_region defaults to None when absent."""
        toml_path = _write_mc_toml(tmp_path)
        result = load_montecarlo_config(toml_path)
        assert result.snr_region is None

    def test_cases(self, tmp_path):
        """cases list has correct length and absolute paths."""
        toml_path = _write_mc_toml(tmp_path)
        result = load_montecarlo_config(toml_path)
        assert len(result.cases) == 1
        assert isinstance(result.cases[0], CaseEntry)
        assert result.cases[0].config.is_absolute()
        assert result.cases[0].bvals.is_absolute()

    def test_no_cases_raises(self, tmp_path):
        """ValueError is raised when [[cases]] is empty."""
        toml_path = tmp_path / "mc.toml"
        toml_path.write_text("[montecarlo]\nn_samples = 5\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no \\[\\[cases\\]\\]"):
            load_montecarlo_config(toml_path)

    def test_export_config(self, tmp_path):
        """ExportConfig fields are populated from [export] section."""
        toml_path = _write_mc_toml(tmp_path)
        result = load_montecarlo_config(toml_path)
        assert isinstance(result.export, ExportConfig)
        assert result.export.filename_template == "{case_id}"
        assert result.export.formats == ["hdf5"]

    def test_export_output_dir_is_absolute(self, tmp_path):
        """output_dir is resolved to an absolute path."""
        toml_path = _write_mc_toml(tmp_path)
        result = load_montecarlo_config(toml_path)
        assert result.export.output_dir.is_absolute()

    def test_mc_regions_empty_by_default(self, tmp_path):
        """mc_regions is empty when no [[montecarlo.regions]] entries present."""
        toml_path = _write_mc_toml(tmp_path)
        result = load_montecarlo_config(toml_path)
        assert result.mc_regions == {}

    def test_mc_regions_s0(self, tmp_path):
        """S0 distribution is parsed for an [[montecarlo.regions]] entry."""
        extra = """
[[montecarlo.regions]]
name = "cortex"
S0   = {dist = "normal", mean = 180.0, std = 10.0}
"""
        toml_path = _write_mc_toml(tmp_path, extra=extra)
        result = load_montecarlo_config(toml_path)
        assert "cortex" in result.mc_regions
        rmc = result.mc_regions["cortex"]
        assert isinstance(rmc, RegionMCConfig)
        assert rmc.S0 is not None
        assert rmc.S0.dist == "normal"
        assert rmc.S0.mean == pytest.approx(180.0)
        assert rmc.S0.std == pytest.approx(10.0)
        assert rmc.D is None
        assert rmc.fractions is None

    def test_mc_regions_d_list(self, tmp_path):
        """D distribution list is parsed for multi-component regions."""
        extra = """
[[montecarlo.regions]]
name = "cortex"
D    = [{dist = "lognormal", mean = 0.003, sigma = 0.1},
        {dist = "lognormal", mean = 0.001, sigma = 0.05}]
"""
        toml_path = _write_mc_toml(tmp_path, extra=extra)
        result = load_montecarlo_config(toml_path)
        rmc = result.mc_regions["cortex"]
        assert rmc.D is not None
        assert len(rmc.D) == 2
        assert rmc.D[0].dist == "lognormal"
        assert rmc.D[0].mean == pytest.approx(0.003)
        assert rmc.D[1].sigma == pytest.approx(0.05)

    def test_mc_regions_fractions(self, tmp_path):
        """Dirichlet fractions are parsed correctly (kappa)."""
        extra = """
[[montecarlo.regions]]
name      = "cortex"
fractions = {dist = "dirichlet", kappa = 10.0}
"""
        toml_path = _write_mc_toml(tmp_path, extra=extra)
        result = load_montecarlo_config(toml_path)
        rmc = result.mc_regions["cortex"]
        assert rmc.fractions is not None
        assert rmc.fractions.dist == "dirichlet"
        assert rmc.fractions.kappa == pytest.approx(10.0)

    def test_multiple_mc_regions(self, tmp_path):
        """Multiple [[montecarlo.regions]] entries are all parsed."""
        extra = """
[[montecarlo.regions]]
name = "cortex"
S0   = {dist = "normal", mean = 180.0, std = 10.0}

[[montecarlo.regions]]
name = "medulla"
S0   = {dist = "lognormal", mean = 160.0, sigma = 0.05}
"""
        toml_path = _write_mc_toml(tmp_path, extra=extra)
        result = load_montecarlo_config(toml_path)
        assert set(result.mc_regions) == {"cortex", "medulla"}

    def test_paths_relative_to_toml(self, tmp_path):
        """Case paths are resolved relative to the TOML directory."""
        case_dir = _CASE_DIR.resolve()
        toml_path = tmp_path / "mc.toml"
        toml_path.write_text(
            f"""
[montecarlo]
n_samples = 5

[[cases]]
config = "{case_dir / 'kid_000_synthetic.json'}"
bvals  = "{case_dir / 'kidney.bval'}"
""",
            encoding="utf-8",
        )
        result = load_montecarlo_config(toml_path)
        assert result.cases[0].config.exists()
        assert result.cases[0].bvals.exists()
