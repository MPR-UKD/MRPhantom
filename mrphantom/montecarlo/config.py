"""Monte Carlo simulation configuration.

Parses a TOML run-config that extends the standard ``[run]`` / ``[export]`` /
``[[cases]]`` schema with a ``[montecarlo]`` block describing per-region
parameter distributions.

TOML schema
-----------
::

    [run]
    snr        = 30.0
    snr_region = "cortex"   # optional
    seed       = 42

    [montecarlo]
    n_samples = 100

    [[montecarlo.regions]]
    name       = "cortex"
    S0         = {dist = "normal",    mean = 180.0, std  = 10.0}
    D          = [{dist = "lognormal", mean = 0.003, sigma = 0.10},
                  {dist = "lognormal", mean = 0.001, sigma = 0.05}]
    fractions  = {dist = "dirichlet", alpha = [2.0, 1.0]}

    [[montecarlo.regions]]
    name = "medulla"
    S0   = {dist = "lognormal", mean = 160.0, sigma = 0.05}

    [export]
    output_dir        = "output/"
    filename_template = "{case_id}"
    formats           = ["hdf5"]

    [[cases]]
    config = "case_000/kid_000_synthetic.json"
    bvals  = "case_000/kidney.bval"

Distribution conventions
------------------------
``"normal"``
    Args: *mean*, *std* — plain Gaussian.

``"lognormal"``
    Args: *mean*, *sigma* — *mean* is the **real-space geometric mean**
    (internally mapped to ``np.log(mean)``), *sigma* is the log-space std.
    So for a diffusivity of roughly 0.002 mm²/s use ``mean = 0.002``.

``"dirichlet"``
    Args: *alpha* (list) — Dirichlet concentration parameters, one per
    compartment.  Sampled fractions always sum to 1.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib  # type: ignore[no-redef]
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError as exc:
            raise ImportError(
                "Python < 3.11 requires the 'tomli' package to read TOML files. "
                "Install it with:  pip install tomli"
            ) from exc

from ..run_config import CaseEntry, ExportConfig


# ---------------------------------------------------------------------------
# Distribution specification
# ---------------------------------------------------------------------------


@dataclass
class ParamDist:
    """Distribution specification for a single scalar or vector parameter.

    Attributes:
        dist:  Distribution name: ``"normal"``, ``"lognormal"``, or
               ``"dirichlet"``.
        mean:  Mean for ``"normal"`` distributions, or real-space geometric
               mean for ``"lognormal"`` distributions.
        std:   Standard deviation for ``"normal"`` distributions.
        sigma: Log-space standard deviation for ``"lognormal"``
               distributions.
        alpha: Concentration parameters for ``"dirichlet"`` distributions,
               one per compartment — kept for internal use (computed from
               ``base_fractions * kappa`` at sampling time).
        kappa: Scalar concentration for ``"dirichlet"`` distributions. The
               alpha vector is computed as ``base_fractions * kappa``, so
               the mean fraction is always anchored to the base config.
               Must be > 0. Only used when *dist* is ``"dirichlet"``.

    Examples:
        >>> ParamDist(dist="normal", mean=180.0, std=10.0)
        ParamDist(dist='normal', mean=180.0, std=10.0, sigma=1.0, alpha=[], kappa=0.0)
        >>> ParamDist(dist="dirichlet", kappa=10.0)
        ParamDist(dist='dirichlet', mean=0.0, std=1.0, sigma=1.0, alpha=[], kappa=10.0)
    """

    dist: str
    mean: float = 0.0
    std: float = 1.0
    sigma: float = 1.0
    alpha: list[float] = field(default_factory=list)
    kappa: float = 0.0


# ---------------------------------------------------------------------------
# Per-region Monte Carlo config
# ---------------------------------------------------------------------------


@dataclass
class RegionMCConfig:
    """Monte Carlo parameter distributions for one segmented region.

    Each field is optional: ``None`` means *use the base config value for
    every sample* (i.e. that parameter is deterministic across draws).

    Attributes:
        name:      Region name — must match a
                   :attr:`~mrphantom.dwi.config.RegionConfig.name` in the
                   base :class:`~mrphantom.dwi.config.DWIConfig`.
        S0:        Distribution for baseline signal intensity.
        D:         Per-component diffusivity distributions. If provided,
                   length must equal the region's ``n_components``.
        fractions: Dirichlet distribution for compartment fractions.
                   :attr:`ParamDist.alpha` must have length == n_components.
    """

    name: str
    S0: ParamDist | None = None
    D: list[ParamDist] | None = None
    fractions: ParamDist | None = None


# ---------------------------------------------------------------------------
# Top-level MC run config
# ---------------------------------------------------------------------------


@dataclass
class MCRunConfig:
    """Configuration for a Monte Carlo batch synthesis run.

    Attributes:
        snr:        Signal-to-noise ratio.  ``None`` for noise-free output.
        snr_region: Name of the region whose S0 is used as the SNR noise
                    reference.  ``None`` uses ``max(S0)`` across all regions.
        seed:       Master random seed.  ``None`` for non-reproducible runs.
        export:     Export settings (output directory, filename template,
                    formats).
        cases:      Ordered list of cases to simulate.
        n_samples:  Number of Monte Carlo draws per case.
        mc_regions: Per-region distribution specs, keyed by region name.
                    Regions absent from this dict use their base config values
                    unchanged across all samples.
    """

    snr: float | None
    snr_region: str | None
    seed: int | None
    export: ExportConfig
    cases: list[CaseEntry]
    n_samples: int
    mc_regions: dict[str, RegionMCConfig]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_param_dist(raw: dict) -> ParamDist:
    """Parse a raw TOML distribution dict into a :class:`ParamDist`.

    Args:
        raw: Dict with at least a ``"dist"`` key.

    Returns:
        ParamDist instance.

    Raises:
        KeyError: If ``"dist"`` key is missing.
        ValueError: If the distribution name is not recognised.
    """
    dist: str = raw["dist"]
    if dist in ("normal", "lognormal"):
        return ParamDist(
            dist=dist,
            mean=float(raw.get("mean", 0.0)),
            std=float(raw.get("std", 1.0)),
            sigma=float(raw.get("sigma", 1.0)),
        )
    elif dist == "dirichlet":
        return ParamDist(
            dist=dist,
            kappa=float(raw.get("kappa", 0.0)),
        )
    else:
        raise ValueError(
            f"Unknown distribution {dist!r}. "
            "Supported values: 'normal', 'lognormal', 'dirichlet'."
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_montecarlo_config(toml_path: Path | str) -> MCRunConfig:
    """Parse a Monte Carlo TOML config file into an :class:`MCRunConfig`.

    All relative paths inside the TOML are resolved against the directory
    containing the TOML file.

    Args:
        toml_path: Path to the ``.toml`` config file.

    Returns:
        MCRunConfig: Fully resolved configuration.

    Raises:
        FileNotFoundError: If *toml_path* does not exist.
        KeyError: If required TOML keys are missing.
        ValueError: If ``[[cases]]`` is empty, a distribution name is
            unrecognised, or an ``[[montecarlo.regions]]`` entry is missing
            the ``"name"`` key.
    """
    toml_path = Path(toml_path)
    if not toml_path.is_file():
        raise FileNotFoundError(f"MC config TOML not found: {toml_path}")

    base_dir = toml_path.parent

    with open(toml_path, "rb") as fh:
        data: dict = tomllib.load(fh)

    # ------------------------------------------------------------------ [run]
    run_section: dict = data.get("run", {})
    snr: float | None = run_section.get("snr", None)
    snr_region: str | None = run_section.get("snr_region", None)
    seed: int | None = run_section.get("seed", None)
    if snr is not None:
        snr = float(snr)
    if seed is not None:
        seed = int(seed)

    # --------------------------------------------------------------- [export]
    export_section: dict = data.get("export", {})
    output_dir = base_dir / export_section.get("output_dir", "output")
    filename_template: str = export_section.get("filename_template", "{case_id}")
    formats: list[str] = export_section.get("formats", ["hdf5"])
    export = ExportConfig(
        output_dir=output_dir,
        filename_template=filename_template,
        formats=formats,
    )

    # ------------------------------------------------------------- [[cases]]
    raw_cases: list[dict] = data.get("cases", [])
    if not raw_cases:
        raise ValueError(
            f"MC config TOML contains no [[cases]] entries: {toml_path}"
        )
    cases: list[CaseEntry] = [
        CaseEntry(
            config=(base_dir / raw["config"]).resolve(),
            bvals=(base_dir / raw["bvals"]).resolve(),
        )
        for raw in raw_cases
    ]

    # ---------------------------------------------------------- [montecarlo]
    mc_section: dict = data.get("montecarlo", {})
    n_samples: int = int(mc_section.get("n_samples", 100))

    raw_mc_regions: list[dict] = mc_section.get("regions", [])
    mc_regions: dict[str, RegionMCConfig] = {}
    for raw_region in raw_mc_regions:
        name: str = raw_region["name"]

        s0_raw = raw_region.get("S0")
        d_raw = raw_region.get("D")
        fractions_raw = raw_region.get("fractions")

        s0_dist = _parse_param_dist(s0_raw) if s0_raw is not None else None
        d_dists = (
            [_parse_param_dist(d) for d in d_raw] if d_raw is not None else None
        )
        fractions_dist = (
            _parse_param_dist(fractions_raw) if fractions_raw is not None else None
        )

        mc_regions[name] = RegionMCConfig(
            name=name,
            S0=s0_dist,
            D=d_dists,
            fractions=fractions_dist,
        )

    return MCRunConfig(
        snr=snr,
        snr_region=snr_region,
        seed=seed,
        export=export,
        cases=cases,
        n_samples=n_samples,
        mc_regions=mc_regions,
    )
