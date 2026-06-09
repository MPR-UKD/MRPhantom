"""mrphantom.montecarlo — Monte Carlo DWI simulation subpackage.

Public API
----------
:class:`MCDWIPhantom`
    Class-centred Monte Carlo phantom synthesizer (batch and lazy modes).

:class:`ParamDist`
    Distribution specification for a scalar or vector tissue parameter.

:class:`RegionMCConfig`
    Per-region distribution specs (S0, D, fractions).

:class:`MCRunConfig`
    Top-level Monte Carlo run configuration (loaded from TOML).

:func:`load_montecarlo_config`
    Parse a Monte Carlo TOML config file.

:class:`SampledParams`
    Pre-sampled parameter arrays for all regions × N draws.

:func:`sample_region_params`
    Draw per-region tissue parameters for N Monte Carlo samples.

:class:`MCResult`
    Stacked (X, Y, Z, n_b, N) signal and GT arrays plus sampled params.

:func:`run_montecarlo`
    Run N independent DWI simulations from a base config.
"""

from .config import (
    MCRunConfig,
    ParamDist,
    RegionMCConfig,
    load_montecarlo_config,
)
from .phantom import MCDWIPhantom
from .runner import MCResult, run_montecarlo
from .sampler import SampledParams, sample_region_params

__all__ = [
    # phantom
    "MCDWIPhantom",
    # config
    "ParamDist",
    "RegionMCConfig",
    "MCRunConfig",
    "load_montecarlo_config",
    # sampler
    "SampledParams",
    "sample_region_params",
    # runner
    "MCResult",
    "run_montecarlo",
]
