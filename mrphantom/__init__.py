"""mrphantom — standalone MR phantom synthesis and export library.

Documentation
-------------
Full documentation lives in the ``docs/`` directory:

- ``docs/api.md``              — public API reference
- ``docs/usage.md``            — usage guide and examples
- ``docs/case_structure.md``   — case directory layout and JSON schema
- ``docs/toml_run_config.md``  — TOML run-config format
"""

from .base import MRPhantom
from .config import PhantomConfig, RegionConfig, load_phantom_config
from .dwi import DWIPhantom
from .io import PhantomData, load_hdf5, save_hdf5, save_nifti
from .roi import EVALUATION_REGION_NAMES, get_evaluation_segments, get_fitting_mask
from .run_config import CaseEntry, ExportConfig, RunConfig, load_run_config
from .synthesis import synthesize_phantom

__all__ = [
    # base
    "MRPhantom",
    # phantom classes
    "DWIPhantom",
    # config
    "RegionConfig",
    "PhantomConfig",
    "load_phantom_config",
    # run config
    "RunConfig",
    "CaseEntry",
    "ExportConfig",
    "load_run_config",
    # synthesis (functional API)
    "synthesize_phantom",
    # roi
    "EVALUATION_REGION_NAMES",
    "get_evaluation_segments",
    "get_fitting_mask",
    # io
    "save_nifti",
    "save_hdf5",
    "load_hdf5",
    "PhantomData",
]
