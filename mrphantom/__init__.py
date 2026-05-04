"""mrphantom — standalone MR phantom synthesis and export library.

Documentation
-------------
Full documentation lives in the ``docs/`` directory:

- ``docs/api.md``              — public API reference
- ``docs/usage.md``            — usage guide and examples
- ``docs/case_structure.md``   — case directory layout and JSON schema
- ``docs/toml_run_config.md``  — TOML run-config format
- ``docs/reference.md``        — signal model formulation and HDF5 file schema
"""

from .base import MRPhantom
from .dwi import (
    DWIConfig,
    DWIPhantom,
    EVALUATION_REGION_NAMES,
    RegionConfig,
    get_evaluation_segments,
    load_dwi_config,
    synthesize_phantom,
)
from .io import PhantomData, load_hdf5, save_hdf5, save_nifti
from .roi import get_fitting_mask
from .run_config import CaseEntry, ExportConfig, RunConfig, load_run_config

__all__ = [
    # base
    "MRPhantom",
    # dwi phantom
    "DWIPhantom",
    # dwi config
    "RegionConfig",
    "DWIConfig",
    "load_dwi_config",
    # run config
    "RunConfig",
    "CaseEntry",
    "ExportConfig",
    "load_run_config",
    # synthesis (functional API)
    "synthesize_phantom",
    # roi — general
    "get_fitting_mask",
    # roi — dwi / kidney
    "EVALUATION_REGION_NAMES",
    "get_evaluation_segments",
    # io
    "save_nifti",
    "save_hdf5",
    "load_hdf5",
    "PhantomData",
]
