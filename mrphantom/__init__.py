"""mrsim — standalone MR phantom synthesis and export library.

Public API
----------
Config / data model::

    from mrsim import RegionConfig, PhantomConfig, load_phantom_config

Synthesis::

    from mrsim import synthesize_phantom

ROI helpers::

    from mrsim import get_evaluation_segments, get_fitting_mask, EVALUATION_REGION_NAMES

I/O::

    from mrsim import save_nifti, save_hdf5, load_hdf5, PhantomData

Quick example::

    from pathlib import Path
    from mrsim import load_phantom_config, synthesize_phantom, save_nifti, save_hdf5

    config = load_phantom_config(Path("phantom.json"), Path("bvals.txt"))
    signal, img = synthesize_phantom(config, snr=30, rng=None)

    save_nifti(img, Path("phantom.nii.gz"))
    save_hdf5(signal, img, config, Path("phantom.h5"), snr=30)
"""

from .config import PhantomConfig, RegionConfig, load_phantom_config
from .io import PhantomData, load_hdf5, save_hdf5, save_nifti
from .roi import EVALUATION_REGION_NAMES, get_evaluation_segments, get_fitting_mask
from .synthesis import synthesize_phantom

__all__ = [
    # config
    "RegionConfig",
    "PhantomConfig",
    "load_phantom_config",
    # synthesis
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
