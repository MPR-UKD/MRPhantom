"""I/O sub-package for mrsim.

Public API::

    from mrsim.io import save_nifti, save_hdf5, load_hdf5, PhantomData
"""

from .hdf5 import PhantomData, load_hdf5, save_hdf5
from .nifti import save_nifti

__all__ = [
    "save_nifti",
    "save_hdf5",
    "load_hdf5",
    "PhantomData",
]
