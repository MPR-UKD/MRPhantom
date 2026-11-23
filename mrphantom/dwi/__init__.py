"""mrphantom.dwi — DWI-specific phantom synthesis subpackage.

All DWI-specific types and functions are re-exported from this subpackage
for convenient access via ``mrphantom.dwi`` or the top-level ``mrphantom``
namespace.

Submodules
----------
- :mod:`mrphantom.dwi.config`    — :class:`DWIConfig`, :class:`RegionConfig`, :func:`load_dwi_config`
- :mod:`mrphantom.dwi.phantom`   — :class:`DWIPhantom`
- :mod:`mrphantom.dwi.synthesis` — :func:`synthesize_phantom`, :func:`synthesize_with_truth`
- :mod:`mrphantom.dwi.resample`  — block-average downsampling helpers
- :mod:`mrphantom.dwi.texture`   — smooth random texture fields
- :mod:`mrphantom.dwi.roi`       — :data:`EVALUATION_REGION_NAMES`, :func:`get_evaluation_segments`
"""

from .config import DWIConfig, RegionConfig, load_dwi_config
from .phantom import DWIPhantom
from .roi import EVALUATION_REGION_NAMES, get_evaluation_segments
from .synthesis import SynthesisResult, synthesize_phantom, synthesize_with_truth

__all__ = [
    "DWIPhantom",
    "DWIConfig",
    "RegionConfig",
    "load_dwi_config",
    "synthesize_phantom",
    "synthesize_with_truth",
    "SynthesisResult",
    "EVALUATION_REGION_NAMES",
    "get_evaluation_segments",
]
