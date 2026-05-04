"""Generate a fully synthetic kidney-phantom segmentation NIfTI.

Creates ``assets/kidney/case_000/kid_000_synthetic.nii.gz`` — a 176×176×3
integer label map where every region defined in the kidney JSON schema is
represented by a simple geometric shape (ellipse or rectangle).

All shapes are drawn in the XY plane (176×176) and replicated unchanged
across the three Z slices.  Drawing order is back-to-front, so foreground
structures overwrite background ones where they overlap.

Region index → anatomical meaning
-----------------------------------
 0  background
 1  hilum
 2  liver
 3  spleen
 4  bones (spine)
 5  lungs
 6  fat
 7  remaining_tissue
 8  spinal_disc
 9  spinal_cord
10  cortex
11  medulla

Usage
-----
    uv run python scripts/make_synthetic_seg.py
    # or
    python scripts/make_synthetic_seg.py
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------

OUT = Path(__file__).parent.parent / "assets" / "kidney" / "case_000" / "kid_000_synthetic.nii.gz"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NX, NY, NZ = 176, 176, 3


def ellipse(cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    """Return a boolean 2-D mask (NX × NY) for the filled ellipse.

    Parameters
    ----------
    cx, cy : float  Centre coordinates along X and Y axes.
    rx, ry : float  Semi-axes along X and Y axes (in voxels).
    """
    X, Y = np.ogrid[:NX, :NY]
    return ((X - cx) ** 2 / rx ** 2 + (Y - cy) ** 2 / ry ** 2) <= 1.0


def fill(seg: np.ndarray, mask2d: np.ndarray, label: int) -> None:
    """Write *label* into every Z-slice where *mask2d* is True."""
    for z in range(NZ):
        seg[mask2d, z] = label


# ---------------------------------------------------------------------------
# Build segmentation
# ---------------------------------------------------------------------------
#
# Coordinate convention (matches NIfTI identity affine):
#   X  axis — left↔right,         0 .. 175
#   Y  axis — inferior↔superior,  0 .. 175
#   Z  axis — anterior↔posterior, 0 ..   2  (3 slices)
#
# Layout overview (coronal abdominal view):
#
#   Y=175 ┌──────────────────────────────────┐
#         │          [ lungs ]               │
#         │   [liver]   |spine|   [spleen]   │
#         │   [L kid]   |disc |   [R kid]    │
#   Y=  0 └──────────────────────────────────┘
#         X=0                             X=175
#
# Drawing order: background → outer body → organs → fine structures.
# Later calls overwrite earlier ones where masks overlap.

def build_seg() -> np.ndarray:
    seg = np.zeros((NX, NY, NZ), dtype=np.int16)

    # -- outer body envelope -----------------------------------------------
    fill(seg, ellipse(88, 95, 72, 78), 6)   # fat           (outer ring)
    fill(seg, ellipse(88, 95, 66, 71), 7)   # remaining_tissue (inner body)

    # -- large organs ------------------------------------------------------
    fill(seg, ellipse(88, 148, 26, 22), 5)  # left lung  (upper, X-centre)
    # two separate lung lobes
    fill(seg, ellipse(60, 145, 22, 26), 5)  # left lung
    fill(seg, ellipse(116, 145, 22, 26), 5) # right lung
    fill(seg, ellipse(46, 105, 22, 18), 2)  # liver  (left in anatomy = low X)
    fill(seg, ellipse(128, 106, 14, 14), 3) # spleen (right in anatomy = high X)

    # -- spine + spinal contents -------------------------------------------
    fill(seg, ellipse(88, 95, 8, 42), 4)    # bones / spine column
    fill(seg, ellipse(88, 95, 6, 6), 8)     # spinal_disc  (small disc)
    fill(seg, ellipse(88, 95, 3, 3), 9)     # spinal_cord  (tiny cord)

    # -- kidneys -----------------------------------------------------------
    # left kidney  (lower X = patient's right)
    fill(seg, ellipse(50, 112, 12, 18), 10) # cortex
    fill(seg, ellipse(50, 112,  7, 11), 11) # medulla
    fill(seg, ellipse(62, 112,  4,  5),  1) # hilum (medial border, towards spine)

    # right kidney (higher X = patient's left)
    fill(seg, ellipse(126, 112, 12, 18), 10)
    fill(seg, ellipse(126, 112,  7, 11), 11)
    fill(seg, ellipse(114, 112,  4,  5),  1)

    return seg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    seg = build_seg()

    labels = sorted(np.unique(seg).tolist())
    print(f"Labels present : {labels}")
    print(f"Expected labels: {list(range(12))}")
    missing = set(range(12)) - set(labels)
    if missing:
        raise RuntimeError(f"Missing label(s): {missing}")

    affine = np.diag([1.0, 1.0, 3.0, 1.0])   # 1 mm isotropic XY, 3 mm slice gap
    img = nib.Nifti1Image(seg, affine)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, OUT)
    print(f"Saved -> {OUT}  (shape {img.shape})")


if __name__ == "__main__":
    main()
