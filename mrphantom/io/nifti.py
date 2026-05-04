"""NIfTI export for synthesized phantom data.

Provides :func:`save_nifti` to persist a :class:`nibabel.Nifti1Image` to disk,
with automatic parent-directory creation and support for both ``.nii`` and
compressed ``.nii.gz`` output.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib


def save_nifti(signal_img: nib.Nifti1Image, out_path: Path | str) -> Path:
    """Save a NIfTI image to *out_path*.

    The parent directory is created automatically if it does not exist.
    The output format is inferred from the file extension:

    - ``.nii``    → uncompressed NIfTI-1
    - ``.nii.gz`` → gzip-compressed NIfTI-1 (recommended)

    Args:
        signal_img: The :class:`nibabel.Nifti1Image` to save (e.g. from
            :func:`~mrphantom.synthesis_dwi.synthesize_phantom`).
        out_path:   Destination file path.  If the path does not end in
            ``.nii`` or ``.nii.gz``, the suffix ``.nii.gz`` is appended
            automatically.

    Returns:
        The resolved :class:`Path` of the written file.

    Raises:
        TypeError: If *signal_img* is not a :class:`nibabel.Nifti1Image`.
    """
    if not isinstance(signal_img, nib.Nifti1Image):
        raise TypeError(
            f"signal_img must be a nibabel.Nifti1Image, got {type(signal_img)}"
        )

    out_path = Path(out_path)

    # Ensure the extension is a recognised NIfTI suffix
    suffixes = "".join(out_path.suffixes).lower()
    if not (suffixes.endswith(".nii") or suffixes.endswith(".nii.gz")):
        out_path = out_path.with_suffix("").with_suffix(".nii.gz")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(signal_img, out_path)
    return out_path
