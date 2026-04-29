"""Phantom signal synthesis.

Generates a 4-D synthetic DWI image from a :class:`PhantomConfig` by computing
the multi-exponential diffusion signal for each segmented region:

    S(b) = S0 * Σ_i ( f_i * exp(-b * D_i) )

Optionally adds Rician noise at a specified SNR.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np

from .config import PhantomConfig


def synthesize_phantom(
    config: PhantomConfig,
    snr: float | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, nib.Nifti1Image]:
    """Synthesize a 4-D DWI phantom from ground-truth diffusion parameters.

    For each region defined in *config*, the multi-exponential signal model::

        S(b) = S0 * Σ_i ( f_i * exp(-b * D_i) )

    is evaluated at every b-value and written into all voxels belonging to that
    segment.

    If *snr* is given, Rician noise is added::

        S_noisy = sqrt( (S + N_re)² + N_im² )

    where ``N_re, N_im ~ Normal(0, σ)`` and ``σ = max(S0) / snr``.

    Args:
        config: Fully loaded :class:`~mrsim.config.PhantomConfig`.
        snr:    Signal-to-noise ratio.  Pass ``None`` to skip noise entirely.
        rng:    Optional seeded :class:`numpy.random.Generator` for
                reproducibility.  A new generator is created if ``None``.

    Returns:
        Tuple of:

        - ``signal`` — 4-D ``float64`` array with shape ``(X, Y, Z, n_bvalues)``
          where X, Y, Z are the spatial dimensions of the segmentation.
        - ``signal_img`` — :class:`nibabel.Nifti1Image` sharing the segmentation
          affine and header, wrapping *signal*.

    Raises:
        FileNotFoundError: If the NIfTI segmentation file does not exist.
    """
    if not config.nifti_path.is_file():
        raise FileNotFoundError(f"Segmentation NIfTI not found: {config.nifti_path}")

    seg_img: nib.Nifti1Image = nib.load(config.nifti_path)
    seg_raw: np.ndarray = np.round(seg_img.get_fdata()).astype(np.int32)

    # Segmentation NIfTI may be 4-D (X, Y, Z, volumes) where every volume
    # holds the same integer label map — use only the first volume.
    seg_data: np.ndarray = seg_raw[..., 0] if seg_raw.ndim == 4 else seg_raw

    n_b = len(config.bvalues)
    signal = np.zeros((*seg_data.shape, n_b), dtype=np.float64)

    for region in config.regions.values():
        mask = seg_data == region.index
        if not np.any(mask):
            continue

        # Evaluate multi-exponential model: shape (n_b,)
        region_signal: np.ndarray = region.S0 * sum(
            f * np.exp(-config.bvalues * d) for f, d in zip(region.fractions, region.D)
        )

        # Assign to all voxels belonging to this region: (N_vox, n_b)
        signal[mask] = region_signal

    if snr is not None:
        if rng is None:
            rng = np.random.default_rng()
        max_s0 = max(r.S0 for r in config.regions.values())
        sigma = max_s0 / snr
        noise_real = rng.normal(0.0, sigma, signal.shape)
        noise_imag = rng.normal(0.0, sigma, signal.shape)
        signal = np.sqrt((signal + noise_real) ** 2 + noise_imag**2)

    signal_img = nib.Nifti1Image(signal, seg_img.affine, seg_img.header)
    return signal, signal_img
