"""DWI phantom signal synthesis.

Generates a 4-D synthetic DWI image from a :class:`DWIConfig` by computing
the multi-exponential diffusion signal for each segmented region:

    S(b) = S0 * Σ_i ( f_i * exp(-b * D_i) )

Optionally adds Rician noise at a specified SNR.
"""

from __future__ import annotations

from dataclasses import dataclass

import nibabel as nib
import numpy as np

from ..motion import MotionConfig, apply_motion, sample_motion
from .config import DWIConfig


@dataclass
class SynthesisResult:
    """Output of :func:`synthesize_dwi`.

    Attributes:
        signal: Final signal ``(X, Y, Z, n_b)``; moved (if motion) and noisy (if snr).
        signal_gt: Noise-free signal in the reference frame (never moved).
        signal_img: NIfTI image wrapping *signal*.
        motion_matrices: ``(n_b, 4, 4)`` pull matrices (mm, spacing-only frame,
            ``volume(x) = object(M x)``), or ``None`` without motion.
        signal_moved_gt: Noise-free moved signal, or ``None`` without motion.
    """

    signal: np.ndarray
    signal_gt: np.ndarray
    signal_img: nib.Nifti1Image
    motion_matrices: np.ndarray | None = None
    signal_moved_gt: np.ndarray | None = None


def _model_signal(
    seg_data: np.ndarray,
    config: DWIConfig,
    bvalues: np.ndarray,
    bvecs: np.ndarray | None,
) -> np.ndarray:
    """Evaluate the signal model of every region for the given b-values / directions."""
    signal = np.zeros((*seg_data.shape, len(bvalues)), dtype=np.float64)

    for region in config.regions.values():
        mask = seg_data == region.index
        if not np.any(mask):
            continue

        if region.tensor is not None:
            # Single-compartment tensor model: S0 * exp(-b g^T D g), shape (n_b,)
            if bvecs is None:
                raise ValueError(
                    f"Region {region.name!r} defines a diffusion tensor but the "
                    "config has no bvecs; pass bvec_path to load_dwi_config."
                )
            adc = np.einsum("ni,ij,nj->n", bvecs, region.tensor, bvecs)
            region_signal: np.ndarray = region.S0 * np.exp(-bvalues * adc)
        else:
            # Evaluate multi-exponential model: shape (n_b,)
            region_signal = region.S0 * sum(
                f * np.exp(-bvalues * d) for f, d in zip(region.fractions, region.D)
            )

        # Assign to all voxels belonging to this region: (N_vox, n_b)
        signal[mask] = region_signal

    return signal


def _moved_signal(
    seg_data: np.ndarray,
    config: DWIConfig,
    motion: MotionConfig,
    voxel_size: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Per volume: model signal with the gradient in the object frame, then resample.

    With ``volume(x) = object(M x)`` a lab-frame gradient ``g`` acts on the object as
    ``R g`` (R = rotation part of M), so the tensor signal uses ``g_obj = R g``.
    """
    n_b = len(config.bvalues)
    center = (np.asarray(seg_data.shape) - 1) / 2 * np.asarray(voxel_size)
    matrices = sample_motion(n_b, motion, voxel_size, center_mm=center)
    moved = np.zeros((*seg_data.shape, n_b), dtype=np.float64)
    for n in range(n_b):
        rot = matrices[n, :3, :3]
        g = None if config.bvecs is None else (rot @ config.bvecs[n])[None]
        vol = _model_signal(seg_data, config, config.bvalues[n : n + 1], g)[..., 0]
        if n == motion.reference_volume:
            moved[..., n] = vol
        else:
            moved[..., n] = apply_motion(vol, matrices[n], voxel_size)
    return moved, matrices


def synthesize_phantom(
    config: DWIConfig,
    snr: float | None = None,
    rng: np.random.Generator | None = None,
    snr_region: str | None = None,
    motion: MotionConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    """Synthesize a 4-D DWI phantom; returns ``(signal, signal_gt, signal_img)``.

    Thin wrapper around :func:`synthesize_dwi` that drops the motion outputs
    (``motion_matrices``, ``signal_moved_gt``).  See there for all arguments.
    """
    res = synthesize_dwi(config, snr, rng, snr_region, motion)
    return res.signal, res.signal_gt, res.signal_img


def synthesize_dwi(
    config: DWIConfig,
    snr: float | None = None,
    rng: np.random.Generator | None = None,
    snr_region: str | None = None,
    motion: MotionConfig | None = None,
) -> SynthesisResult:
    """Synthesize a 4-D DWI phantom from ground-truth diffusion parameters.

    For each region defined in *config*, the multi-exponential signal model::

        S(b) = S0 * Σ_i ( f_i * exp(-b * D_i) )

    is evaluated at every b-value and written into all voxels belonging to that
    segment.  Regions with a diffusion ``tensor`` instead use the
    single-compartment model ``S(b, g) = S0 * exp(-b * g^T D g)`` with the
    gradient directions ``config.bvecs``.

    If *snr* is given, Rician noise is added::

        S_noisy = sqrt( (S + N_re)² + N_im² )

    where ``N_re, N_im ~ Normal(0, σ)``.  The noise standard deviation σ is
    determined by the *snr_region* argument:

    - If *snr_region* is a region name (e.g. ``"cortex"``):
      ``σ = S0_of_that_region / snr``.
    - If *snr_region* is ``None`` (default): ``σ = max(S0) / snr``.

    Args:
        config:     Fully loaded :class:`~mrphantom.dwi.config.DWIConfig`.
        snr:        Signal-to-noise ratio.  Pass ``None`` to skip noise entirely.
        rng:        Optional seeded :class:`numpy.random.Generator` for
                    reproducibility.  A new generator is created if ``None``.
        snr_region: Name of the region whose S0 is used as the noise reference.
                    Must match a :attr:`~mrphantom.dwi.config.RegionConfig.name`
                    in *config*.  Pass ``None`` to fall back to ``max(S0)``
                    across all regions.
        motion:     Optional :class:`~mrphantom.motion.MotionConfig`.  Each volume
                    is then rigidly moved (gradient directions rotated into the
                    object frame, one resampling per volume) *before* noise is
                    added.  ``signal_gt`` stays in the reference frame.

    Returns:
        :class:`SynthesisResult` with the fields below (``motion_matrices`` and
        ``signal_moved_gt`` are ``None`` without *motion*):

        - ``signal`` — 4-D ``float64`` array with shape ``(X, Y, Z, n_bvalues)``
          containing the (possibly noisy) synthesized signal.
        - ``signal_gt`` — 4-D ``float64`` array with the same shape as
          ``signal``, containing the noise-free ground-truth signal.  Identical
          to ``signal`` when *snr* is ``None``.
        - ``signal_img`` — :class:`nibabel.Nifti1Image` sharing the segmentation
          affine and header, wrapping *signal*.

    Raises:
        FileNotFoundError: If the NIfTI segmentation file does not exist.
        ValueError: If *snr_region* is given but does not match any region name
            in *config*.
    """
    if not config.nifti_path.is_file():
        raise FileNotFoundError(f"Segmentation NIfTI not found: {config.nifti_path}")

    seg_img: nib.Nifti1Image = nib.load(config.nifti_path)
    seg_raw: np.ndarray = np.round(seg_img.get_fdata()).astype(np.int32)

    # Segmentation NIfTI may be 4-D (X, Y, Z, volumes) where every volume
    # holds the same integer label map — use only the first volume.
    seg_data: np.ndarray = seg_raw[..., 0] if seg_raw.ndim == 4 else seg_raw

    signal = _model_signal(seg_data, config, config.bvalues, config.bvecs)

    # Capture noise-free ground-truth before any noise is applied.
    signal_gt = signal.copy()

    motion_matrices = signal_moved_gt = None
    if motion is not None:
        voxel_size = tuple(float(z) for z in seg_img.header.get_zooms()[:3])
        signal, motion_matrices = _moved_signal(seg_data, config, motion, voxel_size)
        signal_moved_gt = signal.copy()

    if snr is not None:
        if snr <= 0:
            raise ValueError(f"snr must be > 0, got {snr}")
        if rng is None:
            rng = np.random.default_rng()
        if snr_region is not None:
            matched = next(
                (r for r in config.regions.values() if r.name == snr_region), None
            )
            if matched is None:
                valid_names = sorted(r.name for r in config.regions.values())
                raise ValueError(
                    f"snr_region {snr_region!r} not found in config. "
                    f"Valid region names: {valid_names}."
                )
            ref_s0 = matched.S0
        else:
            ref_s0 = max(r.S0 for r in config.regions.values())
        sigma = ref_s0 / snr
        noise_real = rng.normal(0.0, sigma, signal.shape)
        noise_imag = rng.normal(0.0, sigma, signal.shape)
        signal = np.sqrt((signal + noise_real) ** 2 + noise_imag**2)

    signal_img = nib.Nifti1Image(signal, seg_img.affine, seg_img.header)
    return SynthesisResult(
        signal, signal_gt, signal_img, motion_matrices, signal_moved_gt
    )
