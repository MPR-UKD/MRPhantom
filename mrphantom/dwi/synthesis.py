"""DWI phantom signal synthesis.

Generates a 4-D synthetic DWI image from a :class:`DWIConfig` by computing
the multi-exponential diffusion signal for each segmented region:

    S(b) = S0 * Σ_i ( f_i * exp(-b * D_i) )

Optionally adds Rician noise at a specified SNR.

Opt-in extensions (all off by default, so label-only cases behave exactly as
before):

- **Partial volume** — a region may reference a probability map
  (``RegionConfig.probability_file``).  The signal becomes
  ``S(x, b) = Σ_r p_r(x) · S_r(b)``.
- **Texture** — ``S0_cv`` / ``D_cv`` modulate S0 and D with a smooth, seeded
  random field.
- **Resampling** — ``DWIConfig.voxel_size`` block-averages the maps to a
  coarser grid before synthesis (which itself yields partial volume).
"""

from __future__ import annotations

from dataclasses import dataclass

import nibabel as nib
import numpy as np

from .config import DWIConfig
from .resample import block_average, block_factors, resample_affine
from .texture import scale_field


@dataclass
class SynthesisResult:
    """Everything :func:`synthesize_with_truth` produces.

    Attributes:
        signal: (Possibly noisy) signal, ``(X, Y, Z, n_b)``.
        signal_gt: Noise-free signal, same shape.
        signal_img: NIfTI wrapping *signal*.
        segmentation: ``int32`` label map on the synthesised grid: the region with
            the highest probability, or ``0`` where the implicit background
            ``1 - Σp`` is higher (only when the grid was resampled or probability
            maps are used; otherwise the input labels).
        params: Ground-truth parameters, all indexed by region in ascending
            label order: ``region_index`` ``(n,)``, ``S0`` ``(n,)``,
            ``fractions`` / ``D`` ``(n, C)`` (NaN-padded), ``probability``
            ``(n, X, Y, Z)`` and, only if any region has texture,
            ``S0_scale`` / ``D_scale`` ``(n, X, Y, Z)``.  The per-voxel S0 of
            region ``r`` is ``S0[r] * S0_scale[r]``; its diffusivities are
            ``D[r] * D_scale[r]``.
    """

    signal: np.ndarray
    signal_gt: np.ndarray
    signal_img: nib.Nifti1Image
    segmentation: np.ndarray
    params: dict[str, np.ndarray]


def _load_probabilities(
    config: DWIConfig, seg_img: nib.Nifti1Image, seg_data: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-region probability stack on the synthesis grid.

    Returns:
        ``(indices, probability, segmentation, affine)`` where *probability* has
        shape ``(n_regions, X, Y, Z)``.
    """
    indices = np.array(sorted(config.regions), dtype=np.int32)
    factors = None
    if config.voxel_size is not None:
        factors = block_factors(seg_img.header.get_zooms(), config.voxel_size)
        if np.all(factors == 1):
            factors = None

    out_shape = (
        tuple(-(-s // int(f)) for s, f in zip(seg_data.shape, factors))
        if factors is not None
        else seg_data.shape
    )
    prob = np.zeros((len(indices), *out_shape), dtype=np.float32)

    for k, idx in enumerate(indices):
        region = config.regions[int(idx)]
        if region.probability_file is not None:
            if not region.probability_file.is_file():
                raise FileNotFoundError(
                    f"Probability NIfTI not found: {region.probability_file}"
                )
            p = nib.load(region.probability_file).get_fdata(dtype=np.float32)
            if p.ndim == 4:
                p = p[..., 0]
            if p.shape != seg_data.shape:
                raise ValueError(
                    f"Probability map of region {region.name!r} has shape "
                    f"{p.shape}, expected {seg_data.shape}."
                )
            p = np.clip(p, 0.0, 1.0)
        else:
            p = (seg_data == idx).astype(np.float32)
        prob[k] = block_average(p, factors) if factors is not None else p

    if factors is None and not any(
        r.probability_file is not None for r in config.regions.values()
    ):
        segmentation = seg_data
    else:
        # Dominant class per voxel, counting the implicit background 1 - Σp as a class
        # (faint probability tails in empty space must not be labelled as tissue).
        best = prob.argmax(axis=0)
        background = np.clip(1.0 - prob.sum(axis=0), 0.0, None)
        segmentation = np.where(prob.max(axis=0) > background, indices[best], 0).astype(
            np.int32
        )

    affine = (
        resample_affine(seg_img.affine, factors) if factors is not None else seg_img.affine
    )
    return indices, prob, segmentation, affine


def synthesize_phantom(
    config: DWIConfig,
    snr: float | None = None,
    rng: np.random.Generator | None = None,
    snr_region: str | None = None,
    texture_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    """Synthesize a 4-D DWI phantom (signal, ground-truth signal, NIfTI image).

    Thin wrapper around :func:`synthesize_with_truth` that drops the label map
    and parameter maps.  See there for the model and arguments.
    """
    res = synthesize_with_truth(
        config, snr=snr, rng=rng, snr_region=snr_region, texture_seed=texture_seed
    )
    return res.signal, res.signal_gt, res.signal_img


def synthesize_with_truth(
    config: DWIConfig,
    snr: float | None = None,
    rng: np.random.Generator | None = None,
    snr_region: str | None = None,
    texture_seed: int | None = None,
) -> SynthesisResult:
    """Synthesize a 4-D DWI phantom and return it with its ground truth.

    For each region defined in *config*, the multi-exponential signal model::

        S_r(b) = S0 * Σ_i ( f_i * exp(-b * D_i) )

    is evaluated and weighted by the region's probability ``p_r(x)`` (a hard
    0/1 label mask unless the region has a probability map or the grid is
    resampled)::

        S(x, b) = Σ_r p_r(x) * S_r(x, b)

    If a region sets ``S0_cv`` / ``D_cv``, S0 and D are multiplied by a smooth
    random field ``max(1 + cv * g(x), 0)`` (``g`` unit variance, smoothed with
    ``config.texture_sigma`` voxels), seeded by *texture_seed*.

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
                    the Rician noise.  A new generator is created if ``None``.
        snr_region: Name of the region whose S0 is used as the noise reference.
                    Must match a :attr:`~mrphantom.dwi.config.RegionConfig.name`
                    in *config*.  Pass ``None`` to fall back to ``max(S0)``
                    across all regions.
        texture_seed: Seed for the texture fields (independent of the noise
                    stream).  ``None`` draws fresh entropy.  Unused when no
                    region has texture.

    Returns:
        A :class:`SynthesisResult`.

    Raises:
        FileNotFoundError: If the segmentation or a probability NIfTI does not
            exist.
        ValueError: If *snr_region* is given but does not match any region name
            in *config*, a probability map has the wrong shape, or the target
            voxel size is not an integer multiple of the source's.
    """
    if not config.nifti_path.is_file():
        raise FileNotFoundError(f"Segmentation NIfTI not found: {config.nifti_path}")

    seg_img: nib.Nifti1Image = nib.load(config.nifti_path)
    seg_raw: np.ndarray = np.round(seg_img.get_fdata()).astype(np.int32)

    # Segmentation NIfTI may be 4-D (X, Y, Z, volumes) where every volume
    # holds the same integer label map — use only the first volume.
    seg_data: np.ndarray = seg_raw[..., 0] if seg_raw.ndim == 4 else seg_raw

    indices, prob, segmentation, affine = _load_probabilities(config, seg_img, seg_data)
    grid_shape = prob.shape[1:]

    n_b = len(config.bvalues)
    n_reg = len(indices)
    max_comp = max(len(r.D) for r in config.regions.values())
    signal = np.zeros((*grid_shape, n_b), dtype=np.float64)

    textured = any(r.S0_cv > 0 or r.D_cv > 0 for r in config.regions.values())
    tex_rng = np.random.default_rng(
        None if texture_seed is None else [int(texture_seed), 0x7E57]
    )
    s0_scale = np.ones((n_reg, *grid_shape), dtype=np.float32) if textured else None
    d_scale = np.ones((n_reg, *grid_shape), dtype=np.float32) if textured else None

    p_S0 = np.zeros(n_reg)
    p_frac = np.full((n_reg, max_comp), np.nan)
    p_D = np.full((n_reg, max_comp), np.nan)

    for k, idx in enumerate(indices):
        region = config.regions[int(idx)]
        n_c = len(region.D)
        p_S0[k] = region.S0
        p_frac[k, :n_c] = region.fractions
        p_D[k, :n_c] = region.D

        mask = prob[k] > 0
        if not np.any(mask):
            continue
        p = prob[k][mask].astype(np.float64)[:, None]  # (N, 1)

        if region.S0_cv > 0:
            s0_scale[k] = scale_field(grid_shape, region.S0_cv, config.texture_sigma, tex_rng)
        if region.D_cv > 0:
            d_scale[k] = scale_field(grid_shape, region.D_cv, config.texture_sigma, tex_rng)

        if region.S0_cv > 0 or region.D_cv > 0:
            s0 = region.S0 * s0_scale[k][mask].astype(np.float64)[:, None]  # (N, 1)
            ds = d_scale[k][mask].astype(np.float64)[:, None]  # (N, 1)
            region_signal = s0 * sum(
                f * np.exp(-config.bvalues[None, :] * (d * ds))
                for f, d in zip(region.fractions, region.D)
            )
        else:
            # Evaluate multi-exponential model: shape (n_b,)
            region_signal = region.S0 * sum(
                f * np.exp(-config.bvalues * d)
                for f, d in zip(region.fractions, region.D)
            )

        signal[mask] += p * region_signal

    # Capture noise-free ground-truth before any noise is applied.
    signal_gt = signal.copy()

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

    header = seg_img.header
    signal_img = nib.Nifti1Image(signal, affine, header)
    if affine is not seg_img.affine:
        # Resampled grid: record the new voxel size in the header.
        zooms = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
        signal_img.header.set_zooms(tuple(float(z) for z in zooms) + (1.0,))

    params: dict[str, np.ndarray] = {
        "region_index": indices,
        "S0": p_S0,
        "fractions": p_frac,
        "D": p_D,
        "probability": prob,
    }
    if textured:
        params["S0_scale"] = s0_scale
        params["D_scale"] = d_scale

    return SynthesisResult(
        signal=signal,
        signal_gt=signal_gt,
        signal_img=signal_img,
        segmentation=segmentation,
        params=params,
    )
