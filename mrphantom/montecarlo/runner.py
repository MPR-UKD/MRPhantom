"""Monte Carlo DWI simulation runner.

Generates *N* independent noisy DWI realisations from a base
:class:`~mrphantom.dwi.config.DWIConfig` by:

1. Sampling tissue parameters (S0, D, fractions) for every region using
   distributions from a :class:`~mrphantom.montecarlo.config.MCRunConfig`.
2. Building a modified :class:`~mrphantom.dwi.config.DWIConfig` for each
   draw.
3. Calling :func:`~mrphantom.dwi.synthesis.synthesize_phantom` once per
   draw with an independent per-sample noise seed.

Two independent RNG streams are derived from the master *seed* via
:class:`numpy.random.SeedSequence`:

- **param_rng** — parameter sampling (S0, D, fractions).
- **noise_seq** — one child seed per sample for Rician noise.

This design ensures that changing *n_samples* does not perturb the noise
realisations of earlier samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dwi.config import DWIConfig
from ..dwi.synthesis import synthesize_phantom
from .config import RegionMCConfig
from .sampler import SampledParams, _build_sample_config, sample_region_params


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class MCResult:
    """Monte Carlo simulation results for one phantom case.

    Attributes:
        signal:         Stacked noisy signal, shape ``(X, Y, Z, n_b, N)``.
        signal_gt:      Stacked noise-free ground-truth signal,
                        shape ``(X, Y, Z, n_b, N)``.  Each slice ``[..., i]``
                        is the GT for the *i*-th parameter draw.
        sampled_params: Parameter arrays used to generate each sample.
        affine:         4×4 NIfTI affine matrix (copied from the segmentation
                        image).
    """

    signal: np.ndarray        # (X, Y, Z, n_b, N)
    signal_gt: np.ndarray     # (X, Y, Z, n_b, N)
    sampled_params: SampledParams
    affine: np.ndarray        # (4, 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_montecarlo(
    base_config: DWIConfig,
    mc_regions: dict[str, RegionMCConfig],
    n_samples: int,
    snr: float | None,
    snr_region: str | None,
    seed: int | None,
) -> MCResult:
    """Run *n_samples* Monte Carlo DWI simulations.

    For each draw *i*:

    1. Tissue parameters are taken from *sampled_params[..., i]* (computed
       once for all draws before the loop).
    2. A modified :class:`~mrphantom.dwi.config.DWIConfig` is constructed
       with those parameters.
    3. :func:`~mrphantom.dwi.synthesis.synthesize_phantom` is called with an
       independent noise seed derived from *seed*.

    Args:
        base_config:  Base :class:`~mrphantom.dwi.config.DWIConfig` with
            reference tissue parameters.
        mc_regions:   Per-region distribution specs (by name).  Regions
            absent from this dict use their base config values unchanged.
        n_samples:    Number of Monte Carlo draws.
        snr:          Signal-to-noise ratio.  ``None`` for noise-free output.
        snr_region:   Name of the region whose S0 is used as the noise
            reference.  ``None`` uses ``max(S0)`` across all regions.
        seed:         Master random seed.  ``None`` yields non-reproducible
            results.

    Returns:
        :class:`MCResult` with stacked 5-D signal and GT arrays.

    Raises:
        ValueError: If a MC region name is not found in *base_config*.
        ValueError: If a D distribution list or Dirichlet alpha length does
            not match ``n_components`` for its region.
    """
    # ------------------------------------------------------------------
    # Derive two independent RNG streams from master seed
    # ------------------------------------------------------------------
    ss = np.random.SeedSequence(seed)
    param_ss, noise_ss = ss.spawn(2)
    param_rng = np.random.default_rng(param_ss)
    noise_seeds = noise_ss.spawn(n_samples)

    # ------------------------------------------------------------------
    # Sample parameters for all regions × all samples
    # ------------------------------------------------------------------
    sampled = sample_region_params(
        base_config, mc_regions, n_samples, param_rng
    )

    # ------------------------------------------------------------------
    # First sample — determines spatial / volume shapes and affine
    # ------------------------------------------------------------------
    first_config = _build_sample_config(base_config, sampled, 0)
    first_rng = np.random.default_rng(noise_seeds[0]) if snr is not None else None
    first_sig, first_gt, first_img = synthesize_phantom(
        first_config,
        snr=snr,
        rng=first_rng,
        snr_region=snr_region,
    )

    spatial_shape = first_sig.shape[:3]  # (X, Y, Z)
    n_b = first_sig.shape[3]
    affine: np.ndarray = first_img.affine.copy()

    signal = np.empty((*spatial_shape, n_b, n_samples), dtype=np.float64)
    signal_gt = np.empty((*spatial_shape, n_b, n_samples), dtype=np.float64)
    signal[..., 0] = first_sig
    signal_gt[..., 0] = first_gt

    # ------------------------------------------------------------------
    # Remaining samples
    # ------------------------------------------------------------------
    for i in range(1, n_samples):
        sample_config = _build_sample_config(base_config, sampled, i)
        sample_rng = np.random.default_rng(noise_seeds[i]) if snr is not None else None
        sig, gt, _ = synthesize_phantom(
            sample_config,
            snr=snr,
            rng=sample_rng,
            snr_region=snr_region,
        )
        signal[..., i] = sig
        signal_gt[..., i] = gt

    return MCResult(
        signal=signal,
        signal_gt=signal_gt,
        sampled_params=sampled,
        affine=affine,
    )
