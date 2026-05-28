"""Parameter sampling for Monte Carlo DWI simulation.

Draws per-region tissue parameters (S0, D, fractions) for N independent
Monte Carlo samples.  For regions without a distribution specification the
base config values are broadcast unchanged across all samples.

Distribution conventions match :mod:`mrphantom.montecarlo.config`:

- ``"normal"``    — plain Gaussian; arg ``std``. Mean is auto-anchored to the
  base config value.
- ``"lognormal"`` — arg ``sigma`` (log-space std). Mean is auto-anchored to the
  base config value (real-space geometric mean, ``log(base_value)`` is passed
  to ``numpy``).
- ``"dirichlet"`` — *kappa* is a scalar concentration parameter; the
  Dirichlet alpha vector is computed as ``base_fractions * kappa``, so
  the mean fraction is always anchored to the base config values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dwi.config import DWIConfig, RegionConfig
from .config import ParamDist, RegionMCConfig


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SampledParams:
    """Sampled parameter arrays for all regions across N Monte Carlo draws.

    Attributes:
        region_names: Ordered list of region names, length ``n_regions``.
            Order matches the first axis of :attr:`S0`, :attr:`D`, and
            :attr:`fractions`.
        S0:        Sampled (or fixed) S0 per region per sample,
                   shape ``(n_regions, n_samples)``.
        D:         Sampled (or fixed) diffusivities,
                   shape ``(n_regions, max_n_components, n_samples)``.
                   Components beyond a region's ``n_components`` are zero.
        fractions: Sampled (or fixed) compartment fractions,
                   shape ``(n_regions, max_n_components, n_samples)``.
                   Fractions sum to 1 along the component axis for each
                   region.  Components beyond ``n_components`` are zero.
    """

    region_names: list[str]
    S0: np.ndarray        # (n_regions, n_samples)
    D: np.ndarray         # (n_regions, max_n_comp, n_samples)
    fractions: np.ndarray  # (n_regions, max_n_comp, n_samples)


# ---------------------------------------------------------------------------
# Internal samplers
# ---------------------------------------------------------------------------


def _sample_scalar(
    dist: ParamDist,
    base_value: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw *n* i.i.d. samples from a scalar distribution centred on *base_value*.

    The mean is always anchored to *base_value* — *dist.mean* is ignored.

    Args:
        dist:       Distribution specification (``"normal"`` or ``"lognormal"``).
        base_value: Base config value used as the distribution mean.
        n:          Number of samples.
        rng:        Random generator.

    Returns:
        1-D ``float64`` array of length *n*.

    Raises:
        ValueError: If *dist.dist* is not a scalar distribution.
    """
    if dist.dist == "normal":
        return rng.normal(base_value, dist.std, size=n).astype(np.float64)
    elif dist.dist == "lognormal":
        return rng.lognormal(
            mean=np.log(base_value), sigma=dist.sigma, size=n
        ).astype(np.float64)
    else:
        raise ValueError(
            f"Expected a scalar distribution ('normal' or 'lognormal'), "
            f"got {dist.dist!r}."
        )


def _sample_dirichlet(
    kappa: float,
    base_fractions: list[float],
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw *n* samples from a Dirichlet distribution centred on *base_fractions*.

    The Dirichlet alpha vector is computed as ``base_fractions * kappa``,
    so the mean of the sampled fractions always equals *base_fractions*.

    Args:
        kappa:         Concentration parameter (must be > 0).
        base_fractions: Base compartment fractions from the config (sum to 1).
        n:             Number of samples.
        rng:           Random generator.

    Returns:
        2-D ``float64`` array of shape ``(n, k)`` where ``k = len(base_fractions)``.

    Raises:
        ValueError: If *kappa* is not positive.
    """
    if kappa <= 0:
        raise ValueError(f"Dirichlet kappa must be positive, got {kappa}.")
    alpha = np.asarray(base_fractions, dtype=np.float64) * kappa
    return rng.dirichlet(alpha, size=n).astype(np.float64)  # (n, k)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sample_region_params(
    base_config: DWIConfig,
    mc_regions: dict[str, RegionMCConfig],
    n_samples: int,
    rng: np.random.Generator,
) -> SampledParams:
    """Sample per-region tissue parameters for *n_samples* Monte Carlo draws.

    For each region:

    - If the region appears in *mc_regions* with a distribution for a
      parameter, that parameter is sampled independently for every draw.
    - If the region is absent from *mc_regions*, or its distribution for a
      particular parameter is ``None``, the base config value is used
      unchanged for all draws.

    Args:
        base_config: Base :class:`~mrphantom.dwi.config.DWIConfig` with
            reference tissue parameters.
        mc_regions:  Per-region distribution specs, keyed by region name.
            A region absent from this dict is treated as fully deterministic.
        n_samples:   Number of Monte Carlo draws.
        rng:         Seeded :class:`numpy.random.Generator`.

    Returns:
        :class:`SampledParams` with arrays of sampled (or fixed) parameters
        for every region, ordered by ascending NIfTI label index.

    Raises:
        ValueError: If a key in *mc_regions* does not match any region name
            in *base_config*.
        ValueError: If a ``D`` distribution list length does not match
            ``n_components`` for the corresponding region.
        ValueError: If a ``fractions`` Dirichlet ``alpha`` length does not
            match ``n_components`` for the corresponding region.
    """
    # ------------------------------------------------------------------
    # Validate MC region names
    # ------------------------------------------------------------------
    base_names = {r.name for r in base_config.regions.values()}
    for name in mc_regions:
        if name not in base_names:
            raise ValueError(
                f"MC region {name!r} not found in base config. "
                f"Valid names: {sorted(base_names)}."
            )

    # ------------------------------------------------------------------
    # Order regions deterministically (ascending NIfTI label index)
    # ------------------------------------------------------------------
    ordered_regions = [
        base_config.regions[idx] for idx in sorted(base_config.regions)
    ]
    region_names = [r.name for r in ordered_regions]
    n_regions = len(ordered_regions)
    max_comp = max((r.n_components for r in ordered_regions), default=1)

    S0_arr = np.zeros((n_regions, n_samples), dtype=np.float64)
    D_arr = np.zeros((n_regions, max_comp, n_samples), dtype=np.float64)
    fractions_arr = np.zeros((n_regions, max_comp, n_samples), dtype=np.float64)

    for i, region in enumerate(ordered_regions):
        mc = mc_regions.get(region.name)
        n_comp = region.n_components

        # ---- S0 ----------------------------------------------------------
        if mc is not None and mc.S0 is not None:
            S0_arr[i] = _sample_scalar(mc.S0, region.S0, n_samples, rng)
        else:
            S0_arr[i] = region.S0

        # ---- D -----------------------------------------------------------
        if mc is not None and mc.D is not None:
            if len(mc.D) != n_comp:
                raise ValueError(
                    f"Region {region.name!r}: mc.D has {len(mc.D)} distribution(s) "
                    f"but region has {n_comp} component(s)."
                )
            for c, dist in enumerate(mc.D):
                D_arr[i, c] = _sample_scalar(dist, region.D[c], n_samples, rng)
        else:
            for c in range(n_comp):
                D_arr[i, c] = region.D[c]

        # ---- fractions ---------------------------------------------------
        if mc is not None and mc.fractions is not None:
            if mc.fractions.dist != "dirichlet":
                raise ValueError(
                    f"Region {region.name!r}: fractions distribution must be "
                    f"'dirichlet', got {mc.fractions.dist!r}."
                )
            if mc.fractions.kappa <= 0:
                raise ValueError(
                    f"Region {region.name!r}: fractions Dirichlet kappa must "
                    f"be positive, got {mc.fractions.kappa}."
                )
            sampled_fracs = _sample_dirichlet(
                mc.fractions.kappa, region.fractions, n_samples, rng
            )  # (n_samples, n_comp)
            fractions_arr[i, :n_comp] = sampled_fracs.T  # → (n_comp, n_samples)
        else:
            for c in range(n_comp):
                fractions_arr[i, c] = region.fractions[c]

    return SampledParams(
        region_names=region_names,
        S0=S0_arr,
        D=D_arr,
        fractions=fractions_arr,
    )


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def _build_sample_config(
    base_config: DWIConfig,
    sampled: SampledParams,
    sample_idx: int,
) -> DWIConfig:
    """Return a :class:`DWIConfig` with parameters from draw *sample_idx*.

    Args:
        base_config:  Reference configuration.
        sampled:      Pre-sampled parameter arrays from
                      :func:`sample_region_params`.
        sample_idx:   Which draw to extract (0-indexed).

    Returns:
        DWIConfig with per-region S0, D, fractions replaced by sampled
        values for this draw.
    """
    new_regions: dict[int, RegionConfig] = {}
    for region_idx, region in base_config.regions.items():
        try:
            arr_idx = sampled.region_names.index(region.name)
        except ValueError:
            # Should never happen after upstream validation.
            new_regions[region_idx] = region
            continue

        n_comp = region.n_components
        s0 = float(sampled.S0[arr_idx, sample_idx])
        d = [float(sampled.D[arr_idx, c, sample_idx]) for c in range(n_comp)]
        fracs = [
            float(sampled.fractions[arr_idx, c, sample_idx]) for c in range(n_comp)
        ]

        new_regions[region_idx] = RegionConfig(
            name=region.name,
            index=region.index,
            n_components=n_comp,
            S0=s0,
            fractions=fracs,
            D=d,
        )

    return DWIConfig(
        case_id=base_config.case_id,
        name=base_config.name,
        description=base_config.description,
        nifti_path=base_config.nifti_path,
        image_dims=base_config.image_dims,
        bvalues=base_config.bvalues,
        regions=new_regions,
    )
