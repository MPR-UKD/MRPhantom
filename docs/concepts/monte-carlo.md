# Monte Carlo Simulation

> **TL;DR** — Conceptual overview of MRPhantom's Monte Carlo DWI simulation. Covers why you might want multiple random draws (**Motivation**), how tissue parameters are sampled per draw (**Parameter sampling**), the two independent RNG streams that decouple parameter draws from noise (**Random seeds**), the three distribution types and auto-anchoring (**Normal / lognormal / Dirichlet**), and the trade-off between in-memory batch synthesis vs lazy per-sample writing (**Synthesis modes**).

---

## Motivation

A single `DWIPhantom` synthesis produces one noisy realisation of the DWI signal. In practice you often want *many* realisations to:

- **Benchmark fitting algorithms** — how does an estimator behave over many noise instances?
- **Propagate uncertainty** — spread tissue parameters across draws, not just noise
- **Noise-only ensembles** — keep tissue parameters fixed, vary only the Rician noise

Monte Carlo (MC) simulation wraps the standard DWI synthesis in an outer loop that repeats the process *N* times. Each draw can differ in both its tissue parameters (S0, D, fractions) and its noise realisation, depending on how you configure it.

---

## Parameter sampling

For each region listed in `mc_regions`, you can define per-parameter distributions. The three available distribution types mirror common needs:

| Type | Used for | Property |
|---|---|---|
| `normal` | S0 (baseline signal) | Symmetric scatter around the base S0 |
| `lognormal` | D (diffusivities) | Strictly positive, asymmetric tail |
| `dirichlet` | fractions | Always sums to 1 across compartments |

Parameters of regions **not** listed in `mc_regions` are held fixed at their base-config values across all draws.

### Normal

Plain Gaussian. The mean is auto-anchored to the base config S0 — you only control the spread:

```python
S0 ~ Normal(base_S0, std)
```

### Lognormal

Strictly positive distribution suitable for diffusivities. The log-space mean is set so that the real-space geometric mean equals the base config value:

```python
D ~ LogNormal(mean=log(base_D), sigma=sigma)
```

This ensures the median of the sampled values matches the base config. The `sigma` parameter approximates the geometric coefficient of variation — `sigma=0.10` means ≈10% variation.

### Dirichlet

Compartment fractions must always sum to 1 (they form a probability simplex). The Dirichlet distribution is the natural choice for random vectors on the simplex.

The Dirichlet alpha vector is computed from the base config fractions:

```python
alpha = [base_f1 * kappa, base_f2 * kappa, ...]
```

- **kappa** controls concentration. High kappa (e.g. 100) keeps fractions close to the base values. Low kappa (e.g. 1) allows wide variation.
- The expected value of the Dirichlet is exactly the base fractions.
- Internally, `numpy.random.Generator.dirichlet(alpha)` draws one sample.

> A higher kappa gives more reproducible fractions across draws; a lower kappa explores more of the fraction space. Zero or negative kappa is rejected.

### Auto-anchoring

For normal and lognormal distributions, the mean is **always** determined by the base config. You never specify `mean` — only `std` (normal) or `sigma` (lognormal). This means:

- The base config is the biological reference.
- The distribution is only a perturbation around that reference.
- Changing the base config automatically shifts the entire distribution.

For Dirichlet, the same principle applies: the alpha vector is derived from `base_fractions * kappa`, so the mean fraction is always anchored.

---

## Random seeds

MC draws involve two independent sources of randomness: **tissue parameters** and **noise**. These are separated into two independent RNG streams derived from the master seed:

```
              master seed
              /          \
      param_ss          noise_ss
          |                 |
    param_rng      [seed_0, seed_1, ... seed_{N-1}]
```

1. **param_rng** — used once during `setup()` to draw all tissue parameters across all N samples. Its state determines every sampled S0, D, and fraction value.
2. **noise seeds** — one independent seed per sample, used for Rician noise in `_run_synthesis`.

### Why separate streams?

If parameters and noise shared a single RNG, changing the number of samples (`n_samples`) would shift which noise values are paired with which parameter draws. Independent streams guarantee that the parameter draws are identical regardless of `n_samples`.

### Why per-sample noise seeds?

`noise_ss.spawn(N)` produces N independent child seeds. Each sample always uses the same seed, regardless of the total count. This means:

- Sample 5's noise is identical whether you run 5, 10, or 100 samples.
- Adding more samples does not change earlier samples.
- Reproducibility is preserved across runs with different `N`.

If instead a single RNG was consumed sequentially, sample 5's noise would depend on how many random numbers samples 0-4 consumed — which changes with parameter values.

### Reproducibility

Pass the same `seed` to get identical parameter draws and noise realisations. Pass `seed=None` to use OS entropy (non-reproducible).

---

## Synthesis modes

### Batch mode

```python
phantom.setup().synthesize()
```

All N samples are synthesized in memory. The full 5-D signal array `(X, Y, Z, n_b, N)` is stored on the instance.

- **Pros**: Simple, fast, all samples accessible immediately.
- **Cons**: Memory scales linearly with N. Large spatial volumes × many samples may exceed available RAM.

### Lazy mode

```python
phantom.setup()
with phantom.open_h5("output.h5") as h5:
    for i in range(phantom.n_samples):
        phantom.step(i)
        h5.flush(i)
```

Each sample is synthesized one at a time and written directly to a pre-allocated HDF5 slice. Only one 4-D sample resides in memory at any moment.

- **Pros**: Memory use is constant regardless of N.
- **Cons**: Slightly slower (disk I/O per sample), samples are only available from disk afterwards.

Both modes produce **identical** output files for the same seed.

### Noise-ensemble mode

Pass an empty `mc_regions` dict (the default) to keep all tissue parameters fixed across draws:

```python
phantom = MCDWIPhantom(n_samples=100, snr=20, seed=0)
phantom.load_case(config).setup().synthesize()
```

Now every sample shares the same tissue parameters but gets an independent noise realisation. This isolates the effect of noise on downstream analysis.

---

## Relationship to the runner API

The `MCDWIPhantom` class wraps the same logic exposed by `run_montecarlo()` in `mrphantom.montecarlo.runner`. The runner is a stateless function that performs the same loop in one call:

```python
result = run_montecarlo(
    base_config=cfg,
    mc_regions=mc_regions,
    n_samples=100,
    snr=30,
    snr_region=None,
    seed=42,
)
```

Use the class API when you want lazy writing or method chaining. Use the runner when you want a single function call.

---

## See also

- [TOML Monte Carlo Config](../toml_mc_config.md) — configure MC runs from a `.toml` file
- [Usage Guide — Monte Carlo](../usage.md#monte-carlo-simulation-mcdwiphantom) — code walkthrough
- [Reference — Monte Carlo HDF5 schema](../reference.md#monte-carlo-hdf5-schema) — output file format
- [API Reference — Monte Carlo](../api.md#monte-carlo) — full symbol listing
