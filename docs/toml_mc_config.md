# TOML Monte Carlo Config

> **TL;DR** — Top-level orchestration file for Monte Carlo DWI simulation. Covers the complete TOML schema with a ready-to-copy example (**Schema**), field-by-field reference for `[run]`, `[montecarlo]`, `[[montecarlo.regions]]`, distribution conventions, `[export]`, and `[[cases]]` (**Fields**), output format notes (**Output**), and CLI and programmatic usage (**CLI**).

---

## Schema

```toml
[run]
snr        = 30.0
snr_region = "cortex"   # region whose S0 sets the noise floor; omit for max(S0)
seed       = 42

[montecarlo]
n_samples = 100

[[montecarlo.regions]]
name      = "cortex"
S0        = {dist = "normal",    std  = 15.0}
D         = [{dist = "lognormal", sigma = 0.10},
             {dist = "lognormal", sigma = 0.15},
             {dist = "lognormal", sigma = 0.20}]
fractions = {dist = "dirichlet", kappa = 10.0}

[[montecarlo.regions]]
name = "medulla"
S0   = {dist = "lognormal", sigma = 0.05}

[export]
output_dir        = "output/"
filename_template = "{case_id}_mc"
formats           = ["hdf5"]

[[cases]]
config = "assets/kidney/case_000/kid_000_synthetic.json"
bvals  = "assets/kidney/case_000/kidney.bval"
```

A ready-to-run example is at `assets/kidney/mc.toml`.

---

## Fields

### `[run]`

| Key | Type | Default | Description |
|---|---|---|---|
| `snr` | float | `null` | Signal-to-noise ratio for Rician noise. Omit or set to `null` for noise-free output. |
| `snr_region` | string | `null` | Name of the region whose S0 sets the noise floor (`σ = S0_region / snr`). Omit to use `max(S0)` across all regions. |
| `seed` | int | `null` | Master random seed for reproducible parameter draws and noise. |

### `[montecarlo]`

| Key | Type | Default | Description |
|---|---|---|---|
| `n_samples` | int | `100` | Number of independent Monte Carlo draws per case. |

### `[[montecarlo.regions]]`

Each `[[montecarlo.regions]]` entry defines parameter distributions for one segmented region. Regions not listed here use their base-config values unchanged across all samples.

| Key | Type | Required | Description |
|---|---|---|---|
| `name` | string | yes | Must match a region name in the case JSON. |
| `S0` | distribution table | no | Baseline signal intensity distribution. |
| `D` | array of distribution tables | no | Per-component diffusivity distributions. Length must equal `n_components` for this region. |
| `fractions` | distribution table | no | Compartment fraction distribution. Must use `dist = "dirichlet"` with a positive `kappa` value. The alpha vector is computed as `base_fractions * kappa`. |

Any of `S0`, `D`, `fractions` may be omitted independently; omitted fields use the base-config value for every sample.

---

## Distribution types

### `"normal"`

Plain Gaussian. Useful for S0, which is always positive in practice and can be modeled as approximately normal.
The mean is auto-anchored to the base config value — you only specify the spread.

```toml
S0 = {dist = "normal", std = 15.0}
```

| Parameter | Description |
|---|---|
| `std` | Standard deviation. |

### `"lognormal"`

Log-normal distribution for strictly-positive parameters (diffusivities).
The mean is auto-anchored to the base config value (real-space geometric mean) — you only specify the log-space spread.

```toml
D = [{dist = "lognormal", sigma = 0.10}]
```

| Parameter | Description |
|---|---|
| `sigma` | Log-space standard deviation. A value of `0.10` is a ≈10% geometric coefficient of variation. |

### `"dirichlet"`

Samples compartment fractions that always sum to 1. Must be used for `fractions`.
The Dirichlet alpha vector is computed as ``base_fractions * kappa``, so the
mean fraction is always anchored to the base config values.

```toml
fractions = {dist = "dirichlet", kappa = 10.0}
```

| Parameter | Description |
|---|---|
| `kappa` | Scalar concentration parameter. Larger values tighten the distribution around the base config fractions. Must be > 0. |

**Note:** Previously this distribution accepted an `alpha` vector. The `kappa`-based API is cleaner: you only control the spread, while the mean is always determined by the base config fractions. Files written with the old API are unaffected.

---

### `[export]`

| Key | Type | Default | Description |
|---|---|---|---|
| `output_dir` | string | `"output"` | Directory for output files. Created automatically if absent. |
| `filename_template` | string | `"{case_id}"` | File stem template. `{case_id}` is replaced at runtime. |
| `formats` | list of strings | `["hdf5"]` | Output formats. Only `"hdf5"` is supported for Monte Carlo runs (signal is 5-D). |

### `[[cases]]`

| Key | Type | Required | Description |
|---|---|---|---|
| `config` | string | yes | Path to the phantom JSON file. |
| `bvals` | string | yes | Path to the b-values text file. |

All paths are resolved **relative to the TOML file's location**.

---

## Output

The Monte Carlo runner writes one HDF5 file per case. Signal and ground-truth arrays are 5-D `(X, Y, Z, n_b, N)` where `N` is `n_samples`. Each slice `[..., i]` corresponds to one parameter draw with independent Rician noise.

NIfTI is not supported as an output format for Monte Carlo runs because the 5-D arrays cannot be stored in a standard 4-D NIfTI file.

See the [Reference](reference.md#monte-carlo-hdf5-schema) for the full HDF5 layout.

---

## CLI

```bash
mrphantom montecarlo mc.toml
```

Equivalent programmatic usage:

```python
from mrphantom import load_montecarlo_config, run_montecarlo, save_mc_hdf5

mc = load_montecarlo_config("mc.toml")

for entry in mc.cases:
    from mrphantom import load_dwi_config
    cfg = load_dwi_config(entry.config, entry.bvals)

    result = run_montecarlo(
        base_config=cfg,
        mc_regions=mc.mc_regions,
        n_samples=mc.n_samples,
        snr=mc.snr,
        snr_region=mc.snr_region,
        seed=mc.seed,
    )

    out_paths = mc.export.resolve_paths(cfg.case_id)   # returns [path.h5]
    save_mc_hdf5(
        signal=result.signal,
        signal_gt=result.signal_gt,
        volumes=cfg.bvalues,
        segmentation=cfg.nifti_path,
        affine=result.affine,
        S0_samples=result.sampled_params.S0,
        D_samples=result.sampled_params.D,
        fractions_samples=result.sampled_params.fractions,
        region_names=result.sampled_params.region_names,
        out_path=out_paths[0],
        case_id=cfg.case_id,
        name=cfg.name,
        description=cfg.description,
        snr=mc.snr,
        snr_region=mc.snr_region,
        seed=mc.seed,
    )
```
