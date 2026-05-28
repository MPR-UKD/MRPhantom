# Usage Guide

> **TL;DR** — Practical walkthrough of the `mrphantom` API. Covers the step-by-step class workflow (**Step-by-step walkthrough**), result access and re-use across cases (**Accessing results**, **Re-using a phantom**), parameter adjustment (**Adjusting parameters**), batch synthesis (**Batch synthesis**), ROI helpers (**ROI evaluation helpers**), and reading saved files back (**Reading back saved files**).

---

## Step-by-step walkthrough

```python
from pathlib import Path
from mrphantom import DWIPhantom, load_phantom_config

# 1. Load case data — parses JSON, loads b-values, resolves null regions
config = load_phantom_config(
    Path("assets/kidney/case_000/kid_000_synthetic.json"),
    Path("assets/kidney/case_000/kidney.bval"),
)

# 2. Create the phantom — only acquisition settings go here, no data yet
phantom = DWIPhantom(snr=30, seed=42)

# 3. Attach the case — clears any previously cached signal, returns self
phantom.load_case(config)

# 4. Synthesize — evaluates the signal model across all regions, returns self
phantom.synthesize()

# 5. Save — format inferred from extension, returns list of written paths
phantom.save("output/kidney_001.nii.gz", "output/kidney_001.h5")
```

Steps 3–5 can be chained:

```python
phantom.load_case(config).synthesize().save("output/kidney_001.nii.gz")
```

Or use the convenience factory to load and configure in one line:

```python
phantom = DWIPhantom.from_json(
    "assets/kidney/case_000/kid_000_synthetic.json",
    "assets/kidney/case_000/kidney.bval",
    snr=30,
    seed=42,
)
phantom.synthesize().save("output/kidney_001.nii.gz")
```

---

## Accessing results after synthesis

After `synthesize()` is called, results are available as read-only properties:

```python
phantom.signal        # np.ndarray, shape (X, Y, Z, n_bvalues), dtype float64
phantom.signal_img    # nibabel.Nifti1Image (same data, with affine from segmentation)
phantom.config        # PhantomConfig (available after load_case(), before synthesize())
```

Accessing any property before its corresponding call raises a `RuntimeError`:

```python
phantom = DWIPhantom(snr=30)
phantom.signal     # RuntimeError: call synthesize() first
```

---

## Re-using a phantom across cases

`load_case()` resets the cached signal, so the same `DWIPhantom` instance can be used for multiple cases in a loop without creating new objects each time:

```python
phantom = DWIPhantom(snr=30, seed=42)

for json_path, bval_path in case_paths:
    config = load_phantom_config(json_path, bval_path)
    phantom.load_case(config).synthesize().save(f"output/{config.case_id}.h5")

    # phantom.signal is available here for the current case
    print(f"{config.case_id}: {phantom.signal.shape}")
```

The acquisition parameters (`snr`, `seed`) remain fixed across all cases; only the loaded case data changes.

---

## Adjusting parameters

`get_params()` returns all constructor parameters as a dict. `set_params()` updates them in-place and returns `self`:

```python
phantom = DWIPhantom(snr=30, seed=42)

phantom.get_params()
# {'snr': 30, 'seed': 42}

phantom.set_params(snr=50)
phantom.get_params()
# {'snr': 50, 'seed': 42}

# set_params returns self, so it can be chained
phantom.set_params(snr=100).load_case(config).synthesize()
```

`__repr__` shows only non-`None` parameters:

```python
str(DWIPhantom(snr=30, seed=42))   # "DWIPhantom(snr=30, seed=42)"
str(DWIPhantom())                  # "DWIPhantom()"
```

---

## Batch synthesis from a TOML run-config

```python
from mrphantom import load_run_config, load_phantom_config, DWIPhantom

run = load_run_config("experiment.toml")
phantom = DWIPhantom(snr=run.snr, seed=run.seed)

for entry in run.cases:
    # entry.config and entry.bvals are absolute Path objects
    cfg = load_phantom_config(entry.config, entry.bvals)

    # resolve_paths interpolates {case_id} and appends the right extensions
    out_paths = run.export.resolve_paths(cfg.case_id)

    phantom.load_case(cfg).synthesize().save(*out_paths)
```

Or run the same thing from the CLI:

```bash
mrphantom run experiment.toml
```

See the [TOML run-config guide](toml_run_config.md) for the full TOML schema.

---

## ROI evaluation helpers

`get_evaluation_segments()` returns the NIfTI label indices for the kidney evaluation regions (cortex, medulla, hilum). `get_fitting_mask()` builds a boolean 3-D mask from those indices.

```python
import nibabel as nib
from mrphantom import (
    DWIPhantom, load_phantom_config,
    get_evaluation_segments, get_fitting_mask,
)

config  = load_phantom_config("phantom.json", "bvals.txt")
phantom = DWIPhantom().load_case(config).synthesize()

# Integer label indices for cortex, medulla, hilum
eval_indices = get_evaluation_segments(config)   # e.g. [1, 10, 11]

# Load the segmentation array
seg_img = nib.load(config.nifti_path)
seg     = seg_img.get_fdata().astype(int)

# Boolean mask over the evaluation regions
mask = get_fitting_mask(seg, eval_indices)        # shape (X, Y, Z)

# Apply to the signal: select only evaluation-region voxels
roi_signal = phantom.signal[mask]                 # shape (N_voxels, n_bvalues)
```

The region matching is name-driven, not index-hardcoded, so it remains correct even if label numbers differ between cases.

---

## Monte Carlo simulation (`MCDWIPhantom`)

`MCDWIPhantom` extends `DWIPhantom` with a Monte Carlo outer loop. Tissue parameters are sampled from per-region distributions and every draw gets an independent noise seed.  The lifecycle has one extra step — `setup()` — which must be called between `load_case()` and any synthesis call.

### Batch mode

Synthesize all *N* samples in memory, then save as a single HDF5 file:

```python
from mrphantom import MCDWIPhantom, load_dwi_config
from mrphantom.montecarlo.config import ParamDist, RegionMCConfig

mc_regions = {
    "cortex": RegionMCConfig(
        name="cortex",
        S0=ParamDist(dist="normal", mean=180.0, std=10.0),
        D=[
            ParamDist(dist="lognormal", mean=0.003, sigma=0.10),
            ParamDist(dist="lognormal", mean=0.001, sigma=0.05),
        ],
        fractions=ParamDist(dist="dirichlet", kappa=10.0),
    )
}

config = load_dwi_config("phantom.json", "bvals.txt")
phantom = MCDWIPhantom(n_samples=50, mc_regions=mc_regions, snr=30, seed=42)
phantom.load_case(config).setup().synthesize().save("mc_output.h5")

phantom.signal     # np.ndarray, shape (X, Y, Z, n_b, 50)
phantom.signal_gt  # noise-free GT, same shape
```

### Lazy mode

Write one sample at a time into a pre-allocated HDF5 file — useful when *N* is large and all samples don't fit in memory at once:

```python
phantom = MCDWIPhantom(n_samples=200, mc_regions=mc_regions, snr=30, seed=42)
phantom.load_case(config).setup()

with phantom.open_h5("mc_output.h5") as h5:
    for i in range(phantom.n_samples):
        phantom.step(i)      # synthesize sample i → 4-D array in phantom.signal
        h5.flush(i)          # write sample i to the pre-allocated HDF5 slice
```

Both modes produce identical HDF5 files for the same seed.

### Noise-ensemble mode

Pass `mc_regions={}` (the default) to keep tissue parameters fixed across all draws — each sample differs only in its noise realisation:

```python
phantom = MCDWIPhantom(n_samples=100, snr=20, seed=0)
phantom.load_case(config).setup().synthesize().save("noise_ensemble.h5")
```

### Loading from a TOML config

```python
phantom = MCDWIPhantom.from_toml("assets/kidney/mc.toml")
# phantom._mc_config holds the full MCRunConfig (cases, export, ...)
```

Or run from the CLI:

```bash
mrphantom montecarlo assets/kidney/mc.toml
```

See the [TOML Monte Carlo config guide](toml_mc_config.md) for the full TOML schema.

### Reading MC results back

```python
from mrphantom import load_mc_hdf5

data = load_mc_hdf5("mc_output.h5")

data.signal             # np.ndarray (X, Y, Z, n_b, N)
data.signal_gt          # np.ndarray (X, Y, Z, n_b, N) — noise-free GT
data.S0_samples         # np.ndarray (n_regions, N)
data.D_samples          # np.ndarray (n_regions, max_comp, N)
data.fractions_samples  # np.ndarray (n_regions, max_comp, N)
data.region_names       # list[str]
data.n_samples          # int
data.snr                # float | None
data.seed               # int | None
```

---

## Reading back saved files

**NIfTI:**

```python
import nibabel as nib

img    = nib.load("kidney_001.nii.gz")
signal = img.get_fdata()          # shape (X, Y, Z, n_bvalues)
affine = img.affine               # 4×4 transformation matrix
```

**HDF5:**

```python
from mrphantom import load_hdf5

data = load_hdf5("kidney_001.h5")

data.signal        # np.ndarray  (X, Y, Z, n_bvalues)
data.bvalues       # np.ndarray  (n_bvalues,)
data.segmentation  # np.ndarray  (X, Y, Z) — integer label map
data.affine        # np.ndarray  (4, 4)
data.case_id       # str
data.snr           # float | None

# Reconstruct a NIfTI image from the stored arrays
img = data.as_nifti()
```

The HDF5 format preserves everything needed to reconstruct the full synthesis context — signal, b-values, segmentation, affine, and metadata — in a single portable file.
