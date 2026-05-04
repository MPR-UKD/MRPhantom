# MRPhantom

A Python library and CLI tool for synthesizing synthetic MRI diffusion-weighted (DWI) phantoms from segmented anatomical images.

MRPhantom generates controlled, ground-truth DWI data from NIfTI segmentation maps using a multi-exponential signal model. It is designed for benchmarking diffusion model fitting algorithms, evaluating post-processing pipelines, and producing reproducible synthetic datasets in medical imaging research.

---

## Features

- Multi-exponential diffusion signal model: `S(b) = S0 · Σᵢ fᵢ · exp(-b · Dᵢ)`
- Rician noise simulation at a specified SNR
- Class-based API (`DWIPhantom`) with method chaining
- TOML-driven batch synthesis across multiple cases
- Export to NIfTI (`.nii.gz`) and HDF5 (`.h5`) in a single pass
- CLI with `synth`, `run`, `info`, and `inspect` commands
- Fully functional API preserved alongside the class API

---

## Installation

```bash
pip install mrphantom
```

Python 3.9 or later is required. On Python < 3.11 the `tomli` package is also needed for TOML support:

```bash
pip install mrphantom tomli   # Python 3.9 / 3.10
```

### Development install

```bash
git clone https://github.com/darksim33/mrphantom
cd mrphantom
pip install -e .
```

---

## Case structure

A **case** is the complete set of inputs needed to synthesize one phantom. It consists of three files that live together in a directory:

```
case_001/
├── kid_001_cor_ang.json      ← phantom configuration (regions, diffusion parameters)
├── kid_001_cor_ang.nii.gz    ← NIfTI integer segmentation label map
└── bvals.txt                 ← b-values used during synthesis
```

The JSON file references the NIfTI by a relative path (`"file"` field), so the three files must stay together. The b-values file is supplied separately and does not need to share the same name or directory, though co-locating them is conventional.

---

### Phantom JSON

The JSON file describes the phantom case metadata and the diffusion parameters for every segmented tissue region.

**Full annotated schema:**

```json
{
  "case_id":     "kidney_001",
  "name":        "kid_cor_ang_001",
  "description": "Synthetic abdomen segmentation with multiple regions.",
  "source":      "<Author_Source>",
  "date":        "<YYYY-MM-DD>",

  "file": "kid_001_cor_ang.nii.gz",

  "image": {
    "plane":      "coronal",
    "note":       "coronal_ang",
    "dimensions": [176, 3, 176]
  },

  "regions": {
    "001": { ... },
    "002": { ... }
  }
}
```

**Top-level fields:**

| Field | Required | Description |
|---|---|---|
| `case_id` | yes | Unique identifier. Used as `{case_id}` in TOML filename templates and as HDF5 metadata. |
| `name` | yes | Short display name. |
| `description` | yes | Human-readable description of the case. |
| `source` | no | Data provenance note. |
| `date` | no | Date string (informational only). |
| `file` | yes | Path to the NIfTI segmentation file, **relative to this JSON**. |
| `image.dimensions` | yes | Spatial dimensions `[X, Y, Z]` of the segmentation volume. |
| `image.plane` | no | Imaging plane label (informational only). |
| `image.note` | no | Free-text note (informational only). |

---

### Regions

Each entry under `"regions"` maps one integer NIfTI label to a set of diffusion parameters. The string keys (`"001"`, `"002"`, …) are arbitrary — they control display order but are not used during loading.

```json
"010": {
  "name":  "cortex",
  "index": 10,
  "diffusivity": {
    "n_components": 3,
    "S0":           180,
    "fractions":    [0.6, 0.3, 0.1],
    "D":            [0.0015, 0.005, 0.09]
  }
}
```

**Region fields:**

| Field | Required | Description |
|---|---|---|
| `name` | yes | Human-readable region name. Must be unique within the JSON. The name `"remaining_tissue"` has special fallback meaning (see below). |
| `index` | yes | Integer voxel value in the NIfTI segmentation. Every voxel with this value is assigned the region's signal. Must be unique. |
| `diffusivity.n_components` | yes* | Number of diffusion compartments. |
| `diffusivity.S0` | yes* | Baseline signal intensity at b=0. |
| `diffusivity.fractions` | yes* | Per-compartment signal fractions. Must sum to 1. Length must equal `n_components`. |
| `diffusivity.D` | yes* | Per-compartment diffusivities in mm²/s. Length must equal `n_components`. |

`*` Can be `null` / empty — see **Null regions** below.

---

### Single-compartment vs multi-compartment regions

**Single-compartment** (mono-exponential, e.g. spleen):

```json
"003": {
  "name":  "spleen",
  "index": 3,
  "diffusivity": {
    "n_components": 1,
    "S0":           74,
    "fractions":    [1],
    "D":            [0.0012]
  }
}
```

Signal: `S(b) = 74 · exp(-b · 0.0012)`

**Multi-compartment** (tri-exponential, e.g. kidney cortex):

```json
"010": {
  "name":  "cortex",
  "index": 10,
  "diffusivity": {
    "n_components": 3,
    "S0":           180,
    "fractions":    [0.6, 0.3, 0.1],
    "D":            [0.0015, 0.005, 0.09]
  }
}
```

Signal: `S(b) = 180 · (0.6·exp(-b·0.0015) + 0.3·exp(-b·0.005) + 0.1·exp(-b·0.09))`

---

### Null regions

Regions where diffusion parameters are unknown or irrelevant can be left as null. They automatically **inherit all parameters** from the region named `"remaining_tissue"`:

```json
"004": {
  "name":  "bones",
  "index": 4,
  "diffusivity": {
    "n_components": 1,
    "S0":           null,
    "fractions":    [],
    "D":            []
  }
}
```

A region is treated as null when `S0` is `null` **or** `D` is an empty list. If any null region is present, a `"remaining_tissue"` region must exist in the same JSON — otherwise loading will raise a `ValueError`.

---

### NIfTI segmentation

The segmentation is a **3-D integer-valued label map** `(X, Y, Z)`:

- Each voxel holds the integer label of the tissue it belongs to.
- Label `0` is background — those voxels produce zero signal.
- Every non-zero label should have a matching `index` in the JSON regions. Labels present in the NIfTI but absent from the JSON produce zero signal without raising an error.
- If the file is stored as 4-D `(X, Y, Z, volumes)` only the first volume is used.
- The affine matrix from the segmentation NIfTI is preserved in all output files.

---

### B-values file

A plain-text file of b-values in **s/mm²**, separated by whitespace or newlines:

```
0 50 100 200 400 800
```

or equivalently:

```
0
50
100
200
400
800
```

The length of this file determines the fourth dimension of the synthesized output: `(X, Y, Z, n_bvalues)`.

---

## Quick start

### Single case — programmatic

```python
from pathlib import Path
from mrphantom import DWIPhantom, load_phantom_config

config = load_phantom_config(
    Path("assets/kidney/case_001/kid_001_cor_ang.json"),
    Path("assets/kidney/case_001/bvals.txt"),
)

phantom = DWIPhantom(snr=30, seed=42)
phantom.load_case(config).synthesize().save("kidney_001.nii.gz", "kidney_001.h5")
```

### Convenience factory

```python
from mrphantom import DWIPhantom

phantom = DWIPhantom.from_json(
    "assets/kidney/case_001/kid_001_cor_ang.json",
    "assets/kidney/case_001/bvals.txt",
    snr=30,
    seed=42,
)
phantom.synthesize().save("kidney_001.nii.gz")
```

### Batch run — CLI

```bash
mrphantom run experiment.toml
```

---

## Usage

### Step-by-step walkthrough

```python
from pathlib import Path
from mrphantom import DWIPhantom, load_phantom_config

# 1. Load case data — parses JSON, loads b-values, resolves null regions
config = load_phantom_config(
    Path("assets/kidney/case_001/kid_001_cor_ang.json"),
    Path("assets/kidney/case_001/bvals.txt"),
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

---

### Accessing results after synthesis

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

### Re-using a phantom across cases

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

### Adjusting parameters

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

### Batch synthesis from a TOML run-config

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

---

### ROI evaluation helpers

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

The region matching is name-driven (not index-hardcoded), so it remains correct even if label numbers differ between cases.

---

### Reading back saved files

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

---

## TOML run-config format

The TOML file is the top-level orchestration layer. Case JSON files and their segmentation NIfTIs are referenced from here but remain unchanged.

```toml
[run]
snr  = 30      # omit or set to null for noise-free output
seed = 42      # omit for non-reproducible noise

[export]
output_dir        = "output/"
filename_template = "{case_id}"          # {case_id} is interpolated at runtime
formats           = ["nifti", "hdf5"]    # one or both

[[cases]]
config = "assets/kidney/case_001/kid_001_cor_ang.json"
bvals  = "assets/kidney/case_001/bvals.txt"

[[cases]]
config = "assets/kidney/case_002/kid_002.json"
bvals  = "assets/kidney/case_002/bvals.txt"
```

All paths under `[[cases]]` are resolved relative to the TOML file's location.

**Supported export formats:**

| Value | Extension | Description |
|---|---|---|
| `"nifti"` | `.nii.gz` | NIfTI image (signal only) |
| `"hdf5"` | `.h5` | HDF5 file (signal, b-values, segmentation, affine, metadata) |

---

## CLI reference

### `mrphantom run`

Batch synthesis from a TOML run-config file.

```
mrphantom run TOML
```

```
Arguments:
  TOML  Path to a run-config .toml file.
```

### `mrphantom synth`

Synthesize a single case directly from a JSON config and b-values file.

```
mrphantom synth CONFIG BVAL [--snr FLOAT] [--seed INT] --out PATH [--out PATH ...]
```

```
Arguments:
  CONFIG  Phantom JSON file (regions, diffusion parameters).
  BVAL    B-values file (whitespace- or newline-separated, s/mm²).

Options:
  --snr FLOAT   SNR for Rician noise. Omit for noise-free.
  --seed INT    Random seed for reproducible noise.
  --out PATH    Output file (.nii/.nii.gz or .h5/.hdf5). Repeatable.
```

Examples:

```bash
# Noise-free NIfTI
mrphantom synth phantom.json bvals.txt --out phantom.nii.gz

# SNR=30, both formats
mrphantom synth phantom.json bvals.txt --snr 30 --seed 42 \
    --out phantom.nii.gz --out phantom.h5
```

### `mrphantom info`

Print a summary of a phantom JSON config.

```
mrphantom info CONFIG BVAL
```

### `mrphantom inspect`

Print a summary of a saved HDF5 phantom file.

```
mrphantom inspect FILE
```

---

## API reference

### `DWIPhantom`

```python
class DWIPhantom(snr=None, seed=None)
```

Diffusion-weighted phantom synthesizer.

| Parameter | Type | Description |
|---|---|---|
| `snr` | `float \| None` | Signal-to-noise ratio for Rician noise. `None` = noise-free. |
| `seed` | `int \| None` | Random seed for reproducible noise. Ignored when `snr` is `None`. |

**Methods:**

| Method | Returns | Description |
|---|---|---|
| `load_case(config)` | `self` | Attach a `PhantomConfig`. Clears any cached signal. |
| `synthesize()` | `self` | Run the signal model. Results cached on instance. |
| `save(*paths)` | `list[Path]` | Write to one or more files. Format inferred from extension. |
| `from_json(json_path, bval_path, *, snr, seed)` | `DWIPhantom` | Classmethod — load config and return fitted instance. |
| `get_params()` | `dict` | Return `{"snr": ..., "seed": ...}`. |
| `set_params(**params)` | `self` | Set one or more parameters. |

**Properties** (available after the corresponding call):

| Property | Available after | Type |
|---|---|---|
| `config` | `load_case()` | `PhantomConfig` |
| `signal` | `synthesize()` | `np.ndarray` shape `(X, Y, Z, n_bvalues)` |
| `signal_img` | `synthesize()` | `nibabel.Nifti1Image` |

### `MRPhantom`

Abstract base class for all phantom types. Subclass this to implement other MR contrast mechanisms (T1, T2, etc.).

```python
class MRPhantom(snr=None, seed=None)
```

Abstract methods to implement: `load_case(config)`, `synthesize()`.

Concrete methods provided: `get_params()`, `set_params(**params)`, `__repr__()`.

### `load_run_config`

```python
load_run_config(toml_path) -> RunConfig
```

Parse a TOML run-config file. All case paths are resolved relative to the TOML file's parent directory.

### `RunConfig`

```python
@dataclass
class RunConfig:
    snr: float | None
    seed: int | None
    export: ExportConfig
    cases: list[CaseEntry]
```

### `ExportConfig`

```python
@dataclass
class ExportConfig:
    output_dir: Path
    filename_template: str          # default: "{case_id}"
    formats: list[str]              # "nifti" and/or "hdf5"

    def resolve_paths(self, case_id: str) -> list[Path]: ...
```

### `CaseEntry`

```python
@dataclass
class CaseEntry:
    config: Path    # absolute path to phantom JSON
    bvals: Path     # absolute path to b-values file
```

### `load_phantom_config`

```python
load_phantom_config(json_path, bval_path) -> PhantomConfig
```

Parse a phantom JSON file and load b-values. Null regions (those with `S0: null`) inherit parameters from the `remaining_tissue` region.

---

## Signal model

The DWI signal for each voxel is computed as a sum of exponential decays across diffusion compartments:

```
S(b) = S0 · Σᵢ fᵢ · exp(-b · Dᵢ)
```

Where:
- `S0` — baseline signal intensity
- `fᵢ` — compartment signal fraction (must sum to 1)
- `Dᵢ` — compartment diffusivity in mm²/s
- `b` — b-value in s/mm²

Rician noise is added (when `snr` is set) as:

```
S_noisy = sqrt( (S + N_re)² + N_im² )
```

Where `N_re, N_im ~ Normal(0, σ)` and `σ = max(S0) / snr`.

---

## HDF5 file schema

Files written by `save()` or `mrphantom synth --out phantom.h5` follow this layout:

```
/signal          float64  (X, Y, Z, n_bvalues)   gzip-compressed
/bvalues         float64  (n_bvalues,)
/segmentation    int32    (X, Y, Z)               gzip-compressed
/affine          float64  (4, 4)

Root attributes:
    case_id      str
    name         str
    description  str
    snr          float   (absent for noise-free phantoms)
```

Read back with:

```python
from mrphantom import load_hdf5

data = load_hdf5("phantom.h5")
print(data.signal.shape)    # (X, Y, Z, n_bvalues)
print(data.snr)
img = data.as_nifti()       # nibabel.Nifti1Image
```

---

## License

See [LICENSE](LICENSE).
