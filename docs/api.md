# API Reference

> **TL;DR** — All public symbols exported from `mrphantom`. Covers the abstract base class (**Base class**), the concrete DWI synthesizer (**Phantom classes**), DWI config and data model types (**DWI config / data model**), TOML batch config types (**Run config**), the low-level functional API (**Synthesis**), ROI helpers (**ROI helpers**), I/O functions (**I/O**), and the full Monte Carlo API (**Monte Carlo**).

---

## Base class

```python
from mrphantom import MRPhantom
```

Abstract base class for all MR phantom types. Provides `load_case()`, `synthesize()`, `get_params()`, `set_params()`, and `__repr__()`. Subclass this to implement other MR contrast mechanisms (T1, T2, etc.).

See [`base.py`](../mrphantom/base.py).

---

## Phantom classes

```python
from mrphantom import DWIPhantom
from mrphantom import MCDWIPhantom
```

| Symbol | Description |
|---|---|
| `DWIPhantom` | Concrete synthesizer for diffusion-weighted phantoms. Implements the multi-exponential signal model with optional Rician noise. |
| `MCDWIPhantom` | Monte Carlo extension of `DWIPhantom`. Adds a `setup()` step, batch `synthesize()` (5-D output), lazy per-sample `step(i)`, and incremental `open_h5()` writing. |

See [`dwi/phantom.py`](../mrphantom/dwi/phantom.py), [`montecarlo/phantom.py`](../mrphantom/montecarlo/phantom.py), and the [Usage guide](usage.md).

---

## DWI config / data model

```python
from mrphantom import RegionConfig, DWIConfig, load_dwi_config
```

| Symbol | Description |
|---|---|
| `RegionConfig` | Dataclass holding diffusion parameters for one segmented region. |
| `DWIConfig` | Dataclass holding the complete parsed DWI case (all regions, b-values, NIfTI path). |
| `load_dwi_config(json_path, bval_path)` | Parse a phantom JSON + b-values file into a `DWIConfig`. |

See [`dwi/config.py`](../mrphantom/dwi/config.py) and the [Case structure guide](case_structure.md).

---

## Run config (TOML layer)

```python
from mrphantom import RunConfig, CaseEntry, ExportConfig, load_run_config
```

| Symbol | Description |
|---|---|
| `RunConfig` | Top-level dataclass for a batch synthesis run (`snr`, `seed`, `export`, `cases`). |
| `CaseEntry` | Dataclass holding resolved `config` and `bvals` paths for one case. |
| `ExportConfig` | Dataclass for export settings (`output_dir`, `filename_template`, `formats`). |
| `load_run_config(toml_path)` | Parse a TOML run-config file into a `RunConfig`. |

See [`run_config.py`](../mrphantom/run_config.py) and the [TOML run-config guide](toml_run_config.md).

---

## Synthesis (functional API)

```python
from mrphantom import synthesize_phantom
```

Low-level functional entry point. Evaluates the multi-exponential signal model for all regions in a `DWIConfig` and returns the 4-D signal array and a `nibabel.Nifti1Image`. Used internally by `DWIPhantom.synthesize()`.

See [`dwi/synthesis.py`](../mrphantom/dwi/synthesis.py).

---

## ROI helpers

```python
# General (modality-agnostic)
from mrphantom import get_fitting_mask

# DWI / kidney evaluation regions
from mrphantom import get_evaluation_segments, EVALUATION_REGION_NAMES
```

| Symbol | Where | Description |
|---|---|---|
| `get_fitting_mask(segmentation, indices)` | `mrphantom.roi` | Build a boolean 3-D mask selecting the given label indices. General utility, usable with any segmented phantom. |
| `EVALUATION_REGION_NAMES` | `mrphantom.dwi.roi` | `frozenset` of kidney evaluation region names: `{"cortex", "medulla", "hilum"}`. |
| `get_evaluation_segments(config)` | `mrphantom.dwi.roi` | Return sorted NIfTI label indices for the kidney evaluation regions in a `DWIConfig`. |

See [`roi.py`](../mrphantom/roi.py), [`dwi/roi.py`](../mrphantom/dwi/roi.py), and the [Usage guide — ROI helpers](usage.md#roi-evaluation-helpers).

---

## I/O

```python
from mrphantom import save_nifti, save_hdf5, load_hdf5, PhantomData
```

| Symbol | Description |
|---|---|
| `save_nifti(signal_img, out_path)` | Save a `nibabel.Nifti1Image` to `.nii` or `.nii.gz`. |
| `save_hdf5(signal, signal_img, volumes, segmentation, out_path, ...)` | Write signal, volumes axis, segmentation, affine, and metadata to HDF5. |
| `load_hdf5(path)` | Read a saved HDF5 phantom file into a `PhantomData` dataclass. |
| `PhantomData` | Dataclass bundling all arrays and metadata loaded from HDF5, with `as_nifti()`. The `volumes` field stores the acquisition-parameter axis (b-values for DWI, TIs for T1, etc.). |

See [`io/`](../mrphantom/io/) and the [Usage guide — Reading back saved files](usage.md#reading-back-saved-files).

---

## Monte Carlo

### Phantom class

```python
from mrphantom import MCDWIPhantom
```

| Symbol | Description |
|---|---|
| `MCDWIPhantom(n_samples, mc_regions, snr, seed, snr_region)` | Class-centred Monte Carlo phantom. Lifecycle: `load_case()` → `setup()` → `synthesize()` + `save()` (batch) or `open_h5()` + `step(i)` + `flush(i)` (lazy). |
| `MCDWIPhantom.from_toml(path)` | Classmethod factory. Reads an MC TOML config; returns a configured instance with `_mc_config` attached. |

See [`montecarlo/phantom.py`](../mrphantom/montecarlo/phantom.py) and the [Usage guide — Monte Carlo](usage.md#monte-carlo-simulation-mcdwiphantom).

---

### Config

```python
from mrphantom import ParamDist, RegionMCConfig, MCRunConfig, load_montecarlo_config
```

| Symbol | Description |
|---|---|
| `ParamDist` | Dataclass holding a distribution specification (`dist`, `mean`, `std`, `sigma`, `alpha`, `kappa`). For `"normal"` and `"lognormal"` distributions, the mean is auto-anchored to the base config value — only spread parameters (`std` / `sigma`) are used at sampling time. For `"dirichlet"` on fractions, use `kappa` (scalar concentration) — the alpha vector is computed as `base_fractions * kappa`. |
| `RegionMCConfig` | Dataclass holding per-region MC distributions (`name`, `S0`, `D`, `fractions`). |
| `MCRunConfig` | Top-level dataclass for a Monte Carlo run (`snr`, `snr_region`, `seed`, `export`, `cases`, `n_samples`, `mc_regions`). |
| `load_montecarlo_config(toml_path)` | Parse a Monte Carlo TOML config file into an `MCRunConfig`. |

See [`montecarlo/config.py`](../mrphantom/montecarlo/config.py) and the [TOML Monte Carlo config guide](toml_mc_config.md).

---

### Runner

```python
from mrphantom import run_montecarlo, MCResult
```

| Symbol | Description |
|---|---|
| `run_montecarlo(base_config, mc_regions, n_samples, snr, snr_region, seed)` | Run *n_samples* Monte Carlo DWI simulations. Returns an `MCResult`. |
| `MCResult` | Dataclass with `signal` and `signal_gt` arrays of shape `(X, Y, Z, n_b, N)`, `sampled_params`, and `affine`. |

See [`montecarlo/runner.py`](../mrphantom/montecarlo/runner.py).

---

### Monte Carlo I/O

```python
from mrphantom import save_mc_hdf5, load_mc_hdf5, MCPhantomData
```

| Symbol | Description |
|---|---|
| `save_mc_hdf5(signal, signal_gt, volumes, segmentation, affine, S0_samples, D_samples, fractions_samples, region_names, out_path, ...)` | Write a 5-D Monte Carlo result to HDF5. |
| `load_mc_hdf5(path)` | Read a Monte Carlo HDF5 file into an `MCPhantomData` dataclass. |
| `MCPhantomData` | Dataclass bundling all arrays and metadata from a Monte Carlo HDF5 file (`signal`, `signal_gt`, `volumes`, `segmentation`, `affine`, `S0_samples`, `D_samples`, `fractions_samples`, `region_names`, and metadata). |

See [`io/mc_hdf5.py`](../mrphantom/io/mc_hdf5.py) and the [Reference — Monte Carlo HDF5 schema](reference.md#monte-carlo-hdf5-schema).
