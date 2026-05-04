# API Reference

> **TL;DR** — All public symbols exported from `mrphantom`. Covers the abstract base class (**Base class**), the concrete DWI synthesizer (**Phantom classes**), config and data model types (**Config / data model**), TOML batch config types (**Run config**), the low-level functional API (**Synthesis**), ROI helpers (**ROI helpers**), and I/O functions (**I/O**).

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
```

Concrete synthesizer for diffusion-weighted phantoms. Implements the multi-exponential signal model with optional Rician noise.

See [`dwi.py`](../mrphantom/dwi.py) and the [Usage guide](usage.md).

---

## Config / data model

```python
from mrphantom import RegionConfig, PhantomConfig, load_phantom_config
```

| Symbol | Description |
|---|---|
| `RegionConfig` | Dataclass holding diffusion parameters for one segmented region. |
| `PhantomConfig` | Dataclass holding the complete parsed case (all regions, b-values, NIfTI path). |
| `load_phantom_config(json_path, bval_path)` | Parse a phantom JSON + b-values file into a `PhantomConfig`. |

See [`config.py`](../mrphantom/config.py) and the [Case structure guide](case_structure.md).

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

Low-level functional entry point. Evaluates the multi-exponential signal model for all regions in a `PhantomConfig` and returns the 4-D signal array and a `nibabel.Nifti1Image`. Used internally by `DWIPhantom.synthesize()`.

See [`synthesis.py`](../mrphantom/synthesis.py).

---

## ROI helpers

```python
from mrphantom import get_evaluation_segments, get_fitting_mask, EVALUATION_REGION_NAMES
```

| Symbol | Description |
|---|---|
| `EVALUATION_REGION_NAMES` | `frozenset` of kidney evaluation region names: `{"cortex", "medulla", "hilum"}`. |
| `get_evaluation_segments(config)` | Return sorted NIfTI label indices for the evaluation regions in a `PhantomConfig`. |
| `get_fitting_mask(segmentation, indices)` | Build a boolean 3-D mask selecting the given label indices. |

See [`roi.py`](../mrphantom/roi.py) and the [Usage guide — ROI helpers](usage.md#roi-evaluation-helpers).

---

## I/O

```python
from mrphantom import save_nifti, save_hdf5, load_hdf5, PhantomData
```

| Symbol | Description |
|---|---|
| `save_nifti(signal_img, out_path)` | Save a `nibabel.Nifti1Image` to `.nii` or `.nii.gz`. |
| `save_hdf5(signal, signal_img, config, out_path, snr)` | Write signal, b-values, segmentation, affine, and metadata to HDF5. |
| `load_hdf5(path)` | Read a saved HDF5 phantom file into a `PhantomData` dataclass. |
| `PhantomData` | Dataclass bundling all arrays and metadata loaded from HDF5, with `as_nifti()`. |

See [`io/`](../mrphantom/io/) and the [Usage guide — Reading back saved files](usage.md#reading-back-saved-files).
