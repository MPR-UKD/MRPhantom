# Case Structure

A **case** is the complete set of inputs needed to synthesize one phantom. It consists of three files that live together in a directory.

---

## Directory layout

```
case_001/
├── kid_001_cor_ang.json      ← phantom configuration (regions, diffusion parameters)
├── kid_001_cor_ang.nii.gz    ← NIfTI integer segmentation label map
└── bvals.txt                 ← b-values used during synthesis
```

The JSON file references the NIfTI by a relative path (`"file"` field), so the two files must stay together. The b-values file is supplied separately and does not need to share the same name or directory, though co-locating them is conventional.

---

## Phantom JSON

The JSON file describes the case metadata and the diffusion parameters for every segmented tissue region.

### Top-level fields

| Field | Required | Description |
|---|---|---|
| `case_id` | yes | Unique identifier. Used as `{case_id}` in TOML filename templates and stored as HDF5 metadata. |
| `name` | yes | Short display name. |
| `description` | yes | Human-readable description of the case. |
| `source` | no | Data provenance note. |
| `date` | no | Date string (informational only). |
| `file` | yes | Path to the NIfTI segmentation, **relative to this JSON file**. |
| `image.dimensions` | yes | Spatial dimensions `[X, Y, Z]` of the segmentation volume. |
| `image.plane` | no | Imaging plane label (informational only). |
| `image.note` | no | Free-text note (informational only). |

### Region fields

Each entry under `"regions"` maps one integer NIfTI label to a set of diffusion parameters. The string keys (`"001"`, `"002"`, …) are arbitrary — they control display order but are not used during loading.

| Field | Required | Description |
|---|---|---|
| `name` | yes | Human-readable region name. Must be unique. `"remaining_tissue"` has special fallback meaning (see below). |
| `index` | yes | Integer voxel value in the NIfTI. Every voxel with this value receives the region's signal. Must be unique. |
| `diffusivity.n_components` | yes* | Number of diffusion compartments. |
| `diffusivity.S0` | yes* | Baseline signal intensity at b=0. |
| `diffusivity.fractions` | yes* | Per-compartment signal fractions. Must sum to 1. Length must equal `n_components`. |
| `diffusivity.D` | yes* | Per-compartment diffusivities in mm²/s. Length must equal `n_components`. |

`*` Can be `null` / empty — see **Null regions** below.

---

## Single-compartment vs multi-compartment regions

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

## Null regions

Regions where parameters are unknown or irrelevant can be left as null. They automatically **inherit all parameters** from the region named `"remaining_tissue"`:

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

A region is treated as null when `S0` is `null` **or** `D` is an empty list. If any null region is present, a `"remaining_tissue"` region must exist in the same JSON — otherwise `load_phantom_config()` raises a `ValueError`.

---

## NIfTI segmentation

The segmentation is a **3-D integer-valued label map** `(X, Y, Z)`:

- Each voxel holds the integer label of the tissue it belongs to.
- Label `0` is background — those voxels produce zero signal.
- Every non-zero label should have a matching `index` in the JSON regions. Labels present in the NIfTI but absent from the JSON produce zero signal without raising an error.
- If the file is stored as 4-D `(X, Y, Z, volumes)`, only the first volume is used.
- The affine matrix from the segmentation NIfTI is preserved in all output files.

---

## B-values file

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

The number of values determines the fourth dimension of the synthesized output: `(X, Y, Z, n_bvalues)`.

---

## Complete example

The bundled example case is at `assets/kidney/case_001/`. Load it with:

```python
from pathlib import Path
from mrphantom import load_phantom_config

config = load_phantom_config(
    Path("assets/kidney/case_001/kid_001_cor_ang.json"),
    Path("assets/kidney/case_001/bvals.txt"),
)

print(config.case_id)          # "kidney_001"
print(config.image_dims)       # [176, 3, 176]
print(len(config.regions))     # 11
print(config.bvalues)          # [  0.  50. 100. 200. 400. 800.]
```
