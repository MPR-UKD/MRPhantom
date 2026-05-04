# TOML Run-Config

> **TL;DR** — Top-level orchestration file for batch synthesis. Covers the complete TOML schema with a ready-to-copy example (**Schema**), field-by-field reference for `[run]`, `[export]`, and `[[cases]]` (**Fields**), output path construction rules (**Output paths**), and the equivalent CLI and programmatic usage (**CLI**).

---

## Schema

```toml
[run]
snr  = 30      # omit or set to null for noise-free output
seed = 42      # omit for non-reproducible noise

[export]
output_dir        = "output/"
filename_template = "{case_id}"          # {case_id} is interpolated at runtime
formats           = ["nifti", "hdf5"]    # one or both

[[cases]]
config = "assets/kidney/case_000/kid_000_synthetic.json"
bvals  = "assets/kidney/case_000/kidney.bval"

[[cases]]
config = "assets/kidney/case_002/kid_002.json"
bvals  = "assets/kidney/case_002/bvals.txt"
```

---

## Fields

### `[run]`

| Key | Type | Default | Description |
|---|---|---|---|
| `snr` | float | `null` | Signal-to-noise ratio for Rician noise. Omit or set to `null` for noise-free output. |
| `seed` | int | `null` | Random seed for reproducible noise. Ignored when `snr` is absent. |

### `[export]`

| Key | Type | Default | Description |
|---|---|---|---|
| `output_dir` | string | `"output"` | Directory where synthesized files are written. Created automatically if it does not exist. |
| `filename_template` | string | `"{case_id}"` | File stem template. `{case_id}` is replaced with each case's `case_id` value at runtime. |
| `formats` | list of strings | `["nifti"]` | Output formats to write. Supported values: `"nifti"` (`.nii.gz`) and `"hdf5"` (`.h5`). |

### `[[cases]]`

Each `[[cases]]` entry defines one case to synthesize. The array is processed in order.

| Key | Type | Required | Description |
|---|---|---|---|
| `config` | string | yes | Path to the phantom JSON file. |
| `bvals` | string | yes | Path to the b-values text file. |

All paths are resolved **relative to the TOML file's location**, so the TOML can sit next to the assets folder or anywhere in the project.

---

## Output paths

For each case the output paths are built as:

```
{output_dir}/{filename_template}.{extension}
```

where `{case_id}` in `filename_template` is replaced with the value from the case JSON, and `{extension}` is `nii.gz` or `h5` depending on the format.

**Example** — with `output_dir = "output/"`, `filename_template = "{case_id}_snr30"`, `formats = ["nifti", "hdf5"]`, and `case_id = "kidney_001"`:

```
output/kidney_001_snr30.nii.gz
output/kidney_001_snr30.h5
```

---

## CLI

```bash
mrphantom run experiment.toml
```

Equivalent programmatic usage:

```python
from mrphantom import load_run_config, load_phantom_config, DWIPhantom

run = load_run_config("experiment.toml")
phantom = DWIPhantom(snr=run.snr, seed=run.seed)

for entry in run.cases:
    cfg = load_phantom_config(entry.config, entry.bvals)
    out_paths = run.export.resolve_paths(cfg.case_id)
    phantom.load_case(cfg).synthesize().save(*out_paths)
```
