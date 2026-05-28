# Reference

> **TL;DR** — Mathematical and format reference for MRPhantom. Covers the multi-exponential DWI signal model and Rician noise formulation (**Signal model**), the layout and schema of single-synthesis HDF5 files (**HDF5 schema**), and the extended layout for Monte Carlo results (**Monte Carlo HDF5 schema**).

---

## Signal model

The DWI signal for each voxel is a sum of exponential decays across diffusion compartments:

```
S(b) = S0 · Σᵢ fᵢ · exp(-b · Dᵢ)
```

| Symbol | Description |
|---|---|
| `S0` | Baseline signal intensity at b=0 |
| `fᵢ` | Compartment signal fraction (all fractions must sum to 1) |
| `Dᵢ` | Compartment diffusivity in mm²/s |
| `b` | b-value in s/mm² |

When `snr` is set, Rician noise is added:

```
S_noisy = sqrt( (S + N_re)² + N_im² )
```

where `N_re, N_im ~ Normal(0, σ)` and `σ = max(S0) / snr`.

Set `snr=None` (the default) for noise-free output.

---

## HDF5 schema

Files written by `DWIPhantom.save("phantom.h5")` or `mrphantom synth --out phantom.h5` follow this layout:

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

Read back with `load_hdf5`:

```python
from mrphantom import load_hdf5

data = load_hdf5("phantom.h5")

data.signal        # np.ndarray  (X, Y, Z, n_bvalues)
data.bvalues       # np.ndarray  (n_bvalues,)
data.segmentation  # np.ndarray  (X, Y, Z) — integer label map
data.affine        # np.ndarray  (4, 4)
data.case_id       # str
data.snr           # float | None

img = data.as_nifti()   # nibabel.Nifti1Image
```

---

## Monte Carlo HDF5 schema

Files written by `mrphantom montecarlo mc.toml` or `save_mc_hdf5()` follow this layout:

```
/signal          float64  (X, Y, Z, n_b, N)              gzip-compressed
/signal_gt       float64  (X, Y, Z, n_b, N)              gzip-compressed
/volumes         float64  (n_b,)
/segmentation    int32    (X, Y, Z)                       gzip-compressed
/affine          float64  (4, 4)

/sampled_params/
    S0           float64  (n_regions, N)                  gzip-compressed
    D            float64  (n_regions, max_n_components, N) gzip-compressed
    fractions    float64  (n_regions, max_n_components, N) gzip-compressed
    region_names variable-length UTF-8 string (n_regions,)

Root attributes:
    case_id      str
    name         str
    description  str
    n_samples    int
    snr          float   (absent for noise-free runs)
    snr_region   str     (absent when None)
    seed         int     (absent when None)
```

`N` is the number of Monte Carlo samples. Each `signal[..., i]` and `signal_gt[..., i]` slice corresponds to the *i*-th parameter draw. `signal_gt` is always noise-free regardless of `snr`.

For multi-component regions the `D` and `fractions` arrays are zero-padded in the component dimension to `max_n_components` across all regions.

Read back with `load_mc_hdf5`:

```python
from mrphantom import load_mc_hdf5

data = load_mc_hdf5("kidney_mc.h5")

data.signal             # np.ndarray  (X, Y, Z, n_b, N)
data.signal_gt          # np.ndarray  (X, Y, Z, n_b, N)
data.volumes            # np.ndarray  (n_b,)
data.segmentation       # np.ndarray  (X, Y, Z) — integer label map
data.affine             # np.ndarray  (4, 4)
data.n_samples          # int
data.snr                # float | None
data.snr_region         # str | None
data.seed               # int | None
data.S0_samples         # np.ndarray  (n_regions, N)
data.D_samples          # np.ndarray  (n_regions, max_comp, N)
data.fractions_samples  # np.ndarray  (n_regions, max_comp, N)
data.region_names       # list[str]   length n_regions
```
