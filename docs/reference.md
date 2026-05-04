# Reference

> **TL;DR** — Mathematical and format reference for MRPhantom. Covers the multi-exponential DWI signal model and Rician noise formulation (**Signal model**), and the layout and schema of HDF5 output files (**HDF5 schema**).

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
