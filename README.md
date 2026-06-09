<p align="center">
  <img src="docs/assets/logo.svg" alt="MrPhantom logo" width="200"/>
</p>

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
- CLI with `synth`, `run`, `info`, `inspect`, and `montecarlo` commands
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

## Quick start

```python
from mrphantom import DWIPhantom

phantom = DWIPhantom.from_json(
    "assets/kidney/case_000/kid_000_synthetic.json",
    "assets/kidney/case_000/kidney.bval",
    snr=30,
    seed=42,
)
phantom.synthesize().save("kidney_001.nii.gz", "kidney_001.h5")
```

Or run a batch from the CLI:

```bash
mrphantom run experiment.toml
```

---

## Documentation

| Page | Description |
|---|---|
| [Usage Guide](docs/usage.md) | Walkthrough: single-case, batch, method chaining, params, ROI helpers |
| [Case Structure](docs/case_structure.md) | Directory layout, JSON schema, NIfTI conventions, b-values format |
| [TOML Run Config](docs/toml_run_config.md) | TOML schema for batch synthesis, field reference, path resolution |
| [TOML Monte Carlo Config](docs/toml_mc_config.md) | TOML schema for Monte Carlo simulation, distribution types, output format |
| [Monte Carlo Concepts](docs/concepts/monte-carlo.md) | Background: parameter sampling, Dirichlet, RNG design, batch vs lazy |
| [API Reference](docs/api.md) | All public classes, functions, and constants |
| [Reference](docs/reference.md) | Signal model formulation and HDF5 file schema |

---

## License

See [LICENSE](LICENSE).
