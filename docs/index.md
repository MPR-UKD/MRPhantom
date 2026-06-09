# MRPhantom Documentation

> **TL;DR** — Navigation index for all MRPhantom docs. Each entry below links to a dedicated page: API Reference, Usage Guide, Case Structure, TOML Run Config, TOML Monte Carlo Config, and Reference (signal model + HDF5 schemas).

---

## Contents

| Page | Description |
|---|---|---|
| [API Reference](api.md) | Complete public API — all classes, functions, and constants |
| [Usage Guide](usage.md) | Walkthrough: single-case, batch, method chaining, params, ROI helpers |
| [Case Structure](case_structure.md) | Case directory layout, JSON schema, NIfTI conventions, b-values format |
| [TOML Run Config](toml_run_config.md) | TOML schema for batch synthesis, field reference, path resolution rules |
| [TOML Monte Carlo Config](toml_mc_config.md) | TOML schema for Monte Carlo simulation, distribution types, output format |
| [Monte Carlo Concepts](concepts/monte-carlo.md) | Background: parameter sampling, Dirichlet, RNG design, batch vs lazy |
| [Reference](reference.md) | Signal model formulation, HDF5 file schema, and Monte Carlo HDF5 schema |

## Quick navigation

- **New to mrphantom?** Start with the [Usage Guide](usage.md).
- **Setting up a case?** See [Case Structure](case_structure.md).
- **Running a batch?** See [TOML Run Config](toml_run_config.md) and the ready-to-run example at `assets/kidney/run.toml`.
- **Running Monte Carlo simulations?** See [TOML Monte Carlo Config](toml_mc_config.md) and the ready-to-run example at `assets/kidney/mc.toml`.
- **Looking up a specific class or function?** See the [API Reference](api.md).
- **Signal model or HDF5 schema?** See the [Reference](reference.md).
