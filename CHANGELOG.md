# Changelog

All notable changes to this project are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.4.1] — 2026-10-19

### Changed

- Project is now hosted at `github.com/MPR-UKD/MRPhantom`; README and project URLs updated.
- README documents installation via `pip install git+https://github.com/MPR-UKD/MRPhantom` (not published on PyPI).
- `pyproject.toml` gains package metadata: license (MIT), authors, keywords, classifiers, project URLs.
- Updated logo.

### Removed

- Internal AI-agent guidance files (`.github/skills/`) are no longer tracked.

No code or API changes.

---

## [0.4.0] — 2026-06-09

### Added

- Ground-truth (noise-free) signal storage throughout the pipeline: `synthesize_phantom()` returns `signal_gt` as a third element; `PhantomData.signal_gt` field; HDF5 round-trip preserves `signal_gt`; old HDF5 files load with `signal_gt=None` (backward compatible).
- `DWIPhantom._run_synthesis(rng, config=None)` extracted as a protected method. `synthesize()` delegates to it — zero behaviour change for existing callers.
- Full Monte Carlo subpackage (`mrphantom.montecarlo`):
  - `ParamDist`, `RegionMCConfig`, `MCRunConfig` config dataclasses with TOML loader (`load_montecarlo_config`).
  - `SampledParams`, `sample_region_params` — parameter sampling from normal, lognormal, and Dirichlet distributions (using NumPy, no SciPy dependency).
  - `MCResult`, `run_montecarlo` — functional MC runner with two-stream RNG (parameter + independent noise seeds per sample).
  - `MCPhantomData`, `save_mc_hdf5`, `load_mc_hdf5` — 5-D HDF5 I/O with `/sampled_params/` group.
- `MCDWIPhantom(DWIPhantom)` — class-centred Monte Carlo API with `setup()` step, batch mode `synthesize()` (5-D output), lazy per-sample `step(i)`, and incremental `open_h5()` writing via `_H5Writer`.
- `MCDWIPhantom.from_toml(path)` classmethod factory.
- CLI `mrphantom montecarlo` command.
- `assets/kidney/mc.toml` — ready-to-run MC example with 5 randomised regions.
- Dirichlet fraction sampling simplified: specify a scalar `kappa` (concentration) instead of a manual `alpha` vector. The Dirichlet alpha is computed as `base_fractions * kappa`, so the mean is always anchored to the base config values.
- Normal and lognormal sampling simplified for S0 and D: the mean is auto-anchored to the base config value — only spread parameters (`std` / `sigma`) need to be specified. `mean` field on `ParamDist` is no longer required (ignored at sampling time).
- New Monte Carlo concepts guide (`docs/concepts/monte-carlo.md`) explaining parameter sampling, Dirichlet distributions, two-stream RNG design, and batch vs lazy synthesis modes.

### Changed

- `_build_sample_config` moved from `runner.py` to `sampler.py` (no duplicate).
- Module exports updated: `mrphantom/__init__.py`, `mrphantom/montecarlo/__init__.py` now export `MCDWIPhantom`.
- Removed vestigial `TYPE_CHECKING: pass` block from `montecarlo/phantom.py`.

### Fixed

- `n_samples=0` now raises `ValueError` in `run_montecarlo` and `MCRunConfig` instead of crashing with an index error.
- `snr=0` now raises `ValueError` in `synthesize_phantom` instead of producing infinite noise.
- Lognormal sampling with `base_value <= 0` now raises `ValueError` instead of silently producing NaN.
- Empty region configs, negative S0/D/b-values, fractions not summing to 1, and duplicate region names are now rejected at config load time.
- HDF5 save now warns when overwriting an existing file.
- Liver fractions in bundled kidney asset corrected to sum to 1.
- CLI `montecarlo --help` example no longer shows the obsolete `mean` parameter.
- Updated all doc files to reflect current API names (`load_phantom_config` → `load_dwi_config`, `PhantomConfig` → `DWIConfig`, `/bvalues` → `/volumes`, `data.bvalues` → `data.volumes`).

### Added tests

- `test_mc_phantom.py` — 80 tests covering full `MCDWIPhantom` lifecycle (init, load_case, setup, synthesize, step, save, open_h5, get_params, from_toml, full chains).
- `test_montecarlo_config.py` — TOML parsing, ParamDist/RegionMCConfig dataclasses.
- `test_montecarlo_sampler.py` — scalar + Dirichlet sampling, reproducibility, validation.
- `test_montecarlo_runner.py` — `run_montecarlo` shape, noise, reproducibility, parameter randomisation.
- `test_mc_io.py` — `save_mc_hdf5` / `load_mc_hdf5` round-trip for all fields.
- `test_mc_cli.py` — `mrphantom montecarlo` CLI integration.
- New mean-anchoring tests for `_sample_dirichlet` and `sample_region_params` fractions.

### Documentation

- Full documentation: Monte Carlo TOML config guide (`docs/toml_mc_config.md`), MC phantom walkthrough in Usage Guide, API Reference updates.
- `snr_region` parameter documented in `reference.md`, `toml_run_config.md`, and `usage.md`.
- Added `montecarlo` CLI command to README feature list.
- Added TOML Monte Carlo Config and Monte Carlo Concepts to README and docs index tables.

---

## [0.3.0] — 2026-05-04

### Breaking changes

- `PhantomConfig` renamed to `DWIConfig`; `load_phantom_config` renamed to `load_dwi_config`.  
  All DWI-specific code has moved into the `mrphantom.dwi` subpackage (`mrphantom.dwi.config`, `mrphantom.dwi.phantom`, `mrphantom.dwi.synthesis`, `mrphantom.dwi.roi`).
- HDF5 dataset key and public field renamed from `bvalues` to `volumes` throughout `save_hdf5` / `load_hdf5`.
- `save_hdf5` signature no longer accepts a `DWIConfig` object; it now takes explicit `volumes`, `segmentation`, `case_id`, `name`, `description` arguments.
- `EVALUATION_REGION_NAMES` and `get_evaluation_segments` moved from `mrphantom.roi` to `mrphantom.dwi.roi`.  
  `get_fitting_mask` remains at `mrphantom.roi`.

### Added

- `mrphantom/dwi/` subpackage that cleanly isolates all DWI-specific logic, ready for future modalities (T1, DCE, ASL).
- Fully synthetic test asset `assets/kidney/case_000/` — abstract ellipsoidal segmentation with all 11 region labels; no patient data.
- `scripts/make_synthetic_seg.py` — reproducible generation script for the synthetic segmentation NIfTI.
- `assets/kidney/run.toml` — example batch run-config pointing at `case_000`.

### Changed

- `mrphantom/roi.py` trimmed to the modality-agnostic `get_fitting_mask` helper only.
- All tests, fixtures, docstrings, and documentation updated to reflect the new subpackage layout, renamed API, and `case_000` assets.

### Fixed

- Build system migrated from `setuptools` to `hatchling`.
- `test_run_command` now writes POSIX-style paths into patched TOML strings, fixing a `TOMLDecodeError` on Windows caused by backslash escape sequences.

---

## [0.2.0] — 2026-04-30

### Added

- Abstract `MRPhantom` base class (`mrphantom/base.py`) with sklearn-style `get_params` / `set_params` and a `load_case` / `synthesize` / `save` interface.
- `DWIPhantom` class — full diffusion phantom pipeline:
  - Multi-exponential signal model: `S(b) = S0 · Σᵢ fᵢ · exp(-b · Dᵢ)`
  - Optional Rician noise at a configurable SNR
  - NIfTI (`.nii.gz`) and HDF5 (`.h5`) export in a single `save()` call
  - `from_json` convenience factory
  - Method chaining: `phantom.load_case(cfg).synthesize().save("out.h5")`
- `RunConfig` / `ExportConfig` / `CaseEntry` dataclasses and `load_run_config` for TOML-driven batch synthesis.
- CLI entry point `mrphantom` with four commands:
  - `synth` — synthesize a single case from JSON + bvals
  - `run`   — batch-synthesize from a TOML run-config
  - `info`  — print phantom JSON metadata
  - `inspect` — print HDF5 file summary
- `PhantomConfig` / `load_phantom_config` JSON loader with multi-compartment region model and null-region fallback to `remaining_tissue`.
- `save_hdf5` / `load_hdf5` — HDF5 round-trip storing signal, segmentation, b-values, affine, and metadata.
- `get_evaluation_segments` / `get_fitting_mask` ROI helpers.
- Comprehensive pytest test suite (120 tests).
- Full documentation: Usage Guide, Case Structure, TOML Run-Config, API Reference, Signal Model reference.

### Fixed

- Removed spurious `get_params` override from `DWIPhantom` (inherited correctly from `MRPhantom`).
- Setuptools package discovery fixed to include all `mrphantom.*` sub-packages.

---

## [0.1.0] — 2026-04-29

### Added

- Initial project scaffolding: `pyproject.toml`, package layout, `.gitignore`.
- First implementation of `PhantomConfig` JSON loader, `synthesize_phantom`, `save_hdf5` / `load_hdf5`, `save_nifti`, and a basic CLI (`mrphantom synth`).
- Bundled kidney segmentation asset `assets/kidney/case_001/` with real coronal DWI segmentation, b-values, and diffusion parameters for 11 anatomical regions.
