# Changelog

All notable changes to this project are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
