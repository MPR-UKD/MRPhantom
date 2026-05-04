"""TOML-driven run configuration for batch phantom synthesis.

The run config acts as a top-level orchestration layer: it points to one or
more case JSON files and centralises acquisition settings (SNR, seed) and
export options (output directory, filename template, formats).

The case JSON files themselves (and the segmentation NIfTI files they
reference) remain unchanged — this layer adds nothing to them.

TOML schema
-----------
::

    [run]
    snr  = 30       # optional; omit or set to null for noise-free
    seed = 42       # optional

    [export]
    output_dir        = "output/"
    filename_template = "{case_id}"      # {case_id} is interpolated at runtime
    formats           = ["nifti", "hdf5"]

    [[cases]]
    config = "assets/kidney/case_001/kid_001_cor_ang.json"
    bvals  = "assets/kidney/case_001/bvals.txt"

    [[cases]]
    config = "assets/kidney/case_002/kid_002.json"
    bvals  = "assets/kidney/case_002/bvals.txt"

All paths in ``[[cases]]`` are resolved relative to the TOML file's parent
directory, so the TOML can live next to the assets folder or anywhere else.

Usage
-----
::

    from mrphantom import load_run_config, DWIPhantom

    run = load_run_config("experiment.toml")
    phantom = DWIPhantom(snr=run.snr, seed=run.seed)

    for entry in run.cases:
        from mrphantom import load_phantom_config
        cfg = load_phantom_config(entry.config, entry.bvals)
        out_paths = run.export.resolve_paths(cfg.case_id)
        phantom.load_case(cfg).synthesize().save(*out_paths)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib  # type: ignore[no-redef]
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError as exc:
            raise ImportError(
                "Python < 3.11 requires the 'tomli' package to read TOML files. "
                "Install it with:  pip install tomli"
            ) from exc


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CaseEntry:
    """A single case to synthesize.

    Attributes:
        config: Absolute path to the phantom JSON file.
        bvals: Absolute path to the b-values text file.
    """

    config: Path
    bvals: Path


@dataclass
class ExportConfig:
    """Export settings shared across all cases in a run.

    Attributes:
        output_dir: Directory where synthesized files are written.
        filename_template: File stem template. ``{case_id}`` is replaced with
            the case's ``case_id`` field at synthesis time.
            Example: ``"{case_id}_snr30"``.
        formats: List of output formats. Supported values: ``"nifti"``,
            ``"hdf5"``.
    """

    output_dir: Path
    filename_template: str = "{case_id}"
    formats: list[str] = field(default_factory=lambda: ["nifti"])

    def resolve_paths(self, case_id: str) -> list[Path]:
        """Build output file paths for a given *case_id*.

        Args:
            case_id: The ``case_id`` string from the phantom JSON (used for
                ``{case_id}`` interpolation in :attr:`filename_template`).

        Returns:
            list[Path]: One path per entry in :attr:`formats`, with
            appropriate file extensions (``.nii.gz`` for NIfTI, ``.h5`` for
            HDF5).

        Raises:
            ValueError: If an entry in :attr:`formats` is not ``"nifti"`` or
                ``"hdf5"``.
        """
        stem = self.filename_template.format(case_id=case_id)
        paths: list[Path] = []
        for fmt in self.formats:
            fmt_lower = fmt.lower()
            if fmt_lower == "nifti":
                paths.append(self.output_dir / f"{stem}.nii.gz")
            elif fmt_lower == "hdf5":
                paths.append(self.output_dir / f"{stem}.h5")
            else:
                raise ValueError(
                    f"Unknown export format {fmt!r}. Supported values: 'nifti', 'hdf5'."
                )
        return paths


@dataclass
class RunConfig:
    """Top-level configuration for a batch synthesis run.

    Attributes:
        snr: Signal-to-noise ratio passed to the phantom synthesizer. ``None``
            produces noise-free output.
        seed: Random seed for reproducible Rician noise. Ignored when ``snr``
            is ``None``.
        export: Export settings (output directory, filename template, formats).
        cases: Ordered list of cases to synthesize.
    """

    snr: float | None
    seed: int | None
    export: ExportConfig
    cases: list[CaseEntry]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_run_config(toml_path: Path | str) -> RunConfig:
    """Parse a TOML run-config file into a :class:`RunConfig`.

    All relative paths inside the TOML are resolved against the directory
    that contains the TOML file, so the file can live anywhere in the project.

    Args:
        toml_path: Path to the ``.toml`` run-config file.

    Returns:
        RunConfig: Fully resolved :class:`RunConfig` with absolute case paths.

    Raises:
        FileNotFoundError: If *toml_path* does not exist.
        KeyError: If required TOML keys are missing.
        ValueError: If ``[[cases]]`` is empty or an unsupported export format
            is listed.
    """
    toml_path = Path(toml_path)
    if not toml_path.is_file():
        raise FileNotFoundError(f"Run-config TOML not found: {toml_path}")

    base_dir = toml_path.parent

    with open(toml_path, "rb") as fh:
        data: dict = tomllib.load(fh)

    # --- [run] section (all optional) ------------------------------------
    run_section: dict = data.get("run", {})
    snr: float | None = run_section.get("snr", None)
    seed: int | None = run_section.get("seed", None)
    if snr is not None:
        snr = float(snr)
    if seed is not None:
        seed = int(seed)

    # --- [export] section ------------------------------------------------
    export_section: dict = data.get("export", {})
    output_dir = base_dir / export_section.get("output_dir", "output")
    filename_template: str = export_section.get("filename_template", "{case_id}")
    formats: list[str] = export_section.get("formats", ["nifti"])

    export = ExportConfig(
        output_dir=output_dir,
        filename_template=filename_template,
        formats=formats,
    )

    # --- [[cases]] array -------------------------------------------------
    raw_cases: list[dict] = data.get("cases", [])
    if not raw_cases:
        raise ValueError(f"Run-config TOML contains no [[cases]] entries: {toml_path}")

    cases: list[CaseEntry] = []
    for raw in raw_cases:
        cases.append(
            CaseEntry(
                config=(base_dir / raw["config"]).resolve(),
                bvals=(base_dir / raw["bvals"]).resolve(),
            )
        )

    return RunConfig(snr=snr, seed=seed, export=export, cases=cases)
