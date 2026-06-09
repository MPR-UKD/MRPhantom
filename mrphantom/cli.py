"""mrphantom CLI — MR phantom synthesis entry point.

Usage examples
--------------
Synthesize a single case, noise-free, write NIfTI::

    mrphantom synth phantom.json bvals.txt --out phantom.nii.gz

Synthesize a single case with SNR=30, write both NIfTI and HDF5::

    mrphantom synth phantom.json bvals.txt --snr 30 --seed 42 \\
        --out phantom.nii.gz --out phantom.h5

Batch synthesis from a TOML run-config::
    
    mrphantom run experiment.toml

Print phantom metadata::

    mrphantom info phantom.json bvals.txt

Inspect a saved HDF5 file::

    mrphantom inspect phantom.h5
"""

from __future__ import annotations

from pathlib import Path

import click

from .dwi.config import load_dwi_config
from .dwi.phantom import DWIPhantom
from .io import load_hdf5
from .io.mc_hdf5 import save_mc_hdf5
from .montecarlo.config import load_montecarlo_config
from .montecarlo.runner import run_montecarlo
from .run_config import load_run_config


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="mrphantom")
def main() -> None:
    """mrphantom — standalone MR phantom synthesis and export tool."""


# ---------------------------------------------------------------------------
# synth command  (single case, kept for direct / scripted use)
# ---------------------------------------------------------------------------


@main.command()
@click.argument(
    "config_json", metavar="CONFIG", type=click.Path(exists=True, path_type=Path)
)
@click.argument(
    "bval_file", metavar="BVAL", type=click.Path(exists=True, path_type=Path)
)
@click.option(
    "--snr",
    type=float,
    default=None,
    show_default=True,
    help="Signal-to-noise ratio for Rician noise. Omit for a noise-free phantom.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    show_default=True,
    help="Integer random seed for reproducible noise (ignored when --snr is absent).",
)
@click.option(
    "--out",
    "-o",
    "outputs",
    type=click.Path(path_type=Path),
    multiple=True,
    required=True,
    help=(
        "Output file path.  Format is inferred from the extension "
        "(.nii / .nii.gz → NIfTI, .h5 / .hdf5 → HDF5). "
        "Repeat the flag to write multiple formats in one pass."
    ),
)
def synth(
    config_json: Path,
    bval_file: Path,
    snr: float | None,
    seed: int | None,
    outputs: tuple[Path, ...],
) -> None:
    """Synthesize a DWI phantom from CONFIG and BVAL.

    CONFIG is a phantom JSON file describing segmentation labels and
    ground-truth diffusion parameters.  BVAL is a whitespace- or
    newline-separated b-values file in s/mm².

    At least one --out path is required.  Provide the flag multiple times to
    export to several formats simultaneously.
    """
    click.echo(f"Loading config: {config_json}")
    config = load_dwi_config(config_json, bval_file)
    click.echo(
        f"  case_id={config.case_id!r}  regions={len(config.regions)}"
        f"  b-values={len(config.bvalues)}"
    )

    click.echo(f"Synthesizing phantom (snr={snr!r}) ...")
    phantom = DWIPhantom(snr=snr, seed=seed)
    phantom.load_case(config).synthesize()
    click.echo(f"  signal shape: {phantom.signal.shape}")

    written = phantom.save(*outputs)
    for path in written:
        suffix = "".join(path.suffixes).lower()
        fmt = (
            "NIfTI" if suffix.endswith(".nii.gz") or suffix.endswith(".nii") else "HDF5"
        )
        click.echo(f"  {fmt:<6} → {path}")

    click.echo("Done.")


# ---------------------------------------------------------------------------
# run command  (batch synthesis from TOML)
# ---------------------------------------------------------------------------


@main.command()
@click.argument(
    "toml_file", metavar="TOML", type=click.Path(exists=True, path_type=Path)
)
def run(toml_file: Path) -> None:
    """Batch-synthesize phantoms described in a TOML run-config file.

    TOML is a run-config file that lists one or more [[cases]] (each pointing
    to a phantom JSON and a b-values file) together with shared acquisition
    settings (snr, seed) and export options (output directory, formats).

    Example TOML::

    \b
        [run]
        snr  = 30
        seed = 42

        [export]
        output_dir        = "output/"
        filename_template = "{case_id}"
        formats           = ["nifti", "hdf5"]

        [[cases]]
        config = "assets/kidney/case_000/kid_000_synthetic.json"
        bvals  = "assets/kidney/case_000/kidney.bval"
    """
    click.echo(f"Loading run config: {toml_file}")
    run_cfg = load_run_config(toml_file)
    click.echo(
        f"  cases={len(run_cfg.cases)}"
        f"  snr={run_cfg.snr!r}"
        f"  seed={run_cfg.seed!r}"
        f"  formats={run_cfg.export.formats}"
    )

    phantom = DWIPhantom(snr=run_cfg.snr, seed=run_cfg.seed)

    for i, entry in enumerate(run_cfg.cases, start=1):
        click.echo(f"\n[{i}/{len(run_cfg.cases)}] {entry.config.name}")
        config = load_dwi_config(entry.config, entry.bvals)
        click.echo(
            f"  case_id={config.case_id!r}"
            f"  regions={len(config.regions)}"
            f"  b-values={len(config.bvalues)}"
        )

        out_paths = run_cfg.export.resolve_paths(config.case_id)
        phantom.load_case(config).synthesize()
        click.echo(f"  signal shape: {phantom.signal.shape}")

        written = phantom.save(*out_paths)
        for path in written:
            suffix = "".join(path.suffixes).lower()
            fmt = (
                "NIfTI"
                if suffix.endswith(".nii.gz") or suffix.endswith(".nii")
                else "HDF5"
            )
            click.echo(f"  {fmt:<6} → {path}")

    click.echo("\nDone.")


# ---------------------------------------------------------------------------
# info command
# ---------------------------------------------------------------------------


@main.command()
@click.argument(
    "config_json", metavar="CONFIG", type=click.Path(exists=True, path_type=Path)
)
@click.argument(
    "bval_file", metavar="BVAL", type=click.Path(exists=True, path_type=Path)
)
def info(config_json: Path, bval_file: Path) -> None:
    """Print a summary of a phantom CONFIG file.

    CONFIG is a phantom JSON file.  BVAL is the matching b-values file.
    """
    config = load_dwi_config(config_json, bval_file)

    click.echo(f"case_id     : {config.case_id}")
    click.echo(f"name        : {config.name}")
    click.echo(f"description : {config.description}")
    click.echo(f"image_dims  : {config.image_dims}")
    click.echo(f"nifti_path  : {config.nifti_path}")
    click.echo(f"b-values ({len(config.bvalues)}): {config.bvalues.tolist()}")
    click.echo(f"regions ({len(config.regions)}):")
    for idx, region in sorted(config.regions.items()):
        click.echo(
            f"  [{idx:3d}] {region.name:<22s} "
            f"n={region.n_components}  S0={region.S0:.1f}"
        )


# ---------------------------------------------------------------------------
# inspect command (HDF5)
# ---------------------------------------------------------------------------


@main.command()
@click.argument(
    "hdf5_file", metavar="FILE", type=click.Path(exists=True, path_type=Path)
)
def inspect(hdf5_file: Path) -> None:
    """Print a summary of a saved phantom HDF5 FILE."""
    data = load_hdf5(hdf5_file)

    click.echo(f"case_id     : {data.case_id}")
    click.echo(f"name        : {data.name}")
    click.echo(f"description : {data.description}")
    click.echo(f"snr         : {data.snr}")
    click.echo(f"signal      : shape={data.signal.shape}  dtype={data.signal.dtype}")
    click.echo(f"volumes     : {data.volumes.tolist()}")
    click.echo(
        f"segmentation: shape={data.segmentation.shape}  dtype={data.segmentation.dtype}"
    )
    click.echo(f"affine      :\n{data.affine}")


# ---------------------------------------------------------------------------
# montecarlo command
# ---------------------------------------------------------------------------


@main.command()
@click.argument(
    "toml_file", metavar="TOML", type=click.Path(exists=True, path_type=Path)
)
def montecarlo(toml_file: Path) -> None:
    """Run Monte Carlo DWI simulations described in a TOML config file.

    TOML extends the standard run-config schema with a ``[montecarlo]``
    block and optional ``[[montecarlo.regions]]`` entries that specify
    per-region parameter distributions.

    Example TOML::

    \b
        [run]
        snr      = 30
        seed     = 42

        [montecarlo]
        n_samples = 100

        [[montecarlo.regions]]
        name      = "cortex"
        S0        = {dist = "normal", std = 10.0}

        [export]
        output_dir        = "output/"
        filename_template = "{case_id}"
        formats           = ["hdf5"]

        [[cases]]
        config = "case_000/kid_000_synthetic.json"
        bvals  = "case_000/kidney.bval"

    Results are always written as HDF5 (5-D signal arrays).
    """
    click.echo(f"Loading MC config: {toml_file}")
    mc_cfg = load_montecarlo_config(toml_file)
    click.echo(
        f"  cases={len(mc_cfg.cases)}"
        f"  n_samples={mc_cfg.n_samples}"
        f"  snr={mc_cfg.snr!r}"
        f"  seed={mc_cfg.seed!r}"
        f"  mc_regions={list(mc_cfg.mc_regions)}"
    )

    for i, entry in enumerate(mc_cfg.cases, start=1):
        click.echo(f"\n[{i}/{len(mc_cfg.cases)}] {entry.config.name}")
        base_config = load_dwi_config(entry.config, entry.bvals)
        click.echo(
            f"  case_id={base_config.case_id!r}"
            f"  regions={len(base_config.regions)}"
            f"  b-values={len(base_config.bvalues)}"
        )

        click.echo(
            f"  Running {mc_cfg.n_samples} Monte Carlo samples "
            f"(snr={mc_cfg.snr!r}, snr_region={mc_cfg.snr_region!r}) ..."
        )
        result = run_montecarlo(
            base_config=base_config,
            mc_regions=mc_cfg.mc_regions,
            n_samples=mc_cfg.n_samples,
            snr=mc_cfg.snr,
            snr_region=mc_cfg.snr_region,
            seed=mc_cfg.seed,
        )
        click.echo(f"  signal shape: {result.signal.shape}")

        stem = mc_cfg.export.filename_template.format(case_id=base_config.case_id)
        out_path = mc_cfg.export.output_dir / f"{stem}.h5"

        save_mc_hdf5(
            signal=result.signal,
            signal_gt=result.signal_gt,
            volumes=base_config.bvalues,
            segmentation=base_config.nifti_path,
            affine=result.affine,
            S0_samples=result.sampled_params.S0,
            D_samples=result.sampled_params.D,
            fractions_samples=result.sampled_params.fractions,
            region_names=result.sampled_params.region_names,
            out_path=out_path,
            case_id=base_config.case_id,
            name=base_config.name,
            description=base_config.description,
            snr=mc_cfg.snr,
            snr_region=mc_cfg.snr_region,
            seed=mc_cfg.seed,
        )
        click.echo(f"  HDF5   → {out_path}")

    click.echo("\nDone.")
