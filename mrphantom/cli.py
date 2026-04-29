"""mrsim CLI — MR phantom synthesis entry point.

Usage examples
--------------
Synthesize, noise-free, write NIfTI::

    mrsim synth phantom.json bvals.txt --out phantom.nii.gz

Synthesize with SNR=30, write both NIfTI and HDF5::

    mrsim synth phantom.json bvals.txt --snr 30 --seed 42 \\
        --out phantom.nii.gz --out phantom.h5

Print phantom metadata::

    mrsim info phantom.json bvals.txt
"""

from __future__ import annotations

from pathlib import Path

import click

from .config import load_phantom_config
from .io import load_hdf5, save_hdf5, save_nifti
from .synthesis import synthesize_phantom


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_format(path: Path) -> str:
    """Return ``'nii'``, ``'hdf5'``, or raise :exc:`click.BadParameter`."""
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".nii") or suffixes.endswith(".nii.gz"):
        return "nii"
    if suffixes.endswith(".h5") or suffixes.endswith(".hdf5"):
        return "hdf5"
    raise click.BadParameter(
        f"Cannot detect format from '{path.name}'. "
        "Use a .nii / .nii.gz or .h5 / .hdf5 extension.",
        param_hint="--out",
    )


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="mrsim")
def main() -> None:
    """mrsim — standalone MR phantom synthesis and export tool."""


# ---------------------------------------------------------------------------
# synth command
# ---------------------------------------------------------------------------

@main.command()
@click.argument("config_json", metavar="CONFIG", type=click.Path(exists=True, path_type=Path))
@click.argument("bval_file", metavar="BVAL", type=click.Path(exists=True, path_type=Path))
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

    CONFIG is a phantom JSON file describing segmentation labels and ground-truth
    diffusion parameters.  BVAL is a whitespace- or newline-separated b-values
    file in s/mm².

    At least one --out path is required.  Provide the flag multiple times to
    export to several formats simultaneously.
    """
    # Validate all output paths before synthesizing
    format_map: dict[Path, str] = {}
    for out_path in outputs:
        format_map[out_path] = _detect_format(out_path)

    click.echo(f"Loading config: {config_json}")
    config = load_phantom_config(config_json, bval_file)
    click.echo(
        f"  case_id={config.case_id!r}  regions={len(config.regions)}"
        f"  b-values={len(config.bvalues)}"
    )

    rng = None
    if snr is not None and seed is not None:
        import numpy as np
        rng = np.random.default_rng(seed)

    click.echo(f"Synthesizing phantom (snr={snr!r}) …")
    signal, signal_img = synthesize_phantom(config, snr=snr, rng=rng)
    click.echo(f"  signal shape: {signal.shape}")

    for out_path, fmt in format_map.items():
        if fmt == "nii":
            written = save_nifti(signal_img, out_path)
            click.echo(f"  NIfTI  → {written}")
        else:
            written = save_hdf5(signal, signal_img, config, out_path, snr=snr)
            click.echo(f"  HDF5   → {written}")

    click.echo("Done.")


# ---------------------------------------------------------------------------
# info command
# ---------------------------------------------------------------------------

@main.command()
@click.argument("config_json", metavar="CONFIG", type=click.Path(exists=True, path_type=Path))
@click.argument("bval_file", metavar="BVAL", type=click.Path(exists=True, path_type=Path))
def info(config_json: Path, bval_file: Path) -> None:
    """Print a summary of a phantom CONFIG file.

    CONFIG is a phantom JSON file.  BVAL is the matching b-values file.
    """
    config = load_phantom_config(config_json, bval_file)

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
@click.argument("hdf5_file", metavar="FILE", type=click.Path(exists=True, path_type=Path))
def inspect(hdf5_file: Path) -> None:
    """Print a summary of a saved phantom HDF5 FILE."""
    data = load_hdf5(hdf5_file)

    click.echo(f"case_id     : {data.case_id}")
    click.echo(f"name        : {data.name}")
    click.echo(f"description : {data.description}")
    click.echo(f"snr         : {data.snr}")
    click.echo(f"signal      : shape={data.signal.shape}  dtype={data.signal.dtype}")
    click.echo(f"bvalues     : {data.bvalues.tolist()}")
    click.echo(f"segmentation: shape={data.segmentation.shape}  dtype={data.segmentation.dtype}")
    click.echo(f"affine      :\n{data.affine}")
