"""Command-line interface: ``cellscribe run``, ``cellscribe demo``, ``cellscribe example-data``, ``cellscribe config``."""

from __future__ import annotations

import logging
import sys
import time
import warnings
from enum import Enum
from pathlib import Path
from typing import Optional

import typer

from cellscribe import __version__

app = typer.Typer(
    name="cellscribe",
    help="Automated first-pass QC, clustering and cell-type annotation reports for single-cell and spatial transcriptomics.",
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


class Format(str, Enum):
    auto = "auto"
    tenx = "10x"
    h5ad = "h5ad"
    csv = "csv"
    visium = "visium"
    xenium = "xenium"


class Species(str, Enum):
    auto = "auto"
    human = "human"
    mouse = "mouse"


class QCMode(str, Enum):
    flag = "flag"
    filter = "filter"


class Integrate(str, Enum):
    none = "none"
    harmony = "harmony"


class Orientation(str, Enum):
    auto = "auto"
    genes_x_cells = "genes_x_cells"
    cells_x_genes = "cells_x_genes"


class DemoKind(str, Enum):
    single_cell = "single-cell"
    spatial = "spatial"
    visium = "visium"


class PlotlyJS(str, Enum):
    inline = "inline"
    cdn = "cdn"


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"cellscribe {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", callback=_version_callback, is_eager=True, help="Show the version and exit."),
) -> None:
    """Cellscribe command-line interface."""


def _setup_logging(verbose: bool, quiet: bool) -> None:
    level = logging.DEBUG if verbose else (logging.ERROR if quiet else logging.WARNING)
    logger = logging.getLogger("cellscribe")
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("  note: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    for noisy in ("scanpy", "squidpy", "spatialdata", "spatialdata._logging", "anndata", "numba", "fsspec", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    if not verbose:
        warnings.filterwarnings("ignore")


def _progress(quiet: bool):
    start = time.perf_counter()

    def say(msg: str) -> None:
        if not quiet:
            typer.echo(f"[{time.perf_counter() - start:6.1f}s] {msg}", err=True)

    return say


def _run_pipeline(source, output: Path, cfg, quiet: bool, verbose: bool, save_h5ad, summary_json) -> None:
    from cellscribe.errors import CellscribeError
    from cellscribe.pipeline import run as run_pipeline

    try:
        result = run_pipeline(
            source, output, cfg, save_h5ad=save_h5ad, summary_json=summary_json, progress=_progress(quiet)
        )
    except CellscribeError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from None
    except Exception as exc:  # noqa: BLE001
        if verbose:
            raise
        typer.secho(
            f"Unexpected error ({type(exc).__name__}): {exc}\nRe-run with --verbose for the full traceback, and please "
            "report it at https://github.com/jonahnki/cellscribe/issues",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from None
    if not quiet:
        s = result.summary
        n_warn = sum(r["status"] == "warning" for r in s["cluster_qc"])
        unit = "spots" if s["input"]["modality"] == "visium" else "cells"
        typer.echo("", err=True)
        typer.secho(f"Report written to {result.report_path}", fg=typer.colors.GREEN, bold=True, err=True)
        typer.echo(
            f"  {s['dataset']['n_cells']:,} {unit} · {s['clustering']['n_clusters']} clusters · "
            f"{s['qc']['n_flagged']:,} {unit} flagged · {n_warn} cluster(s) flagged as likely artefacts",
            err=True,
        )
        nar = s.get("narrative", {})
        if nar.get("status") != "generated":
            typer.echo(f"  AI narrative: {nar.get('reason', 'not generated')}", err=True)
        for extra in (save_h5ad, summary_json):
            if extra:
                typer.echo(f"  wrote {extra}", err=True)


def _build_config(config: Optional[Path], overrides: dict):
    from cellscribe.config import CellscribeConfig
    from cellscribe.errors import CellscribeError

    try:
        cfg = CellscribeConfig.from_yaml(config) if config else CellscribeConfig()
        return cfg.with_overrides(overrides)
    except CellscribeError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from None


@app.command()
def run(
    input: Path = typer.Option(..., "--input", "-i", help="Counts matrix: 10x folder/.h5, .h5ad, CSV/TSV, Visium outs/, or Xenium output folder."),
    format: Format = typer.Option(Format.auto, "--format", "-f", help="Input format (auto-detected by default)."),
    output: Path = typer.Option(Path("cellscribe_report.html"), "--output", "-o", help="Where to write the HTML report."),
    resolution: Optional[float] = typer.Option(None, "--resolution", "-r", min=0.01, help="Leiden resolution (default: automatic selection)."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML configuration file (see `cellscribe config`)."),
    species: Optional[Species] = typer.Option(None, "--species", help="Gene naming convention (default: auto-detect)."),
    qc_mode: Optional[QCMode] = typer.Option(None, "--qc-mode", help="'flag' keeps and labels outlier cells (default); 'filter' removes them."),
    batch_key: Optional[str] = typer.Option(None, "--batch-key", help="Cell metadata column with sample/batch labels (default: auto-detect)."),
    integrate: Optional[Integrate] = typer.Option(None, "--integrate", help="Opt-in batch integration (harmony)."),
    csv_orientation: Optional[Orientation] = typer.Option(None, "--csv-orientation", help="Orientation of CSV/TSV input (default: auto)."),
    sample_name: Optional[str] = typer.Option(None, "--sample-name", help="Display name for the sample."),
    title: Optional[str] = typer.Option(None, "--title", help="Report title."),
    marker_file: Optional[Path] = typer.Option(None, "--markers", help="Custom marker reference (YAML or CSV with cell_type,gene)."),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed (default 0)."),
    no_narrative: bool = typer.Option(False, "--no-narrative", help="Never call the Claude API, even if ANTHROPIC_API_KEY is set."),
    save_h5ad: Optional[Path] = typer.Option(None, "--save-h5ad", help="Also save the processed AnnData (.h5ad)."),
    summary_json: Optional[Path] = typer.Option(None, "--summary-json", help="Also save the structured summary as JSON."),
    plotlyjs: Optional[PlotlyJS] = typer.Option(None, "--plotlyjs", help="'inline' (self-contained, default) or 'cdn' (smaller file, needs internet)."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only print errors."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show warnings and full tracebacks."),
) -> None:
    """Generate a QC, clustering and annotation report for a dataset."""
    _setup_logging(verbose, quiet)
    overrides = {
        "input.format": format.value if format != Format.auto else None,
        "input.species": species.value if species else None,
        "input.batch_key": batch_key,
        "input.csv_orientation": csv_orientation.value if csv_orientation else None,
        "input.sample_name": sample_name,
        "clustering.resolution": resolution,
        "qc.mode": qc_mode.value if qc_mode else None,
        "preprocessing.integrate": integrate.value if integrate else None,
        "annotation.marker_file": str(marker_file) if marker_file else None,
        "report.title": title,
        "report.plotlyjs": plotlyjs.value if plotlyjs else None,
        "seed": seed,
        "narrative.enabled": False if no_narrative else None,
    }
    cfg = _build_config(config, overrides)
    _run_pipeline(input, output, cfg, quiet, verbose, save_h5ad, summary_json)


@app.command()
def demo(
    output: Path = typer.Option(Path("cellscribe_demo_report.html"), "--output", "-o", help="Where to write the demo report."),
    kind: DemoKind = typer.Option(DemoKind.single_cell, "--kind", "-k", help="single-cell (PBMC-like), spatial (imaging-based), or visium."),
    n: Optional[int] = typer.Option(None, "--n", min=200, help="Number of cells (or Visium spots) to simulate."),
    seed: int = typer.Option(1, "--seed", help="Random seed for the simulation and the analysis."),
    no_narrative: bool = typer.Option(False, "--no-narrative", help="Never call the Claude API."),
    save_h5ad: Optional[Path] = typer.Option(None, "--save-h5ad", help="Also save the processed AnnData (.h5ad)."),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the full pipeline on bundled synthetic data (no input needed)."""
    from cellscribe.config import CellscribeConfig
    from cellscribe.synthetic import make_demo

    _setup_logging(verbose, quiet)
    say = _progress(quiet)
    say(f"Simulating a synthetic {kind.value} dataset")
    adata = make_demo(kind.value, n, seed)
    cfg = CellscribeConfig(seed=seed).with_overrides(
        {
            "report.title": f"Cellscribe demo report: synthetic {kind.value} data",
            "narrative.enabled": False if no_narrative else None,
        }
    )
    _run_pipeline(adata, output, cfg, quiet, verbose, save_h5ad, None)


@app.command("example-data")
def example_data(
    outdir: Path = typer.Argument(..., help="Folder to write example input files into."),
    kind: DemoKind = typer.Option(DemoKind.single_cell, "--kind", "-k"),
    n: Optional[int] = typer.Option(None, "--n", min=200),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Write synthetic data in every input format Cellscribe reads (for trying the loaders)."""
    from cellscribe import writers
    from cellscribe.synthetic import make_demo

    outdir.mkdir(parents=True, exist_ok=True)
    adata = make_demo(kind.value, n, seed)
    written = []
    if kind == DemoKind.single_cell:
        written.append(writers.write_10x_mtx(adata, outdir / "filtered_feature_bc_matrix"))
        written.append(writers.write_10x_h5(adata, outdir / "filtered_feature_bc_matrix.h5"))
        written.append(writers.write_csv(adata, outdir / "counts_genes_x_cells.csv"))
        out = outdir / "counts.h5ad"
        adata.write_h5ad(out)
        written.append(out)
    elif kind == DemoKind.spatial:
        written.append(writers.write_xenium(adata, outdir / "xenium_output"))
    else:
        written.append(writers.write_visium(adata, outdir / "visium" / "outs"))
    for w in written:
        typer.echo(f"wrote {w}")
    typer.echo(f"\nTry: cellscribe run --input {written[0]} --output example_report.html")


@app.command("config")
def show_config(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write to a file instead of printing."),
) -> None:
    """Print the default configuration as YAML (a starting point for --config)."""
    from cellscribe.config import CellscribeConfig

    text = "# Cellscribe configuration (all keys optional; omitted keys use these defaults)\n" + CellscribeConfig().to_yaml()
    if output:
        output.write_text(text)
        typer.echo(f"wrote {output}")
    else:
        typer.echo(text)


if __name__ == "__main__":  # pragma: no cover
    app()
