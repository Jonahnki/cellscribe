"""Input loaders. Each normalises its format into Cellscribe's AnnData contract.

Use :func:`load` with ``fmt="auto"`` to detect the format from the path.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad

from cellscribe.errors import InputFormatError
from cellscribe.loaders._common import finalize, standardize_in_memory
from cellscribe.loaders.csv import load_csv
from cellscribe.loaders.h5ad import load_h5ad
from cellscribe.loaders.tenx import load_10x, resolve_mtx_dir
from cellscribe.loaders.visium import load_visium
from cellscribe.loaders.xenium import is_xenium_dir, load_xenium

FORMATS = ("10x", "h5ad", "csv", "visium", "xenium")
_TABLE_SUFFIXES = (".csv", ".tsv", ".txt", ".tab", ".csv.gz", ".tsv.gz", ".txt.gz", ".tab.gz")

__all__ = ["load", "detect_format", "finalize", "standardize_in_memory", "FORMATS"]


def detect_format(path: str | Path) -> str:
    """Infer the input format from a file name or folder layout."""
    p = Path(path)
    if not p.exists():
        raise InputFormatError(f"Input path does not exist: {p}")
    if p.is_file():
        name = p.name.lower()
        if name.endswith(".h5ad"):
            return "h5ad"
        if name == "experiment.xenium":
            return "xenium"
        if name.endswith(".h5") or name.endswith((".mtx", ".mtx.gz")):
            return "10x"
        if name.endswith(_TABLE_SUFFIXES):
            return "csv"
        raise InputFormatError(
            f"Unrecognised file type: {p.name}. Supported files: .h5ad, 10x .h5, matrix.mtx(.gz), "
            ".csv/.tsv/.txt (optionally .gz). Folders: 10x matrix, Visium outs/, Xenium output."
        )
    if is_xenium_dir(p):
        return "xenium"
    if (p / "spatial").is_dir() or (p / "outs" / "spatial").is_dir():
        return "visium"
    if resolve_mtx_dir(p) is not None:
        return "10x"
    for base in (p, p / "outs"):
        if (base / "filtered_feature_bc_matrix.h5").is_file():
            return "10x"
        if any(resolve_mtx_dir(base / n) is not None for n in ("filtered_feature_bc_matrix", "filtered_gene_bc_matrices")):
            return "10x"
        if any((base / n).exists() for n in ("raw_feature_bc_matrix", "raw_feature_bc_matrix.h5", "raw_gene_bc_matrices")):
            return "10x"  # load_10x raises a specific error about raw matrices
    present = sorted(q.name for q in p.iterdir())[:20]
    raise InputFormatError(
        f"Could not recognise the contents of {p}. Expected one of:\n"
        "  - a 10x matrix folder (matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz)\n"
        "  - a Cell Ranger outs/ folder with filtered_feature_bc_matrix(.h5)\n"
        "  - a Visium outs/ folder with spatial/ and filtered_feature_bc_matrix(.h5)\n"
        "  - a Xenium output folder with cell_feature_matrix.h5 and cells.parquet/cells.csv.gz\n"
        f"Found: {present or '(empty directory)'}"
    )


def load(
    path: str | Path,
    fmt: str = "auto",
    *,
    sample_name: str | None = None,
    csv_orientation: str = "auto",
) -> ad.AnnData:
    """Load ``path`` into a standardised AnnData (see :mod:`cellscribe.loaders._common`)."""
    if fmt == "auto":
        fmt = detect_format(path)
    if fmt == "10x":
        return load_10x(path, sample_name=sample_name)
    if fmt == "h5ad":
        return load_h5ad(path, sample_name=sample_name)
    if fmt == "csv":
        return load_csv(path, orientation=csv_orientation, sample_name=sample_name)
    if fmt == "visium":
        return load_visium(path, sample_name=sample_name)
    if fmt == "xenium":
        return load_xenium(path, sample_name=sample_name)
    raise InputFormatError(f"Unknown format {fmt!r}; choose from auto, {', '.join(FORMATS)}.")
