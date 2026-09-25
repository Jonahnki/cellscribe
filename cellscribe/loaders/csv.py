"""Plain-text counts tables (CSV/TSV, optionally gzipped).

Orientation (genes x cells vs cells x genes) is inferred from the row and
column labels: known gene symbols / Ensembl IDs on one axis, and barcode-like
labels on the other. When the labels are uninformative the larger dimension
is assumed to be genes, and a warning is recorded in the report.
"""

from __future__ import annotations

import csv
import gzip
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellscribe.errors import InputFormatError
from cellscribe.loaders._common import finalize
from cellscribe.utils import DISSOCIATION_GENES, is_mito, is_ribo, logger

_BARCODE_RE = re.compile(r"([ACGTN]{10,})")
_ENSEMBL_RE = re.compile(r"^ENS[A-Z]*G\d{6,}")


def _known_genes() -> set[str]:
    from cellscribe.markers import load_reference

    genes = set(DISSOCIATION_GENES)
    for species in ("human", "mouse"):
        genes.update(g.upper() for g in load_reference(species).all_genes())
    return genes


def _label_evidence(labels, known: set[str]) -> tuple[int, int]:
    labels = [str(x) for x in labels]
    sample = labels if len(labels) <= 50_000 else labels[:50_000]
    upper = [x.upper() for x in sample]
    gene_hits = sum(u in known for u in upper)
    gene_hits += int(is_mito(sample).sum() + is_ribo(sample).sum())
    gene_hits += sum(bool(_ENSEMBL_RE.match(x)) for x in sample)
    barcode_hits = sum(bool(_BARCODE_RE.search(x)) for x in sample)
    return gene_hits, barcode_hits


def infer_orientation(index, columns) -> tuple[str, bool, str]:
    """Return ``(orientation, confident, reason)``."""
    known = _known_genes()
    row_genes, row_bc = _label_evidence(index, known)
    col_genes, col_bc = _label_evidence(columns, known)
    if row_genes >= 5 and row_genes > 5 * col_genes:
        return "genes_x_cells", True, f"{row_genes} row labels are recognised gene identifiers"
    if col_genes >= 5 and col_genes > 5 * row_genes:
        return "cells_x_genes", True, f"{col_genes} column labels are recognised gene identifiers"
    if col_bc > 0.5 * len(columns) and row_bc < 0.1 * len(index):
        return "genes_x_cells", True, "column labels look like cell barcodes"
    if row_bc > 0.5 * len(index) and col_bc < 0.1 * len(columns):
        return "cells_x_genes", True, "row labels look like cell barcodes"
    if len(index) >= len(columns):
        return "genes_x_cells", False, (
            "row/column labels did not identify genes or barcodes; assumed genes x cells because there are "
            f"more rows ({len(index)}) than columns ({len(columns)})"
        )
    return "cells_x_genes", False, (
        "row/column labels did not identify genes or barcodes; assumed cells x genes because there are "
        f"more columns ({len(columns)}) than rows ({len(index)})"
    )


def _sniff_sep(path: Path) -> str:
    name = path.name.lower()
    if name.endswith((".tsv", ".tsv.gz", ".tab", ".tab.gz")):
        return "\t"
    if name.endswith((".csv", ".csv.gz")):
        return ","
    opener = gzip.open if name.endswith(".gz") else open
    with opener(path, "rt") as fh:
        head = fh.read(20_000)
    try:
        return csv.Sniffer().sniff(head, delimiters=",\t; ").delimiter
    except csv.Error:
        return "\t" if head.count("\t") > head.count(",") else ","


def load_csv(path: str | Path, orientation: str = "auto", sample_name: str | None = None) -> ad.AnnData:
    p = Path(path)
    if not p.is_file():
        raise InputFormatError(f"Counts table not found: {p}")
    sep = _sniff_sep(p)
    try:
        df = pd.read_csv(p, sep=sep, index_col=0)
    except Exception as exc:  # noqa: BLE001
        raise InputFormatError(f"Could not parse {p} as a delimited counts table: {exc}") from exc
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise InputFormatError(f"{p} parsed as an empty table ({df.shape[0]} rows x {df.shape[1]} columns).")

    notes: list[str] = []
    symbols = None
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        leading = list(df.columns[: len(non_numeric)])
        if non_numeric == leading and len(non_numeric) <= 2:
            # e.g. "gene_id,gene_name,cell1,cell2,..." - use the last text column as symbols
            symbols = df[non_numeric[-1]].astype(str).to_numpy()
            df = df.drop(columns=non_numeric)
            notes.append(f"Text column(s) {non_numeric} were treated as gene annotations, not cells.")
        else:
            raise InputFormatError(
                f"{p} has non-numeric data columns: {non_numeric[:5]}{'...' if len(non_numeric) > 5 else ''}. "
                "Expected a numeric counts table with gene and cell labels as the first column / header row."
            )
    if df.isna().any().any():
        raise InputFormatError(f"{p} contains missing values; counts tables must be complete.")

    if orientation == "auto":
        orientation, confident, reason = infer_orientation(df.index, df.columns)
        msg = f"Table orientation inferred as {orientation.replace('_', ' ')}: {reason}."
        if confident:
            logger.info(msg)
        else:
            notes.append(
                "AMBIGUOUS ORIENTATION: " + msg + " If this is wrong, re-run with --csv-orientation "
                "(genes_x_cells or cells_x_genes)."
            )
    elif orientation not in {"genes_x_cells", "cells_x_genes"}:
        raise InputFormatError(f"Unknown CSV orientation {orientation!r}.")

    values = df.to_numpy(dtype=np.float32)
    if orientation == "genes_x_cells":
        X = sp.csr_matrix(values.T)
        genes, cells = df.index.astype(str), df.columns.astype(str)
        if symbols is not None:
            var = pd.DataFrame({"gene_ids": np.asarray(genes)}, index=pd.Index(symbols))
        else:
            var = pd.DataFrame(index=genes)
    else:
        if symbols is not None:
            raise InputFormatError("Gene-annotation text columns are only supported for genes x cells tables.")
        X = sp.csr_matrix(values)
        genes, cells = df.columns.astype(str), df.index.astype(str)
        var = pd.DataFrame(index=genes)
    adata = ad.AnnData(X=X, obs=pd.DataFrame(index=cells), var=var)
    return finalize(
        adata, fmt="csv", modality="single-cell", path=p, notes=notes, sample_name=sample_name,
        extra={"csv_orientation": orientation}, check_orientation=False,
    )
