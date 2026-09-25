"""10x Genomics Cell Ranger feature-barcode matrices (mtx folder or HDF5)."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

from cellscribe.errors import InputFormatError
from cellscribe.loaders._common import finalize

_MTX_NAMES = ("matrix.mtx.gz", "matrix.mtx")
_BARCODE_NAMES = ("barcodes.tsv.gz", "barcodes.tsv")
_FEATURE_NAMES = ("features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv")


def _first(d: Path, names) -> Path | None:
    for n in names:
        if (d / n).is_file():
            return d / n
    return None


def is_mtx_dir(d: Path) -> bool:
    return d.is_dir() and _first(d, _MTX_NAMES) is not None


def resolve_mtx_dir(d: Path) -> Path | None:
    """Return the matrix folder for ``d``, descending into a single genome subfolder.

    Cell Ranger 2.x writes ``filtered_gene_bc_matrices/<genome>/matrix.mtx``;
    users commonly point at the parent folder.
    """
    if is_mtx_dir(d):
        return d
    if not d.is_dir():
        return None
    subs = [q for q in d.iterdir() if is_mtx_dir(q)]
    if len(subs) == 1:
        return subs[0]
    if len(subs) > 1:
        raise InputFormatError(
            f"{d} contains matrices for several genomes ({', '.join(sorted(q.name for q in subs))}). "
            "Point Cellscribe at one genome's folder."
        )
    return None


def read_mtx_dir(d: Path, gex_only: bool = True) -> tuple[ad.AnnData, list[str]]:
    """Read a Cell Ranger mtx folder (v2 ``genes.tsv`` or v3 ``features.tsv.gz``)."""
    notes: list[str] = []
    mtx = _first(d, _MTX_NAMES)
    bc = _first(d, _BARCODE_NAMES)
    ft = _first(d, _FEATURE_NAMES)
    missing = [
        label
        for label, found in (("matrix.mtx[.gz]", mtx), ("barcodes.tsv[.gz]", bc), ("features.tsv[.gz] or genes.tsv", ft))
        if found is None
    ]
    if missing:
        present = sorted(p.name for p in d.iterdir())
        raise InputFormatError(
            f"{d} looks like a 10x matrix folder but is missing: {', '.join(missing)}. Present: {present}"
        )
    try:
        M = scipy.io.mmread(str(mtx))
    except Exception as exc:  # noqa: BLE001 - surface any parse error with context
        raise InputFormatError(f"Could not parse Matrix Market file {mtx}: {exc}") from exc
    X = sp.csr_matrix(M).T.tocsr().astype(np.float32)  # file is genes x cells
    barcodes = pd.read_csv(bc, header=None, sep="\t", dtype=str)[0].to_numpy()
    feats = pd.read_csv(ft, header=None, sep="\t", dtype=str)
    if X.shape != (len(barcodes), len(feats)):
        raise InputFormatError(
            f"Inconsistent 10x folder {d}: matrix is {X.shape[1]} features x {X.shape[0]} barcodes, but "
            f"{ft.name} lists {len(feats)} features and {bc.name} lists {len(barcodes)} barcodes."
        )
    ids = feats[0].to_numpy()
    names = feats[1].to_numpy() if feats.shape[1] > 1 else ids
    types = feats[2].to_numpy() if feats.shape[1] > 2 else np.array(["Gene Expression"] * len(ids))
    var = pd.DataFrame({"gene_ids": ids, "feature_types": types}, index=pd.Index(names.astype(str)))
    adata = ad.AnnData(X=X, obs=pd.DataFrame(index=pd.Index(barcodes)), var=var)
    if gex_only:
        adata, note = _keep_gene_expression(adata)
        if note:
            notes.append(note)
    return adata, notes


def _keep_gene_expression(adata: ad.AnnData) -> tuple[ad.AnnData, str | None]:
    if "feature_types" not in adata.var:
        return adata, None
    gex = adata.var["feature_types"].astype(str) == "Gene Expression"
    if gex.all():
        return adata, None
    if not gex.any():
        raise InputFormatError("The matrix contains no 'Gene Expression' features.")
    dropped = adata.var.loc[~gex, "feature_types"].value_counts().to_dict()
    desc = ", ".join(f"{v} {k}" for k, v in dropped.items())
    return adata[:, gex.to_numpy()].copy(), f"Non-gene-expression features were excluded ({desc})."


def read_10x_h5(path: Path, gex_only: bool = True) -> tuple[ad.AnnData, list[str]]:
    """Read a Cell Ranger HDF5 matrix (v3 ``/matrix`` layout or v2 per-genome groups)."""
    import scanpy as sc

    try:
        with h5py.File(path, "r") as f:
            keys = set(f.keys())
    except OSError as exc:
        raise InputFormatError(f"{path} is not a readable HDF5 file: {exc}") from exc
    if {"X", "obs", "var"} <= keys:
        raise InputFormatError(
            f"{path} is an AnnData file saved with a .h5 extension; rename it to .h5ad or pass --format h5ad."
        )
    if "matrix" not in keys and not keys:
        raise InputFormatError(f"{path} is an empty HDF5 file.")
    try:
        adata = sc.read_10x_h5(str(path), gex_only=False)
    except Exception as exc:  # noqa: BLE001
        raise InputFormatError(
            f"{path} is not a 10x Genomics feature-barcode HDF5 matrix (top-level groups: {sorted(keys)}): {exc}"
        ) from exc
    adata.var_names_make_unique()
    notes: list[str] = []
    if gex_only:
        adata, note = _keep_gene_expression(adata)
        if note:
            notes.append(note)
    return adata, notes


def load_10x(path: str | Path, sample_name: str | None = None) -> ad.AnnData:
    """Load a Cell Ranger ``filtered_feature_bc_matrix`` (folder or ``.h5``) or its ``outs/`` folder."""
    p = Path(path)
    if not p.exists():
        raise InputFormatError(f"Input path does not exist: {p}")
    if p.is_file():
        name = p.name.lower()
        if name.endswith(".h5"):
            adata, notes = read_10x_h5(p)
        elif name.endswith((".mtx", ".mtx.gz")):
            adata, notes = read_mtx_dir(p.parent)
        else:
            raise InputFormatError(f"{p} is not a 10x matrix file (.h5, matrix.mtx[.gz]) or folder.")
        return finalize(adata, fmt="10x", modality="single-cell", path=p, notes=notes, sample_name=sample_name)

    mtx_dir = resolve_mtx_dir(p)
    if mtx_dir is not None:
        adata, notes = read_mtx_dir(mtx_dir)
        if mtx_dir != p:
            notes.append(f"Using the matrix in genome subfolder '{mtx_dir.name}'.")
        return finalize(adata, fmt="10x", modality="single-cell", path=p, notes=notes, sample_name=sample_name)
    for base in (p, p / "outs"):
        if (base / "filtered_feature_bc_matrix.h5").is_file():
            adata, notes = read_10x_h5(base / "filtered_feature_bc_matrix.h5")
            return finalize(adata, fmt="10x", modality="single-cell", path=p, notes=notes, sample_name=sample_name)
        for name in ("filtered_feature_bc_matrix", "filtered_gene_bc_matrices"):
            mtx_dir = resolve_mtx_dir(base / name)
            if mtx_dir is not None:
                adata, notes = read_mtx_dir(mtx_dir)
                return finalize(adata, fmt="10x", modality="single-cell", path=p, notes=notes, sample_name=sample_name)
    raw_names = ("raw_feature_bc_matrix", "raw_feature_bc_matrix.h5", "raw_gene_bc_matrices", "raw_gene_bc_matrices_h5.h5")
    if any((base / n).exists() for base in (p, p / "outs") for n in raw_names):
        raise InputFormatError(
            f"{p} only contains a raw (unfiltered) matrix. Raw matrices are dominated by empty droplets; "
            "point Cellscribe at filtered_feature_bc_matrix instead."
        )
    present = sorted(q.name for q in p.iterdir())[:20]
    raise InputFormatError(
        f"{p} is not a 10x matrix folder. Expected matrix.mtx(.gz) + barcodes.tsv(.gz) + features.tsv(.gz), "
        f"or a filtered_feature_bc_matrix(.h5) inside it. Found: {present or '(empty directory)'}"
    )
