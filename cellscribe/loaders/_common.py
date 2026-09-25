"""Shared normalisation of loaded data into Cellscribe's AnnData contract.

Every loader returns an AnnData that satisfies:

* ``.X`` and ``.layers['counts']``: raw counts, float32 CSR, cells x genes
* ``.obs_names`` / ``.var_names``: unique strings; var_names are gene symbols
  where available (original IDs kept in ``.var['gene_ids']``)
* ``.obs['sample']``: sample label (existing column kept if present)
* ``.obsm['spatial']`` (spatial formats only): (n_cells, 2) float coordinates
* ``.uns['cellscribe_input']``: provenance (format, modality, path, notes)
"""

from __future__ import annotations

import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellscribe.errors import InputFormatError
from cellscribe.utils import CLUSTER_KEY, LABEL_KEY, is_integer_matrix, logger

_ENSEMBL_RE = re.compile(r"^ENS[A-Z]*G\d{6,}(\.\d+)?$")
_SYMBOL_COLUMNS = (
    "gene_symbols", "gene_symbol", "feature_name", "gene_name", "gene_names",
    "symbol", "symbols", "Symbol", "name", "gene",
)
_OWN_OBS_COLUMNS = (CLUSTER_KEY, LABEL_KEY, "qc_outlier", "qc_reasons", "doublet_score", "predicted_doublet")


def sample_name_from_path(path: str | Path) -> str:
    p = Path(path)
    generic = {
        "outs", "filtered_feature_bc_matrix", "raw_feature_bc_matrix", "cell_feature_matrix",
        "spatial", "count", "data", "output", "outputs",
    }
    name = p.name
    for suffix in (".gz", ".h5ad", ".h5", ".csv", ".tsv", ".txt", ".mtx"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    parts = [name] + [q.name for q in p.parents]
    for part in parts:
        if part and part.lower() not in generic and not part.lower().startswith("filtered_feature_bc_matrix"):
            return part
    return name or "sample"


def _swap_ensembl_for_symbols(adata: ad.AnnData, notes: list[str]) -> None:
    names = adata.var_names.astype(str)
    frac_ens = np.mean([bool(_ENSEMBL_RE.match(n)) for n in names[: min(2000, len(names))]]) if len(names) else 0
    if frac_ens < 0.5:
        return
    for col in _SYMBOL_COLUMNS:
        if col in adata.var.columns:
            symbols = adata.var[col].astype(str)
            valid = symbols.notna() & ~symbols.isin(["", "nan", "None"])
            if valid.mean() < 0.5:
                continue
            adata.var["gene_ids"] = names.to_numpy()
            new = np.where(valid, symbols, names)
            adata.var_names = pd.Index(new.astype(str))
            adata.var_names_make_unique()
            notes.append(f"Gene identifiers were Ensembl IDs; gene symbols were taken from var['{col}'].")
            return
    notes.append(
        "Gene identifiers are Ensembl IDs and no gene-symbol column was found. Mitochondrial/ribosomal "
        "gene detection and marker-based annotation require gene symbols, so they will be limited."
    )


def _check_not_transposed(adata: ad.AnnData, path) -> None:
    from cellscribe.loaders.csv import _known_genes, _label_evidence

    known = _known_genes()
    obs_genes, _ = _label_evidence(adata.obs_names[:20_000], known)
    var_genes, _ = _label_evidence(adata.var_names[:20_000], known)
    if obs_genes >= 20 and obs_genes > 10 * var_genes:
        raise InputFormatError(
            f"The matrix at {path} appears to be transposed: {obs_genes} of the cell labels are gene names "
            f"(e.g. {', '.join(map(str, adata.obs_names[:3]))}). Cellscribe expects cells as rows (observations) "
            "and genes as columns; for an AnnData object use adata.T, or for tables use --csv-orientation."
        )


def check_counts(X, context: str) -> str | None:
    """Validate that X holds raw counts. Returns a note for non-integer counts, raises if clearly not counts."""
    data = X.data if sp.issparse(X) else np.asarray(X)
    if data.size == 0:
        return None
    if np.nanmin(data) < 0:
        raise InputFormatError(
            f"{context} contains negative values, so it looks scaled or batch-corrected rather than raw counts. "
            "Cellscribe needs raw UMI/read counts (e.g. store them in adata.layers['counts'])."
        )
    if is_integer_matrix(X):
        return None
    if np.nanmax(data) < 50:
        raise InputFormatError(
            f"{context} contains small non-integer values (max {np.nanmax(data):.2f}), so it looks already "
            "normalised/log-transformed. Cellscribe needs raw counts: put them in adata.layers['counts'] "
            "or adata.raw, or provide the Cell Ranger matrix."
        )
    return (
        f"{context} contains non-integer values on a count-like scale (e.g. ambient-RNA-corrected counts). "
        "Proceeding, but QC metrics assume raw counts."
    )


def finalize(
    adata: ad.AnnData,
    *,
    fmt: str,
    modality: str,
    path: str | Path,
    notes: list[str] | None = None,
    sample_name: str | None = None,
    extra: dict | None = None,
    check_orientation: bool = True,
) -> ad.AnnData:
    """Coerce a freshly loaded AnnData into the Cellscribe contract (in place) and return it."""
    notes = list(notes or [])
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise InputFormatError(f"Input at {path} loaded as an empty matrix ({adata.n_obs} cells x {adata.n_vars} genes).")
    if check_orientation:
        _check_not_transposed(adata, path)

    X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(np.asarray(X))
    X = X.tocsr().astype(np.float32)
    X.eliminate_zeros()
    if X.nnz == 0:
        raise InputFormatError(f"Input at {path} contains no non-zero counts.")
    note = check_counts(X, "The count matrix")
    if note:
        notes.append(note)
    adata.X = X

    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)
    if not adata.obs_names.is_unique:
        adata.obs_names_make_unique()
        notes.append("Duplicate cell barcodes were made unique by appending suffixes.")
    _swap_ensembl_for_symbols(adata, notes)
    if not adata.var_names.is_unique:
        n_dup = int(adata.var_names.duplicated().sum())
        adata.var_names_make_unique()
        notes.append(f"{n_dup} duplicated gene symbols were made unique by appending suffixes (e.g. 'GENE-1').")

    for col in _OWN_OBS_COLUMNS:
        if col in adata.obs.columns:
            adata.obs = adata.obs.drop(columns=col)
    sample = sample_name or sample_name_from_path(path)
    if "sample" not in adata.obs.columns:
        adata.obs["sample"] = pd.Categorical([sample] * adata.n_obs)

    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise InputFormatError(f"obsm['spatial'] must be an (n_cells, 2) array; got shape {coords.shape}.")
        if not np.isfinite(coords[:, :2]).all():
            raise InputFormatError("Spatial coordinates contain missing or non-finite values.")
        adata.obsm["spatial"] = coords[:, :2]

    adata.layers["counts"] = adata.X.copy()
    info = {
        "format": fmt,
        "modality": modality,
        "path": str(path),
        "sample_name": sample,
        "n_cells_loaded": int(adata.n_obs),
        "n_genes_loaded": int(adata.n_vars),
        "notes": notes,
    }
    if extra:
        info.update(extra)
    adata.uns["cellscribe_input"] = info
    for n in notes:
        logger.warning(n)
    return adata


def standardize_in_memory(adata: ad.AnnData, sample_name: str | None = None) -> ad.AnnData:
    """Apply :func:`finalize` to an AnnData created in Python (e.g. synthetic data)."""
    info = dict(adata.uns.get("cellscribe_input", {}))
    modality = info.get("modality")
    if modality is None:
        modality = "spatial" if "spatial" in adata.obsm else "single-cell"
    extra = {k: v for k, v in info.items() if k not in {"format", "modality", "path", "sample_name", "notes"}}
    return finalize(
        adata,
        fmt=info.get("format", "anndata"),
        modality=modality,
        path=info.get("path", "in-memory AnnData"),
        notes=info.get("notes"),
        sample_name=sample_name or info.get("sample_name"),
        extra=extra,
    )
