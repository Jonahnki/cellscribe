"""AnnData ``.h5ad`` files.

Pre-processed h5ad files often keep normalised values in ``.X``; this loader
looks for raw counts in the usual places (``layers['counts']``,
``layers['raw_counts']``, ``.raw``) before falling back to ``.X``. Prior
analysis results (embeddings, graphs, clusterings) are dropped so the report
reflects Cellscribe's own processing; sample metadata in ``.obs`` and spatial
coordinates/images are kept.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd

from cellscribe.errors import InputFormatError
from cellscribe.loaders._common import finalize
from cellscribe.utils import is_integer_matrix

_COUNT_LAYERS = ("counts", "raw_counts", "count", "umi_counts", "UMIs")


def _select_counts(adata: ad.AnnData, notes: list[str]) -> tuple[object, pd.DataFrame]:
    for layer in _COUNT_LAYERS:
        if layer in adata.layers:
            if layer != "counts" or not is_integer_matrix(adata.X):
                notes.append(f"Raw counts were taken from layers['{layer}'].")
            return adata.layers[layer], adata.var
    if is_integer_matrix(adata.X):
        return adata.X, adata.var
    if adata.raw is not None and is_integer_matrix(adata.raw.X):
        notes.append(".X did not contain raw counts; raw counts were taken from .raw (all genes in .raw are used).")
        return adata.raw.X, adata.raw.var
    return adata.X, adata.var  # finalize() decides whether this is acceptable


def load_h5ad(path: str | Path, sample_name: str | None = None) -> ad.AnnData:
    p = Path(path)
    if not p.is_file():
        raise InputFormatError(f"h5ad file not found: {p}")
    try:
        src = ad.read_h5ad(p)
    except Exception as exc:  # noqa: BLE001
        raise InputFormatError(f"Could not read {p} as AnnData (.h5ad): {exc}") from exc
    notes: list[str] = []
    X, var = _select_counts(src, notes)
    var = var.copy()
    for col in ("highly_variable", "means", "dispersions", "dispersions_norm", "highly_variable_rank", "variances", "variances_norm"):
        if col in var.columns:
            var = var.drop(columns=col)
    adata = ad.AnnData(X=X, obs=src.obs.copy(), var=var)
    modality = "single-cell"
    extra: dict = {}
    if "spatial" in src.obsm:
        adata.obsm["spatial"] = src.obsm["spatial"]
        modality = "spatial"
        if "spatial" in src.uns and isinstance(src.uns["spatial"], dict) and src.uns["spatial"]:
            adata.uns["spatial"] = src.uns["spatial"]
            lib = next(iter(src.uns["spatial"]))
            sf = src.uns["spatial"][lib].get("scalefactors", {}) if isinstance(src.uns["spatial"][lib], dict) else {}
            if "spot_diameter_fullres" in sf:
                modality = "visium"
                extra["library_id"] = lib
    dropped = [k for k in src.obsm.keys() if k != "spatial"]
    if dropped:
        notes.append(f"Existing embeddings in the input ({', '.join(dropped)}) were ignored and recomputed.")
    return finalize(adata, fmt="h5ad", modality=modality, path=p, notes=notes, sample_name=sample_name, extra=extra)
