"""Normalisation, feature selection, PCA, and optional batch integration.

``adata.X`` becomes log-normalised expression for all genes (used for marker
detection and plots); raw counts stay in ``adata.layers['counts']``. Scaling
and PCA are performed on a copy restricted to highly variable genes, so the
full log-normalised matrix is never densified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import anndata as ad
import numpy as np
import scipy.sparse as sp

from cellscribe.config import PreprocessingConfig
from cellscribe.errors import ConfigError, PipelineError
from cellscribe.utils import logger


@dataclass
class PreprocessingResult:
    target_sum: float
    n_hvg: int
    hvg_flavor: str
    used_all_genes: bool
    n_pcs_computed: int
    n_pcs_used: int
    variance_explained: list[float]
    batch_key: str | None
    n_batches: int
    integration: str  # "none" | "harmony"
    use_rep: str

    def to_dict(self) -> dict:
        return asdict(self)


def normalize(adata: ad.AnnData, target_sum: float) -> None:
    import scanpy as sc

    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)


def select_hvgs(adata: ad.AnnData, n_top_genes: int, batch_key: str | None) -> tuple[int, bool, str]:
    """Flag highly variable genes. Small targeted panels use all genes."""
    import scanpy as sc

    if adata.n_vars < 2 * n_top_genes:
        adata.var["highly_variable"] = True
        return adata.n_vars, True, "all genes (panel smaller than 2x n_top_genes)"
    kwargs = {"batch_key": batch_key} if batch_key else {}
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat", **kwargs)
    return int(adata.var["highly_variable"].sum()), False, "seurat (log-normalised dispersion)"


def run_pca(adata: ad.AnnData, n_pcs: int, seed: int) -> tuple[int, list[float]]:
    import scanpy as sc

    hv = adata[:, adata.var["highly_variable"].to_numpy()].copy()
    X = hv.X.astype(np.float64)
    mean = np.asarray(X.mean(axis=0)).ravel()
    mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel() if sp.issparse(X) else (np.asarray(X) ** 2).mean(axis=0)
    variable = (mean_sq - mean**2) > 1e-10 * (1.0 + mean**2)
    if variable.sum() < 2:
        raise PipelineError(
            "The cells show essentially no variation in gene expression, so they cannot be clustered. "
            "Check that the input contains distinct cells."
        )
    if not variable.all():
        hv = hv[:, variable].copy()
    # scaling densifies the (HVG-only) matrix anyway; do it explicitly in float32 to halve peak memory
    hv.X = hv.X.toarray().astype(np.float32) if sp.issparse(hv.X) else np.asarray(hv.X, dtype=np.float32)
    sc.pp.scale(hv, max_value=10)
    hv.X = np.nan_to_num(np.asarray(hv.X), copy=False)
    if not np.any(hv.X):
        raise PipelineError("After scaling, no gene varies between cells; the cells cannot be clustered.")
    n_comps = int(min(50, hv.n_obs - 1, hv.n_vars - 1))
    if n_comps < 2:
        raise PipelineError("Too few cells or variable genes to compute a PCA.")
    sc.pp.pca(hv, n_comps=n_comps, svd_solver="arpack", random_state=seed)
    adata.obsm["X_pca"] = hv.obsm["X_pca"]
    adata.uns["pca"] = hv.uns["pca"]
    ratio = hv.uns["pca"]["variance_ratio"]
    return n_comps, [float(x) for x in ratio]


def run_harmony(adata: ad.AnnData, batch_key: str, n_pcs: int, seed: int) -> None:
    try:
        import harmonypy
    except ImportError as exc:
        raise ConfigError(
            "Harmony integration was requested but harmonypy is not installed. "
            "Install it with: pip install 'cellscribe[harmony]'"
        ) from exc
    pcs = adata.obsm["X_pca"][:, :n_pcs]
    meta = adata.obs[[batch_key]].astype(str)
    try:
        ho = harmonypy.run_harmony(pcs, meta, [batch_key], random_state=seed, verbose=False)
    except TypeError:  # older harmonypy without random_state/verbose
        np.random.seed(seed)
        ho = harmonypy.run_harmony(pcs, meta, [batch_key])
    Z = np.asarray(ho.Z_corr)
    if Z.shape[0] != adata.n_obs:  # harmonypy < 2 returns PCs x cells
        Z = Z.T
    adata.obsm["X_pca_harmony"] = Z.astype(np.float32)


def run(adata: ad.AnnData, cfg: PreprocessingConfig, batch_key: str | None, seed: int = 0) -> PreprocessingResult:
    normalize(adata, cfg.target_sum)
    n_hvg, used_all, flavor = select_hvgs(adata, cfg.n_top_genes, batch_key)
    n_comps, ratio = run_pca(adata, cfg.n_pcs, seed)
    n_pcs_used = int(min(cfg.n_pcs, n_comps))
    n_batches = int(adata.obs[batch_key].nunique()) if batch_key else 1
    integration, use_rep = "none", "X_pca"
    if cfg.integrate == "harmony":
        if batch_key is None:
            logger.warning("Harmony integration requested but only one batch was found; skipping integration.")
        else:
            logger.info("Running Harmony integration over '%s' (%d batches)", batch_key, n_batches)
            run_harmony(adata, batch_key, n_pcs_used, seed)
            integration, use_rep = "harmony", "X_pca_harmony"
    return PreprocessingResult(
        target_sum=cfg.target_sum,
        n_hvg=n_hvg,
        hvg_flavor=flavor,
        used_all_genes=used_all,
        n_pcs_computed=n_comps,
        n_pcs_used=n_pcs_used,
        variance_explained=ratio,
        batch_key=batch_key,
        n_batches=n_batches,
        integration=integration,
        use_rep=use_rep,
    )
