"""Spatial statistics for data with coordinates (Visium, Xenium, or h5ad with ``obsm['spatial']``).

* Spatial neighbour graph (squidpy): hexagonal grid adjacency for Visium
  spots (6 neighbours); Delaunay triangulation for segmented cells, pruned
  of edges longer than 5x the median nearest-neighbour distance so that
  long edges across tissue gaps and borders are removed.
* Neighbourhood enrichment (Palla et al., Nat Methods 2022): permutation
  z-scores of how often cells of two clusters are graph neighbours compared
  with random label shuffles. Positive = co-localised, negative = segregated.
* Spatial autocorrelation: Moran's I for the top highly variable genes, with
  analytic p-values and Benjamini-Hochberg correction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import anndata as ad
import numpy as np

from cellscribe.config import SpatialConfig
from cellscribe.utils import CLUSTER_KEY, logger


@dataclass
class SpatialResult:
    enabled: bool
    reason: str = ""
    modality: str = ""
    coord_units: str = ""
    graph: dict = field(default_factory=dict)
    median_nn_distance: float | None = None
    nhood_clusters: list[str] = field(default_factory=list)
    nhood_zscore: list[list[float]] = field(default_factory=list)
    nhood_n_perms: int = 0
    colocalised_pairs: list[dict] = field(default_factory=list)
    segregated_pairs: list[dict] = field(default_factory=list)
    self_enrichment: list[dict] = field(default_factory=list)
    moran_top: list[dict] = field(default_factory=list)
    n_genes_autocorr: int = 0
    n_genes_significant: int = 0
    has_image: bool = False
    library_id: str | None = None
    notes: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def has_spatial(adata: ad.AnnData) -> bool:
    return "spatial" in adata.obsm and np.asarray(adata.obsm["spatial"]).shape[0] == adata.n_obs


def tissue_image(adata: ad.AnnData) -> tuple[np.ndarray | None, float, str | None]:
    """Return ``(image, scalefactor, library_id)`` for the best available tissue image."""
    spatial = adata.uns.get("spatial")
    if not isinstance(spatial, dict) or not spatial:
        return None, 1.0, None
    lib = next(iter(spatial))
    info = spatial[lib] if isinstance(spatial[lib], dict) else {}
    images = info.get("images", {}) or {}
    sf = info.get("scalefactors", {}) or {}
    for key in ("hires", "lowres"):
        if key in images and f"tissue_{key}_scalef" in sf:
            img = np.asarray(images[key])
            if img.dtype != np.uint8:
                img = np.clip(img * (255 if img.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
            return img, float(sf[f"tissue_{key}_scalef"]), lib
    return None, 1.0, lib


def _median_nn_distance(coords: np.ndarray) -> float:
    from sklearn.neighbors import NearestNeighbors

    n = min(len(coords), 20_000)
    idx = np.random.default_rng(0).choice(len(coords), n, replace=False) if len(coords) > n else np.arange(len(coords))
    nn = NearestNeighbors(n_neighbors=2).fit(coords)
    d, _ = nn.kneighbors(coords[idx])
    return float(np.median(d[:, 1]))


def run(
    adata: ad.AnnData,
    cfg: SpatialConfig,
    modality: str,
    cluster_key: str = CLUSTER_KEY,
    seed: int = 0,
) -> SpatialResult:
    if cfg.enabled == "off":
        return SpatialResult(enabled=False, reason="Spatial analysis disabled in the configuration.")
    if not has_spatial(adata):
        return SpatialResult(enabled=False, reason="No spatial coordinates in the input.")
    try:
        import squidpy as sq
    except ImportError as exc:  # pragma: no cover - squidpy is a core dependency
        return SpatialResult(enabled=False, reason=f"squidpy is not available ({exc}); spatial statistics skipped.")
    import logging

    for name in ("spatialdata", "spatialdata._logging"):  # squidpy logs routine INFO messages through spatialdata
        sd_log = logging.getLogger(name)
        sd_log.setLevel(max(sd_log.level, logging.WARNING))

    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    res = SpatialResult(enabled=True, modality=modality)
    res.coord_units = {
        "visium": "full-resolution image pixels",
        "xenium": "µm",
    }.get(modality, "input coordinate units")
    res.median_nn_distance = _median_nn_distance(coords)
    img, _, lib = tissue_image(adata)
    res.has_image, res.library_id = img is not None, lib

    if modality == "visium":
        sq.gr.spatial_neighbors(adata, coord_type="grid", n_neighs=6)
        res.graph = {"method": "hexagonal grid (Visium)", "n_neighs": 6}
        res.notes.append(
            {
                "severity": "info",
                "message": "Visium spots (55 µm) usually contain several cells, so each cluster describes the dominant "
                "expression signal of a tissue region rather than a single cell type.",
            }
        )
    else:
        r_max = 5.0 * res.median_nn_distance if res.median_nn_distance > 0 else None
        kwargs = {"radius": (0.0, r_max)} if r_max else {}
        try:
            sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True, **kwargs)
            res.graph = {"method": "Delaunay triangulation", "max_edge_length": r_max}
        except Exception as exc:  # noqa: BLE001 - e.g. Qhull failure on collinear/degenerate coordinates
            logger.warning("Delaunay triangulation failed (%s); using a 6-nearest-neighbour graph", type(exc).__name__)
            sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=6)
            res.graph = {"method": "6 nearest neighbours (Delaunay failed on these coordinates)", "n_neighs": 6}
            res.notes.append(
                {
                    "severity": "caution",
                    "message": "Delaunay triangulation of the coordinates failed (they may be collinear or degenerate), "
                    "so a 6-nearest-neighbour graph was used. Check that the spatial coordinates are correct.",
                }
            )
    conn = adata.obsp["spatial_connectivities"]
    res.graph["mean_degree"] = float(conn.sum() / adata.n_obs)

    # Neighbourhood enrichment
    clusters = [str(c) for c in adata.obs[cluster_key].cat.categories]
    if len(clusters) >= 2:
        sq.gr.nhood_enrichment(
            adata, cluster_key=cluster_key, n_perms=cfg.n_perms, seed=seed, show_progress_bar=False, n_jobs=1
        )
        z = np.nan_to_num(np.asarray(adata.uns[f"{cluster_key}_nhood_enrichment"]["zscore"], dtype=float))
        res.nhood_clusters, res.nhood_zscore, res.nhood_n_perms = clusters, z.round(3).tolist(), cfg.n_perms
        pairs = [
            {"a": clusters[i], "b": clusters[j], "z": float(z[i, j])}
            for i in range(len(clusters))
            for j in range(i + 1, len(clusters))
        ]
        res.colocalised_pairs = sorted([p for p in pairs if p["z"] > 2], key=lambda p: -p["z"])[:8]
        res.segregated_pairs = sorted([p for p in pairs if p["z"] < -2], key=lambda p: p["z"])[:8]
        res.self_enrichment = [{"cluster": c, "z": float(z[i, i])} for i, c in enumerate(clusters)]

    # Moran's I on top HVGs
    if "highly_variable" in adata.var and not adata.var["highly_variable"].all() and "dispersions_norm" in adata.var:
        order = adata.var["dispersions_norm"].fillna(-np.inf).sort_values(ascending=False)
        genes = order.index[: cfg.n_genes_autocorr].tolist()
    else:
        detected = np.asarray((adata.layers["counts"] > 0).sum(axis=0)).ravel()
        genes = adata.var_names[np.argsort(-detected)][: cfg.n_genes_autocorr].tolist()
    sq.gr.spatial_autocorr(
        adata, mode="moran", genes=genes, n_perms=None, show_progress_bar=False, seed=seed, n_jobs=1
    )
    moran = adata.uns["moranI"].copy()
    padj_col = next((c for c in moran.columns if c.startswith("pval_norm_fdr")), None)
    moran["padj"] = moran[padj_col] if padj_col else moran.get("pval_norm", np.nan)
    moran = moran.sort_values("I", ascending=False)
    res.n_genes_autocorr = int(len(moran))
    res.n_genes_significant = int((moran["padj"] < 0.05).sum())
    res.moran_top = [
        {"gene": str(g), "I": float(r["I"]), "padj": float(r["padj"])} for g, r in moran.head(20).iterrows()
    ]
    logger.info(
        "Spatial: %d clusters, %d co-localised pairs, %d/%d genes spatially autocorrelated (FDR<0.05)",
        len(clusters), len(res.colocalised_pairs), res.n_genes_significant, res.n_genes_autocorr,
    )
    return res
