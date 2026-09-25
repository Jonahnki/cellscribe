"""Neighbour graph, Leiden clustering with automatic resolution, and UMAP.

Automatic resolution selection
------------------------------
For each resolution in a small grid (default 0.4-1.2) Leiden is run
``stability_runs`` times with different random seeds. Two quantities are
computed:

* **stability**: mean pairwise adjusted Rand index (ARI) between the runs.
  Partitions that change with the seed are not well supported by the graph.
* **silhouette**: mean silhouette width of the partition in the PCA space
  used to build the graph (on a fixed random subsample of cells).

The selected resolution maximises ``stability x (silhouette + 1) / 2`` (both
terms in [0, 1]), i.e. it favours well-separated clusterings that are also
reproducible. A resolution that yields a single cluster is scored with a
silhouette of 0 (the neutral "no separation" value), so homogeneous samples
are not forced into unstable, noise-driven splits: a split must be both
reproducible and positively separated to win. Ties are broken towards the
higher resolution. The full sweep table is shown in the report, and
``--resolution`` overrides the choice.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import combinations

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

from cellscribe.config import ClusteringConfig
from cellscribe.utils import CLUSTER_KEY, logger


@dataclass
class ClusteringResult:
    resolution: float
    auto_selected: bool
    n_clusters: int
    n_neighbors: int
    n_pcs: int
    use_rep: str
    sweep: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _leiden(adata: ad.AnnData, resolution: float, seed: int, key: str) -> np.ndarray:
    import scanpy as sc

    sc.tl.leiden(
        adata,
        resolution=resolution,
        random_state=seed,
        key_added=key,
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )
    return adata.obs[key].astype(int).to_numpy()


def _relabel_by_size(labels: np.ndarray) -> np.ndarray:
    """Renumber clusters 0..k-1 by decreasing size (deterministic tie-break on old id)."""
    ids, counts = np.unique(labels, return_counts=True)
    order = sorted(range(len(ids)), key=lambda i: (-counts[i], ids[i]))
    mapping = {ids[i]: rank for rank, i in enumerate(order)}
    return np.array([mapping[x] for x in labels])


def resolution_sweep(adata: ad.AnnData, cfg: ClusteringConfig, rep: np.ndarray, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    n = adata.n_obs
    sub = rng.choice(n, size=min(cfg.silhouette_sample_size, n), replace=False)
    rows = []
    tmp_key = "_cellscribe_sweep"
    for res in cfg.resolution_grid:
        runs = [_leiden(adata, res, seed + k, tmp_key) for k in range(cfg.stability_runs)]
        aris = [adjusted_rand_score(a, b) for a, b in combinations(runs, 2)]
        stability = float(np.mean(aris)) if aris else 1.0
        labels = runs[0]
        k = len(np.unique(labels))
        sil = float("nan")
        if 1 < len(np.unique(labels[sub])) < len(sub):
            sil = float(silhouette_score(rep[sub], labels[sub], metric="euclidean"))
        if k == 1:
            score = stability * 0.5  # single cluster: neutral silhouette of 0
        else:
            score = stability * (sil + 1) / 2 if np.isfinite(sil) else 0.0
        rows.append(
            {"resolution": float(res), "n_clusters": int(k), "stability_ari": stability, "silhouette": sil, "score": float(score)}
        )
        logger.info("  resolution %.2f: %d clusters, stability %.3f, silhouette %.3f", res, k, stability, sil)
    del adata.obs[tmp_key]
    return rows


def choose_resolution(sweep: list[dict]) -> float:
    best = max(sweep, key=lambda r: (round(r["score"], 3), r["resolution"]))
    return best["resolution"]


def run(adata: ad.AnnData, cfg: ClusteringConfig, use_rep: str, n_pcs: int, seed: int = 0) -> ClusteringResult:
    import scanpy as sc

    rep_full = adata.obsm[use_rep]
    n_pcs = int(min(n_pcs, rep_full.shape[1]))
    n_neighbors = int(min(cfg.n_neighbors, adata.n_obs - 1))
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=use_rep, random_state=seed)
    rep = np.asarray(rep_full[:, :n_pcs])

    sweep: list[dict] = []
    if cfg.resolution is None:
        logger.info("Selecting Leiden resolution (grid %s)", cfg.resolution_grid)
        sweep = resolution_sweep(adata, cfg, rep, seed)
        resolution, auto = choose_resolution(sweep), True
    else:
        resolution, auto = float(cfg.resolution), False

    labels = _relabel_by_size(_leiden(adata, resolution, seed, CLUSTER_KEY))
    cats = [str(i) for i in range(labels.max() + 1)]
    adata.obs[CLUSTER_KEY] = pd.Categorical(labels.astype(str), categories=cats)
    sc.tl.umap(adata, random_state=seed)
    return ClusteringResult(
        resolution=resolution,
        auto_selected=auto,
        n_clusters=len(cats),
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        use_rep=use_rep,
        sweep=sweep,
    )
