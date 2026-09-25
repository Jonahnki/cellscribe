import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from cellscribe import clustering, preprocessing, qc, synthetic
from cellscribe.config import ClusteringConfig, PreprocessingConfig, QCConfig
from cellscribe.errors import PipelineError
from cellscribe.loaders import standardize_in_memory
from cellscribe.pipeline import run
from cellscribe.utils import CLUSTER_KEY


def test_clustering_is_deterministic(sc_adata, sc_result, base_config):
    again = run(sc_adata, output=None, config=base_config)
    assert again.clustering.n_clusters == sc_result.clustering.n_clusters
    assert again.clustering.resolution == sc_result.clustering.resolution
    assert (again.adata.obs[CLUSTER_KEY].to_numpy() == sc_result.adata.obs[CLUSTER_KEY].to_numpy()).all()
    assert np.allclose(again.adata.obsm["X_umap"], sc_result.adata.obsm["X_umap"])


def test_expected_cluster_structure(sc_result):
    """On the synthetic set each simulated population maps to its own cluster(s)."""
    obs = sc_result.adata.obs
    k = sc_result.clustering.n_clusters
    assert 8 <= k <= 11
    assert adjusted_rand_score(obs["true_cell_type"], obs[CLUSTER_KEY]) > 0.8
    # clusters are numbered by decreasing size
    sizes = obs[CLUSTER_KEY].value_counts().reindex(obs[CLUSTER_KEY].cat.categories).to_numpy()
    assert (np.diff(sizes) <= 0).all()


def test_sweep_table_and_choice(sc_result):
    c = sc_result.clustering
    assert c.auto_selected and len(c.sweep) == 5
    best = max(c.sweep, key=lambda r: (round(r["score"], 3), r["resolution"]))
    assert c.resolution == best["resolution"]
    for row in c.sweep:
        assert 0 <= row["stability_ari"] <= 1 and -1 <= row["silhouette"] <= 1


def test_choose_resolution_prefers_higher_on_ties():
    sweep = [{"resolution": 0.4, "score": 0.61}, {"resolution": 0.6, "score": 0.6104}, {"resolution": 0.8, "score": 0.5}]
    assert clustering.choose_resolution(sweep) == 0.6


@pytest.fixture(scope="module")
def prepped():
    a = standardize_in_memory(synthetic.make_single_cell(n_cells=800, n_batches=2, seed=2, n_filler_genes=400))
    a, _ = qc.run_qc(a, QCConfig(), "single-cell", batch_key="sample")
    return a


def test_fixed_resolution_skips_sweep(prepped):
    a = prepped.copy()
    p = preprocessing.run(a, PreprocessingConfig(), "sample")
    low = clustering.run(a.copy(), ClusteringConfig(resolution=0.2), p.use_rep, p.n_pcs_used)
    high = clustering.run(a.copy(), ClusteringConfig(resolution=2.0), p.use_rep, p.n_pcs_used)
    assert not low.auto_selected and low.sweep == []
    assert high.n_clusters > low.n_clusters


def test_harmony_is_opt_in(prepped):
    a = prepped.copy()
    p = preprocessing.run(a, PreprocessingConfig(), "sample")
    assert p.integration == "none" and "X_pca_harmony" not in a.obsm
    b = prepped.copy()
    p = preprocessing.run(b, PreprocessingConfig(integrate="harmony"), "sample")
    assert p.integration == "harmony" and p.use_rep == "X_pca_harmony"
    assert b.obsm["X_pca_harmony"].shape == (b.n_obs, p.n_pcs_used)
    # without batches, harmony is skipped rather than failing
    c = prepped.copy()
    assert preprocessing.run(c, PreprocessingConfig(integrate="harmony"), None).integration == "none"


def test_small_panel_uses_all_genes():
    x = standardize_in_memory(synthetic.make_spatial(n_cells=600))
    x, _ = qc.run_qc(x, QCConfig(), "xenium")
    p = preprocessing.run(x, PreprocessingConfig(), None)
    assert p.used_all_genes and p.n_hvg == x.n_vars


def test_identical_cells_fail_clearly():
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    X = np.tile(np.random.default_rng(0).poisson(3, (1, 300)), (200, 1)).astype(np.float32)
    a = standardize_in_memory(
        ad.AnnData(sp.csr_matrix(X), obs=pd.DataFrame(index=[f"c{i}" for i in range(200)]), var=pd.DataFrame(index=[f"G{i}" for i in range(300)]))
    )
    a, _ = qc.run_qc(a, QCConfig(doublets="off"), "single-cell")
    with pytest.raises(PipelineError, match="no variation"):
        preprocessing.run(a, PreprocessingConfig(), None)


def test_homogeneous_population_is_one_cluster_and_labelled():
    """A sorted population must not be split into noise-driven clusters, and should still be annotated."""
    from cellscribe.config import CellscribeConfig

    pbmc = synthetic.make_single_cell(n_cells=3000, seed=2, n_batches=1, doublet_rate=0, low_quality_rate=0,
                                      include_uncharacterised=False)
    b = pbmc[pbmc.obs["true_cell_type"] == "B cell"].copy()
    res = run(b, output=None, config=CellscribeConfig().with_overrides({"narrative.enabled": False}))
    assert res.clustering.n_clusters == 1
    only = res.annotation.clusters[0]
    assert only.label == "B cell" and only.evidence == "expression" and only.confidence == "medium"
    assert any("single population" in n["message"] for n in res.summary["notes"])
    assert res.qc.cluster_table["status"].iloc[0] == "pass"  # no spurious "few markers" flag
