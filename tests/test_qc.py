import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellscribe import qc, synthetic
from cellscribe.config import QCConfig
from cellscribe.errors import PipelineError
from cellscribe.loaders import standardize_in_memory


def test_mad_bounds_known_values():
    x = np.array([10.0, 11, 12, 13, 14])
    med, mad, lo, hi = qc.mad_bounds(x, nmads=2, direction="both")
    assert med == 12 and mad == pytest.approx(1.4826, rel=1e-4)
    assert lo == pytest.approx(12 - 2 * 1.4826, rel=1e-4) and hi == pytest.approx(12 + 2 * 1.4826, rel=1e-4)
    _, _, lo, hi = qc.mad_bounds(x, nmads=2, direction="upper")
    assert lo is None and hi is not None
    # zero MAD -> no thresholds (metric skipped rather than flagging everything)
    assert qc.mad_bounds(np.ones(50), 5, "both")[2:] == (None, None)


def _toy(n=400, n_genes=300, seed=0):
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(n_genes - 5)] + ["MT-CO1", "MT-CO2", "MT-ND1", "RPL10", "RPS3"]
    X = rng.poisson(2.0, (n, n_genes)).astype(np.float32)
    X[:, -5:-2] = rng.poisson(1.0, (n, 3))
    return ad.AnnData(sp.csr_matrix(X), obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]), var=pd.DataFrame(index=genes))


def test_outliers_flagged_with_reasons():
    a = _toy()
    X = a.X.toarray()
    X[:10, -5:-2] = 400  # 10 cells with very high mitochondrial counts
    X[10:15] = np.where(X[10:15] > 0, 0, 0)  # 5 near-empty cells
    X[10:15, :60] = 1
    a.X = sp.csr_matrix(X)
    a = standardize_in_memory(a)
    cfg = QCConfig(min_genes=0)
    has_mt, _ = qc.compute_qc_metrics(a)
    assert has_mt
    thresholds, skipped = qc.flag_outliers(a, cfg, has_mt, batch_key=None)
    assert a.obs["qc_outlier"].iloc[:10].all()
    assert all("high_mt" in r for r in a.obs["qc_reasons"].iloc[:10])
    assert all("low_counts" in r or "low_genes" in r for r in a.obs["qc_reasons"].iloc[10:15])
    assert a.obs["qc_outlier"].iloc[15:].mean() < 0.05
    metrics = {t.metric for t in thresholds}
    assert {"total_counts", "n_genes_by_counts", "pct_counts_mt"} <= metrics
    mt = next(t for t in thresholds if t.metric == "pct_counts_mt")
    assert mt.lower is None and mt.n_high >= 10


def test_per_batch_thresholds():
    a = _toy(n=600)
    a.obs["batch"] = pd.Categorical(np.repeat(["A", "B"], 300))
    X = a.X.toarray()
    X[300:] *= 3  # batch B sequenced 3x deeper: must not be flagged as outliers
    a.X = sp.csr_matrix(X)
    a = standardize_in_memory(a)
    has_mt, _ = qc.compute_qc_metrics(a)
    thresholds, _ = qc.flag_outliers(a, QCConfig(), has_mt, batch_key="batch")
    assert {t.batch for t in thresholds} == {"A", "B"}
    assert a.obs["qc_outlier"].mean() < 0.05


def test_no_mito_genes_is_reported_not_flagged():
    x = standardize_in_memory(synthetic.make_spatial(n_cells=500))
    out, res = qc.run_qc(x, QCConfig(), "xenium")
    assert not res.has_mt
    assert not any(t.metric == "pct_counts_mt" for t in res.thresholds)
    assert any("No mitochondrial genes" in n["message"] for n in res.notes)
    assert res.doublets["status"] == "skipped"  # not meaningful for segmented imaging data


def test_floor_and_filter_mode(sc_adata):
    a = standardize_in_memory(sc_adata.copy())
    flagged_mode, res_flag = qc.run_qc(a.copy(), QCConfig(), "single-cell", batch_key="sample")
    filtered, res_filter = qc.run_qc(a.copy(), QCConfig(mode="filter"), "single-cell", batch_key="sample")
    assert res_flag.n_removed_by_filter == 0
    assert flagged_mode.n_obs == a.n_obs - res_flag.n_cells_below_floor
    assert filtered.n_obs == flagged_mode.n_obs - res_filter.n_flagged
    # damaged cells are almost all flagged, healthy cells almost never
    truth = flagged_mode.obs["true_cell_type"]
    assert flagged_mode.obs.loc[truth == synthetic.LOW_QUALITY, "qc_flagged"].mean() > 0.9
    real = ~truth.isin([synthetic.LOW_QUALITY, synthetic.DOUBLET])
    assert flagged_mode.obs.loc[real, "qc_flagged"].mean() < 0.05


def test_doublet_scores_separate_doublets(sc_result):
    obs = sc_result.adata.obs
    d = obs.loc[obs["true_cell_type"] == synthetic.DOUBLET, "doublet_score"].median()
    s = obs.loc[~obs["true_cell_type"].isin([synthetic.DOUBLET]), "doublet_score"].median()
    assert d > 2 * s
    assert sc_result.qc.doublets["status"] == "run"


def test_too_few_genes_or_cells_raise():
    tiny = _toy(n_genes=15)
    with pytest.raises(PipelineError, match="at least 20"):
        qc.run_qc(standardize_in_memory(tiny), QCConfig(), "single-cell")
    empty = _toy(n=30)
    empty.X = sp.csr_matrix(empty.X.shape, dtype=np.float32)
    empty.X[0, 0] = 1
    with pytest.raises(PipelineError, match="too few to analyse"):
        qc.run_qc(standardize_in_memory(empty), QCConfig(), "single-cell")


def test_cluster_qc_flags_damaged_cluster(sc_result):
    table = sc_result.qc.cluster_table
    obs = sc_result.adata.obs
    truth_major = obs.groupby("cellscribe_cluster", observed=True)["true_cell_type"].agg(lambda x: x.value_counts().index[0])
    damaged = truth_major[truth_major == synthetic.LOW_QUALITY].index
    assert len(damaged) == 1
    row = table.set_index("cluster").loc[damaged[0]]
    assert row["status"] == "warning"
    codes = {r["code"] for r in row["reasons"]}
    assert {"qc_outliers", "high_mt", "mt_markers"} <= codes
    # every genuine cell-type cluster is free of "likely artefact" warnings
    genuine = truth_major[~truth_major.isin([synthetic.LOW_QUALITY, synthetic.DOUBLET, synthetic.UNCHARACTERISED])].index
    assert (table.set_index("cluster").loc[genuine, "status"] != "warning").all()


def test_cluster_qc_low_complexity_is_only_caution():
    """A cluster whose cells fail QC only for low complexity (e.g. plasma cells) is not called an artefact."""
    a = _toy(n=300)
    a = standardize_in_memory(a)
    qc.compute_qc_metrics(a)
    a.obs["qc_outlier"] = np.r_[np.ones(100, bool), np.zeros(200, bool)]
    a.obs["qc_reasons"] = ["low_complexity"] * 100 + [""] * 200
    a.obs["cellscribe_cluster"] = pd.Categorical(["0"] * 100 + ["1"] * 200)
    res = qc.QCResult("flag", 300, 300, 0, 0, 0, True, True)
    table = qc.cluster_qc(a, "cellscribe_cluster", QCConfig(), res).set_index("cluster")
    assert table.loc["0", "status"] == "caution"
    assert "plasma" in table.loc["0", "reasons"][0]["message"]
