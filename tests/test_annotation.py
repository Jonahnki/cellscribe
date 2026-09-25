import pytest

from cellscribe import annotation, synthetic
from cellscribe.annotation import Candidate, decide_label, score_cluster
from cellscribe.config import AnnotationConfig
from cellscribe.markers import MarkerSet
from cellscribe.utils import CLUSTER_KEY, LABEL_KEY


def _cand(name, lineage, score, hits=4, n=8, conf="high", generic=False):
    return Candidate(name=name, lineage=lineage, score=score, n_hits=hits, n_present=n, hits=[f"g{i}" for i in range(hits)],
                     pval=1e-6, padj=1e-5, confidence=conf, generic=generic)


def test_score_cluster_fraction_and_small_set_penalty():
    sets = [
        (MarkerSet("A", "L1", ("a1", "a2", "a3", "a4")), ["a1", "a2", "a3", "a4"]),
        (MarkerSet("B", "L2", ("b1", "b2")), ["b1", "b2"]),
        (MarkerSet("C", "L3", ("c1", "c2", "c3")), ["c1", "c2", "c3"]),
    ]
    support = {"a1", "a2", "a3", "b1", "b2", "x", "y"}
    res = {c.name: c for c in score_cluster(support, 5000, sets)}
    assert res["A"].n_hits == 3 and res["A"].score == pytest.approx(0.75)
    assert res["B"].score == pytest.approx(1.0 * 2 / 3)  # full overlap but only 2 genes -> down-weighted
    assert res["C"].n_hits == 0 and res["C"].pval == 1.0
    assert res["A"].pval < 1e-6


def test_decide_label_cases():
    label, status, conf, _ = decide_label([_cand("CD8+ T cell", "T", 0.8), _cand("B cell", "B", 0.1, hits=1, conf="low")])
    assert (label, status, conf) == ("CD8+ T cell", "annotated", "high")
    # near-tie within a lineage -> lineage-level label
    label, status, _, why = decide_label([_cand("CD4+ T cell", "T/NK lymphocyte", 0.7), _cand("CD8+ T cell", "T/NK lymphocyte", 0.65)])
    assert status == "lineage" and label.startswith("T/NK lymphocyte")
    # near-tie across lineages -> unresolved, with a doublet hint
    label, status, _, why = decide_label([_cand("B cell", "B", 0.8), _cand("CD4+ T cell", "T", 0.78)])
    assert label == "Unresolved" and "doublets" in why
    # ... but for Visium spots it is an expected mixture
    label, status, _, _ = decide_label([_cand("B cell", "B", 0.8), _cand("CD4+ T cell", "T", 0.78)], multicellular=True)
    assert status == "mixed" and label.startswith("Mixed: B cell")
    # generic entries yield to confident specific entries of the same lineage
    label, *_ = decide_label([_cand("Epithelial cell (unspecified)", "Epithelial", 0.9, generic=True), _cand("Enterocyte", "Epithelial", 0.7)])
    assert label == "Enterocyte"
    # nothing confident -> unresolved
    label, status, conf, _ = decide_label([_cand("B cell", "B", 0.2, hits=1, conf="low")])
    assert (label, status, conf) == ("Unresolved", "unresolved", "low")
    assert decide_label([])[0] == "Unresolved"


def test_synthetic_clusters_are_labelled_correctly(sc_result):
    obs = sc_result.adata.obs
    truth = obs.groupby(CLUSTER_KEY, observed=True)["true_cell_type"].agg(lambda x: x.value_counts().index[0])
    artefacts = {synthetic.LOW_QUALITY, synthetic.DOUBLET, synthetic.UNCHARACTERISED}
    for c in sc_result.annotation.clusters:
        t = truth[c.cluster]
        if t in artefacts:
            assert c.status == "unresolved", f"cluster {c.cluster} ({t}) should be unresolved, got {c.label}"
        else:
            assert c.label == t and c.confidence in ("high", "medium"), f"cluster {c.cluster}: {t} -> {c.label}"
    assert LABEL_KEY in obs
    # the uncharacterised population is explicitly left unresolved rather than guessed
    unchar = truth[truth == synthetic.UNCHARACTERISED].index[0]
    assert sc_result.annotation.by_cluster()[unchar].label == "Unresolved"


def test_markers_and_rationale_are_reported(sc_result):
    for c in sc_result.annotation.clusters:
        assert 0 < len(c.top_markers) <= 10
        assert c.rationale
        if c.status == "annotated":
            assert c.candidates[0].name == c.label and c.candidates[0].hits


def test_custom_marker_file(sc_result, tmp_path):
    f = tmp_path / "markers.csv"
    f.write_text("cell_type,gene\nMy B,MS4A1\nMy B,CD79A\nMy B,CD79B\nMy B,CD19\nMy mono,LYZ\nMy mono,CD14\nMy mono,S100A8\nMy mono,FCN1\n")
    ann = annotation.run(sc_result.adata.copy(), AnnotationConfig(marker_file=f), "human")
    labels = {c.label for c in ann.clusters}
    assert {"My B", "My mono"} <= labels
    assert ann.reference_source.startswith("Custom")


def test_mouse_annotation():
    from cellscribe.config import CellscribeConfig
    from cellscribe.pipeline import run

    res = run(
        synthetic.make_single_cell(n_cells=900, species="mouse", seed=1, n_filler_genes=400),
        output=None,
        config=CellscribeConfig().with_overrides({"narrative.enabled": False}),
    )
    assert res.summary["input"]["species"] == "mouse"
    labels = {c.label for c in res.annotation.clusters}
    assert {"B cell", "NK cell", "Classical monocyte"} <= labels


def test_limited_coverage_note(sc_result):
    a = sc_result.adata.copy()
    a.var_names = [f"ENSG{i:011d}" for i in range(a.n_vars)]
    ann = annotation.run(a, AnnotationConfig(), "human")
    assert all(c.status == "unresolved" for c in ann.clusters)
    assert any("coverage is limited" in n["message"] for n in ann.notes)
