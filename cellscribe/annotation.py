"""Marker genes per cluster and reference-based cell-type candidates.

Method
------
1. Marker genes: Wilcoxon rank-sum test of each cluster against all other
   cells on log-normalised expression (``scanpy.tl.rank_genes_groups``).
2. Specific marker evidence: a gene *supports* a cluster when it is

   * **up-regulated** vs. the rest: BH-adjusted p < ``max_padj``, log2 fold
     change >= ``min_log2fc``, detected in >= ``min_pct`` of the cluster; or
   * **enriched**: detected in >= 20% of the cluster's cells, with a cluster
     mean >= 1.5x the all-cell mean and >= 50% of the highest cluster mean.
     This keeps lineage markers shared by sibling clusters (e.g. CD3E in both
     CD4+ and CD8+ T-cell clusters), which a one-vs-rest test misses.

   Fallback for homogeneous samples (sorted cells, cell lines), where no
   contrast between clusters exists: if no reference type is at least a
   medium-confidence match on specific evidence, the scoring is repeated on
   *expression-level* evidence (genes detected in >= 50% of the cluster's
   cells and in its top quartile by mean expression). A type is assigned
   only if it is a clear winner: score >= 0.5 with >= 3 markers
   (hypergeometric p < 0.01) and at least 0.25 above every other reference
   type. Such calls are capped at medium confidence and labelled as
   expression-based in the report.
3. Reference scoring: for each reference cell type, only its marker genes
   present in the dataset are considered (types with fewer than
   ``min_markers_present`` are not scored). With ``k`` of its ``n`` present
   markers among the cluster's supported genes:

   * overlap fraction ``f = k / n``
   * score ``= f * min(1, k / 3)`` (down-weights calls supported by < 3 genes)
   * enrichment p-value: hypergeometric test of the overlap given the number
     of genes tested and of supported genes, BH-adjusted over all
     cluster x cell-type tests.

4. Confidence: **high** (score >= 0.5, k >= 3, padj < 0.01), **medium**
   (score >= 0.3, k >= 2, padj < 0.05), otherwise **low**. A cluster is only
   labelled when its best candidate is at least medium. Near-ties (within
   0.1) between subtypes of one lineage yield the lineage label; near-ties
   across lineages leave the cluster **unresolved** (often doublets or mixed
   populations). Catch-all ("generic") reference entries only win when no
   specific entry of the same lineage is at least medium. Cell *states*
   (e.g. proliferation) are reported alongside the cell-type call.

The output is a ranked list of candidates with their supporting genes, not a
forced single label: this is a first-pass hypothesis for expert review.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from cellscribe.config import AnnotationConfig
from cellscribe.markers import MarkerReference, MarkerSet, load_reference
from cellscribe.utils import CLUSTER_KEY, LABEL_KEY, logger

AMBIGUITY_MARGIN = 0.1
_CONF_RANK = {"high": 3, "medium": 2, "low": 1, "none": 0}


@dataclass
class Candidate:
    name: str
    lineage: str
    score: float
    n_hits: int
    n_present: int
    hits: list[str]
    pval: float
    padj: float = 1.0
    confidence: str = "low"
    generic: bool = False
    kind: str = "cell_type"


@dataclass
class ClusterAnnotation:
    cluster: str
    label: str
    status: str  # "annotated" | "lineage" | "mixed" | "unresolved"
    confidence: str  # "high" | "medium" | "low" | "none"
    rationale: str
    candidates: list[Candidate] = field(default_factory=list)
    states: list[Candidate] = field(default_factory=list)
    top_markers: list[dict] = field(default_factory=list)
    n_sig_markers: int = 0
    evidence: str = "specific"  # "specific" | "expression"
    mixed_lineages: list[str] = field(default_factory=list)


@dataclass
class AnnotationResult:
    species: str
    reference_source: str
    reference_version: str
    n_reference_types: int
    n_scored_types: int
    clusters: list[ClusterAnnotation]
    params: dict
    notes: list[dict] = field(default_factory=list)
    marker_tables: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("marker_tables", None)
        return d

    def by_cluster(self) -> dict[str, ClusterAnnotation]:
        return {c.cluster: c for c in self.clusters}


# ------------------------------------------------------------------ markers


def find_markers(adata: ad.AnnData, cluster_key: str = CLUSTER_KEY) -> dict[str, pd.DataFrame]:
    """Wilcoxon markers per cluster (vs. rest). Returns cluster -> ranked table."""
    import scanpy as sc

    sizes = adata.obs[cluster_key].value_counts()
    groups = [str(g) for g in adata.obs[cluster_key].cat.categories if sizes.get(g, 0) >= 3]
    if len(groups) < 2 or adata.obs[cluster_key].nunique() < 2:
        return {}
    sc.tl.rank_genes_groups(
        adata,
        cluster_key,
        groups=groups,
        reference="rest",
        method="wilcoxon",
        use_raw=False,
        pts=True,
        key_added="rank_genes_groups",
    )
    tables = {}
    for g in groups:
        df = sc.get.rank_genes_groups_df(adata, group=g, key="rank_genes_groups")
        df = df.rename(
            columns={
                "names": "gene",
                "scores": "score",
                "logfoldchanges": "log2fc",
                "pvals": "pval",
                "pvals_adj": "padj",
                "pct_nz_group": "pct_in",
                "pct_nz_reference": "pct_out",
            }
        )
        df["log2fc"] = df["log2fc"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        tables[g] = df.sort_values("score", ascending=False).reset_index(drop=True)
    return tables


def significant_markers(df: pd.DataFrame, cfg: AnnotationConfig) -> pd.DataFrame:
    return df[(df["padj"] < cfg.max_padj) & (df["log2fc"] >= cfg.min_log2fc) & (df["pct_in"] >= cfg.min_pct)]


def cluster_expression(adata: ad.AnnData, cluster_key: str = CLUSTER_KEY) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Per-cluster mean expression (linear scale, from log-normalised X) and detection rate.

    Returns ``(means, pct, overall_mean)`` with clusters as rows and genes as columns.
    """
    import scipy.sparse as sp

    labels = adata.obs[cluster_key].astype(str).to_numpy()
    clusters = [str(c) for c in adata.obs[cluster_key].cat.categories]
    ind = sp.csr_matrix(
        (np.ones(adata.n_obs), (pd.Categorical(labels, categories=clusters).codes, np.arange(adata.n_obs))),
        shape=(len(clusters), adata.n_obs),
    )
    sizes = np.asarray(ind.sum(axis=1)).ravel()
    sizes[sizes == 0] = 1
    X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    E = X.copy().astype(np.float64)
    E.data = np.expm1(E.data)
    D = X.copy()
    D.data = np.ones_like(D.data)
    means = np.asarray((ind @ E).todense()) / sizes[:, None]
    pct = np.asarray((ind @ D).todense()) / sizes[:, None]
    overall = np.asarray(E.mean(axis=0)).ravel()
    return (
        pd.DataFrame(means, index=clusters, columns=adata.var_names),
        pd.DataFrame(pct, index=clusters, columns=adata.var_names),
        overall,
    )


def supported_genes(
    cluster: str,
    table: pd.DataFrame | None,
    means: pd.DataFrame,
    pct: pd.DataFrame,
    overall: np.ndarray,
    cfg: AnnotationConfig,
) -> set[str]:
    """Genes that support a cluster's identity (see module docstring)."""
    genes: set[str] = set()
    if table is not None:
        genes |= set(significant_markers(table, cfg)["gene"])
    m = means.loc[cluster].to_numpy()
    p = pct.loc[cluster].to_numpy()
    top = means.to_numpy().max(axis=0)
    enriched = (p >= 0.2) & (m >= 1.5 * overall) & (m >= 0.5 * top) & (m > 0)
    genes |= set(means.columns.to_numpy()[enriched])
    return genes


def expressed_genes(cluster: str, means: pd.DataFrame, pct: pd.DataFrame) -> set[str]:
    """Genes detected in >= 50% of the cluster's cells and in its top quartile by mean (no contrast)."""
    m = means.loc[cluster].to_numpy()
    p = pct.loc[cluster].to_numpy()
    high = (p >= 0.5) & (m >= np.quantile(m, 0.75)) & (m > 0)
    return set(means.columns.to_numpy()[high])


# ------------------------------------------------------------------ scoring


def _present(mset: MarkerSet, lookup: dict[str, str]) -> list[str]:
    return list(dict.fromkeys(lookup[g.upper()] for g in mset.genes if g.upper() in lookup))


def score_cluster(
    sig_genes: set[str],
    n_tested: int,
    reference_sets: list[tuple[MarkerSet, list[str]]],
) -> list[Candidate]:
    """Score one cluster's significant markers against each reference set."""
    out = []
    n_sig = len(sig_genes)
    for mset, present in reference_sets:
        hits = [g for g in present if g in sig_genes]
        k, n = len(hits), len(present)
        f = k / n
        score = f * min(1.0, k / 3.0)
        pval = float(hypergeom.sf(k - 1, n_tested, n_sig, n)) if k > 0 else 1.0
        out.append(
            Candidate(
                name=mset.name,
                lineage=mset.lineage,
                score=float(score),
                n_hits=k,
                n_present=n,
                hits=hits,
                pval=pval,
                generic=mset.generic,
                kind=mset.kind,
            )
        )
    return out


def _bh(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    ranked = pvals[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def _confidence(c: Candidate) -> str:
    if c.score >= 0.5 and c.n_hits >= 3 and c.padj < 0.01:
        return "high"
    if c.score >= 0.3 and c.n_hits >= 2 and c.padj < 0.05:
        return "medium"
    return "low" if c.n_hits >= 1 else "none"


def _rank(cands: list[Candidate]) -> list[Candidate]:
    return sorted(cands, key=lambda c: (-c.score, -c.n_hits, c.pval, c.name))


def decide_label(cands: list[Candidate], multicellular: bool = False) -> tuple[str, str, str, str]:
    """Return (label, status, confidence, rationale) for a cluster's ranked cell-type candidates.

    ``multicellular`` (Visium spots) turns cross-lineage ties into a "Mixed" label,
    since each spot is expected to contain several cell types.
    """
    ranked = _rank(cands)
    confident = [c for c in ranked if _CONF_RANK[c.confidence] >= 2]
    # generic entries yield to confident specific entries of the same lineage
    specific_lineages = {c.lineage for c in confident if not c.generic}
    confident = [c for c in confident if not (c.generic and c.lineage in specific_lineages)]
    if not confident:
        best = ranked[0] if ranked and ranked[0].n_hits > 0 else None
        if best is None:
            return "Unresolved", "unresolved", "none", "None of the reference cell types' markers are enriched or highly expressed in this cluster."
        return (
            "Unresolved",
            "unresolved",
            "low",
            f"No confident match. Best candidate: {best.name} with {best.n_hits} of {best.n_present} markers "
            f"({', '.join(best.hits[:5])}), which is too weak to assign.",
        )
    best = confident[0]
    rivals = [c for c in confident[1:] if c.score >= best.score - AMBIGUITY_MARGIN]
    evidence = f"{best.n_hits} of {best.n_present} {best.name} markers support this cluster ({', '.join(best.hits[:6])})"
    if not rivals:
        runner = next((c for c in ranked if c.name != best.name and c.n_hits > 0), None)
        tail = f"; next best: {runner.name} ({runner.n_hits}/{runner.n_present})." if runner else "."
        return best.name, "annotated", best.confidence, evidence + tail
    names = ", ".join(r.name for r in rivals[:2])
    if all(r.lineage == best.lineage for r in rivals):
        conf = "medium" if best.confidence == "high" else best.confidence
        return (
            f"{best.lineage} (subtype unclear)",
            "lineage",
            conf,
            f"{evidence}, but {names} fit about equally well; the subtype cannot be resolved from canonical markers.",
        )
    lineages = sorted({best.lineage} | {r.lineage for r in rivals})
    if multicellular:
        parts = [best.name] + [r.name for r in rivals[:2]]
        return (
            "Mixed: " + " + ".join(parts),
            "mixed",
            "medium" if best.confidence == "high" else best.confidence,
            f"{evidence}; markers of {names} are also present. Spots capture several cells, so this cluster most "
            "likely represents a tissue region containing these cell types.",
        )
    return (
        "Unresolved",
        "unresolved",
        "low",
        f"Markers of unrelated lineages ({' and '.join(lineages)}) are co-expressed: {best.name} and {names} fit "
        "about equally well. This pattern is typical of doublets or a mixed/poorly separated cluster.",
    )


def run(
    adata: ad.AnnData,
    cfg: AnnotationConfig,
    species: str,
    reference: MarkerReference | None = None,
    cluster_key: str = CLUSTER_KEY,
    modality: str = "single-cell",
) -> AnnotationResult:
    reference = reference or load_reference(species, cfg.marker_file)
    multicellular = modality == "visium"
    tables = find_markers(adata, cluster_key)
    lookup = {g.upper(): g for g in adata.var_names}
    ref_sets = []
    for mset in reference.sets:
        present = _present(mset, lookup)
        if len(present) >= cfg.min_markers_present:
            ref_sets.append((mset, present))
    n_scored = sum(1 for m, _ in ref_sets if m.kind == "cell_type")
    notes: list[dict] = []
    n_ref_genes_present = len({g for _, p in ref_sets for g in p})
    if n_scored < 0.25 * max(len(reference.cell_types), 1):
        notes.append(
            {
                "severity": "caution",
                "message": f"Only {n_scored} of {len(reference.cell_types)} reference cell types have at least "
                f"{cfg.min_markers_present} marker genes in this dataset ({n_ref_genes_present} reference genes present). "
                "Annotation coverage is limited (targeted panel, non-symbol gene IDs, or a species mismatch).",
            }
        )

    clusters = [str(c) for c in adata.obs[cluster_key].cat.categories]
    n_tested = adata.n_vars
    means, pct, overall = cluster_expression(adata, cluster_key)
    all_cands: dict[str, list[Candidate]] = {}
    expr_cands: dict[str, list[Candidate]] = {}
    sig_counts: dict[str, int] = {}
    for c in clusters:
        table = tables.get(c)
        sig_counts[c] = len(significant_markers(table, cfg)) if table is not None else 0
        all_cands[c] = score_cluster(supported_genes(c, table, means, pct, overall, cfg), n_tested, ref_sets)
        expr_cands[c] = score_cluster(expressed_genes(c, means, pct), n_tested, ref_sets)

    for family in (all_cands, expr_cands):  # BH within each evidence family
        flat = [cand for cl in clusters for cand in family[cl]]
        padj = _bh(np.array([cand.pval for cand in flat])) if flat else np.array([])
        for cand, q in zip(flat, padj):
            cand.padj = float(q)
            cand.confidence = _confidence(cand)

    results = []
    for c in clusters:
        cands = [x for x in all_cands[c] if x.kind == "cell_type"]
        states = [x for x in all_cands[c] if x.kind == "state" and _CONF_RANK[x.confidence] >= 2]
        label, status, conf, why = decide_label(cands, multicellular)
        evidence = "specific"
        if status == "unresolved" and not any(_CONF_RANK[x.confidence] >= 2 for x in cands):
            e_cands = _rank([x for x in expr_cands[c] if x.kind == "cell_type" and x.n_hits > 0])
            if e_cands:
                best = e_cands[0]
                runner = e_cands[1] if len(e_cands) > 1 else None
                margin = best.score - (runner.score if runner else 0.0)
                genes = ", ".join(best.hits[:6])
                if best.score >= 0.5 and best.n_hits >= 3 and best.pval < 0.01 and margin >= 0.25:
                    cands, evidence = e_cands, "expression"
                    label, status, conf = best.name, "annotated", "medium"
                    best.confidence = "medium"
                    tail = f"; next best: {runner.name} ({runner.n_hits}/{runner.n_present})." if runner else "."
                    why = (
                        "Expression-based call (no cluster-specific contrast, as in sorted or homogeneous samples): "
                        f"{best.n_hits} of {best.n_present} {best.name} markers are detected in most cells and highly "
                        f"expressed ({genes}){tail}"
                    )
                elif conf == "none":
                    conf = "low"
                    why = (
                        f"No confident match. Best candidate from expression level alone: {best.name} with {best.n_hits} "
                        f"of {best.n_present} markers ({genes}), which is not specific enough to assign."
                    )
        mixed: list[str] = []
        if status == "unresolved" and evidence == "specific":
            conf_c = [x for x in _rank(cands) if _CONF_RANK[x.confidence] >= 2]
            if conf_c:
                near = [x for x in conf_c if x.score >= conf_c[0].score - AMBIGUITY_MARGIN]
                lineages = sorted({x.lineage for x in near})
                if len(lineages) > 1:
                    mixed = lineages
        top = []
        if c in tables:
            for _, r in tables[c].head(cfg.n_markers_display).iterrows():
                top.append(
                    {
                        "gene": r["gene"],
                        "log2fc": float(r["log2fc"]),
                        "pct_in": float(r["pct_in"]),
                        "pct_out": float(r["pct_out"]),
                        "padj": float(r["padj"]),
                    }
                )
        results.append(
            ClusterAnnotation(
                cluster=c,
                label=label,
                status=status,
                confidence=conf,
                rationale=why,
                candidates=[x for x in _rank(cands) if x.n_hits > 0][:3],
                states=_rank(states),
                top_markers=top,
                n_sig_markers=sig_counts[c],
                evidence=evidence,
                mixed_lineages=mixed,
            )
        )

    label_map = {r.cluster: r.label for r in results}
    adata.obs[LABEL_KEY] = pd.Categorical(adata.obs[cluster_key].astype(str).map(label_map))
    n_res = sum(r.status != "unresolved" for r in results)
    logger.info("Annotated %d of %d clusters (%s reference, %s)", n_res, len(results), reference.source, species)
    return AnnotationResult(
        species=species,
        reference_source=reference.source,
        reference_version=reference.version,
        n_reference_types=len(reference.cell_types),
        n_scored_types=n_scored,
        clusters=results,
        params={
            "test": "Wilcoxon rank-sum (cluster vs rest), BH-adjusted",
            "min_log2fc": cfg.min_log2fc,
            "max_padj": cfg.max_padj,
            "min_pct": cfg.min_pct,
            "min_markers_present": cfg.min_markers_present,
            "ambiguity_margin": AMBIGUITY_MARGIN,
        },
        notes=notes,
        marker_tables=tables,
    )
