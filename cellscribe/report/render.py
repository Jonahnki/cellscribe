"""Render a :class:`~cellscribe.pipeline.CellscribeResult` into a single self-contained HTML file."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, select_autoescape
from markupsafe import Markup

from cellscribe import __version__
from cellscribe.qc import REASONS
from cellscribe.report import figures as F

if TYPE_CHECKING:  # pragma: no cover
    from cellscribe.pipeline import CellscribeResult

REPO_URL = "https://github.com/jonahnki/cellscribe"
ICONS = {"warning": "⚠", "caution": "!", "info": "i"}
MODALITY_LABELS = {
    "single-cell": "Single-cell RNA-seq",
    "visium": "10x Visium spatial transcriptomics",
    "xenium": "Imaging-based spatial transcriptomics (Xenium)",
    "spatial": "Spatial transcriptomics",
}

REFERENCES = {
    "scanpy": "Wolf FA, Angerer P, Theis FJ. SCANPY: large-scale single-cell gene expression data analysis. Genome Biology 19, 15 (2018).",
    "anndata": "Virshup I, Rybakov S, Theis FJ, Angerer P, Wolf FA. anndata: Access and store annotated data matrices. Journal of Open Source Software 9(101), 4371 (2024).",
    "scater": "McCarthy DJ, Campbell KR, Lun ATL, Wills QF. Scater: pre-processing, quality control, normalization and visualization of single-cell RNA-seq data in R. Bioinformatics 33(8), 1179-1186 (2017).",
    "pipecomp": "Germain PL, Sonrel A, Robinson MD. pipeComp, a general framework for the evaluation of computational pipelines, reveals performant single cell RNA-seq preprocessing tools. Genome Biology 21, 227 (2020).",
    "bestpractices": "Heumos L, Schaar AC, Lance C, et al. Best practices for single-cell analysis across modalities. Nature Reviews Genetics 24, 550-572 (2023).",
    "scrublet": "Wolock SL, Lopez R, Klein AM. Scrublet: computational identification of cell doublets in single-cell transcriptomic data. Cell Systems 8(4), 281-291 (2019).",
    "seurat": "Satija R, Farrell JA, Gennert D, Schier AF, Regev A. Spatial reconstruction of single-cell gene expression data. Nature Biotechnology 33, 495-502 (2015).",
    "harmony": "Korsunsky I, Millard N, Fan J, et al. Fast, sensitive and accurate integration of single-cell data with Harmony. Nature Methods 16, 1289-1296 (2019).",
    "leiden": "Traag VA, Waltman L, van Eck NJ. From Louvain to Leiden: guaranteeing well-connected communities. Scientific Reports 9, 5233 (2019).",
    "silhouette": "Rousseeuw PJ. Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics 20, 53-65 (1987).",
    "ari": "Hubert L, Arabie P. Comparing partitions. Journal of Classification 2, 193-218 (1985).",
    "umap": "McInnes L, Healy J, Melville J. UMAP: Uniform Manifold Approximation and Projection for dimension reduction. arXiv:1802.03426 (2018).",
    "bh": "Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal of the Royal Statistical Society B 57(1), 289-300 (1995).",
    "dissociation": "van den Brink SC, Sage F, Vértesy Á, et al. Single-cell sequencing reveals dissociation-induced gene expression in tissue subpopulations. Nature Methods 14, 935-936 (2017).",
    "squidpy": "Palla G, Spitzer H, Klein M, et al. Squidpy: a scalable framework for spatial omics analysis. Nature Methods 19, 171-178 (2022).",
    "moran": "Moran PAP. Notes on continuous stochastic phenomena. Biometrika 37(1/2), 17-23 (1950).",
}


def _fmt(x) -> str:
    if x is None:
        return "—"
    x = float(x)
    if abs(x) >= 100:
        return f"{x:,.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"


def _status_badge(status: str) -> Markup:
    text = {"pass": "✓ pass", "caution": "! review", "warning": "⚠ likely artefact"}.get(status, status)
    return Markup(f'<span class="badge {status}">{text}</span>')


def citation(version: str = __version__) -> tuple[str, str]:
    text = (
        f"Cellscribe contributors. Cellscribe: automated first-pass quality control, clustering and annotation "
        f"reports for single-cell and spatial transcriptomics. Version {version}. {REPO_URL} (2026)."
    )
    bib = (
        "@software{cellscribe,\n"
        "  title   = {Cellscribe: automated first-pass quality control, clustering and annotation reports\n"
        "             for single-cell and spatial transcriptomics},\n"
        "  author  = {{Cellscribe contributors}},\n"
        "  year    = {2026},\n"
        f"  version = {{{version}}},\n"
        f"  url     = {{{REPO_URL}}},\n"
        "  license = {MIT}\n"
        "}"
    )
    return text, bib


def methods_paragraphs(summary: dict) -> tuple[list[str], list[str]]:
    """Manuscript-style methods text populated with this run's parameters, plus the references it cites."""
    cfg, qc, prep, clust, ann = summary["config"], summary["qc"], summary["preprocessing"], summary["clustering"], summary["annotation"]
    sw = summary["software"]
    refs = ["scanpy", "anndata"]
    by_batch = any(t.get("batch") for t in qc["thresholds"])
    p1 = (
        f"Raw counts ({summary['input']['format']} input) were analysed with Cellscribe v{summary['cellscribe_version']} "
        f"(scanpy v{sw.get('scanpy', '?')}). Barcodes with fewer than {qc['min_genes']} detected genes "
        f"({qc['n_cells_below_floor']:,} barcodes) and genes detected in fewer than {cfg['qc']['min_cells_per_gene']} cells were removed, "
        f"leaving {summary['dataset']['n_cells']:,} cells and {summary['dataset']['n_genes']:,} genes"
        f"{' after QC filtering' if qc['mode'] == 'filter' else ''}. Per-cell QC metrics (total counts, detected genes"
        f"{', percentage of counts from mitochondrial genes' if qc['has_mt'] else ''}"
        f"{', ribosomal-protein genes' if qc['has_ribo'] else ''} and from the 20 most highly expressed genes) were computed. "
        f"Outliers were identified with data-adaptive thresholds based on the median absolute deviation (MAD; scaled to be "
        f"consistent with the standard deviation){' within each batch' if by_batch else ''}: cells more than {cfg['qc']['nmads']:g} MADs "
        f"from the median of log-transformed total counts or detected genes, more than {cfg['qc']['nmads']:g} MADs above the median "
        f"percentage of counts in the top 20 genes"
        f"{', or more than ' + format(cfg['qc']['nmads_mt'], 'g') + ' MADs above the median mitochondrial percentage' if qc['has_mt'] else ''} "
        f"were flagged, following established recommendations (McCarthy et al. 2017; Germain et al. 2020; Heumos et al. 2023)."
    )
    refs += ["scater", "pipecomp", "bestpractices"]
    if qc["doublets"].get("status") == "run":
        p1 += (
            f" Doublet scores were computed with Scrublet (Wolock et al. 2019) as implemented in scanpy, with an expected doublet "
            f"rate of {qc['doublets']['expected_rate']:.0%}{' and per-batch simulation' if by_batch else ''}; "
            f"{qc['doublets']['n_predicted']:,} cells exceeded the automatically determined threshold."
        )
        refs.append("scrublet")
    else:
        p1 += f" Doublet detection was not performed ({qc['doublets'].get('reason', '').rstrip('.')})."
    p1 += (
        " Flagged cells were retained and labelled rather than removed, so that their distribution across clusters could be assessed."
        if qc["mode"] == "flag"
        else f" Flagged cells ({qc['n_removed_by_filter']:,}) were removed before downstream analysis."
    )

    hvg = (
        f"all {prep['n_hvg']:,} genes were used for dimensionality reduction (fewer than twice the "
        f"{cfg['preprocessing']['n_top_genes']:,} highly variable genes that would otherwise be selected)"
        if prep["used_all_genes"]
        else f"the {prep['n_hvg']:,} most highly variable genes were selected (dispersion-based method of Satija et al. 2015"
        f"{', computed per batch' if prep['batch_key'] else ''})"
    )
    if not prep["used_all_genes"]:
        refs.append("seurat")
    p2 = (
        f"Counts were normalised to {prep['target_sum']:,.0f} counts per cell and log(x+1)-transformed, and {hvg}. "
        f"Expression values were scaled to unit variance (clipped at 10) and reduced with principal component analysis "
        f"({prep['n_pcs_computed']} components). "
    )
    if prep["integration"] == "harmony":
        p2 += (
            f"Batch effects between the {prep['n_batches']} samples in '{prep['batch_key']}' were corrected by applying Harmony "
            f"(Korsunsky et al. 2019) to the first {prep['n_pcs_used']} components. "
        )
        refs.append("harmony")
    p2 += (
        f"A {clust['n_neighbors']}-nearest-neighbour graph was built from the first {clust['n_pcs']} components"
        f"{' of the integrated embedding' if prep['integration'] == 'harmony' else ''}, and cells were clustered with the Leiden "
        f"algorithm (Traag et al. 2019)"
    )
    refs.append("leiden")
    if clust["auto_selected"]:
        grid = ", ".join(f"{r['resolution']:g}" for r in clust["sweep"])
        p2 += (
            f". The resolution was selected from a grid ({grid}) by maximising the product of partition stability (mean adjusted "
            f"Rand index across {cfg['clustering']['stability_runs']} runs with different random seeds; Hubert and Arabie 1985) and "
            f"the rescaled mean silhouette width in PCA space (Rousseeuw 1987; a single-cluster solution was assigned a neutral "
            f"silhouette of 0), giving a resolution of {clust['resolution']:g} and {clust['n_clusters']} "
            f"cluster{'s' if clust['n_clusters'] != 1 else ''}"
        )
        refs += ["ari", "silhouette"]
    else:
        p2 += f" at a user-specified resolution of {clust['resolution']:g} ({clust['n_clusters']} clusters)"
    p2 += f". Cells were embedded in two dimensions with UMAP (McInnes et al. 2018). The random seed was {cfg['seed']} throughout."
    refs.append("umap")

    p3 = (
        "Marker genes of each cluster were identified with a Wilcoxon rank-sum test against all other cells, with "
        "Benjamini-Hochberg correction (Benjamini and Hochberg 1995). Clusters were compared with the "
        f"{ann['reference_source']} (version {ann['reference_version']}; {ann['n_scored_types']} {ann['species']} cell types with at "
        f"least {ann['params']['min_markers_present']} markers present). A reference marker supported a cluster if it was "
        f"significantly up-regulated (adjusted p < {ann['params']['max_padj']:g}, log2 fold change ≥ {ann['params']['min_log2fc']:g}, "
        f"detected in ≥ {ann['params']['min_pct']:.0%} of the cluster's cells) or enriched (detected in ≥ 20% of cells, cluster mean "
        "≥ 1.5 times the all-cell mean and ≥ 50% of the highest cluster mean). For each cell type, the fraction k/n of its markers "
        "supporting the cluster was multiplied by min(1, k/3) and its over-representation tested with a hypergeometric test "
        "(BH-adjusted across all cluster and cell-type pairs). Labels were assigned only with high (score ≥ 0.5, k ≥ 3, adjusted "
        "p < 0.01) or medium (score ≥ 0.3, k ≥ 2, adjusted p < 0.05) confidence; near-ties within a lineage were reported at "
        "lineage level, and all other clusters were reported as unresolved. For clusters with no specific match, the scoring "
        "was repeated on highly expressed genes (detected in ≥ 50% of cells and in the cluster's top quartile); a type was "
        "assigned (at most medium confidence) only if it scored ≥ 0.5 with ≥ 3 markers (p < 0.01) and exceeded every other "
        "type by ≥ 0.25. QC metrics were summarised per cluster and clusters were flagged when most cells failed QC, "
        "when median mitochondrial fraction or library size deviated strongly from the dataset, when doublet scores were "
        "elevated, or when their top markers were dominated by mitochondrial, ribosomal-protein, or dissociation-induced "
        "genes (van den Brink et al. 2017)."
    )
    refs += ["bh", "dissociation"]
    paragraphs = [p1, p2, p3]
    sp = summary.get("spatial")
    if sp and sp.get("enabled"):
        paragraphs.append(
            f"Spatial analyses used Squidpy v{sw.get('squidpy', '?')} (Palla et al. 2022). A spatial neighbour graph was built "
            f"({sp['graph'].get('method', '').lower()}); neighbourhood enrichment z-scores were computed from {sp['nhood_n_perms']} "
            f"permutations of cluster labels, and spatial autocorrelation of {sp['n_genes_autocorr']} highly variable genes was "
            "quantified with Moran's I (Moran 1950), with analytic p-values and Benjamini-Hochberg correction."
        )
        refs += ["squidpy", "moran"]
    return paragraphs, [REFERENCES[r] for r in dict.fromkeys(refs)]


def _plotly_js(mode: str) -> str:
    import plotly
    from plotly.offline import get_plotlyjs

    if mode == "cdn":
        return f'<script src="https://cdn.plot.ly/plotly-{plotly.offline.get_plotlyjs_version()}.min.js" charset="utf-8"></script>'
    return f'<script type="text/javascript">{get_plotlyjs()}</script>'


def build_context(result: CellscribeResult) -> dict:
    from cellscribe.spatial import tissue_image

    adata, summary, cfg = result.adata, result.summary, result.config
    qc, ann, clust, prep = result.qc, result.annotation, result.clustering, result.preprocessing
    max_points = cfg.report.max_points
    table = qc.cluster_table
    clusters = [str(c) for c in adata.obs["cellscribe_cluster"].cat.categories]
    colors = F.cluster_colors(clusters)
    labels = {c.cluster: c.label for c in ann.clusters}
    status = dict(zip(table["cluster"], table["status"]))
    batch_key = summary["input"]["batch_key"]

    figs = {
        "qc_dist": F.to_div(F.qc_distributions(adata, qc), "qc_distributions"),
        "qc_scatter": F.to_div(F.qc_scatter(adata, max_points), "qc_scatter"),
        "umap_clusters": F.to_div(F.umap_clusters(adata, ann, max_points), "umap_clusters"),
        "umap_metrics": F.to_div(F.umap_metrics(adata, max_points), "umap_qc_metrics"),
        "cluster_boxes": F.to_div(F.cluster_qc_boxes(adata, table), "cluster_qc"),
    }
    dbl = F.doublet_histogram(adata, qc)
    figs["doublet"] = F.to_div(dbl, "doublet_scores") if dbl is not None else None
    figs["batch_comp"] = F.to_div(F.batch_composition(adata, batch_key, ann), "batch_composition") if batch_key else None
    heat, _ = F.marker_heatmap(adata, ann)
    figs["heatmap"] = F.to_div(heat, "marker_heatmap") if heat is not None else None

    sp = result.spatial if (result.spatial is not None and result.spatial.enabled) else None
    if sp is not None:
        image_info = tissue_image(adata)
        units = sp.coord_units
        figs["spatial_clusters"] = F.to_div(F.spatial_clusters(adata, ann, image_info, max_points), "spatial_clusters")
        figs["spatial_counts"] = F.to_div(
            F.spatial_feature(adata, adata.obs["total_counts"].to_numpy(), "counts", image_info, max_points, log=True),
            "spatial_counts",
        )
        if summary["input"]["modality"] == "visium":
            # spots sit on a regular grid, so density is uninformative; show gene detection instead
            figs["spatial_density"] = F.to_div(
                F.spatial_feature(adata, adata.obs["n_genes_by_counts"].to_numpy(), "genes", image_info, max_points),
                "spatial_genes_detected",
            )
        else:
            figs["spatial_density"] = F.to_div(F.spatial_density(adata, units), "spatial_density")
        figs["nhood"] = F.to_div(F.nhood_heatmap(sp, ann), "nhood_enrichment") if sp.nhood_clusters else None
        figs["moran"] = F.to_div(F.moran_bars(sp), "morans_i") if sp.moran_top else ""
        sg = F.spatial_genes(adata, [m["gene"] for m in sp.moran_top], image_info, max_points)
        figs["spatial_genes"] = F.to_div(sg, "spatial_genes") if sg is not None else None

    n_after_floor = qc.n_cells_input - qc.n_cells_below_floor
    pct_flagged = 100.0 * qc.n_flagged / max(n_after_floor, 1)
    n_warn = sum(s == "warning" for s in status.values())
    n_caution = sum(s == "caution" for s in status.values())
    n_unres = sum(c.status == "unresolved" for c in ann.clusters)
    n_ann = len(ann.clusters) - n_unres
    unit = "spots" if summary["input"]["modality"] == "visium" else "cells"
    tiles = [
        {"label": f"{unit.capitalize()} analysed", "value": f"{adata.n_obs:,}", "sub": f"of {summary['input'].get('n_cells_loaded', adata.n_obs):,} loaded"},
        {"label": "Genes", "value": f"{adata.n_vars:,}", "sub": f"of {summary['input'].get('n_genes_loaded', adata.n_vars):,} loaded"},
        {"label": "Clusters", "value": str(clust.n_clusters), "sub": f"resolution {clust.resolution:g}{' (auto)' if clust.auto_selected else ''}"},
        {"label": "Clusters annotated", "value": f"{n_ann}/{len(ann.clusters)}", "sub": f"{n_unres} unresolved"},
        {"label": f"{unit.capitalize()} flagged by QC", "value": f"{pct_flagged:.1f}%", "sub": f"{qc.n_flagged:,} {unit}", "warn": pct_flagged > 20},
        {"label": "Clusters flagged", "value": str(n_warn), "sub": f"+ {n_caution} to review", "warn": n_warn > 0},
    ]

    info = summary["input"]
    input_rows = [
        ("Sample", info.get("sample_name", "")),
        ("Input", info.get("path", "")),
        ("Format", info.get("format", "")),
        ("Data type", MODALITY_LABELS.get(info.get("modality"), info.get("modality"))),
        ("Species", f"{info['species']} ({info['species_reason']})"),
        ("Cells × genes loaded", f"{info.get('n_cells_loaded', 0):,} × {info.get('n_genes_loaded', 0):,}"),
        ("Batch / sample column", info["batch_key"] or "none detected (single sample)"),
    ]
    for key, label in (
        ("csv_orientation", "Table orientation"), ("library_id", "Visium library"), ("coordinate_units", "Coordinate units"),
        ("xenium_panel_name", "Xenium panel"), ("xenium_run_name", "Xenium run"), ("xenium_analysis_sw_version", "Xenium software"),
    ):
        if info.get(key):
            input_rows.append((label, str(info[key]).replace("_", " ")))

    cluster_rows = []
    for r in table.to_dict(orient="records"):
        r["label"] = labels.get(r["cluster"], "")
        r["color"] = colors.get(r["cluster"], "#999")
        cluster_rows.append(r)

    ann_rows = []
    for c in ann.clusters:
        best = c.candidates[0] if c.candidates else None
        ann_rows.append(
            {
                "cluster": c.cluster,
                "color": colors.get(c.cluster, "#999"),
                "label": c.label,
                "confidence": c.confidence,
                "rationale": c.rationale,
                "evidence": c.evidence,
                "states": [s.name for s in c.states],
                "others": [
                    f"{x.name} ({x.n_hits}/{x.n_present})"
                    for x in c.candidates
                    if x.n_hits >= 2 and (best is None or x.name != c.label)
                ][:3],
                "markers": [m["gene"] for m in c.top_markers],
                "hit_genes": set(best.hits) if best is not None and c.status != "unresolved" else set(),
                "qc_status": status.get(c.cluster, "pass"),
            }
        )
    marker_rows = [{"cluster": c.cluster, **m} for c in ann.clusters for m in c.top_markers]

    notes = summary["notes"]
    has_spatial = sp is not None
    narrative = result.narrative if (result.narrative is not None and result.narrative.status == "generated") else None
    num = 8 + int(has_spatial)
    narrative_num = num if narrative else None
    methods_num = num + int(narrative is not None)
    cite_num = methods_num + 1
    sections = [
        {"id": "summary", "title": "Executive summary"},
        {"id": "input", "title": "Sample & input"},
        {"id": "qc", "title": "Quality control"},
        {"id": "clustering", "title": "Clustering overview"},
        {"id": "cluster-qc", "title": "Per-cluster QC", "flags": n_warn},
        {"id": "annotation", "title": "Cell-type annotation"},
        {"id": "markers", "title": "Marker genes"},
    ]
    if has_spatial:
        sections.append({"id": "spatial", "title": "Spatial analysis"})
    if narrative:
        sections.append({"id": "narrative", "title": "AI narrative summary"})
    sections += [{"id": "methods", "title": "Methods & parameters"}, {"id": "cite", "title": "How to cite"}]

    methods, refs = methods_paragraphs(summary)
    cite_text, bibtex = citation()
    title = cfg.report.title or f"Cellscribe report: {info.get('sample_name', 'sample')}"
    reason_rows = [
        (REASONS.get(code, code).capitalize(), n, 100.0 * n / max(adata.n_obs if qc.mode == "flag" else n_after_floor, 1))
        for code, n in sorted(qc.reason_counts.items(), key=lambda kv: -kv[1])
        if n > 0
    ]
    timings = [(k, v) for k, v in result.timings.items() if k != "total"]
    nar = summary.get("narrative", {})
    narrative_status = None
    if nar and nar.get("status") != "generated":
        narrative_status = nar.get("reason")

    return {
        "version": __version__,
        "title": title,
        "css": resources.files("cellscribe.report").joinpath("static/report.css").read_text(),
        "plotly_js": _plotly_js(cfg.report.plotlyjs),
        "summary": summary,
        "cfg": cfg,
        "qc": qc,
        "clust": clust,
        "prep": prep,
        "ann": ann,
        "spatial": sp,
        "narrative": narrative,
        "narrative_num": narrative_num,
        "methods_num": methods_num,
        "cite_num": cite_num,
        "ann_num": 6,
        "sections": sections,
        "tiles": tiles,
        "icons": ICONS,
        "modality_label": MODALITY_LABELS.get(info.get("modality"), "Transcriptomics"),
        "input_rows": input_rows,
        "batch_rows": sorted(info.get("batches", {}).items(), key=lambda kv: -kv[1]),
        "input_notes": [n for n in notes if n.get("source") in ("input", "preprocessing", "clustering")],
        "qc_notes": [n for n in notes if n.get("source") == "qc"],
        "ann_notes": [n for n in notes if n.get("source") == "annotation"],
        "spatial_notes": [n for n in notes if n.get("source") == "spatial"],
        "pct_flagged": pct_flagged,
        "has_batch_thresholds": any(t.batch for t in qc.thresholds),
        "reason_rows": reason_rows,
        "figs": figs,
        "subsampled": adata.n_obs > max_points,
        "max_points": max_points,
        "cluster_rows": cluster_rows,
        "has_doublets": "doublet_score" in adata.obs,
        "has_batch": bool(batch_key) and "dominant_batch" in table.columns,
        "cqc_cols": 7 + int(qc.has_mt) + int("doublet_score" in adata.obs)
        + int(bool(batch_key) and "dominant_batch" in table.columns),
        "ann_rows": ann_rows,
        "marker_rows": marker_rows,
        "labels": labels,
        "methods_paragraphs": methods,
        "references": refs,
        "params": [(k, "null" if v is None else v) for k, v in cfg.flat().items()],
        "timings": timings,
        "narrative_status": narrative_status,
        "citation_text": cite_text,
        "bibtex": bibtex,
        "fmt": _fmt,
        "status_badge": _status_badge,
    }


def render_html(result: CellscribeResult) -> str:
    env = Environment(autoescape=select_autoescape(["html", "j2"]), trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(resources.files("cellscribe.report").joinpath("templates/report.html.j2").read_text())
    return template.render(**build_context(result))


def render_report(result: CellscribeResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(result), encoding="utf-8")
    return path

