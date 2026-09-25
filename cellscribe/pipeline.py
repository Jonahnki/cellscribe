"""End-to-end orchestration: load -> QC -> preprocess -> cluster -> annotate -> spatial -> report.

:func:`run` is the programmatic entry point used by the CLI. It returns a
:class:`CellscribeResult` holding the processed AnnData, every stage's result
object, and a JSON-serialisable ``summary`` (the only thing the optional
narrative module ever sees).
"""

from __future__ import annotations

import json
import time
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np

from cellscribe import __version__
from cellscribe.config import CellscribeConfig
from cellscribe.utils import CLUSTER_KEY, detect_batch_key, infer_species, logger, seed_everything

ProgressFn = Callable[[str], None]


@dataclass
class CellscribeResult:
    adata: ad.AnnData
    summary: dict
    config: CellscribeConfig
    qc: Any = None
    preprocessing: Any = None
    clustering: Any = None
    annotation: Any = None
    spatial: Any = None
    narrative: Any = None
    report_path: Path | None = None
    timings: dict[str, float] = field(default_factory=dict)


@contextmanager
def _quiet_dependencies():
    """Silence deprecation chatter from scientific dependencies during a run."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module=r"(anndata|scanpy|squidpy|numba|umap|sklearn)")
        warnings.filterwarnings("ignore", message=r".*[Nn]umba.*")
        yield


def software_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {"cellscribe": __version__}
    for pkg in (
        "scanpy", "anndata", "squidpy", "numpy", "scipy", "pandas", "scikit-learn",
        "leidenalg", "python-igraph", "umap-learn", "plotly", "harmonypy", "anthropic",
    ):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    import sys

    out["python"] = sys.version.split()[0]
    return out


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _key_findings(summary: dict) -> list[dict]:
    """Deterministic, plain-language bullet points for the executive summary."""
    out: list[dict] = []
    qc, ds = summary["qc"], summary["dataset"]
    clust, ann = summary["clustering"], summary["annotation"]
    n = ds["n_cells"]
    pct_flag = 100.0 * qc["n_flagged"] / max(qc["n_cells_input"] - qc["n_cells_below_floor"], 1)
    mode_txt = (
        "kept in the analysis and labelled" if qc["mode"] == "flag" else "removed before clustering"
    )
    sev = "warning" if pct_flag > 20 else ("caution" if pct_flag > 10 else "info")
    out.append(
        {
            "severity": sev,
            "message": f"{qc['n_flagged']:,} of {qc['n_cells_input'] - qc['n_cells_below_floor']:,} cells "
            f"({pct_flag:.1f}%) were flagged by QC ({mode_txt}).",
        }
    )
    how = "automatically selected" if clust["auto_selected"] else "user-specified"
    out.append(
        {
            "severity": "info",
            "message": f"{n:,} cells form {clust['n_clusters']} cluster{'s' if clust['n_clusters'] != 1 else ''} "
            f"(Leiden resolution {clust['resolution']:g}, {how}).",
        }
    )
    statuses = [c["status"] for c in ann["clusters"]]
    n_ann = sum(s in ("annotated", "lineage", "mixed") for s in statuses)
    n_high = sum(c["confidence"] == "high" for c in ann["clusters"])
    n_unres = statuses.count("unresolved")
    out.append(
        {
            "severity": "info" if n_unres == 0 else "caution",
            "message": f"{n_ann} of {len(statuses)} clusters matched a reference cell type ({n_high} with high "
            f"confidence); {n_unres} {'is' if n_unres == 1 else 'are'} unresolved and need expert review.",
        }
    )
    labels = {c["cluster"]: c["label"] for c in ann["clusters"]}
    for row in summary["cluster_qc"]:
        if row["status"] == "warning":
            first = row["reasons"][0]["message"]
            out.append(
                {
                    "severity": "warning",
                    "message": f"Cluster {row['cluster']} ({labels.get(row['cluster'], '?')}, {row['n_cells']:,} cells) "
                    f"looks like a technical artefact: {first}",
                }
            )
    for note in summary["notes"]:
        important = note["severity"] in ("warning", "caution") or note.get("source") == "clustering"
        if important and note.get("source") != "annotation":
            out.append({"severity": note["severity"], "message": note["message"]})
    sp = summary.get("spatial")
    if sp and sp.get("enabled"):
        if sp["colocalised_pairs"]:
            p = sp["colocalised_pairs"][0]
            out.append(
                {
                    "severity": "info",
                    "message": f"Strongest spatial co-localisation: cluster {p['a']} ({labels.get(p['a'], '?')}) with "
                    f"cluster {p['b']} ({labels.get(p['b'], '?')}), z = {p['z']:.1f}.",
                }
            )
        out.append(
            {
                "severity": "info",
                "message": f"{sp['n_genes_significant']} of {sp['n_genes_autocorr']} tested genes show significant spatial "
                "autocorrelation (Moran's I, FDR < 0.05).",
            }
        )
    return out


def run(
    input: str | Path | ad.AnnData,
    output: str | Path | None = "cellscribe_report.html",
    config: CellscribeConfig | None = None,
    *,
    save_h5ad: str | Path | None = None,
    summary_json: str | Path | None = None,
    progress: ProgressFn | None = None,
) -> CellscribeResult:
    """Run the full Cellscribe pipeline and (optionally) write the HTML report.

    Parameters
    ----------
    input
        Path to a supported input (see :func:`cellscribe.loaders.load`) or an
        AnnData object with raw counts (it is copied, not modified).
    output
        Report path; ``None`` skips rendering.
    config
        Run configuration; defaults to :class:`CellscribeConfig()`.
    save_h5ad / summary_json
        Optional paths for the processed AnnData and the structured summary.
    """
    from cellscribe import annotation, clustering, preprocessing, qc, spatial
    from cellscribe.loaders import load, standardize_in_memory

    cfg = config or CellscribeConfig()
    say = progress or (lambda msg: logger.info(msg))
    timings: dict[str, float] = {}
    t_start = time.perf_counter()
    seed_everything(cfg.seed)

    def stage(name: str):
        timings[name] = time.perf_counter()

    def done(name: str):
        timings[name] = time.perf_counter() - timings[name]

    with _quiet_dependencies():
        stage("load")
        if isinstance(input, ad.AnnData):
            say("Loading in-memory AnnData")
            adata = standardize_in_memory(input.copy(), sample_name=cfg.input.sample_name)
        else:
            say(f"Loading {input}")
            adata = load(
                input, cfg.input.format, sample_name=cfg.input.sample_name, csv_orientation=cfg.input.csv_orientation
            )
        info = dict(adata.uns["cellscribe_input"])
        modality = info["modality"]
        done("load")

        if cfg.input.species == "auto":
            species, species_reason = infer_species(adata.var_names)
        else:
            species, species_reason = cfg.input.species, "set in configuration"
        batch_key = detect_batch_key(adata.obs, cfg.input.batch_key)
        batch_counts = adata.obs[batch_key].value_counts().to_dict() if batch_key else {}
        say(
            f"{adata.n_obs:,} cells x {adata.n_vars:,} genes | format={info['format']} modality={modality} "
            f"species={species}" + (f" | {len(batch_counts)} batches in '{batch_key}'" if batch_key else "")
        )

        stage("qc")
        say("Quality control: MAD outliers and doublets")
        adata, qc_res = qc.run_qc(adata, cfg.qc, modality, batch_key=batch_key, seed=cfg.seed)
        done("qc")

        stage("preprocessing")
        say("Normalisation, highly variable genes and PCA")
        prep_res = preprocessing.run(adata, cfg.preprocessing, batch_key, seed=cfg.seed)
        done("preprocessing")

        stage("clustering")
        say("Leiden clustering" + (" with automatic resolution selection" if cfg.clustering.resolution is None else ""))
        clust_res = clustering.run(adata, cfg.clustering, prep_res.use_rep, prep_res.n_pcs_used, seed=cfg.seed)
        done("clustering")

        stage("annotation")
        say(f"Marker genes and reference annotation ({clust_res.n_clusters} clusters)")
        ann_res = annotation.run(adata, cfg.annotation, species, modality=modality)
        done("annotation")

        stage("cluster_qc")
        n_sig = {c.cluster: c.n_sig_markers for c in ann_res.clusters}
        mixed = {c.cluster: c.mixed_lineages for c in ann_res.clusters if c.mixed_lineages}
        cluster_table = qc.cluster_qc(
            adata, CLUSTER_KEY, cfg.qc, qc_res, markers=ann_res.marker_tables, batch_key=batch_key,
            n_sig_markers=n_sig, mixed_lineages=mixed,
        )
        qc_res.cluster_table = cluster_table
        done("cluster_qc")

        spatial_res = None
        if cfg.spatial.enabled != "off" and spatial.has_spatial(adata):
            stage("spatial")
            say("Spatial statistics (neighbourhood enrichment, Moran's I)")
            spatial_res = spatial.run(adata, cfg.spatial, modality, seed=cfg.seed)
            done("spatial")
        elif cfg.spatial.enabled == "on":
            logger.warning("Spatial analysis was requested but the input has no spatial coordinates.")

    # ------------------------------------------------------------ summary
    notes: list[dict] = []
    for msg in info.get("notes", []):
        sev = "caution" if msg.startswith("AMBIGUOUS") or "non-integer" in msg else "info"
        notes.append({"severity": sev, "message": msg, "source": "input"})
    for n in qc_res.notes:
        notes.append({**n, "source": "qc"})
    for n in ann_res.notes:
        notes.append({**n, "source": "annotation"})
    if spatial_res is not None:
        for n in spatial_res.notes:
            notes.append({**n, "source": "spatial"})
    if batch_key and prep_res.integration == "none":
        notes.append(
            {
                "severity": "caution",
                "message": f"{len(batch_counts)} samples/batches were detected in '{batch_key}' but no batch integration "
                "was applied. If clusters separate by sample (see the per-cluster QC table), consider re-running with "
                "--integrate harmony.",
                "source": "preprocessing",
            }
        )
    if cfg.preprocessing.integrate == "harmony" and prep_res.integration == "none":
        notes.append(
            {
                "severity": "caution",
                "message": "Harmony integration was requested but only one sample/batch was found "
                f"({'no batch column detected' if not cfg.input.batch_key else repr(cfg.input.batch_key) + ' has a single value'}), "
                "so no integration was applied.",
                "source": "preprocessing",
            }
        )
    sils = [r["silhouette"] for r in clust_res.sweep if r["n_clusters"] > 1 and np.isfinite(r["silhouette"])]
    if clust_res.n_clusters == 1:
        notes.append(
            {
                "severity": "info",
                "message": "The cells form a single population: no reproducible cluster structure was found at any tested "
                "resolution. This is expected for sorted or homogeneous samples; use --resolution to force a finer split.",
                "source": "clustering",
            }
        )
    elif sils and max(sils) < 0.05:
        notes.append(
            {
                "severity": "caution",
                "message": f"Clusters are only weakly separated (best mean silhouette {max(sils):.2f}). The cells may form one "
                "continuous population, so cluster boundaries should not be over-interpreted.",
                "source": "clustering",
            }
        )
    if prep_res.integration == "harmony":
        notes.append(
            {
                "severity": "info",
                "message": f"Harmony batch integration was applied over '{batch_key}' (opt-in). Clusters and UMAP are "
                "computed on the integrated embedding; marker genes use uncorrected expression.",
                "source": "preprocessing",
            }
        )

    obs = adata.obs
    summary: dict[str, Any] = {
        "cellscribe_version": __version__,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "input": {
            **{k: v for k, v in info.items() if k != "notes"},
            "species": species,
            "species_reason": species_reason,
            "batch_key": batch_key,
            "batches": {str(k): int(v) for k, v in batch_counts.items()},
        },
        "dataset": {
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "median_counts": float(np.median(obs["total_counts"])),
            "median_genes": float(np.median(obs["n_genes_by_counts"])),
            "median_pct_mt": float(np.median(obs["pct_counts_mt"])) if qc_res.has_mt else None,
        },
        "qc": qc_res.to_dict(),
        "preprocessing": prep_res.to_dict(),
        "clustering": clust_res.to_dict(),
        "annotation": ann_res.to_dict(),
        "cluster_qc": cluster_table.to_dict(orient="records"),
        "spatial": spatial_res.to_dict() if spatial_res is not None else None,
        "notes": notes,
        "config": cfg.model_dump(mode="json"),
        "software": software_versions(),
    }
    summary = _jsonable(summary)
    summary["key_findings"] = _key_findings(summary)

    result = CellscribeResult(
        adata=adata,
        summary=summary,
        config=cfg,
        qc=qc_res,
        preprocessing=prep_res,
        clustering=clust_res,
        annotation=ann_res,
        spatial=spatial_res,
        timings=timings,
    )

    # ------------------------------------------------------------ narrative
    from cellscribe import narrative

    result.narrative = narrative.generate(summary, cfg.narrative)
    if result.narrative.status == "generated":
        say("AI narrative generated")
    summary["narrative"] = result.narrative.to_dict()

    timings["total"] = time.perf_counter() - t_start
    summary["runtime_seconds"] = round(timings["total"], 1)

    if output is not None:
        from cellscribe.report.render import render_report

        say("Rendering report")
        result.report_path = render_report(result, output)
    if summary_json is not None:
        Path(summary_json).write_text(json.dumps(summary, indent=2))
    if save_h5ad is not None:
        _write_h5ad(adata, save_h5ad)
    return result


def _write_h5ad(adata: ad.AnnData, path: str | Path) -> None:
    out = adata.copy()
    # uns values that h5ad cannot store (nested lists of dicts) are serialised as JSON strings
    info = out.uns.get("cellscribe_input", {})
    out.uns["cellscribe_input"] = json.dumps(_jsonable(info))
    out.uns.pop("_all_spots", None)
    out.write_h5ad(path, compression="gzip")
