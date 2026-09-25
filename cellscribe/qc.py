"""Quality control: per-cell metrics, MAD-based outliers, doublets, per-cluster QC.

Cell-level outliers are defined with the median absolute deviation (MAD)
rather than fixed universal cutoffs, following the data-adaptive approach of
scater's ``isOutlier`` (McCarthy et al. 2017; Lun et al. 2016) as recommended
in current best-practice guides (Heumos et al., Nat Rev Genet 2023; Germain
et al., Genome Biol 2020). A cell is flagged when it lies more than ``nmads``
MADs from the median on any metric (log scale for counts and genes), with
per-batch thresholds when several batches are present.

The per-cluster breakdown (:func:`cluster_qc`) runs after clustering and
surfaces clusters that are likely artefacts (damaged cells, doublets,
dissociation stress) rather than genuine cell types.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

from cellscribe.config import QCConfig
from cellscribe.utils import DISSOCIATION_GENES, is_hb, is_mito, is_ribo, logger

_DAMAGE_CODES = {"high_mt", "above_max_mt", "low_counts", "low_genes"}
_BENIGN_HINTS = {
    "low_complexity": "This is typical of cell types dominated by a few very highly expressed transcripts (e.g. plasma "
    "cells, erythrocytes, acinar cells) but can also indicate poor-quality libraries; check the markers.",
    "high_counts": "Large libraries can come from large cells or from doublets; check whether the cluster co-expresses "
    "markers of two unrelated cell types.",
    "high_genes": "Many detected genes can come from large, transcriptionally complex cells or from doublets.",
}

REASONS = {
    "low_counts": "very low total counts (possible empty droplet or damaged cell)",
    "high_counts": "unusually high total counts (possible doublet)",
    "low_genes": "very few detected genes",
    "high_genes": "unusually many detected genes (possible doublet)",
    "low_complexity": "a few genes dominate the library (low complexity)",
    "high_mt": "high mitochondrial fraction (possible damaged or dying cell)",
    "above_max_mt": "mitochondrial fraction above the configured hard ceiling",
    "doublet": "predicted doublet (Scrublet)",
}

METRIC_LABELS = {
    "total_counts": "Total counts (UMIs) per cell",
    "n_genes_by_counts": "Detected genes per cell",
    "pct_counts_in_top_20_genes": "% counts in top 20 genes",
    "pct_counts_mt": "% mitochondrial counts",
    "pct_counts_ribo": "% ribosomal-protein counts",
    "pct_counts_hb": "% haemoglobin counts",
    "doublet_score": "Scrublet doublet score",
}


@dataclass
class Threshold:
    metric: str
    label: str
    direction: str  # "both" | "upper"
    nmads: float
    log_scale: bool
    median: float
    mad: float
    lower: float | None
    upper: float | None
    n_low: int
    n_high: int
    batch: str | None = None


@dataclass
class QCResult:
    mode: str
    n_cells_input: int
    n_genes_input: int
    min_genes: int
    n_cells_below_floor: int
    n_genes_removed: int
    has_mt: bool
    has_ribo: bool
    thresholds: list[Threshold] = field(default_factory=list)
    skipped_metrics: list[str] = field(default_factory=list)
    n_outliers: int = 0
    n_flagged: int = 0
    n_removed_by_filter: int = 0
    reason_counts: dict[str, int] = field(default_factory=dict)
    doublets: dict = field(default_factory=dict)
    notes: list[dict] = field(default_factory=list)
    cluster_table: pd.DataFrame | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("cluster_table", None)
        return d


# ------------------------------------------------------------------ metrics


def compute_qc_metrics(adata: ad.AnnData) -> tuple[bool, bool]:
    """Add scanpy QC metrics to ``adata.obs`` / ``adata.var``. Returns (has_mt, has_ribo)."""
    import scanpy as sc

    genes = adata.var_names
    adata.var["mt"] = is_mito(genes)
    adata.var["ribo"] = is_ribo(genes)
    adata.var["hb"] = is_hb(genes)
    percent_top = [20] if adata.n_vars > 20 else None
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo", "hb"], percent_top=percent_top, log1p=True, inplace=True
    )
    if "control_counts" in adata.obs:
        denom = adata.obs["total_counts"] + adata.obs["control_counts"]
        adata.obs["pct_counts_control"] = np.where(denom > 0, 100 * adata.obs["control_counts"] / denom, 0.0)
    return bool(adata.var["mt"].any()), bool(adata.var["ribo"].any())


def _auto_min_genes(adata: ad.AnnData, modality: str) -> int:
    if modality == "xenium" or adata.n_vars < 1000:
        # targeted panels: scale the floor with panel size (3 genes for <150-gene panels, up to 10)
        return int(max(3, min(10, adata.n_vars // 50)))
    if modality == "visium":
        return 100
    return 200


def apply_floor(adata: ad.AnnData, cfg: QCConfig, modality: str) -> tuple[ad.AnnData, int, int, int]:
    """Remove near-empty barcodes and genes seen in almost no cells.

    This minimal floor is applied in both QC modes; it only removes barcodes
    that carry essentially no information.
    """
    from cellscribe.errors import PipelineError

    if adata.n_vars < 20:
        raise PipelineError(
            f"The input has only {adata.n_vars} genes; Cellscribe needs at least 20 to cluster and annotate cells."
        )
    min_genes = cfg.min_genes if cfg.min_genes is not None else _auto_min_genes(adata, modality)
    n_genes_per_cell = np.asarray((adata.X > 0).sum(axis=1)).ravel()
    keep_cells = n_genes_per_cell >= max(min_genes, 1)
    n_cells_removed = int((~keep_cells).sum())
    if keep_cells.sum() < 20:
        raise PipelineError(
            f"Only {int(keep_cells.sum())} cells have at least {min_genes} detected genes; too few to analyse. "
            "Check that the input is a cell-level counts matrix (and not, e.g., transposed)."
        )
    if n_cells_removed:
        adata = adata[keep_cells].copy()
    n_cells_per_gene = np.asarray((adata.X > 0).sum(axis=0)).ravel()
    keep_genes = n_cells_per_gene >= cfg.min_cells_per_gene
    n_genes_removed = int((~keep_genes).sum())
    if n_genes_removed:
        adata = adata[:, keep_genes].copy()
    return adata, min_genes, n_cells_removed, n_genes_removed


def mad_bounds(values: np.ndarray, nmads: float, direction: str) -> tuple[float, float, float | None, float | None]:
    """Median, scaled MAD and lower/upper bounds (same scale as ``values``).

    Uses the normal-consistent MAD (scale 1.4826), as in scater's isOutlier.
    Returns ``(median, mad, lower, upper)``; bounds are None when MAD is zero.
    """
    values = np.asarray(values, dtype=np.float64)
    med = float(np.median(values))
    mad = float(median_abs_deviation(values, scale="normal"))
    if mad <= 0 or not np.isfinite(mad):
        return med, mad, None, None
    lower = med - nmads * mad if direction == "both" else None
    upper = med + nmads * mad
    return med, mad, lower, upper


def _metric_specs(adata: ad.AnnData, cfg: QCConfig, has_mt: bool) -> list[tuple[str, str, bool, float, tuple[str, str]]]:
    """(obs column, direction, log scale, nmads, (low reason, high reason))."""
    specs = [
        ("total_counts", "both", True, cfg.nmads, ("low_counts", "high_counts")),
        ("n_genes_by_counts", "both", True, cfg.nmads, ("low_genes", "high_genes")),
    ]
    if adata.n_vars >= 1000 and "pct_counts_in_top_20_genes" in adata.obs:
        specs.append(("pct_counts_in_top_20_genes", "upper", False, cfg.nmads, ("", "low_complexity")))
    if has_mt:
        specs.append(("pct_counts_mt", "upper", False, cfg.nmads_mt, ("", "high_mt")))
    return specs


def flag_outliers(adata: ad.AnnData, cfg: QCConfig, has_mt: bool, batch_key: str | None) -> tuple[list[Threshold], list[str]]:
    """Set ``obs['qc_outlier']`` and ``obs['qc_reasons']``; return thresholds and skipped metrics."""
    n = adata.n_obs
    reasons = [[] for _ in range(n)]
    thresholds: list[Threshold] = []
    skipped: list[str] = []
    groups = [(None, np.ones(n, dtype=bool))]
    if batch_key is not None:
        groups = [(str(b), (adata.obs[batch_key] == b).to_numpy()) for b in adata.obs[batch_key].unique()]

    for col, direction, log_scale, nmads, (low_code, high_code) in _metric_specs(adata, cfg, has_mt):
        raw = adata.obs[col].to_numpy(dtype=np.float64)
        vals = np.log1p(raw) if log_scale else raw
        for batch, mask in groups:
            if mask.sum() < 10:
                continue
            med, mad, lo, hi = mad_bounds(vals[mask], nmads, direction)
            if hi is None:
                skipped.append(f"{METRIC_LABELS.get(col, col)}{f' ({batch})' if batch else ''}: no variation (MAD = 0)")
                continue
            idx = np.flatnonzero(mask)
            low = vals[idx] < lo if lo is not None else np.zeros(idx.size, dtype=bool)
            high = vals[idx] > hi
            for i in idx[low]:
                reasons[i].append(low_code)
            for i in idx[high]:
                reasons[i].append(high_code)
            back = np.expm1 if log_scale else (lambda x: x)
            thresholds.append(
                Threshold(
                    metric=col,
                    label=METRIC_LABELS.get(col, col),
                    direction=direction,
                    nmads=nmads,
                    log_scale=log_scale,
                    median=float(back(med)),
                    mad=mad,
                    lower=float(max(back(lo), 0.0)) if lo is not None else None,
                    upper=float(back(hi)),
                    n_low=int(low.sum()),
                    n_high=int(high.sum()),
                    batch=batch,
                )
            )
    if has_mt and cfg.max_pct_mt is not None:
        over = adata.obs["pct_counts_mt"].to_numpy() > cfg.max_pct_mt
        for i in np.flatnonzero(over):
            reasons[i].append("above_max_mt")

    adata.obs["qc_reasons"] = [",".join(dict.fromkeys(r)) for r in reasons]
    adata.obs["qc_outlier"] = np.array([bool(r) for r in reasons])
    return thresholds, skipped


def run_doublets(adata: ad.AnnData, cfg: QCConfig, modality: str, batch_key: str | None, seed: int) -> dict:
    """Scrublet doublet scores (scanpy's implementation of Wolock et al. 2019).

    Requires raw counts in ``adata.X``. Returns a status dict for the report.
    """
    import scanpy as sc

    if cfg.doublets == "off":
        return {"status": "skipped", "reason": "Doublet detection disabled in the configuration."}
    if cfg.doublets == "auto" and modality != "single-cell":
        reason = (
            "Visium spots contain several cells by design, so droplet-doublet detection does not apply."
            if modality == "visium"
            else "Scrublet models droplet doublets; it is not designed for segmented cells from imaging-based spatial data."
        )
        return {"status": "skipped", "reason": reason}
    if adata.n_obs < 200:
        return {"status": "skipped", "reason": f"Too few cells ({adata.n_obs}) for reliable doublet detection."}
    n_pcs = int(min(30, adata.n_vars - 1, adata.n_obs - 1))
    try:
        sc.pp.scrublet(
            adata,
            batch_key=batch_key,
            expected_doublet_rate=cfg.expected_doublet_rate,
            n_prin_comps=n_pcs,
            # exact kNN is both exact and ~15x faster here than scanpy's default approximate search
            use_approx_neighbors=False,
            random_state=seed,
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 - never let doublet calling abort the report
        logger.warning("Scrublet failed: %s", str(exc).splitlines()[0][:200])
        for col in ("doublet_score", "predicted_doublet"):
            if col in adata.obs:
                del adata.obs[col]
        short = str(exc).splitlines()[0][:200]
        return {"status": "failed", "reason": f"Scrublet could not run on this dataset ({type(exc).__name__}: {short})."}
    pred = adata.obs["predicted_doublet"].astype("boolean").fillna(False).astype(bool)
    adata.obs["predicted_doublet"] = pred.to_numpy()
    info = adata.uns.get("scrublet", {})
    thresholds: dict[str, float] = {}
    if isinstance(info, dict):
        if info.get("threshold") is not None:
            thresholds["all cells"] = float(info["threshold"])
        for b, binfo in (info.get("batches") or {}).items():
            if isinstance(binfo, dict) and binfo.get("threshold") is not None:
                thresholds[str(b)] = float(binfo["threshold"])
    n_pred = int(pred.sum())
    return {
        "status": "run",
        "method": "Scrublet (scanpy.pp.scrublet)",
        "expected_rate": cfg.expected_doublet_rate,
        "thresholds": thresholds,
        "n_predicted": n_pred,
        "pct_predicted": 100.0 * n_pred / adata.n_obs,
        "median_score": float(np.median(adata.obs["doublet_score"])),
    }


def run_qc(
    adata: ad.AnnData,
    cfg: QCConfig,
    modality: str,
    batch_key: str | None = None,
    seed: int = 0,
) -> tuple[ad.AnnData, QCResult]:
    """Full cell-level QC. Returns the (floor-filtered, possibly outlier-filtered) AnnData and a QCResult."""
    n_cells_in, n_genes_in = adata.n_obs, adata.n_vars
    adata, min_genes, n_below, n_genes_removed = apply_floor(adata, cfg, modality)
    has_mt, has_ribo = compute_qc_metrics(adata)
    thresholds, skipped = flag_outliers(adata, cfg, has_mt, batch_key)
    doublets = run_doublets(adata, cfg, modality, batch_key, seed)

    flagged = adata.obs["qc_outlier"].to_numpy().copy()
    if "predicted_doublet" in adata.obs:
        flagged |= adata.obs["predicted_doublet"].to_numpy(dtype=bool)
    adata.obs["qc_flagged"] = flagged

    reason_counts: dict[str, int] = {}
    for r in adata.obs["qc_reasons"]:
        for code in filter(None, r.split(",")):
            reason_counts[code] = reason_counts.get(code, 0) + 1
    if "predicted_doublet" in adata.obs:
        reason_counts["doublet"] = int(adata.obs["predicted_doublet"].sum())

    result = QCResult(
        mode=cfg.mode,
        n_cells_input=n_cells_in,
        n_genes_input=n_genes_in,
        min_genes=min_genes,
        n_cells_below_floor=n_below,
        n_genes_removed=n_genes_removed,
        has_mt=has_mt,
        has_ribo=has_ribo,
        thresholds=thresholds,
        skipped_metrics=skipped,
        n_outliers=int(adata.obs["qc_outlier"].sum()),
        n_flagged=int(flagged.sum()),
        reason_counts=reason_counts,
        doublets=doublets,
    )
    result.notes = _dataset_notes(adata, result, modality)

    if cfg.mode == "filter" and flagged.any():
        result.n_removed_by_filter = int(flagged.sum())
        adata = adata[~flagged].copy()
        # genes may now be all-zero; drop them so downstream steps are well defined
        detected = np.asarray((adata.X > 0).sum(axis=0)).ravel() > 0
        if not detected.all():
            adata = adata[:, detected].copy()
    return adata, result


def _dataset_notes(adata: ad.AnnData, res: QCResult, modality: str) -> list[dict]:
    notes: list[dict] = []
    if not res.has_mt:
        msg = "No mitochondrial genes (MT-/mt- prefix) were found, so mitochondrial QC was not possible."
        if modality == "xenium":
            msg += " This is expected for targeted imaging panels."
        notes.append({"severity": "info", "message": msg})
    for s in res.skipped_metrics:
        notes.append({"severity": "info", "message": f"Outlier detection skipped for {s}."})
    if res.doublets.get("status") != "run":
        notes.append({"severity": "info", "message": "Doublet detection: " + res.doublets.get("reason", "not run.")})
    else:
        detected = res.doublets["pct_predicted"]
        expected = 100 * res.doublets["expected_rate"]
        if detected < expected / 3:
            notes.append(
                {
                    "severity": "caution",
                    "message": f"Scrublet's automatic threshold called only {detected:.1f}% of cells doublets, well below the "
                    f"~{expected:.0f}% expected rate. The automatic threshold is often conservative: check the doublet-score "
                    "histogram and the per-cluster doublet scores, and treat clusters with elevated scores with suspicion.",
                }
            )
    pct = 100.0 * res.n_flagged / max(adata.n_obs, 1)
    if pct > 20:
        notes.append(
            {
                "severity": "warning",
                "message": f"{pct:.0f}% of cells were flagged by QC. This is high and may indicate a sample-level "
                "problem (e.g. low viability, over-digestion, or a wetting/clog failure).",
            }
        )
    med_genes = float(np.median(adata.obs["n_genes_by_counts"]))
    if modality == "single-cell" and adata.n_vars >= 5000 and med_genes < 500:
        notes.append(
            {
                "severity": "caution",
                "message": f"The median cell has only {med_genes:.0f} detected genes, which is low for whole-transcriptome "
                "data and suggests low sequencing depth or poor library quality.",
            }
        )
    if adata.n_obs < 500:
        notes.append(
            {
                "severity": "caution",
                "message": f"Only {adata.n_obs} cells passed the minimal floor; clusters and markers will be less stable.",
            }
        )
    return notes


# ----------------------------------------------------------------- per-cluster


def _reason(code: str, severity: str, message: str) -> dict:
    return {"code": code, "severity": severity, "message": message}


def cluster_qc(
    adata: ad.AnnData,
    cluster_key: str,
    cfg: QCConfig,
    qc: QCResult,
    markers: dict[str, pd.DataFrame] | None = None,
    batch_key: str | None = None,
    n_sig_markers: dict[str, int] | None = None,
    mixed_lineages: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Per-cluster QC summary with plain-language flags.

    ``markers`` maps cluster -> ranked marker table (column ``gene``), used to
    detect clusters whose top markers are mitochondrial, ribosomal, or
    dissociation-stress genes. ``n_sig_markers`` gives the number of
    significant up-regulated markers per cluster. ``mixed_lineages`` maps
    clusters that co-express markers of unrelated lineages (from annotation)
    to those lineages; combined with elevated doublet scores this escalates
    to a "likely doublets" warning.
    """
    mixed_lineages = mixed_lineages or {}
    obs = adata.obs
    clusters = obs[cluster_key].cat.categories if hasattr(obs[cluster_key], "cat") else sorted(obs[cluster_key].unique())
    log_counts = np.log1p(obs["total_counts"].to_numpy())
    log_genes = np.log1p(obs["n_genes_by_counts"].to_numpy())
    g_med_c, g_mad_c = np.median(log_counts), median_abs_deviation(log_counts, scale="normal")
    g_med_g, g_mad_g = np.median(log_genes), median_abs_deviation(log_genes, scale="normal")
    has_dbl = "doublet_score" in obs
    if qc.has_mt:
        mt = obs["pct_counts_mt"].to_numpy()
        g_med_mt, g_mad_mt = np.median(mt), median_abs_deviation(mt, scale="normal")
    if has_dbl:
        dscore = obs["doublet_score"].to_numpy()
        g_p95_dbl = float(np.quantile(dscore, 0.95))
        g_med_dbl = float(np.median(dscore))
    batch_frac = None
    if batch_key is not None and batch_key in obs and obs[batch_key].nunique() > 1:
        batch_frac = obs[batch_key].value_counts(normalize=True)

    rows = []
    for c in clusters:
        m = (obs[cluster_key] == c).to_numpy()
        sub = obs.loc[m]
        n = int(m.sum())
        row = {
            "cluster": str(c),
            "n_cells": n,
            "pct_cells": 100.0 * n / adata.n_obs,
            "median_counts": float(sub["total_counts"].median()),
            "median_genes": float(sub["n_genes_by_counts"].median()),
            "median_pct_mt": float(sub["pct_counts_mt"].median()) if qc.has_mt else np.nan,
            "median_pct_ribo": float(sub["pct_counts_ribo"].median()) if qc.has_ribo else np.nan,
            "mean_doublet_score": float(sub["doublet_score"].mean()) if has_dbl else np.nan,
            "pct_doublets": 100.0 * float(sub["predicted_doublet"].mean()) if has_dbl else np.nan,
            "pct_outliers": 100.0 * float(sub["qc_outlier"].mean()),
        }
        reasons: list[dict] = []

        # (a) most cells fail cell-level QC. Damage signals (mitochondrial fraction, very small
        # libraries) make this a likely artefact; other reasons (low complexity, large libraries)
        # are also typical of some genuine cell types, so they only ask for review.
        frac_out = sub["qc_outlier"].mean()
        if frac_out >= cfg.cluster_flag_fraction:
            out_reasons = [r.split(",") for r in sub.loc[sub["qc_outlier"], "qc_reasons"]]
            codes = pd.Series([c2 for r in out_reasons for c2 in r if c2]).value_counts()
            top = ", ".join(REASONS[k].split(" (")[0] for k in codes.index[:2])
            damaged = np.mean([bool(set(r) & _DAMAGE_CODES) for r in out_reasons]) if out_reasons else 0.0
            if damaged >= 0.5:
                reasons.append(
                    _reason(
                        "qc_outliers",
                        "warning",
                        f"{frac_out:.0%} of cells in this cluster fail cell-level QC (mainly: {top}).",
                    )
                )
            else:
                hint = _BENIGN_HINTS.get(codes.index[0], "")
                reasons.append(
                    _reason(
                        "qc_outliers",
                        "caution",
                        f"{frac_out:.0%} of cells in this cluster are QC outliers (mainly: {top}). {hint}".strip(),
                    )
                )
        # (b) elevated mitochondrial fraction
        if qc.has_mt and g_mad_mt > 0:
            med_mt = row["median_pct_mt"]
            if med_mt > g_med_mt + cfg.cluster_nmads * g_mad_mt and med_mt > 5:
                reasons.append(
                    _reason(
                        "high_mt",
                        "warning",
                        f"Median mitochondrial fraction is {med_mt:.1f}% vs {g_med_mt:.1f}% across the dataset, "
                        "typical of damaged or dying cells.",
                    )
                )
        # (c) low library size / gene detection
        low_c = g_mad_c > 0 and np.median(log_counts[m]) < g_med_c - cfg.cluster_nmads * g_mad_c
        low_g = g_mad_g > 0 and np.median(log_genes[m]) < g_med_g - cfg.cluster_nmads * g_mad_g
        if low_c or low_g:
            reasons.append(
                _reason(
                    "low_library",
                    "caution",
                    f"Median of {row['median_counts']:,.0f} counts and {row['median_genes']:,.0f} genes per cell is far below "
                    f"the dataset ({np.expm1(g_med_c):,.0f} counts, {np.expm1(g_med_g):,.0f} genes). This can indicate damaged "
                    "cells or empty droplets, although some cell types (e.g. neutrophils, platelets, erythrocytes) are "
                    "naturally small; check the markers before discarding.",
                )
            )
        # (d) doublets
        if has_dbl:
            frac_dbl = sub["predicted_doublet"].mean()
            med_dbl = float(sub["doublet_score"].median())
            if frac_dbl >= cfg.cluster_doublet_fraction:
                reasons.append(
                    _reason(
                        "doublets",
                        "warning",
                        f"{frac_dbl:.0%} of cells are predicted doublets (median Scrublet score {med_dbl:.2f} vs "
                        f"{g_med_dbl:.2f} overall). This cluster may consist of cell doublets rather than a distinct "
                        "cell type.",
                    )
                )
            else:
                # Scrublet's automatic threshold is often conservative, so also look at the
                # score distribution itself: by chance ~5% of a cluster's cells fall in the
                # dataset's top 5% of scores; 4x that (or a doubled median) is suspicious.
                frac_top = float((sub["doublet_score"] > g_p95_dbl).mean())
                elevated = frac_top >= 0.2 or (g_med_dbl > 0 and med_dbl >= 2 * g_med_dbl)
                if elevated and str(c) in mixed_lineages:
                    lin = " and ".join(mixed_lineages[str(c)])
                    reasons.append(
                        _reason(
                            "doublets",
                            "warning",
                            f"Likely doublets: doublet scores are elevated ({frac_top:.0%} of cells in the dataset's top 5%; "
                            f"median {med_dbl:.2f} vs {g_med_dbl:.2f} overall) and the cluster co-expresses markers of "
                            f"unrelated lineages ({lin}).",
                        )
                    )
                elif elevated:
                    reasons.append(
                        _reason(
                            "doublet_scores",
                            "caution",
                            f"Doublet scores are elevated: {frac_top:.0%} of cells are in the dataset's top 5% of Scrublet "
                            f"scores (median {med_dbl:.2f} vs {g_med_dbl:.2f} overall). If this cluster co-expresses markers "
                            "of two unrelated lineages, treat it as doublets.",
                        )
                    )
        # (e) top markers dominated by technical gene classes
        if markers is not None and str(c) in markers:
            top = markers[str(c)]["gene"].head(10).tolist()
            mt_top = [g for g in top if is_mito([g])[0]]
            ribo_top = [g for g in top if is_ribo([g])[0]]
            diss_top = [g for g in top if g.upper() in DISSOCIATION_GENES]
            if len(mt_top) >= 3:
                reasons.append(
                    _reason(
                        "mt_markers",
                        "warning",
                        f"Top marker genes are mitochondrial ({', '.join(mt_top[:4])}), a hallmark of low-quality cells.",
                    )
                )
            if len(ribo_top) >= 5:
                reasons.append(
                    _reason(
                        "ribo_markers",
                        "caution",
                        f"Top marker genes are ribosomal-protein genes ({', '.join(ribo_top[:4])}). This can be biological "
                        "(e.g. naive lymphocytes) but also reflects low-complexity libraries.",
                    )
                )
            if len(diss_top) >= 3:
                reasons.append(
                    _reason(
                        "stress_markers",
                        "caution",
                        f"Top markers include immediate-early/heat-shock genes ({', '.join(diss_top[:4])}), which are induced "
                        "by tissue dissociation and may reflect handling stress rather than cell identity.",
                    )
                )
        # (f) no distinctive markers
        if n_sig_markers is not None and len(clusters) > 1 and n_sig_markers.get(str(c), 0) < 5:
            reasons.append(
                _reason(
                    "few_markers",
                    "caution",
                    f"Only {n_sig_markers.get(str(c), 0)} genes are specifically up-regulated in this cluster. It may be an "
                    "intermediate state, a low-quality population, or the result of over-clustering.",
                )
            )
        # (g) single-batch clusters
        if batch_frac is not None and n >= 20:
            comp = sub[batch_key].value_counts(normalize=True)
            top_b = comp.index[0]
            if comp.iloc[0] >= 0.9 and batch_frac[top_b] < 0.7:
                reasons.append(
                    _reason(
                        "single_batch",
                        "caution",
                        f"{comp.iloc[0]:.0%} of cells come from '{top_b}' (which contributes {batch_frac[top_b]:.0%} of all "
                        "cells). This may be sample-specific biology or a batch effect.",
                    )
                )
            row["dominant_batch"] = str(top_b)
            row["dominant_batch_pct"] = 100.0 * float(comp.iloc[0])

        severities = {r["severity"] for r in reasons}
        row["status"] = "warning" if "warning" in severities else ("caution" if "caution" in severities else "pass")
        row["reasons"] = reasons
        rows.append(row)
    table = pd.DataFrame(rows)
    return table
