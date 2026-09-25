"""Plotly figures for the HTML report.

Design rules (applied consistently across every figure):

* One visual system: system sans font, hairline recessive grid, light surface.
* Categorical colour = identity (clusters, batches), assigned in a fixed order
  from a colour-vision-deficiency-validated 8-hue palette. Beyond 8 clusters,
  lighter/darker tiers of the same hues are used and every cluster is also
  labelled directly at its centroid, so identity never relies on colour alone.
* Sequential colour (one blue hue, light -> dark) = magnitude.
* Diverging colour (blue <-> neutral grey <-> red) = signed scores (z-scores).
* Status colours (warning/critical) are reserved for QC flags and always paired
  with a text label or icon.
* Large datasets are randomly subsampled (fixed seed) for interactive scatter
  plots; every figure has a table equivalent elsewhere in the report.
"""

from __future__ import annotations

import base64
import io

import anndata as ad
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from cellscribe.utils import CLUSTER_KEY

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STATUS = {"pass": "#8a96a8", "caution": "#fab219", "warning": "#d03b3b", "good": "#0ca30c"}
STATUS_ICON = {"pass": "✓", "caution": "!", "warning": "⚠"}
SEQUENTIAL = [
    [0.0, "#e8f1fd"], [0.15, "#cde2fb"], [0.3, "#9ec5f4"], [0.45, "#6da7ec"],
    [0.6, "#3987e5"], [0.75, "#256abf"], [0.9, "#184f95"], [1.0, "#0d366b"],
]
DIVERGING = [
    [0.0, "#184f95"], [0.2, "#3987e5"], [0.4, "#9ec5f4"], [0.5, "#f0efec"],
    [0.6, "#f4b3b2"], [0.8, "#e34948"], [1.0, "#a52a2a"],
]
FLAG_SCALE = [[0.0, "#b9c3d1"], [0.5, "#b9c3d1"], [0.5, STATUS["warning"]], [1.0, STATUS["warning"]]]

_template = go.layout.Template()
_axis = dict(
    gridcolor=GRID, linecolor=AXIS, zerolinecolor=GRID, showline=True, ticks="outside",
    tickcolor=AXIS, ticklen=4, title_font=dict(color=INK2, size=12), tickfont=dict(color=INK2, size=11),
)
_template.layout = go.Layout(
    font=dict(family=FONT, size=12, color=INK2),
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    colorway=SERIES,
    margin=dict(l=60, r=20, t=40, b=50),
    xaxis=_axis,
    yaxis=_axis,
    hoverlabel=dict(bgcolor="#ffffff", bordercolor=AXIS, font=dict(family=FONT, color=INK, size=12)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK2, size=11), itemsizing="constant"),
    title=dict(font=dict(color=INK, size=14), x=0.0, xanchor="left"),
    hovermode="closest",
)
pio.templates["cellscribe"] = _template


# ---------------------------------------------------------------- helpers


def _mix(hex_color: str, target: tuple[int, int, int], amount: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    r, g, b = (round(c + (t - c) * amount) for c, t in zip((r, g, b), target))
    return f"#{r:02x}{g:02x}{b:02x}"


def cluster_colors(clusters: list[str]) -> dict[str, str]:
    """Fixed-order palette: 8 validated hues, then darker and lighter tiers of the same hues."""
    tiers = [
        SERIES,
        [_mix(c, (0, 0, 0), 0.35) for c in SERIES],
        [_mix(c, (255, 255, 255), 0.45) for c in SERIES],
    ]
    flat = [c for tier in tiers for c in tier]
    return {cl: flat[i % len(flat)] for i, cl in enumerate(clusters)}


def plot_index(n: int, max_points: int, seed: int = 0) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, max_points, replace=False))


def _layout(fig: go.Figure, height: int, **kw) -> go.Figure:
    fig.update_layout(template="cellscribe", height=height, **kw)
    return fig


def to_div(fig: go.Figure, name: str) -> str:
    """Serialise a figure to an HTML <div> (plotly.js is included once by the page)."""
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        default_width="100%",
        config={
            "displaylogo": False,
            "responsive": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
            "toImageButtonOptions": {"format": "svg", "filename": f"cellscribe_{name}"},
        },
    )


def _log_ticks(lo: float, hi: float) -> tuple[list[float], list[str]]:
    vals, text = [], []
    for k in range(int(np.floor(lo)), int(np.ceil(hi)) + 1):
        for m in (1, 3):
            v = np.log10(m * 10**k)
            if lo - 0.05 <= v <= hi + 0.05:
                vals.append(v)
                x = m * 10**k
                text.append(f"{x / 1000:g}k" if x >= 1000 else f"{x:g}")
    return vals, text


def _nice_log_ticks(values: np.ndarray) -> tuple[list[float], list[str]]:
    """Ticks at 1, 2, 5 x 10^k covering ``values`` (for axes of type 'log')."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return [], []
    lo, hi = np.floor(np.log10(v.min())), np.ceil(np.log10(v.max()))
    mults = (1, 2, 5) if hi - lo <= 3 else (1,)
    ticks = [m * 10**k for k in range(int(lo), int(hi) + 1) for m in mults]
    ticks = [t for t in ticks if v.min() / 1.5 <= t <= v.max() * 1.5]
    return ticks, [f"{t / 1000:g}k" if t >= 1000 else f"{t:g}" for t in ticks]


def _cluster_names(annotation) -> dict[str, str]:
    return {c.cluster: c.label for c in annotation.clusters}


# ---------------------------------------------------------------- QC


def qc_distributions(adata: ad.AnnData, qc) -> go.Figure:
    """Small multiples: distribution of each QC metric with its MAD thresholds."""
    obs = adata.obs
    panels = [("total_counts", "Total counts per cell", True), ("n_genes_by_counts", "Detected genes per cell", True)]
    if qc.has_mt:
        panels.append(("pct_counts_mt", "% mitochondrial counts", False))
    if "pct_counts_in_top_20_genes" in obs and adata.n_vars >= 1000:
        panels.append(("pct_counts_in_top_20_genes", "% counts in top 20 genes", False))
    elif qc.has_ribo:
        panels.append(("pct_counts_ribo", "% ribosomal-protein counts", False))
    ncol = 2
    nrow = int(np.ceil(len(panels) / ncol))
    fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=[p[1] for p in panels], horizontal_spacing=0.1, vertical_spacing=0.2)
    thr_by_metric: dict[str, list] = {}
    for t in qc.thresholds:
        thr_by_metric.setdefault(t.metric, []).append(t)
    n_batch_thr = len({t.batch for t in qc.thresholds})
    for i, (col, title, log) in enumerate(panels):
        r, c = i // ncol + 1, i % ncol + 1
        vals = obs[col].to_numpy(dtype=float)
        x = np.log10(np.maximum(vals, 1)) if log else vals
        fig.add_trace(
            go.Histogram(
                x=x, nbinsx=60, marker=dict(color=SERIES[0], line=dict(width=0)), opacity=0.9,
                name=title, showlegend=False,
                hovertemplate=("%{x:.2f} (log10)" if log else "%{x:.1f}") + "<br>%{y} cells<extra></extra>",
            ),
            row=r, col=c,
        )
        for j, t in enumerate(thr_by_metric.get(col, [])):
            for bound, side in ((t.lower, "lower"), (t.upper, "upper")):
                if bound is None or (bound <= 0 and log):
                    continue
                xv = float(np.log10(max(bound, 1))) if log else float(bound)
                if not (np.nanmin(x) - 1 <= xv <= np.nanmax(x) + 1):
                    continue
                label = f"{side}: {bound:,.0f}" if log else f"{side}: {bound:.1f}"
                if t.batch and n_batch_thr <= 2:
                    label += f" ({t.batch})"
                show_label = n_batch_thr <= 2 or j == 0
                if n_batch_thr > 2 and j == 0:
                    label = f"{side} (per batch, see table)"
                # alternate label sides so neighbouring per-batch lines stay readable
                pos = "top left" if (j % 2 == 0) == (side == "lower") else "top right"
                fig.add_vline(
                    x=xv, line=dict(color=STATUS["warning"], width=1.5), row=r, col=c,
                    annotation=dict(text=label, font=dict(size=10, color=STATUS["warning"]), textangle=-90) if show_label else None,
                    annotation_position=pos if show_label else None,
                )
        if log:
            tv, tt = _log_ticks(float(np.nanmin(x)), float(np.nanmax(x)))
            fig.update_xaxes(tickvals=tv, ticktext=tt, row=r, col=c)
        fig.update_yaxes(title_text="cells" if c == 1 else None, row=r, col=c)
    fig.update_layout(bargap=0.04)
    for ann in fig.layout.annotations[: len(panels)]:
        ann.font = dict(size=12, color=INK)
    return _layout(fig, 300 * nrow + 40)


def qc_scatter(adata: ad.AnnData, max_points: int) -> go.Figure:
    """Library size vs detected genes; cells flagged by QC highlighted."""
    obs = adata.obs
    idx = plot_index(adata.n_obs, max_points)
    sub = obs.iloc[idx]
    flagged = sub["qc_flagged"].to_numpy(dtype=bool)
    fig = go.Figure()
    has_mt = "pct_counts_mt" in sub
    for mask, name, color, size in ((~flagged, "Passes QC", "#8fb4e3", 4), (flagged, "Flagged by QC", STATUS["warning"], 5)):
        s = sub[mask]
        reasons = s["qc_reasons"].str.replace(",", ", ").replace("", "-")
        if "predicted_doublet" in s:
            reasons = np.where(s["predicted_doublet"], reasons + " +doublet", reasons)
        custom = np.column_stack([s["pct_counts_mt"].to_numpy() if has_mt else np.zeros(len(s)), np.asarray(reasons, dtype=object)])
        fig.add_trace(
            go.Scattergl(
                x=s["total_counts"].to_numpy(), y=s["n_genes_by_counts"].to_numpy(), mode="markers", name=f"{name} ({mask.sum():,})",
                marker=dict(size=size, color=color, opacity=0.7, line=dict(width=0)),
                customdata=custom,
                hovertemplate="%{x:,.0f} counts · %{y:,.0f} genes<br>mito %{customdata[0]:.1f}%<br>flags: %{customdata[1]}<extra></extra>",
            )
        )
    tx, ttx = _nice_log_ticks(sub["total_counts"].to_numpy())
    ty, tty = _nice_log_ticks(sub["n_genes_by_counts"].to_numpy())
    fig.update_xaxes(type="log", title_text="Total counts per cell (log scale)", tickvals=tx, ticktext=ttx)
    fig.update_yaxes(type="log", title_text="Detected genes (log scale)", tickvals=ty, ticktext=tty)
    return _layout(fig, 420, legend=dict(orientation="h", y=1.08, x=0))


def doublet_histogram(adata: ad.AnnData, qc) -> go.Figure | None:
    if "doublet_score" not in adata.obs:
        return None
    fig = go.Figure(
        go.Histogram(
            x=adata.obs["doublet_score"].to_numpy(), nbinsx=70, marker=dict(color=SERIES[0], line=dict(width=0)),
            hovertemplate="score %{x:.2f}<br>%{y} cells<extra></extra>", showlegend=False,
        )
    )
    for name, thr in (qc.doublets.get("thresholds") or {}).items():
        fig.add_vline(
            x=thr, line=dict(color=STATUS["warning"], width=1.5),
            annotation=dict(text=f"Scrublet threshold{'' if name == 'all cells' else f' ({name})'}: {thr:.2f}",
                            font=dict(size=10, color=STATUS["warning"]), textangle=-90, yanchor="top"),
        )
    fig.update_xaxes(title_text="Scrublet doublet score")
    fig.update_yaxes(title_text="cells (log scale)", type="log", dtick=1)
    fig.update_layout(bargap=0.04)
    return _layout(fig, 300)


# ---------------------------------------------------------------- embedding


def _embedding(adata: ad.AnnData, basis: str) -> np.ndarray:
    return np.asarray(adata.obsm[basis])[:, :2]


def umap_clusters(adata: ad.AnnData, annotation, max_points: int, basis: str = "X_umap", title: str = "") -> go.Figure:
    names = _cluster_names(annotation)
    clusters = [str(c) for c in adata.obs[CLUSTER_KEY].cat.categories]
    colors = cluster_colors(clusters)
    xy = _embedding(adata, basis)
    idx = plot_index(adata.n_obs, max_points)
    labels = adata.obs[CLUSTER_KEY].astype(str).to_numpy()
    fig = go.Figure()
    for cl in clusters:
        m_all = labels == cl
        m = m_all[idx]
        pts = xy[idx][m]
        fig.add_trace(
            go.Scattergl(
                x=pts[:, 0], y=pts[:, 1], mode="markers", name=f"{cl} · {names.get(cl, '')}",
                marker=dict(size=4, color=colors[cl], opacity=0.85, line=dict(width=0)),
                hovertemplate=f"<b>Cluster {cl}</b><br>{names.get(cl, '')}<br>{int(m_all.sum()):,} cells<extra></extra>",
            )
        )
        if m_all.any():
            cx, cy = np.median(xy[m_all], axis=0)
            fig.add_annotation(
                x=cx, y=cy, text=f"<b>{cl}</b>", showarrow=False,
                font=dict(size=13, color=INK), bgcolor="rgba(252,252,251,0.75)", borderpad=2,
            )
    fig.update_xaxes(showticklabels=False, title_text=f"{'UMAP' if 'umap' in basis else basis} 1", showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, title_text=f"{'UMAP' if 'umap' in basis else basis} 2", showgrid=False, zeroline=False,
                     scaleanchor="x", scaleratio=1)
    return _layout(fig, 560, legend=dict(title=dict(text="Cluster"), itemclick="toggleothers"), title=title or None)


def umap_metrics(adata: ad.AnnData, max_points: int) -> go.Figure:
    """UMAP coloured by a QC metric, switchable with a dropdown."""
    obs = adata.obs
    idx = plot_index(adata.n_obs, max_points)
    xy = _embedding(adata, "X_umap")[idx]
    metrics = [("Total counts (log10)", np.log10(np.maximum(obs["total_counts"].to_numpy(), 1))),
               ("Detected genes (log10)", np.log10(np.maximum(obs["n_genes_by_counts"].to_numpy(), 1)))]
    if "pct_counts_mt" in obs and obs["pct_counts_mt"].max() > 0:
        metrics.append(("% mitochondrial", obs["pct_counts_mt"].to_numpy()))
    if "doublet_score" in obs:
        metrics.append(("Doublet score", obs["doublet_score"].to_numpy()))
    metrics.append(("Flagged by QC (red)", obs["qc_flagged"].to_numpy(dtype=float)))
    first_name, first_vals = metrics[0]
    fig = go.Figure(
        go.Scattergl(
            x=xy[:, 0], y=xy[:, 1], mode="markers",
            marker=dict(size=4, color=first_vals[idx], colorscale=SEQUENTIAL, showscale=True,
                        colorbar=dict(title=dict(text=first_name, side="right"), thickness=12, outlinewidth=0)),
            customdata=first_vals[idx],
            hovertemplate="%{customdata:.2f}<extra></extra>",
            showlegend=False,
        )
    )
    buttons = []
    for name, vals in metrics:
        is_flag = name.startswith("Flagged")
        buttons.append(
            dict(
                label=name, method="restyle",
                args=[{
                    "marker.color": [vals[idx]],
                    "customdata": [vals[idx]],
                    "marker.colorscale": [FLAG_SCALE if is_flag else SEQUENTIAL],
                    "marker.cmin": [0 if is_flag else float(np.nanmin(vals))],
                    "marker.cmax": [1 if is_flag else float(np.nanpercentile(vals, 99.5))],
                    "marker.showscale": [not is_flag],
                    "marker.colorbar.title.text": [name],
                }],
            )
        )
    fig.update_layout(
        updatemenus=[dict(buttons=buttons, direction="down", x=0, y=1.12, xanchor="left", yanchor="top",
                          bgcolor="#ffffff", bordercolor=AXIS, font=dict(size=12))],
    )
    fig.update_xaxes(showticklabels=False, title_text="UMAP 1", showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, title_text="UMAP 2", showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1)
    return _layout(fig, 520, margin=dict(l=40, r=20, t=60, b=40))


def batch_composition(adata: ad.AnnData, batch_key: str, annotation) -> go.Figure:
    """100% stacked bars: sample composition of each cluster."""
    names = _cluster_names(annotation)
    ct = pd.crosstab(adata.obs[CLUSTER_KEY].astype(str), adata.obs[batch_key].astype(str), normalize="index") * 100
    clusters = [str(c) for c in adata.obs[CLUSTER_KEY].cat.categories]
    ct = ct.reindex(clusters)
    batches = list(ct.columns)
    if len(batches) > 8:
        keep = adata.obs[batch_key].astype(str).value_counts().index[:7].tolist()
        ct["Other"] = ct.drop(columns=keep).sum(axis=1)
        ct = ct[keep + ["Other"]]
        batches = keep + ["Other"]
    fig = go.Figure()
    ylabels = [f"{c} · {names.get(c, '')[:28]}" for c in clusters]
    for i, b in enumerate(batches):
        fig.add_trace(
            go.Bar(
                y=ylabels, x=ct[b].to_numpy(), name=b, orientation="h",
                marker=dict(color=SERIES[i % 8], line=dict(color=SURFACE, width=2)),
                hovertemplate=f"{b}: %{{x:.1f}}% of cluster<extra></extra>",
            )
        )
    overall = adata.obs[batch_key].astype(str).value_counts(normalize=True) * 100
    overall_txt = ", ".join(f"{b} {overall.get(b, 0):.0f}%" for b in batches if b != "Other")
    fig.update_layout(
        barmode="stack", bargap=0.25,
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, traceorder="normal",
                    title=dict(text=f"{batch_key} (overall: {overall_txt})  ")),
    )
    fig.update_xaxes(title_text="% of cluster's cells", range=[0, 100], ticksuffix="%")
    fig.update_yaxes(autorange="reversed", title_text=None)
    return _layout(fig, max(260, 28 * len(clusters) + 130), margin=dict(l=210, r=20, t=50, b=50))


# ---------------------------------------------------------------- per-cluster QC


def cluster_qc_boxes(adata: ad.AnnData, table: pd.DataFrame) -> go.Figure:
    """Per-cluster distributions of key QC metrics (precomputed box statistics)."""
    obs = adata.obs
    status = dict(zip(table["cluster"], table["status"]))
    clusters = [str(c) for c in obs[CLUSTER_KEY].cat.categories]
    ticks = [f"{STATUS_ICON[status.get(c, 'pass')]} {c}" if status.get(c, "pass") != "pass" else c for c in clusters]
    metrics = [("total_counts", "Total counts", True), ("n_genes_by_counts", "Detected genes", True)]
    if "pct_counts_mt" in obs and obs["pct_counts_mt"].max() > 0:
        metrics.append(("pct_counts_mt", "% mitochondrial", False))
    if "doublet_score" in obs:
        metrics.append(("doublet_score", "Doublet score", False))
    elif "pct_counts_ribo" in obs and obs["pct_counts_ribo"].max() > 0:
        metrics.append(("pct_counts_ribo", "% ribosomal", False))
    nrow = int(np.ceil(len(metrics) / 2))
    fig = make_subplots(rows=nrow, cols=2, subplot_titles=[m[1] for m in metrics], vertical_spacing=0.16, horizontal_spacing=0.08)
    labels = obs[CLUSTER_KEY].astype(str).to_numpy()
    for i, (col, title, log) in enumerate(metrics):
        r, c = i // 2 + 1, i % 2 + 1
        vals = obs[col].to_numpy(dtype=float)
        for cl, tick in zip(clusters, ticks):
            v = vals[labels == cl]
            if v.size == 0:
                continue
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            iqr = q3 - q1
            lo = float(max(v.min(), q1 - 1.5 * iqr))
            hi = float(min(v.max(), q3 + 1.5 * iqr))
            color = STATUS[status.get(cl, "pass")]
            fig.add_trace(
                go.Box(
                    x=[tick], q1=[q1], median=[med], q3=[q3], lowerfence=[lo], upperfence=[hi],
                    name=tick, marker_color=color, fillcolor=_mix(color, (255, 255, 255), 0.55),
                    line=dict(color=color, width=1.2), showlegend=False,
                    hovertemplate=f"<b>Cluster {cl}</b><br>{title}<br>median %{{median:,.2f}}<br>IQR %{{q1:,.2f}}–%{{q3:,.2f}}<extra></extra>",
                ),
                row=r, col=c,
            )
        if log:
            tv, tt = _nice_log_ticks(vals)
            fig.update_yaxes(type="log", tickvals=tv, ticktext=tt, row=r, col=c)
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=ticks, row=r, col=c, tickangle=0)
    for ann in fig.layout.annotations[: len(metrics)]:
        ann.font = dict(size=12, color=INK)
    return _layout(fig, 280 * nrow + 40)


# ---------------------------------------------------------------- markers


def marker_heatmap(adata: ad.AnnData, annotation, n_per_cluster: int = 4) -> tuple[go.Figure | None, pd.DataFrame | None]:
    """Z-scored mean expression of each cluster's top markers."""
    from cellscribe.annotation import cluster_expression

    genes: list[str] = []
    for c in annotation.clusters:
        added = 0
        for m in c.top_markers:
            if added >= n_per_cluster:
                break
            if m["log2fc"] > 0 and m["gene"] not in genes:
                genes.append(m["gene"])
                added += 1
    if not genes:
        return None, None
    means, pct, _ = cluster_expression(adata[:, genes])
    logm = np.log1p(means)
    z = (logm - logm.mean(axis=0)) / logm.std(axis=0).replace(0, 1)
    names = _cluster_names(annotation)
    ylabels = [f"{c} · {names.get(c, '')[:30]}" for c in z.index]
    custom = np.dstack([means.to_numpy(), pct.to_numpy() * 100])
    fig = go.Figure(
        go.Heatmap(
            z=z.to_numpy(), x=genes, y=ylabels, colorscale=DIVERGING, zmid=0, zmin=-2.5, zmax=2.5,
            customdata=custom, xgap=1, ygap=1,
            colorbar=dict(title=dict(text="z-score", side="right"), thickness=12, outlinewidth=0),
            hovertemplate="<b>%{x}</b> in %{y}<br>z = %{z:.2f}<br>mean expr %{customdata[0]:.2f}<br>"
            "expressed in %{customdata[1]:.0f}% of cells<extra></extra>",
        )
    )
    fig.update_xaxes(tickangle=-60, showgrid=False, tickfont=dict(size=10))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    height = max(320, 26 * len(ylabels) + 170)
    return _layout(fig, height, margin=dict(l=230, r=20, t=20, b=110)), z


# ---------------------------------------------------------------- spatial


def _image_uri(img: np.ndarray, max_side: int = 1400) -> tuple[str, float]:
    from PIL import Image

    im = Image.fromarray(img)
    scale = 1.0
    if max(im.size) > max_side:
        scale = max_side / max(im.size)
        im = im.resize((int(im.size[0] * scale), int(im.size[1] * scale)))
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), scale


def _spatial_base(adata: ad.AnnData, image_info) -> tuple[go.Figure, np.ndarray, bool]:
    """Figure with optional tissue image; returns coordinates in plotting space."""
    coords = np.asarray(adata.obsm["spatial"], dtype=float)[:, :2]
    fig = go.Figure()
    img, sf, _ = image_info
    has_img = img is not None
    if has_img:
        uri, extra = _image_uri(img)
        h, w = img.shape[:2]
        fig.add_layout_image(
            dict(source=uri, xref="x", yref="y", x=0, y=0, sizex=w, sizey=h, sizing="stretch", layer="below", opacity=0.9)
        )
        coords = coords * sf
        fig.update_xaxes(range=[0, w])
        fig.update_yaxes(range=[h, 0])
    else:
        fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, showline=False, ticks="")
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, showline=False, ticks="", scaleanchor="x", scaleratio=1)
    return fig, coords, has_img


def _spot_size(adata: ad.AnnData, coords: np.ndarray, n: int) -> float:
    return float(np.clip(700 / np.sqrt(max(n, 1)), 2.5, 9))


def spatial_clusters(adata: ad.AnnData, annotation, image_info, max_points: int) -> go.Figure:
    names = _cluster_names(annotation)
    clusters = [str(c) for c in adata.obs[CLUSTER_KEY].cat.categories]
    colors = cluster_colors(clusters)
    fig, coords, has_img = _spatial_base(adata, image_info)
    idx = plot_index(adata.n_obs, max_points)
    labels = adata.obs[CLUSTER_KEY].astype(str).to_numpy()
    size = _spot_size(adata, coords, len(idx))
    for cl in clusters:
        m = labels[idx] == cl
        pts = coords[idx][m]
        fig.add_trace(
            go.Scattergl(
                x=pts[:, 0], y=pts[:, 1], mode="markers", name=f"{cl} · {names.get(cl, '')}",
                marker=dict(size=size, color=colors[cl], opacity=0.85 if has_img else 0.9,
                            line=dict(width=0.5 if has_img else 0, color=SURFACE)),
                hovertemplate=f"<b>Cluster {cl}</b><br>{names.get(cl, '')}<extra></extra>",
            )
        )
    return _layout(fig, 600, legend=dict(title=dict(text="Cluster"), itemclick="toggleothers"), margin=dict(l=10, r=10, t=10, b=10))


def spatial_feature(adata: ad.AnnData, values: np.ndarray, title: str, image_info, max_points: int, log: bool = False) -> go.Figure:
    fig, coords, has_img = _spatial_base(adata, image_info)
    idx = plot_index(adata.n_obs, max_points)
    v = np.asarray(values, dtype=float)
    shown = np.log10(np.maximum(v, 1)) if log else v
    order = idx[np.argsort(shown[idx])]  # draw high values on top
    fig.add_trace(
        go.Scattergl(
            x=coords[order, 0], y=coords[order, 1], mode="markers",
            marker=dict(size=_spot_size(adata, coords, len(idx)), color=shown[order], colorscale=SEQUENTIAL,
                        cmin=float(np.nanpercentile(shown, 1)), cmax=float(np.nanpercentile(shown, 99)),
                        colorbar=dict(title=dict(text=title + (" (log10)" if log else ""), side="right"), thickness=12, outlinewidth=0)),
            customdata=v[order], hovertemplate=f"{title}: %{{customdata:,.2f}}<extra></extra>", showlegend=False,
        )
    )
    return _layout(fig, 460, margin=dict(l=10, r=10, t=10, b=10))


def spatial_density(adata: ad.AnnData, units: str) -> go.Figure:
    coords = np.asarray(adata.obsm["spatial"], dtype=float)[:, :2]
    fig = go.Figure(
        go.Histogram2d(
            x=coords[:, 0], y=coords[:, 1], nbinsx=60, nbinsy=60, colorscale=SEQUENTIAL,
            colorbar=dict(title=dict(text="cells / bin", side="right"), thickness=12, outlinewidth=0),
            hovertemplate="%{z} cells<extra></extra>",
        )
    )
    fig.update_xaxes(title_text=f"x ({units})", showgrid=False)
    fig.update_yaxes(title_text=f"y ({units})", autorange="reversed", showgrid=False, scaleanchor="x", scaleratio=1)
    return _layout(fig, 460)


def nhood_heatmap(spatial_res, annotation) -> go.Figure:
    names = _cluster_names(annotation)
    labels = [f"{c} · {names.get(c, '')[:28]}" for c in spatial_res.nhood_clusters]
    z = np.asarray(spatial_res.nhood_zscore, dtype=float)
    lim = float(max(3.0, np.nanpercentile(np.abs(z), 95)))
    fig = go.Figure(
        go.Heatmap(
            z=z, x=labels, y=labels, colorscale=DIVERGING, zmid=0, zmin=-lim, zmax=lim, xgap=1, ygap=1,
            colorbar=dict(title=dict(text="z-score", side="right"), thickness=12, outlinewidth=0),
            hovertemplate="%{y}<br>next to %{x}<br>z = %{z:.1f}<extra></extra>",
        )
    )
    fig.update_xaxes(tickangle=-45, showgrid=False, tickfont=dict(size=10))
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=10))
    n = len(labels)
    return _layout(fig, max(380, 30 * n + 200), margin=dict(l=190, r=20, t=20, b=170))


def moran_bars(spatial_res, n: int = 15) -> go.Figure:
    top = spatial_res.moran_top[:n][::-1]
    fig = go.Figure(
        go.Bar(
            x=[m["I"] for m in top], y=[m["gene"] for m in top], orientation="h",
            marker=dict(color=SERIES[0], line=dict(width=0)),
            customdata=[m["padj"] for m in top],
            hovertemplate="%{y}<br>Moran's I = %{x:.3f}<br>FDR = %{customdata:.2g}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_xaxes(title_text="Moran's I")
    fig.update_layout(bargap=0.3)
    return _layout(fig, max(300, 22 * len(top) + 90), margin=dict(l=110, r=20, t=20, b=50))


def spatial_genes(adata: ad.AnnData, genes: list[str], image_info, max_points: int) -> go.Figure | None:
    genes = [g for g in genes if g in adata.var_names][:3]
    if not genes:
        return None
    fig = make_subplots(rows=1, cols=len(genes), subplot_titles=genes, horizontal_spacing=0.03)
    img, sf, _ = image_info
    coords = np.asarray(adata.obsm["spatial"], dtype=float)[:, :2] * (sf if img is not None else 1.0)
    idx = plot_index(adata.n_obs, max_points)
    for i, g in enumerate(genes):
        v = adata[:, g].X
        v = np.asarray(v.toarray() if hasattr(v, "toarray") else v).ravel()
        order = idx[np.argsort(v[idx])]
        fig.add_trace(
            go.Scattergl(
                x=coords[order, 0], y=coords[order, 1], mode="markers",
                marker=dict(size=_spot_size(adata, coords, len(idx)) * 0.7, color=v[order], colorscale=SEQUENTIAL,
                            cmin=0, cmax=float(np.nanpercentile(v, 99.5)) or 1.0, showscale=(i == len(genes) - 1),
                            colorbar=dict(title=dict(text="log expr", side="right"), thickness=10, outlinewidth=0)),
                hovertemplate=f"{g}: %{{marker.color:.2f}}<extra></extra>", showlegend=False,
            ),
            row=1, col=i + 1,
        )
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, showline=False, ticks="", row=1, col=i + 1)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, showline=False, ticks="", autorange="reversed",
                         scaleanchor=f"x{i + 1 if i else ''}", scaleratio=1, row=1, col=i + 1)
    for ann in fig.layout.annotations[: len(genes)]:
        ann.font = dict(size=12, color=INK)
    return _layout(fig, 360, margin=dict(l=10, r=10, t=40, b=10))
