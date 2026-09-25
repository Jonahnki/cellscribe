"""Optional AI-written narrative summary (Claude API).

This module is deliberately isolated and optional:

* It runs only when ``ANTHROPIC_API_KEY`` is set in the environment, the
  ``anthropic`` package is installed, and ``narrative.enabled`` is true.
  Otherwise it returns a "skipped" result and the report renders without
  the narrative section; nothing else changes.
* It receives the structured run summary (JSON), never the AnnData object.
  :func:`build_payload` further reduces that to summary statistics, cluster
  labels, and marker gene *names*; no per-cell data or expression values
  are ever sent.
* Any failure (network, authentication, refusal) is caught and reported as
  "failed"; it never aborts the report.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from cellscribe.config import NarrativeConfig
from cellscribe.utils import logger

# Models for which Anthropic's server-side refusal fallback is enabled by default.
_FALLBACK_MODELS = {"claude-opus-5", "claude-fable-5-1"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = """\
You are an experienced single-cell bioinformatician. You are writing the short \
interpretive summary of an automated first-pass quality-control and annotation \
report. The reader is a wet-lab researcher without in-house bioinformatics \
support who needs to decide what to do next with their data.

You receive only a JSON summary produced by the pipeline (cell counts, QC \
thresholds and flags, cluster sizes, candidate cell-type labels with confidence, \
marker gene names, and spatial statistics where available). Base every statement \
on that JSON: do not invent numbers, genes, cell types, or tissue context that it \
does not contain. Automated labels are hypotheses from canonical markers, so \
describe them with the stated confidence and never as definitive identities. \
This is research-use software; do not offer clinical or diagnostic interpretation.

Write two or three paragraphs of plain prose, about 200-300 words in total, with \
no headings, lists, or Markdown formatting:
1. What the data look like overall: size, overall quality, and the main populations.
2. What to be cautious about: flagged or unresolved clusters, likely doublets or \
damaged cells, batch structure, and limits of the automated analysis.
3. Concrete, practical next steps for this dataset."""


@dataclass
class NarrativeResult:
    status: str  # "generated" | "skipped" | "failed"
    reason: str = ""
    paragraphs: list[str] = field(default_factory=list)
    model: str | None = None
    payload_chars: int = 0

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)

    def to_dict(self) -> dict:
        return asdict(self)


def build_payload(summary: dict) -> dict:
    """Reduce the run summary to what the narrative needs (no per-cell or expression data)."""
    ann = {c["cluster"]: c for c in summary["annotation"]["clusters"]}
    cqc = {r["cluster"]: r for r in summary["cluster_qc"]}
    qc = summary["qc"]
    n_after_floor = qc["n_cells_input"] - qc["n_cells_below_floor"]
    clusters = []
    for cl, a in ann.items():
        q = cqc.get(cl, {})
        clusters.append(
            {
                "cluster": cl,
                "n_cells": q.get("n_cells"),
                "pct_of_cells": round(q.get("pct_cells", 0.0), 1),
                "label": a["label"],
                "label_status": a["status"],
                "confidence": a["confidence"],
                "evidence": a.get("evidence", "specific"),
                "candidates": [
                    {"name": c["name"], "markers_matched": f"{c['n_hits']}/{c['n_present']}"} for c in a["candidates"][:3]
                ],
                "states": [s["name"] for s in a.get("states", [])],
                "top_marker_genes": [m["gene"] for m in a["top_markers"][:8]],
                "qc_status": q.get("status"),
                "qc_flags": [r["message"] for r in q.get("reasons", [])],
                "median_counts": q.get("median_counts"),
                "median_pct_mt": q.get("median_pct_mt"),
            }
        )
    payload: dict[str, Any] = {
        "sample": {
            "modality": summary["input"]["modality"],
            "input_format": summary["input"]["format"],
            "species": summary["input"]["species"],
            "cells_loaded": summary["input"].get("n_cells_loaded"),
            "cells_analysed": summary["dataset"]["n_cells"],
            "genes_analysed": summary["dataset"]["n_genes"],
            "batches": summary["input"].get("batches", {}),
        },
        "qc": {
            "mode": qc["mode"],
            "median_counts_per_cell": round(summary["dataset"]["median_counts"]),
            "median_genes_per_cell": round(summary["dataset"]["median_genes"]),
            "median_pct_mitochondrial": summary["dataset"]["median_pct_mt"],
            "cells_flagged": qc["n_flagged"],
            "pct_flagged": round(100.0 * qc["n_flagged"] / max(n_after_floor, 1), 1),
            "flag_reasons": qc["reason_counts"],
            "thresholds": [
                {k: t[k] for k in ("label", "lower", "upper", "batch")} for t in qc["thresholds"]
            ],
            "doublet_detection": {
                k: qc["doublets"].get(k) for k in ("status", "pct_predicted", "expected_rate", "reason") if k in qc["doublets"]
            },
        },
        "clustering": {
            "n_clusters": summary["clustering"]["n_clusters"],
            "resolution": summary["clustering"]["resolution"],
            "auto_selected": summary["clustering"]["auto_selected"],
            "batch_integration": summary["preprocessing"]["integration"],
        },
        "clusters": clusters,
        "notes": [n["message"] for n in summary["notes"] if n["severity"] in ("caution", "warning")],
    }
    sp = summary.get("spatial")
    if sp and sp.get("enabled"):
        payload["spatial"] = {
            "colocalised_cluster_pairs": sp["colocalised_pairs"][:5],
            "segregated_cluster_pairs": sp["segregated_pairs"][:5],
            "top_spatially_autocorrelated_genes": [m["gene"] for m in sp["moran_top"][:10]],
            "n_genes_spatially_autocorrelated": sp["n_genes_significant"],
            "n_genes_tested": sp["n_genes_autocorr"],
        }
    return payload


def build_user_message(payload: dict) -> str:
    return (
        "Here is the pipeline summary for this dataset as JSON. Write the interpretive summary.\n\n"
        + json.dumps(payload, indent=1, default=str)
    )


def _clean_paragraphs(text: str) -> list[str]:
    paras = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        lines = [ln.strip().lstrip("#").strip() for ln in block.strip().splitlines()]
        para = " ".join(ln for ln in lines if ln)
        para = para.replace("**", "").replace("__", "")
        if para:
            paras.append(para)
    return paras


def generate(summary: dict, cfg: NarrativeConfig, client: Any = None) -> NarrativeResult:
    """Generate the narrative, or explain why it was skipped/failed. Never raises."""
    if not cfg.enabled:
        return NarrativeResult("skipped", "Narrative synthesis disabled (--no-narrative or narrative.enabled: false).")
    if client is None and not os.environ.get("ANTHROPIC_API_KEY"):
        return NarrativeResult("skipped", "ANTHROPIC_API_KEY is not set; the optional AI narrative was not generated.")
    try:
        import anthropic
    except ImportError:
        return NarrativeResult(
            "skipped", "The 'anthropic' package is not installed (pip install 'cellscribe[narrative]')."
        )

    payload = build_payload(summary)
    user = build_user_message(payload)
    try:
        if client is None:
            client = anthropic.Anthropic(timeout=cfg.timeout_seconds, max_retries=2)
        kwargs = {
            "model": cfg.model,
            "max_tokens": cfg.max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user}],
        }
        if cfg.model in _FALLBACK_MODELS:
            # Server-side refusal fallback: a declined request is re-run on Anthropic's
            # recommended model for that refusal category instead of failing.
            try:
                response = client.beta.messages.create(betas=[_FALLBACK_BETA], fallbacks="default", **kwargs)
            except TypeError:  # older SDK without the typed parameter
                response = client.beta.messages.create(
                    betas=[_FALLBACK_BETA], extra_body={"fallbacks": "default"}, **kwargs
                )
        else:
            response = client.messages.create(**kwargs)
    except anthropic.AuthenticationError:
        return NarrativeResult("failed", "The Anthropic API rejected the API key; narrative not generated.")
    except anthropic.RateLimitError:
        return NarrativeResult("failed", "The Anthropic API rate limit was reached; narrative not generated.")
    except anthropic.APIStatusError as exc:
        return NarrativeResult("failed", f"The Anthropic API returned an error (HTTP {exc.status_code}); narrative not generated.")
    except anthropic.APIConnectionError:
        return NarrativeResult("failed", "Could not reach the Anthropic API (network error); narrative not generated.")
    except Exception as exc:  # noqa: BLE001 - the report must never fail because of the narrative
        logger.warning("Narrative generation failed: %s", exc)
        return NarrativeResult("failed", f"Narrative generation failed ({type(exc).__name__}).")

    if getattr(response, "stop_reason", None) == "refusal":
        return NarrativeResult("failed", "The model declined to write this summary; narrative not generated.")
    text = "".join(getattr(b, "text", "") for b in response.content if getattr(b, "type", None) == "text")
    paragraphs = _clean_paragraphs(text)
    if not paragraphs:
        return NarrativeResult("failed", "The model returned no text; narrative not generated.")
    if getattr(response, "stop_reason", None) == "max_tokens":
        paragraphs[-1] += " [truncated]"
    return NarrativeResult(
        "generated",
        "",
        paragraphs=paragraphs,
        model=getattr(response, "model", cfg.model),
        payload_chars=len(user),
    )
