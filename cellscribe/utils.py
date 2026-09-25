"""Small shared helpers: logging, seeding, gene classes, species inference."""

from __future__ import annotations

import logging
import random
import re
from collections.abc import Iterable

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger("cellscribe")

CLUSTER_KEY = "cellscribe_cluster"
LABEL_KEY = "cellscribe_annotation"

# Case-insensitive gene-class patterns (matched against upper-cased symbols so
# human "MT-CO1" and mouse "mt-Co1" are treated identically).
_MT_RE = re.compile(r"^MT-")
# Cytosolic ribosomal proteins. Written to avoid matching kinases such as
# RPS6KA1/RPS6KB1 that a naive startswith(("RPS", "RPL")) would include.
_RIBO_RE = re.compile(r"^(RP[LS]\d+[AXY]?\d*L?|RPLP[0-2]|RPSA)$")
# Haemoglobin genes (HBA1, HBB, Hba-a1, ...), excluding HBP1/HBEGF/HBS1L.
_HB_RE = re.compile(r"^HB(A|B|D|E|G|M|Q|Z)(\d|-|$)")

# Early-response / heat-shock genes induced by enzymatic dissociation
# (van den Brink et al., Nat Methods 2017; core subset). Their dominance in a
# cluster's markers suggests a dissociation-stress artefact.
DISSOCIATION_GENES = frozenset(
    {
        "FOS", "FOSB", "JUN", "JUNB", "JUND", "ATF3", "EGR1", "EGR2", "EGR3",
        "IER2", "IER3", "IER5", "HSPA1A", "HSPA1B", "HSPA8", "HSPB1", "HSPE1",
        "HSPH1", "DNAJB1", "DUSP1", "ZFP36", "NR4A1", "KLF2", "KLF4", "KLF6",
        "SOCS3", "CYR61", "CCN1", "PPP1R15A", "BTG2", "MYC", "GADD45B",
    }
)

BATCH_KEY_CANDIDATES = (
    "batch", "sample", "sample_id", "samples", "library_id", "library",
    "donor", "donor_id", "orig.ident", "patient", "patient_id", "replicate",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def is_mito(genes: Iterable[str]) -> np.ndarray:
    return np.array([bool(_MT_RE.match(str(g).upper())) for g in genes], dtype=bool)


def is_ribo(genes: Iterable[str]) -> np.ndarray:
    return np.array([bool(_RIBO_RE.match(str(g).upper())) for g in genes], dtype=bool)


def is_hb(genes: Iterable[str]) -> np.ndarray:
    return np.array([bool(_HB_RE.match(str(g).upper())) for g in genes], dtype=bool)


def infer_species(genes: Iterable[str]) -> tuple[str, str]:
    """Guess human vs mouse from gene-symbol casing.

    Human symbols are upper case (``CD3E``); mouse symbols are title case
    (``Cd3e``). Returns ``(species, reason)``. Falls back to human when the
    evidence is weak (e.g. Ensembl IDs only), and says so in ``reason``.
    """
    genes = [str(g) for g in genes]
    if not genes:
        return "human", "no genes available; defaulted to human"
    ens_mouse = sum(g.startswith("ENSMUSG") for g in genes)
    ens_human = sum(g.startswith("ENSG") for g in genes)
    alpha = [g for g in genes if re.match(r"^[A-Za-z][A-Za-z0-9-]{1,}$", g) and not g.startswith("ENS")]
    upper = sum(g == g.upper() and any(c.isalpha() for c in g[1:]) for g in alpha)
    title = sum(g[0].isupper() and g[1:] != g[1:].upper() and g[1:] == g[1:].lower() for g in alpha)
    n = max(len(alpha), 1)
    if ens_mouse > max(ens_human, 0.3 * len(genes)):
        return "mouse", "Ensembl mouse gene IDs (ENSMUSG)"
    if title / n > 0.5 and title > upper:
        return "mouse", f"{title / n:.0%} of gene symbols are title case (mouse convention)"
    if upper / n > 0.5:
        return "human", f"{upper / n:.0%} of gene symbols are upper case (human convention)"
    return "human", "gene symbol casing was inconclusive; defaulted to human"


def is_integer_matrix(X, n_check: int = 20_000) -> bool:
    """True if the (sampled) non-zero values of X are all integers."""
    data = X.data if sp.issparse(X) else np.asarray(X).ravel()
    if data.size == 0:
        return True
    if data.size > n_check:
        idx = np.random.default_rng(0).choice(data.size, n_check, replace=False)
        data = data[idx]
    data = np.asarray(data, dtype=np.float64)
    return bool(np.all(np.abs(data - np.round(data)) < 1e-6) and np.all(data >= 0))


def detect_batch_key(obs, explicit: str | None = None) -> str | None:
    """Return the obs column describing batches, or None if there is only one batch.

    An explicitly configured key must exist; auto-detection only accepts
    categorical-like columns with 2..(n_cells/10) levels.
    """
    from cellscribe.errors import ConfigError

    if explicit is not None:
        if explicit not in obs.columns:
            raise ConfigError(
                f"batch_key '{explicit}' is not a column of the input's cell metadata. "
                f"Available columns: {', '.join(map(str, obs.columns)) or '(none)'}"
            )
        return explicit if obs[explicit].nunique() > 1 else None
    lower = {str(c).lower(): c for c in obs.columns}
    for cand in BATCH_KEY_CANDIDATES:
        col = lower.get(cand)
        if col is None:
            continue
        n_levels = obs[col].nunique()
        if 1 < n_levels <= max(2, len(obs) // 10):
            return col
    return None


def fmt_int(x) -> str:
    try:
        return f"{int(round(float(x))):,}"
    except (TypeError, ValueError):
        return str(x)
