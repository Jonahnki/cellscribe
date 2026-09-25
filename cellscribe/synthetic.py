"""Synthetic single-cell and spatial datasets with known ground truth.

These generators power ``cellscribe demo`` and serve as test fixtures. They
are deliberately realistic in the ways that matter for a QC report:

* Counts are drawn from a gamma-Poisson (negative binomial) model with
  log-normal library sizes, a skewed gene-abundance distribution, a
  mitochondrial fraction, and a ribosomal-protein fraction.
* Cell types are defined by *real* marker gene symbols from the bundled
  reference (a random ~80% subset of each type's markers is up-regulated),
  plus type-specific genes absent from the reference.
* Three failure modes a first-pass report must catch are built in:
  damaged cells (low counts, high mitochondrial fraction), heterotypic
  doublets (mostly T cell + B cell, so they form a cluster), and an
  "uncharacterised" population that matches no reference cell type.

Ground truth is stored in ``obs['true_cell_type']``.
"""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellscribe.markers import load_reference

_SYNTH_NOTE = (
    "Simulated data: marker genes use real gene symbols, while genes named SG00001, SG00002, ... are simulated "
    "background genes. Ground-truth cell types are stored in obs['true_cell_type']."
)
LOW_QUALITY = "Low-quality (damaged)"
DOUBLET = "Doublet"
UNCHARACTERISED = "Uncharacterised"

_MT_HUMAN = [
    "MT-ND1", "MT-ND2", "MT-CO1", "MT-CO2", "MT-ATP8", "MT-ATP6", "MT-CO3",
    "MT-ND3", "MT-ND4L", "MT-ND4", "MT-ND5", "MT-ND6", "MT-CYB",
]
_RIBO_HUMAN = (
    [f"RPL{i}" for i in range(3, 42) if i not in (16, 20, 25, 33, 40)]
    + [f"RPS{i}" for i in range(2, 30) if i not in (22,)]
    + ["RPLP0", "RPLP1", "RPLP2", "RPSA", "RPL13A", "RPL27A", "RPS27A"]
)


@dataclass(frozen=True)
class _TypeSpec:
    name: str
    fraction: float
    median_umis: float
    ribo_frac: float = 0.20


PBMC_TYPES = (
    _TypeSpec("CD4+ T cell", 0.29, 4200, 0.30),
    _TypeSpec("CD8+ T cell", 0.16, 4400, 0.28),
    _TypeSpec("NK cell", 0.10, 4000, 0.20),
    _TypeSpec("B cell", 0.14, 4600, 0.26),
    _TypeSpec("Classical monocyte", 0.19, 6500, 0.14),
    _TypeSpec("Non-classical monocyte", 0.06, 6000, 0.14),
    _TypeSpec(UNCHARACTERISED, 0.06, 5000, 0.18),
)

TISSUE_TYPES = (
    "Colonocyte", "Goblet cell", "Fibroblast", "Endothelial cell",
    "Smooth muscle cell", "Macrophage", "B cell", "CD4+ T cell",
)


def _case(genes: list[str], species: str) -> list[str]:
    if species == "human":
        return genes
    out = []
    for g in genes:
        if g.startswith("MT-"):
            out.append("mt-" + g[3:].capitalize())
        else:
            out.append(g[0] + g[1:].lower())
    return out


def _barcodes(n: int, rng: np.random.Generator, kind: str = "10x") -> list[str]:
    seen: set[str] = set()
    out = []
    letters = np.array(list("ACGT")) if kind == "10x" else np.array(list("abcdefghijklmnop"))
    length = 16 if kind == "10x" else 8
    while len(out) < n:
        s = "".join(rng.choice(letters, length)) + "-1"
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


class _Simulator:
    """Holds the gene universe and per-type expression programs."""

    def __init__(
        self,
        type_names: list[str],
        species: str,
        rng: np.random.Generator,
        n_filler: int,
        include_mt_ribo: bool = True,
    ):
        self.rng = rng
        self.species = species
        ref = load_reference(species)
        marker_genes = ref.all_genes()
        from cellscribe.utils import DISSOCIATION_GENES

        extra = [g for g in _case(sorted(DISSOCIATION_GENES), species) if g not in marker_genes]
        mt = _case(_MT_HUMAN, species) if include_mt_ribo else []
        ribo = _case(_RIBO_HUMAN, species) if include_mt_ribo else []
        prefix = "SG" if species == "human" else "Sg"
        filler = [f"{prefix}{i:05d}" for i in range(1, n_filler + 1)]
        genes = list(dict.fromkeys(marker_genes + extra + ribo + mt + filler))
        self.genes = np.array(genes)
        self.n_genes = len(genes)
        idx = {g: i for i, g in enumerate(genes)}
        self.mt_idx = np.array([idx[g] for g in mt], dtype=int)
        self.ribo_idx = np.array([idx[g] for g in ribo], dtype=int)
        other = np.ones(self.n_genes, dtype=bool)
        other[self.mt_idx] = False
        other[self.ribo_idx] = False
        self.other_idx = np.flatnonzero(other)
        filler_idx = np.array([idx[g] for g in filler], dtype=int)
        marker_idx = np.array([idx[g] for g in marker_genes], dtype=int)

        # Baseline abundance: skewed, markers lowly expressed outside their type.
        base = rng.normal(0.0, 1.3, self.n_genes)
        base[marker_idx] = rng.normal(-2.2, 0.7, marker_idx.size)
        housekeeping = rng.choice(filler_idx, size=min(60, filler_idx.size), replace=False)
        base[housekeeping] += 2.5
        self.base = base
        self.mt_weights = rng.dirichlet(np.full(max(len(mt), 1), 3.0))
        self.ribo_weights = rng.dirichlet(np.full(max(len(ribo), 1), 5.0))

        # Type programs (log fold changes over baseline). Each type up-regulates a
        # random ~80% of its reference markers; markers listed for several types of
        # the same lineage (e.g. CD3E for CD4+ and CD8+ T cells) are shared, as in
        # real tissue.
        self.programs: dict[str, np.ndarray] = {}
        free_filler = [i for i in filler_idx if i not in set(housekeeping)]
        rng.shuffle(free_filler)
        up_sets: dict[str, set[int]] = {}
        for name in type_names:
            if name == UNCHARACTERISED:
                continue
            mset = [idx[g] for g in ref.get(name).genes if g in idx]
            n_up = max(4, int(round(0.8 * len(mset))))
            up_sets[name] = set(rng.choice(mset, size=min(n_up, len(mset)), replace=False).tolist())
        for a_name in list(up_sets):
            a_set = ref.get(a_name)
            for b_name in up_sets:
                b_set = ref.get(b_name)
                if a_name != b_name and a_set.lineage == b_set.lineage:
                    shared = {idx[g] for g in a_set.genes if g in idx} & {idx[g] for g in b_set.genes if g in idx}
                    up_sets[a_name] |= shared & (up_sets[b_name] | up_sets[a_name])
        for name in type_names:
            lfc = np.zeros(self.n_genes)
            if name == UNCHARACTERISED:
                own = free_filler[:40]
                free_filler = free_filler[40:]
                lfc[own] = rng.uniform(2.8, 4.0, len(own))
            else:
                up = np.array(sorted(up_sets[name]), dtype=int)
                lfc[up] = rng.uniform(3.4, 4.6, up.size)
                own = free_filler[:25]
                free_filler = free_filler[25:]
                lfc[own] = rng.uniform(1.0, 2.2, len(own))
            self.programs[name] = lfc

    def proportions(self, name: str, mt_frac: float, ribo_frac: float, batch_lfc=None) -> np.ndarray:
        logits = self.base + self.programs[name]
        if batch_lfc is not None:
            logits = logits + batch_lfc
        p = np.zeros(self.n_genes)
        w = np.exp(logits[self.other_idx])
        rest = 1.0 - (mt_frac if self.mt_idx.size else 0) - (ribo_frac if self.ribo_idx.size else 0)
        p[self.other_idx] = rest * w / w.sum()
        if self.mt_idx.size:
            p[self.mt_idx] = mt_frac * self.mt_weights
        if self.ribo_idx.size:
            p[self.ribo_idx] = ribo_frac * self.ribo_weights
        return p

    def sample(self, props: np.ndarray, lib: np.ndarray, dispersion: float = 0.2) -> np.ndarray:
        """Gamma-Poisson counts for a block of cells (rows of ``props``)."""
        mu = props * lib[:, None]
        shape = 1.0 / dispersion
        lam = self.rng.gamma(shape, mu / shape)
        return self.rng.poisson(lam).astype(np.float32)


def _simulate_cells(
    sim: _Simulator,
    labels: list[str],
    specs: dict[str, _TypeSpec],
    batches: np.ndarray,
    batch_lfcs: dict,
    lib_scale: float = 1.0,
    mt_mean: float = 0.045,
) -> np.ndarray:
    rng = sim.rng
    n = len(labels)
    X = np.zeros((n, sim.n_genes), dtype=np.float32)
    labels_arr = np.asarray(labels)
    for name in dict.fromkeys(labels):
        for b in np.unique(batches):
            rows = np.flatnonzero((labels_arr == name) & (batches == b))
            if rows.size == 0:
                continue
            spec = specs[name]
            mt = np.clip(rng.normal(mt_mean, 0.012, rows.size), 0.005, 0.2)
            props = np.stack(
                [sim.proportions(name, m, spec.ribo_frac, batch_lfcs.get(b)) for m in mt]
            )
            lib = rng.lognormal(np.log(spec.median_umis * lib_scale), 0.35, rows.size)
            X[rows] = sim.sample(props, lib)
    return X


def make_single_cell(
    n_cells: int = 3000,
    species: str = "human",
    seed: int = 0,
    n_batches: int = 2,
    doublet_rate: float = 0.08,
    low_quality_rate: float = 0.04,
    include_uncharacterised: bool = True,
    n_filler_genes: int = 1800,
) -> ad.AnnData:
    """Simulate a PBMC-like dissociated single-cell dataset.

    Returns raw counts (float32 CSR) with ``obs['true_cell_type']`` and
    ``obs['sample']`` (when ``n_batches > 1``).
    """
    rng = np.random.default_rng(seed)
    specs = [s for s in PBMC_TYPES if include_uncharacterised or s.name != UNCHARACTERISED]
    spec_map = {s.name: s for s in specs}
    sim = _Simulator([s.name for s in specs], species, rng, n_filler_genes)

    n_dbl = int(round(n_cells * doublet_rate))
    n_lq = int(round(n_cells * low_quality_rate))
    n_clean = n_cells - n_dbl - n_lq
    fr = np.array([s.fraction for s in specs])
    labels = list(rng.choice([s.name for s in specs], size=n_clean, p=fr / fr.sum()))
    batches_all = rng.integers(0, max(n_batches, 1), n_cells)
    batch_lfcs = {
        b: rng.normal(0, 0.12, sim.n_genes) + np.log(rng.uniform(0.85, 1.2)) for b in range(max(n_batches, 1))
    }

    X_clean = _simulate_cells(sim, labels, spec_map, batches_all[:n_clean], batch_lfcs)

    # Damaged cells: a real type's profile, but with a collapsed library and high mtRNA.
    lq_types = rng.choice([s.name for s in specs if s.name != UNCHARACTERISED], size=n_lq)
    X_lq = np.zeros((n_lq, sim.n_genes), dtype=np.float32)
    for i, name in enumerate(lq_types):
        spec = spec_map[name]
        mt = rng.uniform(0.25, 0.55)
        props = sim.proportions(name, mt, spec.ribo_frac * 0.6)[None, :]
        lib = np.array([rng.lognormal(np.log(spec.median_umis * 0.22), 0.3)])
        X_lq[i] = sim.sample(props, lib)[0]

    # Heterotypic doublets: 70% CD4 T + B cell (similar RNA content, so the
    # doublets sit between the two lineages and form a cluster), 30% random pairs.
    real = [s.name for s in specs if s.name != UNCHARACTERISED]
    pairs = []
    for _ in range(n_dbl):
        if rng.random() < 0.7:
            pairs.append(("CD4+ T cell", "B cell"))
        else:
            a, b = rng.choice(real, size=2, replace=False)
            pairs.append((a, b))
    X_dbl = np.zeros((n_dbl, sim.n_genes), dtype=np.float32)
    if n_dbl:
        bdb = batches_all[n_clean + n_lq:]
        first = _simulate_cells(sim, [p[0] for p in pairs], spec_map, bdb, batch_lfcs)
        second = _simulate_cells(sim, [p[1] for p in pairs], spec_map, bdb, batch_lfcs)
        X_dbl = first + second

    X = np.vstack([X_clean, X_lq, X_dbl])
    truth = labels + [LOW_QUALITY] * n_lq + [DOUBLET] * n_dbl
    order = rng.permutation(n_cells)
    X = X[order]
    truth = np.asarray(truth)[order]
    batches_all = batches_all  # already random, independent of order

    obs = pd.DataFrame(index=_barcodes(n_cells, rng))
    obs["true_cell_type"] = pd.Categorical(truth)
    if n_batches > 1:
        obs["sample"] = pd.Categorical([f"sample_{chr(65 + b)}" for b in batches_all[order]])
    adata = ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=pd.DataFrame(index=sim.genes))
    adata.uns["cellscribe_input"] = {
        "format": "synthetic",
        "modality": "single-cell",
        "path": f"synthetic PBMC-like dataset (seed={seed})",
        "sample_name": "Synthetic PBMC demo",
        "notes": [_SYNTH_NOTE],
    }
    return adata


# --------------------------------------------------------------------- spatial


def _tissue_labels(xy: np.ndarray, W: float, H: float, rng: np.random.Generator) -> np.ndarray:
    """Assign colon-like tissue compartments to positions (µm)."""
    x, y = xy[:, 0], xy[:, 1]
    n = len(x)
    labels = np.empty(n, dtype=object)

    def pick(mask, names, probs):
        k = int(mask.sum())
        if k:
            labels[mask] = rng.choice(names, size=k, p=np.asarray(probs) / np.sum(probs))

    muscle = y < 0.18 * H
    fc = np.array([0.74 * W, 0.58 * H])
    r = 0.17 * min(W, H)
    d = np.hypot(x - fc[0], y - fc[1])
    core = (d < 0.65 * r) & ~muscle
    rim = (d >= 0.65 * r) & (d < r) & ~muscle
    period = 0.12 * W
    phase = np.mod(x, period) - period / 2
    crypt = (np.abs(phase) < 0.28 * period) & (y > 0.24 * H) & ~core & ~rim & ~muscle
    vessel_y = 0.21 * H + 0.03 * H * np.sin(x / (0.08 * W))
    vessel = (np.abs(y - vessel_y) < 0.012 * H) & ~crypt & ~core & ~rim
    lp = ~(muscle | core | rim | crypt | vessel)

    pick(muscle & ~vessel, ["Smooth muscle cell", "Fibroblast", "Endothelial cell"], [0.84, 0.11, 0.05])
    pick(core, ["B cell", "CD4+ T cell", "Macrophage"], [0.80, 0.15, 0.05])
    pick(rim, ["CD4+ T cell", "B cell", "Fibroblast", "Endothelial cell"], [0.6, 0.2, 0.1, 0.1])
    pick(crypt, ["Colonocyte", "Goblet cell"], [0.64, 0.36])
    pick(vessel, ["Endothelial cell", "Fibroblast"], [0.85, 0.15])
    pick(lp, ["Fibroblast", "Macrophage", "CD4+ T cell", "Endothelial cell", "B cell"], [0.45, 0.2, 0.15, 0.15, 0.05])
    return labels


def make_spatial(n_cells: int = 5000, species: str = "human", seed: int = 0) -> ad.AnnData:
    """Simulate an imaging-based (Xenium-like) single-cell spatial dataset.

    A targeted panel (reference markers plus 40 other genes, no
    mitochondrial or ribosomal probes), low per-cell counts, and a colon-like
    tissue layout: muscularis, crypts, lamina propria, a vessel, and a
    lymphoid follicle. Coordinates are in µm in ``obsm['spatial']``.
    """
    rng = np.random.default_rng(seed)
    W, H = 1600.0 * np.sqrt(n_cells / 5000), 1200.0 * np.sqrt(n_cells / 5000)
    xy = np.column_stack([rng.uniform(0, W, n_cells), rng.uniform(0, H, n_cells)])
    labels = _tissue_labels(xy, W, H, rng)
    sim = _Simulator(list(TISSUE_TYPES), species, rng, n_filler=40, include_mt_ribo=False)
    specs = {t: _TypeSpec(t, 0, 260, 0.0) for t in TISSUE_TYPES}
    X = _simulate_cells(sim, list(labels), specs, np.zeros(n_cells, dtype=int), {}, mt_mean=0.0)

    obs = pd.DataFrame(index=_barcodes(n_cells, rng, kind="xenium"))
    obs["true_cell_type"] = pd.Categorical(labels.astype(str))
    obs["cell_area"] = rng.lognormal(np.log(60), 0.3, n_cells).round(2)
    adata = ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=pd.DataFrame(index=sim.genes))
    adata.obsm["spatial"] = xy.astype(np.float64)
    adata.uns["cellscribe_input"] = {
        "format": "synthetic",
        "modality": "xenium",
        "path": f"synthetic imaging-based spatial dataset (seed={seed})",
        "sample_name": "Synthetic spatial demo (imaging-based)",
        "notes": [_SYNTH_NOTE],
    }
    return adata


def _he_image(W: float, H: float, px_per_um: float, tissue_mask_fn, rng) -> np.ndarray:
    """Render a crude H&E-like RGB image (uint8) of the synthetic tissue."""
    from scipy.ndimage import gaussian_filter

    w_px, h_px = int(W * px_per_um), int(H * px_per_um)
    xs = (np.arange(w_px) + 0.5) / px_per_um
    ys = (np.arange(h_px) + 0.5) / px_per_um
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    lab = _tissue_labels(pts, W, H, np.random.default_rng(1)).reshape(h_px, w_px)
    palette = {
        "Smooth muscle cell": (214, 110, 150),
        "Fibroblast": (236, 170, 196),
        "Endothelial cell": (200, 90, 120),
        "Macrophage": (190, 120, 170),
        "B cell": (98, 62, 150),
        "CD4+ T cell": (120, 80, 165),
        "Colonocyte": (170, 110, 180),
        "Goblet cell": (215, 190, 225),
    }
    img = np.zeros((h_px, w_px, 3), dtype=np.float64)
    for name, col in palette.items():
        img[lab == name] = col
    img += rng.normal(0, 14, img.shape)
    img = np.stack([gaussian_filter(img[..., c], 1.2) for c in range(3)], axis=-1)
    inside = tissue_mask_fn(pts).reshape(h_px, w_px)
    img[~inside] = (246, 244, 242)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_visium(n_spots_target: int = 1500, species: str = "human", seed: int = 0) -> ad.AnnData:
    """Simulate a 10x Visium dataset (55 µm spots on a 100 µm hexagonal grid).

    Each spot sums the counts of ~8 simulated cells whose types follow the
    same colon-like layout as :func:`make_spatial`. The returned AnnData
    mimics what :func:`cellscribe.loaders.load` produces for a Visium
    directory, including a synthetic H&E image in ``uns['spatial']``.
    """
    rng = np.random.default_rng(seed)
    spacing = 100.0
    row_h = spacing * np.sqrt(3) / 2
    # choose field so that ~n_spots_target spots fall inside the tissue ellipse
    area_per_spot = spacing * row_h
    field_area = n_spots_target * area_per_spot / (np.pi / 4)
    W = np.sqrt(field_area * 4 / 3)
    H = W * 3 / 4
    n_rows = int(H // row_h)
    n_cols = int(W // spacing)
    centres, rc = [], []
    for r in range(n_rows):
        for c in range(n_cols):
            x = c * spacing + (spacing / 2 if r % 2 else 0) + spacing / 2
            y = r * row_h + row_h / 2
            centres.append((x, y))
            rc.append((r, 2 * c + (r % 2)))
    centres = np.array(centres)

    def in_tissue(p):
        return ((p[:, 0] - W / 2) / (0.49 * W)) ** 2 + ((p[:, 1] - H / 2) / (0.49 * H)) ** 2 < 1

    tissue = in_tissue(centres)
    sim = _Simulator(list(TISSUE_TYPES), species, rng, n_filler=1200)
    specs = {t: _TypeSpec(t, 0, 1100, 0.18) for t in TISSUE_TYPES}
    n_tissue = int(tissue.sum())
    counts = np.zeros((n_tissue, sim.n_genes), dtype=np.float32)
    dominant = []
    for i, (cx, cy) in enumerate(centres[tissue]):
        k = rng.poisson(8) + 1
        ang = rng.uniform(0, 2 * np.pi, k)
        rad = 27.5 * np.sqrt(rng.uniform(0, 1, k))
        pts = np.column_stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)])
        labs = _tissue_labels(pts, W, H, rng)
        cells = _simulate_cells(sim, list(labs), specs, np.zeros(k, dtype=int), {})
        counts[i] = cells.sum(axis=0)
        vals, cts = np.unique(labs.astype(str), return_counts=True)
        dominant.append(vals[np.argmax(cts)])

    px_per_um = 2.0  # "full resolution" image scale
    barcodes = _barcodes(len(centres), rng)
    obs = pd.DataFrame(index=np.array(barcodes)[tissue])
    obs["true_cell_type"] = pd.Categorical(dominant)
    obs["in_tissue"] = 1
    obs["array_row"] = np.array([r for r, _ in rc])[tissue]
    obs["array_col"] = np.array([c for _, c in rc])[tissue]
    adata = ad.AnnData(X=sp.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=sim.genes))
    adata.obsm["spatial"] = (centres[tissue] * px_per_um).astype(np.float64)

    full_w = W * px_per_um
    hires_scalef = 2000.0 / max(full_w, H * px_per_um)
    lowres_scalef = 600.0 / max(full_w, H * px_per_um)
    hires = _he_image(W, H, px_per_um * hires_scalef, in_tissue, rng)
    lowres = _he_image(W, H, px_per_um * lowres_scalef, in_tissue, rng)
    library_id = "synthetic_visium"
    adata.uns["spatial"] = {
        library_id: {
            "images": {"hires": hires, "lowres": lowres},
            "scalefactors": {
                "tissue_hires_scalef": hires_scalef,
                "tissue_lowres_scalef": lowres_scalef,
                "spot_diameter_fullres": 55.0 * px_per_um,
                "fiducial_diameter_fullres": 65.0 * px_per_um,
            },
        }
    }
    # Positions of every spot (including off-tissue) for writing a Visium folder.
    adata.uns["_all_spots"] = {
        "barcode": barcodes,
        "in_tissue": tissue.astype(int).tolist(),
        "array_row": [r for r, _ in rc],
        "array_col": [c for _, c in rc],
        "pxl_row": (centres[:, 1] * px_per_um).tolist(),
        "pxl_col": (centres[:, 0] * px_per_um).tolist(),
    }
    adata.uns["cellscribe_input"] = {
        "format": "synthetic",
        "modality": "visium",
        "path": f"synthetic Visium dataset (seed={seed})",
        "sample_name": "Synthetic spatial demo (Visium)",
        "library_id": library_id,
        "notes": [_SYNTH_NOTE],
    }
    return adata


def make_demo(kind: str = "single-cell", n: int | None = None, seed: int = 0) -> ad.AnnData:
    """Dataset used by ``cellscribe demo``."""
    if kind == "single-cell":
        return make_single_cell(n_cells=n or 3000, seed=seed)
    if kind in {"spatial", "xenium"}:
        return make_spatial(n_cells=n or 5000, seed=seed)
    if kind == "visium":
        return make_visium(n_spots_target=n or 1500, seed=seed)
    raise ValueError(f"Unknown demo kind {kind!r}; choose single-cell, spatial or visium.")
