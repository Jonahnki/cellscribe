"""10x Genomics Xenium output folders.

Expected layout (Xenium Onboard Analysis 1.x-3.x)::

    output-XETG.../
      cell_feature_matrix.h5          (or cell_feature_matrix/ mtx folder)
      cells.parquet                   (or cells.csv.gz / cells.csv)
      experiment.xenium               (optional, run metadata)

Negative-control probe/codeword features are removed from the gene matrix and
summarised per cell in ``obs['control_counts']`` (a Xenium-specific QC metric).
No tissue image is assumed.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from cellscribe.errors import InputFormatError
from cellscribe.loaders._common import finalize
from cellscribe.loaders.tenx import is_mtx_dir, read_10x_h5, read_mtx_dir


def is_xenium_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    names = {q.name for q in p.iterdir()}
    has_cells = bool(names & {"cells.parquet", "cells.csv.gz", "cells.csv"})
    has_matrix = "cell_feature_matrix.h5" in names or "cell_feature_matrix" in names
    return "experiment.xenium" in names or (has_cells and has_matrix)


def _read_cells(d: Path) -> pd.DataFrame:
    for name in ("cells.parquet", "cells.csv.gz", "cells.csv"):
        f = d / name
        if f.is_file():
            cells = pd.read_parquet(f) if name.endswith(".parquet") else pd.read_csv(f)
            break
    else:
        raise InputFormatError(f"No cells table in {d}: expected cells.parquet, cells.csv.gz or cells.csv.")
    missing = [c for c in ("cell_id", "x_centroid", "y_centroid") if c not in cells.columns]
    if missing:
        raise InputFormatError(f"The Xenium cells table is missing columns {missing}; found {list(cells.columns)}.")
    ids = cells["cell_id"]
    if ids.map(type).eq(bytes).any():
        ids = ids.map(lambda v: v.decode() if isinstance(v, bytes) else v)
    cells.index = ids.astype(str)
    return cells


def load_xenium(path: str | Path, sample_name: str | None = None) -> ad.AnnData:
    p = Path(path)
    if p.is_file() and p.name == "experiment.xenium":
        p = p.parent
    if not p.is_dir():
        raise InputFormatError(f"Xenium input must be the output folder; {p} is not a directory.")
    if (p / "cell_feature_matrix.h5").is_file():
        full, notes = read_10x_h5(p / "cell_feature_matrix.h5", gex_only=False)
    elif is_mtx_dir(p / "cell_feature_matrix"):
        full, notes = read_mtx_dir(p / "cell_feature_matrix", gex_only=False)
    else:
        present = sorted(q.name for q in p.iterdir())[:20]
        raise InputFormatError(
            f"Xenium folder {p} has no cell_feature_matrix.h5 or cell_feature_matrix/ folder. Found: {present}"
        )

    ftypes = full.var.get("feature_types", pd.Series("Gene Expression", index=full.var_names)).astype(str)
    gex = (ftypes == "Gene Expression").to_numpy()
    if not gex.any():
        raise InputFormatError(f"{p}: the cell-feature matrix has no 'Gene Expression' features.")
    control = np.asarray(full[:, ~gex].X.sum(axis=1)).ravel() if (~gex).any() else np.zeros(full.n_obs)
    adata = full[:, gex].copy()
    adata.obs["control_counts"] = control.astype(np.float32)
    if (~gex).any():
        kinds = ftypes[~gex].value_counts().to_dict()
        notes.append(
            "Control features were excluded from the gene matrix and summarised in 'control_counts' ("
            + ", ".join(f"{v} {k}" for k, v in kinds.items()) + ")."
        )

    cells = _read_cells(p)
    shared = adata.obs_names.intersection(cells.index)
    if len(shared) < 0.9 * adata.n_obs:
        raise InputFormatError(
            f"Only {len(shared)} of {adata.n_obs} cells in the matrix have coordinates in the cells table; "
            "the files do not seem to come from the same Xenium run."
        )
    if len(shared) < adata.n_obs:
        notes.append(f"{adata.n_obs - len(shared)} cells without coordinates were excluded.")
    adata = adata[shared].copy()
    cells = cells.loc[shared]
    adata.obsm["spatial"] = cells[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float64)
    for col in ("cell_area", "nucleus_area"):
        if col in cells.columns:
            adata.obs[col] = cells[col].to_numpy()

    extra: dict = {"coordinate_units": "µm"}
    meta = p / "experiment.xenium"
    if meta.is_file():
        try:
            info = json.loads(meta.read_text())
            for key in ("panel_name", "run_name", "analysis_sw_version", "region_name"):
                if key in info:
                    extra[f"xenium_{key}"] = str(info[key])
        except json.JSONDecodeError:
            notes.append("experiment.xenium could not be parsed; run metadata omitted.")
    return finalize(adata, fmt="xenium", modality="xenium", path=p, notes=notes, sample_name=sample_name, extra=extra)
