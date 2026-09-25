"""10x Genomics Visium (Space Ranger ``outs/``) folders.

Expected layout (Space Ranger 1.x-3.x; Visium HD bin folders also work)::

    outs/
      filtered_feature_bc_matrix.h5   (or filtered_feature_bc_matrix/)
      spatial/
        tissue_positions.csv          (or tissue_positions_list.csv / .parquet)
        scalefactors_json.json
        tissue_hires_image.png        (optional)
        tissue_lowres_image.png       (optional)
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from cellscribe.errors import InputFormatError
from cellscribe.loaders._common import finalize, sample_name_from_path
from cellscribe.loaders.tenx import is_mtx_dir, read_10x_h5, read_mtx_dir

_POS_COLUMNS = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]


def find_outs(p: Path) -> Path:
    if (p / "spatial").is_dir():
        return p
    if (p / "outs" / "spatial").is_dir():
        return p / "outs"
    if p.name == "spatial" and p.is_dir():
        return p.parent
    raise InputFormatError(
        f"{p} is not a Visium output folder: no 'spatial/' subfolder found (looked in {p} and {p / 'outs'})."
    )


def _read_positions(spatial: Path) -> pd.DataFrame:
    for name in ("tissue_positions.csv", "tissue_positions.parquet", "tissue_positions_list.csv"):
        f = spatial / name
        if not f.is_file():
            continue
        if name.endswith(".parquet"):
            pos = pd.read_parquet(f)
        elif name == "tissue_positions_list.csv":
            pos = pd.read_csv(f, header=None)
            if pos.shape[1] != 6:
                raise InputFormatError(f"{f} should have 6 columns, found {pos.shape[1]}.")
            pos.columns = _POS_COLUMNS
        else:
            pos = pd.read_csv(f)
        missing = [c for c in _POS_COLUMNS if c not in pos.columns]
        if missing:
            raise InputFormatError(f"{f} is missing columns {missing}; found {list(pos.columns)}.")
        pos["barcode"] = pos["barcode"].astype(str)
        return pos.set_index("barcode")
    raise InputFormatError(
        f"No spot positions found in {spatial}: expected tissue_positions.csv, tissue_positions_list.csv, "
        "or tissue_positions.parquet."
    )


def _read_image(f: Path) -> np.ndarray | None:
    if not f.is_file():
        return None
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(f) as im:
        return np.asarray(im.convert("RGB"))


def load_visium(path: str | Path, sample_name: str | None = None) -> ad.AnnData:
    p = Path(path)
    if not p.exists():
        raise InputFormatError(f"Input path does not exist: {p}")
    outs = find_outs(p)
    spatial = outs / "spatial"

    if (outs / "filtered_feature_bc_matrix.h5").is_file():
        adata, notes = read_10x_h5(outs / "filtered_feature_bc_matrix.h5")
    elif is_mtx_dir(outs / "filtered_feature_bc_matrix"):
        adata, notes = read_mtx_dir(outs / "filtered_feature_bc_matrix")
    else:
        present = sorted(q.name for q in outs.iterdir())
        raise InputFormatError(
            f"Visium folder {outs} has no filtered_feature_bc_matrix.h5 or filtered_feature_bc_matrix/ "
            f"folder. Found: {present}"
        )

    pos = _read_positions(spatial)
    shared = adata.obs_names.intersection(pos.index)
    if len(shared) < 0.5 * adata.n_obs:
        raise InputFormatError(
            f"Only {len(shared)} of {adata.n_obs} matrix barcodes appear in the spot positions file; the matrix "
            "and spatial/ folder do not seem to come from the same Space Ranger run."
        )
    pos = pos.loc[shared]
    keep = pos.index[pos["in_tissue"].astype(int) == 1]
    n_off = len(shared) - len(keep)
    adata = adata[keep].copy()
    pos = pos.loc[keep]
    if n_off:
        notes.append(f"{n_off} spots outside the tissue (in_tissue = 0) were excluded.")
    if len(shared) < adata.n_obs:
        notes.append(f"{adata.n_obs - len(shared)} matrix barcodes had no spot position and were excluded.")
    adata.obs["in_tissue"] = pos["in_tissue"].astype(int).to_numpy()
    adata.obs["array_row"] = pos["array_row"].astype(int).to_numpy()
    adata.obs["array_col"] = pos["array_col"].astype(int).to_numpy()
    adata.obsm["spatial"] = pos[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=np.float64)

    library_id = sample_name or sample_name_from_path(outs)
    scalefactors = {}
    sf_file = spatial / "scalefactors_json.json"
    if sf_file.is_file():
        try:
            scalefactors = json.loads(sf_file.read_text())
        except json.JSONDecodeError as exc:
            raise InputFormatError(f"{sf_file} is not valid JSON: {exc}") from exc
    else:
        notes.append("spatial/scalefactors_json.json not found; the tissue image overlay is disabled.")
    images = {}
    for key in ("hires", "lowres"):
        img = _read_image(spatial / f"tissue_{key}_image.png")
        if img is not None:
            images[key] = img
    if not images:
        notes.append("No tissue image (tissue_hires_image.png / tissue_lowres_image.png) found; spots are drawn without an image.")
    adata.uns["spatial"] = {library_id: {"images": images, "scalefactors": scalefactors, "metadata": {}}}
    return finalize(
        adata, fmt="visium", modality="visium", path=p, notes=notes, sample_name=sample_name,
        extra={"library_id": library_id},
    )
