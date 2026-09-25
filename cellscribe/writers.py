"""Write AnnData objects to the on-disk formats Cellscribe reads.

Used to materialise synthetic datasets as realistic input folders (for tests
and for users who want to try each loader), mirroring the layouts produced by
Cell Ranger, Space Ranger, and the Xenium Onboard Analysis pipeline.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
from PIL import Image


def _csr(adata: ad.AnnData) -> sp.csr_matrix:
    X = adata.X
    return sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()


def _gene_ids(adata: ad.AnnData) -> list[str]:
    if "gene_ids" in adata.var:
        return adata.var["gene_ids"].astype(str).tolist()
    prefix = "ENSMUSG" if any(g[:1].isupper() and g[1:2].islower() for g in adata.var_names[:50]) else "ENSG"
    return [f"{prefix}{i:011d}" for i in range(adata.n_vars)]


def write_10x_mtx(adata: ad.AnnData, directory: str | Path, feature_types: list[str] | None = None) -> Path:
    """Write a Cell Ranger v3 ``filtered_feature_bc_matrix`` folder (gzipped)."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    X = _csr(adata)
    with gzip.open(d / "matrix.mtx.gz", "wb") as fh:
        scipy.io.mmwrite(fh, sp.coo_matrix(X.T.astype(np.int64)), field="integer")
    with gzip.open(d / "barcodes.tsv.gz", "wt") as fh:
        fh.write("\n".join(adata.obs_names) + "\n")
    ft = feature_types or ["Gene Expression"] * adata.n_vars
    feats = pd.DataFrame({"id": _gene_ids(adata), "name": adata.var_names, "type": ft})
    with gzip.open(d / "features.tsv.gz", "wt") as fh:
        feats.to_csv(fh, sep="\t", header=False, index=False)
    return d


def write_10x_h5(adata: ad.AnnData, path: str | Path, feature_types: list[str] | None = None, genome: str = "GRCh38") -> Path:
    """Write a Cell Ranger v3-style HDF5 feature-barcode matrix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    X = _csr(adata)  # cells x genes CSR == genes x cells CSC, as 10x stores it
    ft = feature_types or ["Gene Expression"] * adata.n_vars
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("barcodes", data=np.array(adata.obs_names, dtype="S"))
        g.create_dataset("data", data=X.data.astype(np.int32))
        g.create_dataset("indices", data=X.indices.astype(np.int64))
        g.create_dataset("indptr", data=X.indptr.astype(np.int64))
        g.create_dataset("shape", data=np.array([adata.n_vars, adata.n_obs], dtype=np.int32))
        fg = g.create_group("features")
        fg.create_dataset("_all_tag_keys", data=np.array(["genome"], dtype="S"))
        fg.create_dataset("feature_type", data=np.array(ft, dtype="S"))
        fg.create_dataset("genome", data=np.array([genome] * adata.n_vars, dtype="S"))
        fg.create_dataset("id", data=np.array(_gene_ids(adata), dtype="S"))
        fg.create_dataset("name", data=np.array(adata.var_names, dtype="S"))
    return path


def write_csv(adata: ad.AnnData, path: str | Path, orientation: str = "genes_x_cells") -> Path:
    """Write a dense counts table (genes x cells by default, as most cores deliver)."""
    path = Path(path)
    X = _csr(adata).toarray().astype(np.int64)
    if orientation == "genes_x_cells":
        df = pd.DataFrame(X.T, index=adata.var_names, columns=adata.obs_names)
    elif orientation == "cells_x_genes":
        df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
    else:
        raise ValueError(orientation)
    sep = "\t" if path.suffix in {".tsv", ".txt"} or path.name.endswith(".tsv.gz") else ","
    df.to_csv(path, sep=sep)
    return path


def write_visium(adata: ad.AnnData, outs: str | Path, use_h5: bool = True) -> Path:
    """Write a Space Ranger-like ``outs/`` folder from :func:`synthetic.make_visium` output."""
    outs = Path(outs)
    spatial = outs / "spatial"
    spatial.mkdir(parents=True, exist_ok=True)
    if use_h5:
        write_10x_h5(adata, outs / "filtered_feature_bc_matrix.h5")
    else:
        write_10x_mtx(adata, outs / "filtered_feature_bc_matrix")
    lib = next(iter(adata.uns["spatial"]))
    info = adata.uns["spatial"][lib]
    allspots = adata.uns.get("_all_spots")
    if allspots is not None:
        pos = pd.DataFrame(
            {
                "barcode": allspots["barcode"],
                "in_tissue": allspots["in_tissue"],
                "array_row": allspots["array_row"],
                "array_col": allspots["array_col"],
                "pxl_row_in_fullres": np.round(allspots["pxl_row"]).astype(int),
                "pxl_col_in_fullres": np.round(allspots["pxl_col"]).astype(int),
            }
        )
    else:
        pos = pd.DataFrame(
            {
                "barcode": adata.obs_names,
                "in_tissue": 1,
                "array_row": adata.obs.get("array_row", 0),
                "array_col": adata.obs.get("array_col", 0),
                "pxl_row_in_fullres": np.round(adata.obsm["spatial"][:, 1]).astype(int),
                "pxl_col_in_fullres": np.round(adata.obsm["spatial"][:, 0]).astype(int),
            }
        )
    pos.to_csv(spatial / "tissue_positions.csv", index=False)
    (spatial / "scalefactors_json.json").write_text(json.dumps(dict(info["scalefactors"]), indent=2))
    for key in ("hires", "lowres"):
        if key in info.get("images", {}):
            Image.fromarray(np.asarray(info["images"][key])).save(spatial / f"tissue_{key}_image.png")
    return outs


def write_xenium(adata: ad.AnnData, directory: str | Path, n_control_probes: int = 20, seed: int = 0) -> Path:
    """Write a Xenium-like output folder (cell_feature_matrix.h5 + cells.csv.gz).

    Adds negative-control probe features with sparse background counts so the
    loader's control-feature handling is exercised.
    """
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    X = _csr(adata)
    ctrl = sp.csr_matrix(rng.poisson(0.02, (adata.n_obs, n_control_probes)).astype(np.float32))
    full = ad.AnnData(
        X=sp.hstack([X, ctrl]).tocsr(),
        obs=pd.DataFrame(index=adata.obs_names),
        var=pd.DataFrame(index=list(adata.var_names) + [f"NegControlProbe_{i:05d}" for i in range(n_control_probes)]),
    )
    ft = ["Gene Expression"] * adata.n_vars + ["Negative Control Probe"] * n_control_probes
    write_10x_h5(full, d / "cell_feature_matrix.h5", feature_types=ft)
    xy = adata.obsm["spatial"]
    tx = np.asarray(X.sum(axis=1)).ravel()
    cc = np.asarray(ctrl.sum(axis=1)).ravel()
    cells = pd.DataFrame(
        {
            "cell_id": adata.obs_names,
            "x_centroid": xy[:, 0],
            "y_centroid": xy[:, 1],
            "transcript_counts": tx.astype(int),
            "control_probe_counts": cc.astype(int),
            "control_codeword_counts": 0,
            "total_counts": (tx + cc).astype(int),
            "cell_area": adata.obs.get("cell_area", pd.Series(50.0, index=adata.obs_names)).to_numpy(),
            "nucleus_area": adata.obs.get("cell_area", pd.Series(50.0, index=adata.obs_names)).to_numpy() * 0.4,
        }
    )
    cells.to_csv(d / "cells.csv.gz", index=False, compression="gzip")
    (d / "experiment.xenium").write_text(
        json.dumps({"major_version": 5, "run_name": "synthetic", "analysis_sw_version": "synthetic", "pixel_size": 0.2125})
    )
    return d
