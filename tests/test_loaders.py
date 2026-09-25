"""Every loader: round-trips, auto-detection, the AnnData contract, and loud, specific failures."""

import gzip
import shutil

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellscribe import synthetic, writers
from cellscribe.errors import InputFormatError
from cellscribe.loaders import detect_format, load


@pytest.fixture(scope="module")
def small():
    return synthetic.make_single_cell(n_cells=250, seed=3, n_filler_genes=150, n_batches=1)


def _assert_contract(adata, modality):
    assert sp.issparse(adata.X) and adata.X.dtype == np.float32
    assert "counts" in adata.layers
    assert adata.obs_names.is_unique and adata.var_names.is_unique
    assert "sample" in adata.obs
    info = adata.uns["cellscribe_input"]
    assert info["modality"] == modality and info["n_cells_loaded"] == adata.n_obs
    if modality != "single-cell":
        assert adata.obsm["spatial"].shape == (adata.n_obs, 2)


def _same_counts(loaded, original):
    return np.array_equal(loaded[original.obs_names, original.var_names].X.toarray(), original.X.toarray())


def test_10x_mtx_folder(tmp_path, small):
    d = writers.write_10x_mtx(small, tmp_path / "filtered_feature_bc_matrix")
    assert detect_format(d) == "10x"
    a = load(d)
    _assert_contract(a, "single-cell")
    assert _same_counts(a, small)
    assert "gene_ids" in a.var
    # a Cell Ranger outs/ folder and the matrix file itself also work
    outs = tmp_path / "run" / "outs"
    outs.mkdir(parents=True)
    shutil.copytree(d, outs / "filtered_feature_bc_matrix")
    assert load(tmp_path / "run").n_obs == small.n_obs
    assert load(d / "matrix.mtx.gz").n_obs == small.n_obs


def test_10x_v2_genome_subfolder(tmp_path, small):
    d = tmp_path / "filtered_gene_bc_matrices" / "hg19"
    writers.write_10x_mtx(small, d)
    for f in d.iterdir():  # v2 layout: uncompressed, genes.tsv without feature type
        with gzip.open(f) as src:
            data = src.read()
        name = f.name[:-3].replace("features", "genes")
        if name == "genes.tsv":
            data = b"\n".join(b"\t".join(line.split(b"\t")[:2]) for line in data.splitlines()) + b"\n"
        (d / name).write_bytes(data)
        f.unlink()
    a = load(d.parent)
    assert _same_counts(a, small)
    assert any("genome subfolder" in n for n in a.uns["cellscribe_input"]["notes"])


def test_10x_h5(tmp_path, small):
    ft = ["Gene Expression"] * small.n_vars
    ft[-3:] = ["Antibody Capture"] * 3
    p = writers.write_10x_h5(small, tmp_path / "filtered_feature_bc_matrix.h5", feature_types=ft)
    assert detect_format(p) == "10x"
    a = load(p)
    _assert_contract(a, "single-cell")
    assert a.n_vars == small.n_vars - 3  # non-gene-expression features dropped
    assert any("Antibody Capture" in n for n in a.uns["cellscribe_input"]["notes"])


def test_h5ad_prefers_raw_counts_layer(tmp_path, small):
    import scanpy as sc

    b = small.copy()
    b.layers["counts"] = b.X.copy()
    sc.pp.normalize_total(b)
    sc.pp.log1p(b)
    b.obsm["X_pca"] = np.zeros((b.n_obs, 2))
    p = tmp_path / "processed.h5ad"
    b.write_h5ad(p)
    a = load(p)
    _assert_contract(a, "single-cell")
    assert _same_counts(a, small)
    assert any("X_pca" in n for n in a.uns["cellscribe_input"]["notes"])


def test_h5ad_normalised_without_counts_is_rejected(tmp_path, small):
    import scanpy as sc

    b = small.copy()
    sc.pp.normalize_total(b)
    sc.pp.log1p(b)
    b.write_h5ad(tmp_path / "lognorm.h5ad")
    with pytest.raises(InputFormatError, match="normalised"):
        load(tmp_path / "lognorm.h5ad")


def test_h5ad_ensembl_ids_are_swapped_for_symbols(tmp_path, small):
    b = small.copy()
    b.var["gene_symbols"] = b.var_names
    b.var_names = [f"ENSG{i:011d}" for i in range(b.n_vars)]
    b.write_h5ad(tmp_path / "ens.h5ad")
    a = load(tmp_path / "ens.h5ad")
    assert "CD3E" in a.var_names and a.var["gene_ids"].str.startswith("ENSG").all()


def test_transposed_matrix_fails_loudly(tmp_path, small):
    t = ad.AnnData(small.X.T.tocsr(), obs=pd.DataFrame(index=small.var_names), var=pd.DataFrame(index=small.obs_names))
    t.write_h5ad(tmp_path / "t.h5ad")
    with pytest.raises(InputFormatError, match="transposed"):
        load(tmp_path / "t.h5ad")


@pytest.mark.parametrize("orientation,name", [("genes_x_cells", "g.csv"), ("cells_x_genes", "c.tsv")])
def test_csv_both_orientations(tmp_path, small, orientation, name):
    p = writers.write_csv(small, tmp_path / name, orientation=orientation)
    assert detect_format(p) == "csv"
    a = load(p)
    _assert_contract(a, "single-cell")
    assert a.uns["cellscribe_input"]["csv_orientation"] == orientation
    assert _same_counts(a, small)
    assert not any(n.startswith("AMBIGUOUS") for n in a.uns["cellscribe_input"]["notes"])


def test_csv_ambiguous_orientation_warns(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.poisson(3, (60, 40)), index=[f"g{i}" for i in range(60)], columns=[f"c{i}" for i in range(40)])
    df.to_csv(tmp_path / "anon.csv")
    a = load(tmp_path / "anon.csv")
    notes = a.uns["cellscribe_input"]["notes"]
    assert any(n.startswith("AMBIGUOUS ORIENTATION") for n in notes)
    assert a.shape == (40, 60)  # more rows than columns -> rows assumed to be genes
    forced = load(tmp_path / "anon.csv", csv_orientation="cells_x_genes")
    assert forced.shape == (60, 40)


def test_csv_with_gene_name_column(tmp_path, small):
    df = pd.DataFrame(small.X.toarray().T.astype(int), index=[f"ENSG{i:011d}" for i in range(small.n_vars)], columns=small.obs_names)
    df.insert(0, "gene_name", small.var_names)
    df.index.name = "gene_id"
    df.to_csv(tmp_path / "annot.csv")
    a = load(tmp_path / "annot.csv")
    assert "CD3E" in a.var_names and a.n_obs == small.n_obs


def test_csv_errors(tmp_path):
    (tmp_path / "text.csv").write_text("gene,a,b\nX,1,foo\nY,2,bar\n")
    with pytest.raises(InputFormatError, match="non-numeric"):
        load(tmp_path / "text.csv")
    (tmp_path / "neg.csv").write_text("gene,a,b\nCD3E,1,-2\nMS4A1,2,3\n")
    with pytest.raises(InputFormatError, match="negative"):
        load(tmp_path / "neg.csv")


def test_visium_folder(tmp_path):
    v = synthetic.make_visium(n_spots_target=250, seed=1)
    outs = writers.write_visium(v, tmp_path / "sample1" / "outs")
    for p in (tmp_path / "sample1", outs):
        assert detect_format(p) == "visium"
    a = load(tmp_path / "sample1")
    _assert_contract(a, "visium")
    assert a.n_obs == v.n_obs  # off-tissue spots excluded
    lib = a.uns["cellscribe_input"]["library_id"]
    assert set(a.uns["spatial"][lib]["images"]) == {"hires", "lowres"}
    assert np.allclose(a[v.obs_names].obsm["spatial"], np.round(v.obsm["spatial"]))


def test_visium_legacy_positions_and_missing_pieces(tmp_path):
    v = synthetic.make_visium(n_spots_target=150, seed=2)
    outs = writers.write_visium(v, tmp_path / "outs", use_h5=False)
    pos = pd.read_csv(outs / "spatial" / "tissue_positions.csv")
    pos.to_csv(outs / "spatial" / "tissue_positions_list.csv", header=False, index=False)
    (outs / "spatial" / "tissue_positions.csv").unlink()
    for img in (outs / "spatial").glob("*.png"):
        img.unlink()
    a = load(outs)
    assert a.n_obs == v.n_obs
    assert any("No tissue image" in n for n in a.uns["cellscribe_input"]["notes"])
    (outs / "spatial" / "tissue_positions_list.csv").unlink()
    with pytest.raises(InputFormatError, match="spot positions"):
        load(outs, "visium")


def test_xenium_folder(tmp_path):
    x = synthetic.make_spatial(n_cells=400, seed=1)
    d = writers.write_xenium(x, tmp_path / "output-XETG001")
    assert detect_format(d) == "xenium"
    a = load(d)
    _assert_contract(a, "xenium")
    assert a.n_vars == x.n_vars  # negative-control probes removed
    assert "control_counts" in a.obs
    assert np.allclose(a[x.obs_names].obsm["spatial"], x.obsm["spatial"])


def test_xenium_mismatched_cells_fail(tmp_path):
    x = synthetic.make_spatial(n_cells=300, seed=1)
    d = writers.write_xenium(x, tmp_path / "xen")
    cells = pd.read_csv(d / "cells.csv.gz")
    cells["cell_id"] = [f"other-{i}" for i in range(len(cells))]
    cells.to_csv(d / "cells.csv.gz", index=False, compression="gzip")
    with pytest.raises(InputFormatError, match="same Xenium run"):
        load(d)


def test_structure_errors_are_specific(tmp_path, small):
    with pytest.raises(InputFormatError, match="does not exist"):
        load(tmp_path / "missing")
    (tmp_path / "empty").mkdir()
    with pytest.raises(InputFormatError, match="empty directory"):
        load(tmp_path / "empty")
    partial = tmp_path / "partial"
    writers.write_10x_mtx(small, partial)
    (partial / "barcodes.tsv.gz").unlink()
    with pytest.raises(InputFormatError, match="barcodes"):
        load(partial)
    raw = tmp_path / "rawonly" / "outs" / "raw_feature_bc_matrix"
    raw.mkdir(parents=True)
    with pytest.raises(InputFormatError, match="raw"):
        load(tmp_path / "rawonly")
    (tmp_path / "notes.docx").write_text("x")
    with pytest.raises(InputFormatError, match="Unrecognised file type"):
        load(tmp_path / "notes.docx")
    with pytest.raises(InputFormatError, match="Unknown format"):
        load(partial, "loom")
