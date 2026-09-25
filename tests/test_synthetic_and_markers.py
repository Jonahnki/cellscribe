import numpy as np
import pytest

from cellscribe import synthetic
from cellscribe.markers import load_reference
from cellscribe.utils import infer_species, is_hb, is_mito, is_ribo


def test_single_cell_generator_shape_and_truth(sc_adata):
    assert sc_adata.n_obs == 1500
    truth = set(sc_adata.obs["true_cell_type"])
    assert {synthetic.LOW_QUALITY, synthetic.DOUBLET, synthetic.UNCHARACTERISED, "B cell"} <= truth
    X = sc_adata.X
    assert np.all(X.data >= 0) and np.allclose(X.data, np.round(X.data))
    assert sc_adata.obs["sample"].nunique() == 2


def test_generator_is_reproducible():
    a = synthetic.make_single_cell(n_cells=300, seed=5, n_filler_genes=200)
    b = synthetic.make_single_cell(n_cells=300, seed=5, n_filler_genes=200)
    assert (a.X != b.X).nnz == 0
    assert list(a.obs_names) == list(b.obs_names)


def test_damaged_cells_have_high_mito(sc_adata):
    mt = is_mito(sc_adata.var_names)
    tot = np.asarray(sc_adata.X.sum(axis=1)).ravel()
    pct = np.asarray(sc_adata.X[:, mt].sum(axis=1)).ravel() / tot
    lq = (sc_adata.obs["true_cell_type"] == synthetic.LOW_QUALITY).to_numpy()
    assert np.median(pct[lq]) > 0.2 > np.median(pct[~lq])


def test_spatial_generators():
    x = synthetic.make_spatial(n_cells=600)
    assert x.obsm["spatial"].shape == (600, 2)
    assert not is_mito(x.var_names).any()  # imaging panels carry no mitochondrial probes
    v = synthetic.make_visium(n_spots_target=200)
    lib = next(iter(v.uns["spatial"]))
    assert v.uns["spatial"][lib]["images"]["lowres"].dtype == np.uint8
    assert "spot_diameter_fullres" in v.uns["spatial"][lib]["scalefactors"]


def test_mouse_generator_uses_mouse_symbols():
    m = synthetic.make_single_cell(n_cells=300, species="mouse", n_filler_genes=100)
    assert "Cd3e" in m.var_names and "mt-Co1" in m.var_names
    assert infer_species(m.var_names)[0] == "mouse"


@pytest.mark.parametrize("species", ["human", "mouse"])
def test_bundled_reference(species):
    ref = load_reference(species)
    names = [s.name for s in ref.sets]
    assert len(names) == len(set(names))
    assert len(ref.cell_types) >= 70 and len(ref.states) >= 2
    for s in ref.sets:
        assert len(s.genes) >= 5
        assert all(isinstance(g, str) and g for g in s.genes)


def test_gene_class_patterns():
    assert list(is_mito(["MT-CO1", "mt-Nd1", "MTOR"])) == [True, True, False]
    assert list(is_ribo(["RPL10A", "RPS27A", "Rpl13", "RPS6KA1", "RPLP0", "RPL36AL"])) == [True, True, True, False, True, True]
    assert list(is_hb(["HBB", "Hba-a1", "HBEGF", "HBP1"])) == [True, True, False, False]


def test_species_inference():
    assert infer_species(["CD3E", "MS4A1", "LYZ", "GAPDH"])[0] == "human"
    assert infer_species(["Cd3e", "Ms4a1", "Lyz2", "Gapdh"])[0] == "mouse"
    assert infer_species(["ENSMUSG00000000001", "ENSMUSG00000000028"])[0] == "mouse"
