import numpy as np

from cellscribe import spatial
from cellscribe.config import SpatialConfig
from cellscribe.utils import CLUSTER_KEY


def test_spatial_statistics_on_imaging_data(spatial_result):
    sp = spatial_result.spatial
    assert sp.enabled and sp.modality == "xenium" and sp.coord_units == "µm"
    assert sp.graph["method"].startswith("Delaunay")
    labels = {c.cluster: c.label for c in spatial_result.annotation.clusters}
    pairs = {frozenset((labels[p["a"]], labels[p["b"]])) for p in sp.colocalised_pairs}
    # the simulated crypts mix colonocytes and goblet cells; the follicle mixes B and T cells
    assert frozenset({"Colonocyte", "Goblet cell"}) in pairs
    assert frozenset({"B cell", "CD4+ T cell"}) in pairs
    z = np.asarray(sp.nhood_zscore)
    assert z.shape == (len(sp.nhood_clusters),) * 2
    assert sp.n_genes_significant > 10
    top_genes = {m["gene"] for m in sp.moran_top[:10]}
    assert top_genes & {"LMOD1", "MYL9", "ACTA2", "MYH11", "DES", "SYNPO2", "TAGLN"}  # the muscularis band


def test_visium_uses_grid_graph_and_image(visium_result):
    sp = visium_result.spatial
    assert sp.graph["method"].startswith("hexagonal") and sp.has_image
    img, sf, lib = spatial.tissue_image(visium_result.adata)
    assert img.dtype == np.uint8 and 0 < sf < 1 and lib
    assert any(c.status == "mixed" for c in visium_result.annotation.clusters)


def test_spatial_skipped_without_coordinates(sc_result):
    assert sc_result.spatial is None
    assert sc_result.summary["spatial"] is None
    assert spatial.run(sc_result.adata, SpatialConfig(), "single-cell").enabled is False


def test_spatial_disabled_by_config(spatial_result):
    assert spatial.run(spatial_result.adata.copy(), SpatialConfig(enabled="off"), "xenium").enabled is False


def test_collinear_coordinates_fall_back_to_knn(spatial_result):
    a = spatial_result.adata.copy()
    a.obsm["spatial"] = a.obsm["spatial"].copy()
    a.obsm["spatial"][:, 1] = 0.0
    res = spatial.run(a, SpatialConfig(n_perms=50), "xenium", CLUSTER_KEY)
    assert res.enabled and "nearest" in res.graph["method"]
    assert any("Delaunay" in n["message"] for n in res.notes)
