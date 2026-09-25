"""Shared fixtures. Expensive pipeline runs are session-scoped and reused across tests."""

from __future__ import annotations

import logging
import warnings

import pytest

from cellscribe import synthetic
from cellscribe.config import CellscribeConfig


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    """Tests never talk to the real Claude API."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture(scope="session", autouse=True)
def _quiet():
    warnings.filterwarnings("ignore")
    logging.getLogger("cellscribe").setLevel(logging.ERROR)


@pytest.fixture(scope="session")
def sc_adata():
    """Small PBMC-like dataset with damaged cells, doublets and an uncharacterised population."""
    return synthetic.make_single_cell(n_cells=1500, seed=0, n_filler_genes=800)


@pytest.fixture(scope="session")
def base_config():
    return CellscribeConfig().with_overrides({"narrative.enabled": False, "spatial.n_perms": 100})


@pytest.fixture(scope="session")
def sc_result(sc_adata, base_config):
    from cellscribe.pipeline import run

    return run(sc_adata, output=None, config=base_config)


@pytest.fixture(scope="session")
def spatial_result(base_config):
    from cellscribe.pipeline import run

    return run(synthetic.make_spatial(n_cells=2000, seed=0), output=None, config=base_config)


@pytest.fixture(scope="session")
def visium_result(base_config):
    from cellscribe.pipeline import run

    return run(synthetic.make_visium(n_spots_target=500, seed=0), output=None, config=base_config)
