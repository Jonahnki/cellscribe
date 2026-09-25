import pytest

from cellscribe.config import CellscribeConfig
from cellscribe.errors import ConfigError


def test_defaults_are_valid_and_round_trip(tmp_path):
    cfg = CellscribeConfig()
    assert cfg.qc.mode == "flag"
    assert cfg.clustering.resolution is None
    path = tmp_path / "c.yaml"
    path.write_text(cfg.to_yaml())
    assert CellscribeConfig.from_yaml(path) == cfg


def test_yaml_partial_override(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("qc:\n  nmads: 4\nclustering:\n  resolution: 0.6\n")
    cfg = CellscribeConfig.from_yaml(path)
    assert cfg.qc.nmads == 4
    assert cfg.clustering.resolution == 0.6
    assert cfg.qc.nmads_mt == 3.0  # untouched default


def test_unknown_keys_fail_loudly(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("qc:\n  nmad: 4\n")  # typo
    with pytest.raises(ConfigError, match="qc.nmad"):
        CellscribeConfig.from_yaml(path)


def test_invalid_values_rejected():
    with pytest.raises(ConfigError):
        CellscribeConfig.from_dict({"qc": {"mode": "delete"}})
    with pytest.raises(ConfigError):
        CellscribeConfig.from_dict({"clustering": {"resolution_grid": [0.5, -1]}})


def test_missing_and_malformed_files(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        CellscribeConfig.from_yaml(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a list\n")
    with pytest.raises(ConfigError, match="mapping"):
        CellscribeConfig.from_yaml(bad)


def test_overrides_ignore_none_and_validate():
    cfg = CellscribeConfig().with_overrides({"clustering.resolution": None, "qc.mode": "filter", "seed": 7})
    assert cfg.clustering.resolution is None and cfg.qc.mode == "filter" and cfg.seed == 7
    with pytest.raises(ConfigError):
        CellscribeConfig().with_overrides({"clustering.resolution": -2})


def test_flat_lists_every_parameter():
    flat = CellscribeConfig().flat()
    assert "qc.nmads" in flat and "narrative.model" in flat and "seed" in flat
    assert flat["narrative.model"] == "claude-opus-5"
