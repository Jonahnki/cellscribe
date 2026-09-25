import json

import yaml
from typer.testing import CliRunner

from cellscribe import __version__, synthetic
from cellscribe.cli import app

runner = CliRunner()


def test_version_and_help():
    r = runner.invoke(app, ["--version"])
    assert r.exit_code == 0 and __version__ in r.output
    r = runner.invoke(app, ["run", "--help"])
    assert r.exit_code == 0 and "--resolution" in r.output and "--format" in r.output


def test_config_command_outputs_valid_yaml(tmp_path):
    r = runner.invoke(app, ["config", "--output", str(tmp_path / "c.yaml")])
    assert r.exit_code == 0
    data = yaml.safe_load((tmp_path / "c.yaml").read_text())
    assert data["qc"]["mode"] == "flag"


def test_demo_end_to_end(tmp_path):
    out = tmp_path / "demo.html"
    r = runner.invoke(app, ["demo", "--output", str(out), "--n", "700", "--quiet"])
    assert r.exit_code == 0, r.output
    html = out.read_text()
    assert "Executive summary" in html and "Per-cluster QC" in html


def test_run_with_options_and_outputs(tmp_path):
    a = synthetic.make_single_cell(n_cells=600, seed=4, n_filler_genes=300, n_batches=1)
    a.write_h5ad(tmp_path / "in.h5ad")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("qc:\n  nmads: 4\n")
    r = runner.invoke(
        app,
        [
            "run", "-i", str(tmp_path / "in.h5ad"), "-o", str(tmp_path / "r.html"), "--resolution", "0.5",
            "--config", str(cfg), "--species", "human", "--summary-json", str(tmp_path / "s.json"),
            "--save-h5ad", str(tmp_path / "out.h5ad"), "--no-narrative", "--quiet",
        ],
    )
    assert r.exit_code == 0, r.output
    summary = json.loads((tmp_path / "s.json").read_text())
    assert summary["clustering"]["resolution"] == 0.5 and not summary["clustering"]["auto_selected"]
    assert summary["config"]["qc"]["nmads"] == 4
    import anndata as ad

    out = ad.read_h5ad(tmp_path / "out.h5ad")
    assert "cellscribe_cluster" in out.obs and "X_umap" in out.obsm


def test_bad_input_gives_clean_error(tmp_path):
    (tmp_path / "empty").mkdir()
    r = runner.invoke(app, ["run", "-i", str(tmp_path / "empty"), "-o", str(tmp_path / "x.html")])
    assert r.exit_code == 2
    assert "Could not recognise" in r.output and "Traceback" not in r.output


def test_bad_config_gives_clean_error(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("clustering:\n  resolutoin: 1\n")
    r = runner.invoke(app, ["run", "-i", str(tmp_path), "--config", str(cfg)])
    assert r.exit_code == 2 and "resolutoin" in r.output


def test_example_data_command(tmp_path):
    r = runner.invoke(app, ["example-data", str(tmp_path / "ex"), "--n", "300"])
    assert r.exit_code == 0
    assert (tmp_path / "ex" / "filtered_feature_bc_matrix" / "matrix.mtx.gz").exists()
    assert (tmp_path / "ex" / "counts.h5ad").exists()
