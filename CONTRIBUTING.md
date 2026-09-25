# Contributing to Cellscribe

Thank you for helping improve Cellscribe. It is community-maintained, research-oriented infrastructure, and contributions of every size are welcome: bug reports, documentation, marker sets, loaders, QC heuristics, and tests.

## Reporting problems

Please open an issue with:

- the command you ran (and your `--config` file, if any),
- the Cellscribe version (`cellscribe --version`) and the software table from the report's *Methods & parameters* section,
- the error message (re-run with `--verbose` for the full traceback), and
- if possible, a small dataset that reproduces the problem. `cellscribe example-data` can generate synthetic inputs in every supported format.

Please do not attach patient-identifiable or otherwise sensitive data.

## Development setup

```bash
git clone https://github.com/jonahnki/cellscribe.git
cd cellscribe
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The test suite runs the full pipeline on small synthetic datasets and takes a couple of minutes. `cellscribe demo` is a quick end-to-end check of the report.

## Guidelines

- **Keep the stages separable.** Loaders, QC, preprocessing, clustering, annotation, spatial statistics, the narrative, and the report are independent modules with small, testable functions. New analysis steps should follow the same pattern and record what they did (parameters, notes) so the report can show it.
- **Fail loudly and specifically.** User-fixable problems should raise a `cellscribe.errors.CellscribeError` subclass with a message that says what was expected, what was found, and how to fix it.
- **Record every parameter.** New tunables belong in `cellscribe/config.py` (validated with pydantic) so they appear in the report's parameter table.
- **Be conservative in what the report claims.** Prefer "unresolved" or "review" over a confident but weakly supported statement, and hedge messages where genuine biology can look like an artefact.
- **Test what you add.** Use the synthetic generators in `cellscribe/synthetic.py` (they have known ground truth) and the writers in `cellscribe/writers.py` to build realistic input folders.
- Code style: type hints, docstrings on public functions, and comments that explain *why* rather than *what*.

## Contributing marker genes

The bundled reference (`cellscribe/data/markers/canonical_markers.yaml`) must remain freely redistributable under the MIT license. When adding or editing entries:

- use canonical markers supported by the primary literature or public atlases, and mention the source in your pull request;
- do **not** copy exports of databases whose redistribution terms are unclear (for example PanglaoDB or CellMarker). Users can still load such files locally with `--markers`;
- provide both `human` (HGNC) and `mouse` (MGI) symbols, 5 to 10 markers per entry, a `lineage`, and `kind: state` for cell states rather than cell types;
- run `pytest tests/test_synthetic_and_markers.py` to validate the file.

## Pull requests

1. Create a branch, make your change, and add or update tests.
2. Run `pytest` and `cellscribe demo` locally.
3. Update `README.md` if you change user-facing behaviour. The README must describe only what is implemented.
4. Open a pull request describing the change and its motivation.

By contributing, you agree that your contributions are licensed under the MIT license. Please be respectful and constructive in all project spaces.
