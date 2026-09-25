"""Cellscribe: automated first-pass QC, clustering, and annotation reports.

Cellscribe turns a counts matrix (10x, AnnData, CSV, Visium, or Xenium) into a
single self-contained HTML report covering quality control, clustering,
marker-based cell-type annotation, and (for spatial data) spatial statistics.

Typical programmatic use::

    import cellscribe
    result = cellscribe.run("filtered_feature_bc_matrix/", output="report.html")
    result.adata          # processed AnnData
    result.summary        # JSON-serialisable structured summary
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__", "run", "load", "CellscribeConfig"]


def __getattr__(name: str):
    # Lazy imports keep `import cellscribe` (and `cellscribe --help`) fast;
    # scanpy/squidpy are only imported when the pipeline actually runs.
    if name == "run":
        from cellscribe.pipeline import run

        return run
    if name == "load":
        from cellscribe.loaders import load

        return load
    if name == "CellscribeConfig":
        from cellscribe.config import CellscribeConfig

        return CellscribeConfig
    raise AttributeError(f"module 'cellscribe' has no attribute {name!r}")
