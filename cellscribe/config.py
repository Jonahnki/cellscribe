"""Validated configuration for a Cellscribe run.

All tunable parameters live here so that (a) YAML config files are validated
with precise error messages, and (b) the report's methods appendix can list
every parameter that was actually used. Unknown keys are rejected so typos in
a config file fail loudly instead of being silently ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from cellscribe.errors import ConfigError

InputFormat = Literal["auto", "10x", "h5ad", "csv", "visium", "xenium"]
Species = Literal["auto", "human", "mouse"]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class InputConfig(_Section):
    format: InputFormat = "auto"
    species: Species = Field(
        "auto",
        description="Gene naming convention; 'auto' infers it from gene symbol casing.",
    )
    sample_name: str | None = Field(None, description="Display name; defaults to the input file/folder name.")
    csv_orientation: Literal["auto", "genes_x_cells", "cells_x_genes"] = "auto"
    batch_key: str | None = Field(
        None,
        description=(
            "obs column identifying samples/batches. Auto-detected from common names "
            "(sample, batch, ...) when unset; used for per-batch QC thresholds and optional integration."
        ),
    )


class QCConfig(_Section):
    mode: Literal["flag", "filter"] = Field(
        "flag",
        description=(
            "'flag' keeps outlier cells in the analysis and labels them, so low-quality "
            "clusters stay visible in the per-cluster QC; 'filter' removes them before clustering."
        ),
    )
    nmads: float = Field(5.0, gt=0, description="MADs from the median for library size, genes and complexity.")
    nmads_mt: float = Field(3.0, gt=0, description="MADs above the median for mitochondrial fraction.")
    max_pct_mt: float | None = Field(
        None, ge=0, le=100, description="Optional hard ceiling on mitochondrial %, applied in addition to MADs."
    )
    min_genes: int | None = Field(
        None, ge=0, description="Hard floor on detected genes per cell; None picks 200 (whole transcriptome) or 10 (targeted panel)."
    )
    min_cells_per_gene: int = Field(3, ge=0)
    doublets: Literal["auto", "on", "off"] = Field(
        "auto", description="'auto' runs Scrublet for dissociated single-cell data and skips it for spatial data."
    )
    expected_doublet_rate: float = Field(0.06, gt=0, lt=1)
    cluster_flag_fraction: float = Field(
        0.5, gt=0, le=1, description="Flag a cluster when at least this fraction of its cells are QC outliers."
    )
    cluster_doublet_fraction: float = Field(
        0.3, gt=0, le=1, description="Flag a cluster when at least this fraction of its cells are predicted doublets."
    )
    cluster_nmads: float = Field(
        3.0, gt=0, description="Flag a cluster whose median QC metric is this many (cell-level) MADs from the dataset median."
    )


class PreprocessingConfig(_Section):
    target_sum: float = Field(1e4, gt=0)
    n_top_genes: int = Field(2000, gt=0)
    n_pcs: int = Field(30, gt=1, description="Principal components used to build the neighbour graph.")
    integrate: Literal["none", "harmony"] = Field(
        "none", description="Opt-in batch integration. Harmony is only applied if explicitly requested."
    )


class ClusteringConfig(_Section):
    resolution: float | None = Field(None, gt=0, description="Fixed Leiden resolution; None runs the automatic sweep.")
    resolution_grid: list[float] = Field(default_factory=lambda: [0.4, 0.6, 0.8, 1.0, 1.2])
    n_neighbors: int = Field(15, gt=1)
    stability_runs: int = Field(3, ge=2, description="Leiden runs with different seeds per resolution for stability.")
    silhouette_sample_size: int = Field(4000, gt=100)

    @field_validator("resolution_grid")
    @classmethod
    def _grid_positive(cls, v: list[float]) -> list[float]:
        if not v or any(r <= 0 for r in v):
            raise ValueError("resolution_grid must contain positive values")
        return sorted(set(v))


class AnnotationConfig(_Section):
    marker_file: Path | None = Field(
        None, description="Custom marker reference (YAML or CSV); defaults to the bundled canonical set."
    )
    n_markers_display: int = Field(10, gt=0)
    min_log2fc: float = Field(1.0, description="Minimum log2 fold change for a gene to count as a cluster marker.")
    max_padj: float = Field(0.05, gt=0, le=1)
    min_pct: float = Field(0.1, ge=0, le=1, description="Minimum fraction of cluster cells expressing a marker.")
    min_markers_present: int = Field(2, ge=1, description="Reference cell types need this many markers in the dataset to be scored.")


class SpatialConfig(_Section):
    enabled: Literal["auto", "on", "off"] = "auto"
    n_perms: int = Field(1000, ge=10, description="Permutations for neighbourhood enrichment.")
    n_genes_autocorr: int = Field(200, gt=0, description="Top highly variable genes tested for Moran's I.")


class NarrativeConfig(_Section):
    enabled: bool = Field(
        True, description="Generate an AI narrative when ANTHROPIC_API_KEY is set. Set False to never contact the API."
    )
    model: str = Field("claude-opus-5", description="Claude model used for the optional narrative.")
    max_tokens: int = Field(16000, gt=0, description="Output ceiling (includes the model's thinking).")
    timeout_seconds: float = Field(300.0, gt=0)


class ReportConfig(_Section):
    title: str | None = None
    plotlyjs: Literal["inline", "cdn"] = Field(
        "inline", description="'inline' makes the report fully self-contained (~5 MB larger)."
    )
    max_points: int = Field(50_000, gt=100, description="Cells drawn in interactive scatter plots (random subsample above this).")


class CellscribeConfig(_Section):
    seed: int = 0
    input: InputConfig = Field(default_factory=InputConfig)
    qc: QCConfig = Field(default_factory=QCConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    annotation: AnnotationConfig = Field(default_factory=AnnotationConfig)
    spatial: SpatialConfig = Field(default_factory=SpatialConfig)
    narrative: NarrativeConfig = Field(default_factory=NarrativeConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> CellscribeConfig:
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"Config file not found: {path}")
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"Could not parse YAML config {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"Config file {path} must contain a mapping at the top level.")
        return cls.from_dict(data, source=str(path))

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: str = "config") -> CellscribeConfig:
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            lines = []
            for err in exc.errors():
                loc = ".".join(str(p) for p in err["loc"])
                lines.append(f"  - {loc}: {err['msg']}")
            raise ConfigError(f"Invalid {source}:\n" + "\n".join(lines)) from exc

    def with_overrides(self, overrides: dict[str, Any]) -> CellscribeConfig:
        """Return a copy with dotted-key overrides applied (e.g. ``{"clustering.resolution": 0.8}``).

        ``None`` values are ignored so CLI options that were not given leave the
        config-file value in place.
        """
        data = self.model_dump()
        for dotted, value in overrides.items():
            if value is None:
                continue
            node = data
            *parents, leaf = dotted.split(".")
            for key in parents:
                node = node[key]
            node[leaf] = value
        return CellscribeConfig.from_dict(data, source="options")

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)

    def flat(self) -> dict[str, Any]:
        """Flatten to ``{"section.key": value}`` for the report's parameter table."""
        out: dict[str, Any] = {}

        def walk(prefix: str, obj: Any) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    walk(f"{prefix}.{k}" if prefix else k, v)
            else:
                out[prefix] = obj

        walk("", self.model_dump(mode="json"))
        return out
