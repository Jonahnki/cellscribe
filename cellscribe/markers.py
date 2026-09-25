"""Loading marker-gene references (bundled canonical set or user supplied)."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import pandas as pd
import yaml

from cellscribe.errors import ConfigError

BUNDLED_REFERENCE = "canonical_markers.yaml"


@dataclass(frozen=True)
class MarkerSet:
    name: str
    lineage: str
    genes: tuple[str, ...]
    kind: str = "cell_type"  # or "state"
    generic: bool = False


@dataclass
class MarkerReference:
    source: str
    version: str
    species: str
    sets: list[MarkerSet] = field(default_factory=list)

    @property
    def cell_types(self) -> list[MarkerSet]:
        return [s for s in self.sets if s.kind == "cell_type"]

    @property
    def states(self) -> list[MarkerSet]:
        return [s for s in self.sets if s.kind == "state"]

    def get(self, name: str) -> MarkerSet:
        for s in self.sets:
            if s.name == name:
                return s
        raise KeyError(name)

    def all_genes(self) -> list[str]:
        seen: dict[str, None] = {}
        for s in self.sets:
            for g in s.genes:
                seen.setdefault(g, None)
        return list(seen)


def _parse_yaml_entries(entries: list, species: str, source: str) -> list[MarkerSet]:
    sets: list[MarkerSet] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or "name" not in entry:
            raise ConfigError(f"{source}: entry {i} must be a mapping with a 'name' field.")
        genes = entry.get(species, entry.get("genes"))
        if genes is None:
            continue  # no genes for this species
        if not isinstance(genes, list):
            raise ConfigError(f"{source}: genes for '{entry['name']}' must be a list.")
        genes = tuple(dict.fromkeys(str(g).strip() for g in genes if str(g).strip()))
        if not genes:
            continue
        kind = str(entry.get("kind", "cell_type"))
        if kind not in {"cell_type", "state"}:
            raise ConfigError(f"{source}: '{entry['name']}' has unknown kind '{kind}'.")
        sets.append(
            MarkerSet(
                name=str(entry["name"]),
                lineage=str(entry.get("lineage", entry["name"])),
                genes=genes,
                kind=kind,
                generic=bool(entry.get("generic", False)),
            )
        )
    return sets


def _load_csv(path: Path, species: str) -> list[MarkerSet]:
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=sep)
    cols = {c.lower().strip(): c for c in df.columns}
    ct_col = cols.get("cell_type") or cols.get("celltype") or cols.get("cell type")
    gene_col = cols.get("gene") or cols.get("genes") or cols.get("marker") or cols.get("symbol")
    if ct_col is None or gene_col is None:
        raise ConfigError(
            f"Marker CSV {path} needs columns 'cell_type' and 'gene' (optionally 'species', 'lineage'). "
            f"Found: {list(df.columns)}"
        )
    sp_col = cols.get("species")
    if sp_col is not None:
        df = df[df[sp_col].astype(str).str.lower().isin([species, "both", "any", "all"])]
    lin_col = cols.get("lineage")
    sets = []
    for name, grp in df.groupby(ct_col, sort=False):
        genes = tuple(dict.fromkeys(grp[gene_col].astype(str).str.strip()))
        lineage = str(grp[lin_col].iloc[0]) if lin_col else str(name)
        sets.append(MarkerSet(name=str(name), lineage=lineage, genes=genes))
    return sets


def load_reference(species: str, path: str | Path | None = None) -> MarkerReference:
    """Load the marker reference for ``species`` ("human" or "mouse").

    ``path`` may point to a YAML file in the bundled schema or a CSV/TSV with
    ``cell_type`` and ``gene`` columns. With no path, the bundled canonical
    reference is used.
    """
    if species not in {"human", "mouse"}:
        raise ConfigError(f"species must be 'human' or 'mouse', got {species!r}")
    if path is None:
        text = resources.files("cellscribe.data.markers").joinpath(BUNDLED_REFERENCE).read_text()
        data = yaml.safe_load(text)
        sets = _parse_yaml_entries(data["cell_types"], species, BUNDLED_REFERENCE)
        return MarkerReference(
            source="Cellscribe canonical markers (bundled, hand-curated)",
            version=str(data.get("version", "unknown")),
            species=species,
            sets=sets,
        )

    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Marker file not found: {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text())
        entries = data.get("cell_types") if isinstance(data, dict) else data
        if not isinstance(entries, list):
            raise ConfigError(f"{path}: expected a list of entries under 'cell_types'.")
        sets = _parse_yaml_entries(entries, species, str(path))
        version = str(data.get("version", "custom")) if isinstance(data, dict) else "custom"
    else:
        sets = _load_csv(path, species)
        version = "custom"
    if not sets:
        raise ConfigError(f"Marker file {path} contains no usable entries for species '{species}'.")
    return MarkerReference(source=f"Custom: {path.name}", version=version, species=species, sets=sets)
