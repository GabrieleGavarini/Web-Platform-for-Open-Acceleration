from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from wbpoa.core.corpus.corpus_config import (
    CorpusConfig,
    InstrumentConfig,
    InstrumentRelationConfig,
    InstrumentRelationType,
)

DEFAULT_CORPORA_DIR = Path("data/corpora")

_CORPUS_KEY_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class CorpusRegistryError(ValueError):
    """Raised when a corpus key or manifest cannot be loaded safely."""


@dataclass(frozen=True, slots=True)
class CorpusWorkspace:
    """A validated manifest together with its corpus-owned paths.

    Attributes:
        key: Canonical lowercase corpus selector.
        config: Parsed corpus manifest.
        root: Root directory owned by the corpus.
        instrument_source_paths: Local source paths keyed by instrument key.
        guidelines_dir: Directory containing corpus-local guideline PDFs.
        rulings_dir: Directory containing corpus-local ruling PDFs.
        markdown_guidelines_dir: Derived Markdown directory for guidelines.
        markdown_rulings_dir: Derived Markdown directory for rulings.
    """

    key: str
    config: CorpusConfig
    root: Path
    instrument_source_paths: dict[str, Path]
    guidelines_dir: Path
    rulings_dir: Path
    markdown_guidelines_dir: Path
    markdown_rulings_dir: Path

    @property
    def instruments_by_key(self) -> dict[str, InstrumentConfig]:
        """Return configured instruments keyed by their stable identifiers.

        Returns:
            Instrument configuration lookup for this corpus.
        """
        return {instrument.key: instrument for instrument in self.config.instruments}

    def instrument_relations(
        self,
        *,
        source: str | None = None,
        target: str | None = None,
        relation_type: InstrumentRelationType | None = None,
    ) -> list[InstrumentRelationConfig]:
        """Return legal relationships matching the supplied endpoint and type filters.

        Args:
            source: Optional source instrument key.
            target: Optional target instrument key.
            relation_type: Optional directed legal relationship type.

        Returns:
            Matching relations in manifest order.
        """
        return [
            relation
            for relation in self.config.instrument_relations
            if (source is None or relation.source == source)
            and (target is None or relation.target == target)
            and (relation_type is None or relation.type is relation_type)
        ]


class CorpusRegistry:
    """Discover and load per-corpus manifests beneath one registry directory.

    Attributes:
        corpora_dir: Root containing one ``<key>/corpus.toml`` directory per corpus.
    """

    def __init__(self, corpora_dir: Path) -> None:
        """Initialize a corpus registry.

        Args:
            corpora_dir: Root containing corpus workspace directories.
        """
        self.corpora_dir = corpora_dir.resolve()

    def discover(self) -> list[CorpusWorkspace]:
        """Load every manifest in deterministic corpus-key order.

        Returns:
            Validated corpus workspaces sorted by key.

        Raises:
            CorpusRegistryError: If the registry is missing or any discovered manifest is invalid.
        """
        if not self.corpora_dir.is_dir():
            raise CorpusRegistryError(f"Corpus registry directory does not exist: {self.corpora_dir}")

        keys = sorted(path.parent.name for path in self.corpora_dir.glob("*/corpus.toml"))
        return [self.load(key) for key in keys]

    def load(self, key: str) -> CorpusWorkspace:
        """Load one corpus manifest by its canonical key.

        Args:
            key: Lowercase slug naming a direct child of the registry directory.

        Returns:
            The validated corpus configuration and resolved owned paths.

        Raises:
            CorpusRegistryError: If the key, workspace path, TOML, or manifest is invalid.
        """
        if not _CORPUS_KEY_PATTERN.fullmatch(key):
            raise CorpusRegistryError(
                f"Invalid corpus key {key!r}; expected lowercase letters and digits separated by single hyphens."
            )

        root = (self.corpora_dir / key).resolve()
        self._require_within(root, self.corpora_dir, label="Corpus workspace")
        manifest_path = root / "corpus.toml"
        if not manifest_path.is_file():
            raise CorpusRegistryError(f"Corpus manifest does not exist: {manifest_path}")

        try:
            manifest_data = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
            config = CorpusConfig.model_validate(manifest_data)
        except (OSError, tomllib.TOMLDecodeError, ValidationError) as error:
            raise CorpusRegistryError(f"Invalid corpus manifest {manifest_path}: {error}") from error

        instrument_source_paths: dict[str, Path] = {}
        for instrument in config.instruments:
            source_path = (root / "source" / instrument.filename).resolve()
            self._require_within(source_path, root, label=f"Instrument {instrument.key!r} source")
            instrument_source_paths[instrument.key] = source_path
        return CorpusWorkspace(
            key=key,
            config=config,
            root=root,
            instrument_source_paths=instrument_source_paths,
            guidelines_dir=root / "guidelines",
            rulings_dir=root / "rulings",
            markdown_guidelines_dir=root / "derived" / "markdown" / "guidelines",
            markdown_rulings_dir=root / "derived" / "markdown" / "rulings",
        )

    @staticmethod
    def _require_within(path: Path, parent: Path, *, label: str) -> None:
        """Reject a resolved path that escapes its expected owner directory."""
        try:
            path.relative_to(parent)
        except ValueError as error:
            raise CorpusRegistryError(f"{label} escapes its owner directory: {path}") from error
