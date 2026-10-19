from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from wbpoa.core.corpus.corpus_registry import CorpusRegistry, CorpusRegistryError, CorpusWorkspace
from wbpoa.multi_corpora.corpus.multi_corpora_config import MultiCorporaConfig

MANIFEST_FILENAME: Final = "multi_corpora.toml"

_MULTI_CORPORA_KEY_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class MultiCorporaRegistryError(CorpusRegistryError):
    """Raised when a multi-corpora key, manifest, or one of its sub-corpora cannot be loaded safely."""


@dataclass(frozen=True, slots=True)
class MultiCorporaWorkspace:
    """A validated multi-corpora manifest together with its two loaded sub-corpora.

    Attributes:
        key: Canonical lowercase multi-corpora selector.
        config: Parsed multi-corpora manifest.
        root: Directory owning the manifest.
        corpora: The two sub-corpus workspaces, in manifest order.
    """

    key: str
    config: MultiCorporaConfig
    root: Path
    corpora: tuple[CorpusWorkspace, CorpusWorkspace]

    @property
    def instrument_keys(self) -> list[str]:
        """Return every instrument key of both sub-corpora, in manifest order.

        Returns:
            The first sub-corpus's instrument keys followed by the second's.
        """
        return [instrument.key for corpus in self.corpora for instrument in corpus.config.instruments]


class MultiCorporaRegistry:
    """Discover and load multi-corpora manifests, resolving their sub-corpora through a corpus registry.

    Attributes:
        multi_corpora_dir: Root containing one ``<key>/multi_corpora.toml`` directory per pairing.
        corpus_registry: Registry the sub-corpora keys are resolved against.
    """

    def __init__(self, multi_corpora_dir: Path, corpus_registry: CorpusRegistry) -> None:
        """Initialize a multi-corpora registry.

        Args:
            multi_corpora_dir: Root containing multi-corpora workspace directories.
            corpus_registry: Registry loading each named sub-corpus.
        """
        self.multi_corpora_dir = multi_corpora_dir.resolve()
        self.corpus_registry = corpus_registry

    def discover(self) -> list[MultiCorporaWorkspace]:
        """Load every multi-corpora manifest in deterministic key order.

        Returns:
            Validated multi-corpora workspaces sorted by key.

        Raises:
            MultiCorporaRegistryError: If the registry is missing or any discovered manifest is invalid.
        """
        if not self.multi_corpora_dir.is_dir():
            message = f"Multi-corpora registry directory does not exist: {self.multi_corpora_dir}"
            raise MultiCorporaRegistryError(message)

        keys = sorted(path.parent.name for path in self.multi_corpora_dir.glob(f"*/{MANIFEST_FILENAME}"))
        return [self.load(key) for key in keys]

    def load(self, key: str) -> MultiCorporaWorkspace:
        """Load one multi-corpora manifest and both of its sub-corpora.

        Args:
            key: Lowercase slug naming a direct child of the registry directory.

        Returns:
            The validated manifest with its two loaded sub-corpora.

        Raises:
            MultiCorporaRegistryError: If the key, manifest, or either sub-corpus is invalid, or if the
                two sub-corpora share an instrument key.
        """
        if not _MULTI_CORPORA_KEY_PATTERN.fullmatch(key):
            message = (
                f"Invalid multi-corpora key {key!r}; expected lowercase letters and digits separated by single hyphens."
            )
            raise MultiCorporaRegistryError(message)

        root = (self.multi_corpora_dir / key).resolve()
        try:
            root.relative_to(self.multi_corpora_dir)
        except ValueError as error:
            message = f"Multi-corpora workspace escapes its owner directory: {root}"
            raise MultiCorporaRegistryError(message) from error

        manifest_path = root / MANIFEST_FILENAME
        if not manifest_path.is_file():
            message = f"Multi-corpora manifest does not exist: {manifest_path}"
            raise MultiCorporaRegistryError(message)

        try:
            config = MultiCorporaConfig.model_validate(tomllib.loads(manifest_path.read_text(encoding="utf-8")))
        except (OSError, tomllib.TOMLDecodeError, ValidationError) as error:
            message = f"Invalid multi-corpora manifest {manifest_path}: {error}"
            raise MultiCorporaRegistryError(message) from error

        first_key, second_key = config.corpora
        workspace = MultiCorporaWorkspace(
            key=key,
            config=config,
            root=root,
            corpora=(self._load_sub_corpus(key, first_key), self._load_sub_corpus(key, second_key)),
        )
        self._validate_instruments(workspace)
        return workspace

    def _load_sub_corpus(self, key: str, corpus_key: str) -> CorpusWorkspace:
        """Load one named sub-corpus, attributing any failure to the multi-corpora manifest."""
        try:
            return self.corpus_registry.load(corpus_key)
        except CorpusRegistryError as error:
            message = f"Multi-corpora {key!r} names sub-corpus {corpus_key!r}, which cannot be loaded: {error}"
            raise MultiCorporaRegistryError(message) from error

    @staticmethod
    def _validate_instruments(workspace: MultiCorporaWorkspace) -> None:
        """
        Reject pairings whose instrument keys collide.

        Instrument keys identify a provision's instrument in every `QualifiedCitation`, so the two
        sub-corpora must not share one, or the union of their cited provisions would be ambiguous.

        Args:
            workspace: The freshly loaded workspace to check.

        Raises:
            MultiCorporaRegistryError: On a shared instrument key.
        """
        first, second = workspace.corpora
        shared = set(first.instruments_by_key) & set(second.instruments_by_key)
        if shared:
            message = (
                f"Multi-corpora {workspace.key!r} sub-corpora {first.key!r} and {second.key!r} share "
                f"instrument keys: {sorted(shared)}"
            )
            raise MultiCorporaRegistryError(message)
