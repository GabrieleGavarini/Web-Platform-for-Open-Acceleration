from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from wbpoa.core.corpus import CorpusWorkspace


class ChatTargetKind(StrEnum):
    """Which agent answers a chat target: one corpus, or a pairing of two."""

    SINGLE = "single"
    MULTI = "multi"


@dataclass(frozen=True, slots=True)
class ChatTarget:
    """
    Something the chat can be asked about: a single corpus or a multi-corpora pairing.

    Attributes:
        kind: Whether the single-corpus or the multi-corpus agent answers.
        key: The corpus key (single) or the pairing key (multi).
        name: Human-readable name from the manifest, e.g. ``GDPR + NIS2``.
        corpora: The corpus workspaces answered from: one for a single corpus, two for a pairing.
        examples: Example questions for the target, possibly empty.
    """

    kind: ChatTargetKind
    key: str
    name: str
    corpora: tuple[CorpusWorkspace, ...]
    examples: tuple[str, ...] = field(default=())

    @property
    def id(self) -> str:
        """Stable identifier of the target across kinds, e.g. ``multi:gdpr-nis2``."""
        return f"{self.kind}:{self.key}"

    @property
    def instrument_names(self) -> list[str]:
        """Names of every instrument answered from, in manifest order."""
        return [instrument.name for corpus in self.corpora for instrument in corpus.config.instruments]
