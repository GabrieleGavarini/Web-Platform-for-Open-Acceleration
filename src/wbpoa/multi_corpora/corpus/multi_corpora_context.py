from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from wbpoa.core.corpus.context import CorpusContext
from wbpoa.core.corpus.corpus_config import InstrumentRelationConfig, InstrumentRole, InstrumentStatus

if TYPE_CHECKING:
    from wbpoa.multi_corpora.corpus.multi_corpora_registry import MultiCorporaWorkspace


@dataclass(frozen=True, slots=True)
class MultiCorporaContext:
    """Runtime identity of a multi-corpora pairing: its own name plus each sub-corpus's `CorpusContext`.

    Exposes the same instrument maps as `CorpusContext`, merged across both sub-corpora, so that
    code reading a context's instruments (e.g. run metadata) works unchanged on a
    pairing. Instrument keys never collide across the two, which `MultiCorporaRegistry` enforces.

    Attributes:
        key: Canonical multi-corpora selector.
        name: Human-readable name of the pairing.
        sub_contexts: The two sub-corpus contexts, in manifest order.
    """

    key: str
    name: str
    sub_contexts: tuple[CorpusContext, CorpusContext]

    @classmethod
    def from_workspace(cls, workspace: MultiCorporaWorkspace) -> MultiCorporaContext:
        """Build the runtime identity of a loaded multi-corpora pairing.

        Args:
            workspace: Validated multi-corpora workspace.

        Returns:
            A context keyed by the pairing key, with one `CorpusContext` per sub-corpus.
        """
        first, second = workspace.corpora
        return cls(
            key=workspace.key,
            name=workspace.config.name,
            sub_contexts=(CorpusContext.from_workspace(first), CorpusContext.from_workspace(second)),
        )

    @property
    def corpus_keys(self) -> tuple[str, str]:
        """Keys of the two sub-corpora, in manifest order."""
        first, second = self.sub_contexts
        return first.key, second.key

    @property
    def instrument_names(self) -> dict[str, str]:
        """Instrument key to human-readable name, across both sub-corpora."""
        return {key: name for context in self.sub_contexts for key, name in context.instrument_names.items()}

    @property
    def instrument_roles(self) -> dict[str, InstrumentRole]:
        """Instrument key to retrieval role, across both sub-corpora."""
        return {key: role for context in self.sub_contexts for key, role in context.instrument_roles.items()}

    @property
    def instrument_statuses(self) -> dict[str, InstrumentStatus]:
        """Instrument key to legal status, across both sub-corpora."""
        return {key: status for context in self.sub_contexts for key, status in context.instrument_statuses.items()}

    @property
    def instrument_celex_ids(self) -> dict[str, str]:
        """Instrument key to CELEX identifier, across both sub-corpora."""
        return {key: celex for context in self.sub_contexts for key, celex in context.instrument_celex_ids.items()}

    @property
    def instrument_relations(self) -> tuple[InstrumentRelationConfig, ...]:
        """Declared instrument relations of both sub-corpora; relations never cross the pairing."""
        return tuple(relation for context in self.sub_contexts for relation in context.instrument_relations)
