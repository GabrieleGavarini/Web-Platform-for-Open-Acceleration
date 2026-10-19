from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from wbpoa.core.corpus.corpus_config import InstrumentRelationConfig, InstrumentRole, InstrumentStatus

if TYPE_CHECKING:
    from wbpoa.core.corpus.corpus_registry import CorpusWorkspace


@dataclass(frozen=True, slots=True)
class CorpusContext:
    """Runtime identity used by prompts and tracing.

    Attributes:
        key: Canonical corpus selector.
        name: Human-readable legislation name.
        instrument_names: Instrument key to human-readable instrument name, for per-provision
            citation formatting in generation.
        instrument_roles: Instrument key to retrieval role, for role-aware generation guidance
            (e.g. preferring amending text, treating implementing regulations as independent).
        instrument_statuses: Instrument key to current legal status, for the run
            metadata's corpus snapshot.
        instrument_celex_ids: Instrument key to CELEX identifier, for the run
            metadata's corpus snapshot.
        instrument_relations: Directed legal relationships declared between corpus instruments,
            for the run metadata's corpus snapshot.
    """

    key: str
    name: str
    instrument_names: dict[str, str] = field(default_factory=dict)
    instrument_roles: dict[str, InstrumentRole] = field(default_factory=dict)
    instrument_statuses: dict[str, InstrumentStatus] = field(default_factory=dict)
    instrument_celex_ids: dict[str, str] = field(default_factory=dict)
    instrument_relations: tuple[InstrumentRelationConfig, ...] = field(default_factory=tuple)

    @classmethod
    def from_workspace(cls, workspace: CorpusWorkspace) -> CorpusContext:
        """Build the runtime identity of a loaded corpus from its manifest.

        Args:
            workspace: Validated corpus workspace whose manifest supplies the instruments.

        Returns:
            A context keyed by the workspace key, carrying every instrument's name, role,
            status, CELEX identifier, and the manifest's declared relations.
        """
        instruments = workspace.config.instruments
        return cls(
            key=workspace.key,
            name=workspace.config.name,
            instrument_names={instrument.key: instrument.name for instrument in instruments},
            instrument_roles={instrument.key: instrument.role for instrument in instruments},
            instrument_statuses={instrument.key: instrument.status for instrument in instruments},
            instrument_celex_ids={instrument.key: instrument.celex_id for instrument in instruments},
            instrument_relations=tuple(workspace.config.instrument_relations),
        )
