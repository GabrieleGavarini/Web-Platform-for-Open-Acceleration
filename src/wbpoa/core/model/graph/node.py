from datetime import date
from uuid import UUID

from pydantic import BaseModel, Field, computed_field

from wbpoa.core.corpus.corpus_config import InstrumentRole, InstrumentStatus
from wbpoa.core.model.qualified_citation import QualifiedCitation


class Node(BaseModel):
    corpus_id: str = Field(
        ...,
        description="The corpus partition owning this node.",
    )
    instrument_key: str | None = Field(
        default=None,
        description="Stable source instrument key for provision and instrument nodes.",
    )
    celex_id: str | None = Field(
        default=None,
        description="Official source instrument CELEX identifier for provision and instrument nodes.",
    )
    provision_path: str | None = Field(
        default=None,
        description="Canonical top-level provision path for provision nodes.",
    )
    instrument_role: InstrumentRole | None = Field(
        default=None,
        description="Retrieval role of the owning instrument, for provision and instrument nodes.",
    )
    instrument_status: InstrumentStatus | None = Field(
        default=None,
        description="Legal status of the owning instrument, for provision and instrument nodes.",
    )
    effective_from: date | None = Field(
        default=None,
        description="First date of legal effect of the owning instrument, for provision and instrument nodes.",
    )
    effective_to: date | None = Field(
        default=None,
        description="Final date of legal effect of the owning instrument, for provision and instrument nodes.",
    )
    member_instrument_keys: list[str] = Field(
        default_factory=list,
        description="Distinct instrument keys among a community node's descendant provisions.",
    )
    member_instrument_roles: list[InstrumentRole] = Field(
        default_factory=list,
        description="Distinct instrument roles among a community node's descendant provisions.",
    )
    id: UUID = Field(
        ...,
        description="The unique identifier of the node.",
    )
    name: str = Field(
        ...,
        description="The name of the node.",
    )
    type: str = Field(
        ...,
        description="The type of the node.",
    )
    text: str = Field(
        ...,
        description="The text content of the node.",
    )
    summary: str = Field(
        ...,
        description="A summary of the node's text content.",
    )
    summary_embedding: list[float] = Field(
        ...,
        description="The embedding vector of the node's summary.",
    )
    layer: int = Field(
        0,
        description="The layer of the node in the graph. 0 is for elements, h >= 1 is for the communities of "
        "layer h, -1 is for instrument nodes (excluded from vector search and community detection).",
    )

    @computed_field
    @property
    def qualified_citation(self) -> QualifiedCitation | None:
        """Return the node's instrument-qualified identity, or None when incomplete (guidelines/rulings, communities)."""
        if self.instrument_key is None or self.celex_id is None or self.provision_path is None:
            return None
        return QualifiedCitation(
            instrument_key=self.instrument_key, celex_id=self.celex_id, provision_path=self.provision_path
        )

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Node) and self.id == other.id
