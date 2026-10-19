from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from wbpoa.core.corpus.corpus_config import InstrumentRelationType
from wbpoa.core.model.operative_relation import OperativeOperation


class Edge(BaseModel):
    id: UUID = Field(
        default_factory=uuid4,
        description="The unique identifier of the edge.",
    )
    weight: float = Field(
        1.0,
        description="The weight of the edge, representing the strength or importance of the connection.",
    )
    type: str = Field(
        ...,
        description="The type of the edge, indicating the nature of the relationship between nodes.",
    )
    weight_contributions: dict[UUID, float] = Field(
        default_factory=dict,
        description="Maps the id of each element that contributed to this edge, to the weight it contributed. "
        "Left empty by graph builders that do not track this.",
    )
    relation_type: InstrumentRelationType | None = Field(
        default=None,
        description="Declared legal relationship type this edge derives from, for instrument-level and "
        "operative provision/affinity edges.",
    )
    operation: OperativeOperation | None = Field(
        default=None,
        description="Textual effect of an OPERATIVE_RELATION edge.",
    )

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Edge) and self.id == other.id
