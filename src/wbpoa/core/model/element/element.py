from __future__ import annotations

from abc import ABC
from typing import ClassVar
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Element(BaseModel, ABC):
    """Any citable unit of text."""

    citation_weight: ClassVar[float] = 1.0

    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for the element.",
    )
    name: str = Field(
        ...,
        description="Label as it appears in the source, including its type, e.g. 'Article 5', 'Annex III'.",
    )
    text: str = Field(..., description="Verbatim text of the element.")

    provision_references: set[UUID] = Field(
        default_factory=set,
        description="Set of element UUIDs referenced by this element.",
    )

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Element) and self.id == other.id
