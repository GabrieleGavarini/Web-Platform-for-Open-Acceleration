from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import Field, computed_field

from wbpoa.core.model.element.element import Element
from wbpoa.core.model.qualified_citation import QualifiedCitation


class Provision(Element, ABC):
    """A top-level citable unit belonging to one legal instrument.

    Attributes:
        corpus_id: Legal-family corpus owning this provision copy.
        instrument_key: Stable key of the provision's source instrument.
        celex_id: Official CELEX identifier of the source instrument.
        provision_path: Canonical top-level path within the instrument.
    """

    citation_weight: ClassVar[float] = 1.0

    corpus_id: str = Field(min_length=1)
    instrument_key: str = Field(min_length=1)
    celex_id: str = Field(min_length=1)
    provision_path: str = Field(min_length=1)

    @computed_field
    @property
    def qualified_citation(self) -> QualifiedCitation:
        """Return the provision's stable instrument-qualified identity."""
        return QualifiedCitation(
            instrument_key=self.instrument_key,
            celex_id=self.celex_id,
            provision_path=self.provision_path,
        )

    @computed_field
    @property
    @abstractmethod
    def to_citation_names(self) -> set[str]:
        """Return accepted textual citation names for this provision."""
        raise NotImplementedError
