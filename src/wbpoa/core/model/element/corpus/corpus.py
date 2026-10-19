from pydantic import BaseModel, Field, computed_field

from wbpoa.core.corpus.corpus_config import InstrumentRole
from wbpoa.core.model.element.corpus.instrument import Instrument
from wbpoa.core.model.element.corpus.types import ValidProvision
from wbpoa.core.model.operative_relation import OperativeRelation


class Corpus(BaseModel):
    """A queryable legal family containing related authoritative instruments.

    Attributes:
        id: Canonical corpus key used as the graph partition.
        name: Human-readable legal-family name.
        instruments: Instruments available to corpus retrieval.
        operative_relations: Provision-level relations derived from declared
            instrument relations.
    """

    id: str = Field(
        ...,
        description="Unique identifier of the corpus.",
    )
    name: str = Field(
        ...,
        description="Human-readable legal-family name.",
    )
    instruments: list[Instrument] = Field(
        default_factory=list,
        description="Authoritative legal instruments contained in the corpus.",
    )
    operative_relations: list[OperativeRelation] = Field(
        default_factory=list,
        description="Provision-level relations derived from declared instrument relations.",
    )

    @computed_field
    @property
    def provisions(self) -> list[ValidProvision]:
        """Return all instrument provisions in manifest and document order."""
        return [provision for instrument in self.instruments for provision in instrument.provisions]

    @property
    def base_instrument(self) -> Instrument:
        """Return the corpus's sole base instrument.

        Safe by construction: `CorpusConfig` validation guarantees exactly
        one base instrument per corpus.

        Returns:
            The instrument whose `role` is `InstrumentRole.BASE`.
        """
        return next(instrument for instrument in self.instruments if instrument.role is InstrumentRole.BASE)
