from datetime import date

from pydantic import BaseModel, Field

from wbpoa.core.corpus.corpus_config import InstrumentRole, InstrumentStatus
from wbpoa.core.model.element.corpus.types import ValidProvision


class Instrument(BaseModel):
    """One independently authoritative legal act within a corpus.

    Attributes:
        key: Stable instrument key unique within the corpus.
        name: Human-readable official or conventional instrument name.
        celex_id: Official CELEX identifier.
        role: Instrument category used for retrieval filtering.
        aliases: Alternative names accepted in qualified references.
        status: Current legal status.
        effective_from: Optional first date of legal effect.
        effective_to: Optional final date of legal effect.
        provisions: Top-level citable provisions extracted from the instrument.
    """

    key: str = Field(min_length=1)
    name: str = Field(min_length=1)
    celex_id: str = Field(min_length=1)
    role: InstrumentRole
    aliases: list[str] = Field(default_factory=list)
    status: InstrumentStatus
    effective_from: date | None = None
    effective_to: date | None = None
    provisions: list[ValidProvision] = Field(default_factory=list)
