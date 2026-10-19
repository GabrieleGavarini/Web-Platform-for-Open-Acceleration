from pydantic import BaseModel, ConfigDict, Field


class QualifiedCitation(BaseModel):
    """Stable identity of a top-level provision within a legal instrument.

    Attributes:
        instrument_key: Stable instrument identifier within its corpus.
        celex_id: Official CELEX identifier of the instrument.
        provision_path: Canonical top-level provision path, such as ``Article 5``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    instrument_key: str = Field(min_length=1)
    celex_id: str = Field(min_length=1)
    provision_path: str = Field(min_length=1)
