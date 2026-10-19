from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

SUB_CORPORA_COUNT: Final = 2


class MultiCorporaConfig(BaseModel):
    """Validated contents of a ``multi_corpora.toml`` manifest pairing exactly two corpora.

    The manifest owns no legal material: its two sub-corpora are ordinary ``data/corpora/<key>``
    corpora, ingested and stored exactly as for single-corpus use. It only names the pair.

    Attributes:
        name: Human-readable name of the pairing, used in prompts and run metadata.
        corpora: Keys of the two sub-corpora, in manifest order.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, description="Human-readable name of the corpora pairing.")
    corpora: list[str] = Field(description="Keys of exactly two distinct sub-corpora.")

    @field_validator("name")
    @classmethod
    def _validate_name(cls, name: str) -> str:
        stripped = name.strip()
        if not stripped:
            message = "name must contain non-whitespace characters"
            raise ValueError(message)
        return stripped

    @field_validator("corpora")
    @classmethod
    def _validate_corpora(cls, corpora: list[str]) -> list[str]:
        keys = [key.strip() for key in corpora]
        if len(keys) != SUB_CORPORA_COUNT:
            message = f"a multi-corpora manifest must name exactly {SUB_CORPORA_COUNT} corpora, got {len(keys)}"
            raise ValueError(message)
        if len(set(keys)) != len(keys):
            message = "multi-corpora sub-corpora must be distinct"
            raise ValueError(message)
        return keys
