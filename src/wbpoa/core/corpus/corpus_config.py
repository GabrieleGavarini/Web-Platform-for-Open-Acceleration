import re
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

_KEY_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_CELEX_PATTERN = re.compile(r"[0-9]{4,}[A-Z][A-Z0-9]*")


class InstrumentRole(StrEnum):
    """The role an instrument has within its legal-family corpus."""

    BASE = "base"
    AMENDING = "amending"
    IMPLEMENTING = "implementing"
    SUPPLEMENTING = "supplementing"
    REPEALING = "repealing"
    REPLACING = "replacing"


class InstrumentStatus(StrEnum):
    """The current legal status used by corpus retrieval."""

    ACTIVE = "active"
    INACTIVE = "inactive"


class InstrumentRelationType(StrEnum):
    """A directed legal relationship from one instrument to another."""

    AMENDS = "amends"
    IMPLEMENTS = "implements"
    SUPPLEMENTS = "supplements"
    REPEALS = "repeals"
    REPLACES = "replaces"


class InstrumentConfig(BaseModel):
    """One authoritative legal instrument declared by a corpus manifest.

    Attributes:
        key: Stable lowercase identifier unique within the corpus.
        name: Human-readable official or conventional instrument name.
        filename: Exact filename expected beneath the corpus ``source`` directory.
        celex_id: CELEX identifier used to construct the official source URL.
        role: The instrument's retrieval category within the corpus.
        aliases: Additional names accepted when resolving qualified citations.
        status: Current legal status used by default retrieval.
        effective_from: Optional first date on which the instrument has legal effect.
        effective_to: Optional final date on which the instrument has legal effect.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(min_length=1, description="Stable instrument key unique within the corpus.")
    name: str = Field(min_length=1, description="Human-readable instrument name.")
    filename: Path = Field(description="Filename expected beneath the corpus source directory.")
    celex_id: str = Field(min_length=1, description="CELEX identifier of the official source.")
    role: InstrumentRole
    aliases: list[str] = Field(default_factory=list, description="Alternative names used in legal citations.")
    status: InstrumentStatus = InstrumentStatus.ACTIVE
    effective_from: date | None = None
    effective_to: date | None = None

    @property
    def url(self) -> HttpUrl:
        """Return the official Publications Office resource URL for the source.

        Returns:
            The derived HTTP(S) source URL.
        """
        return HttpUrl(f"https://publications.europa.eu/resource/celex/{self.celex_id}")

    @field_validator("key")
    @classmethod
    def _validate_key(cls, key: str) -> str:
        if not _KEY_PATTERN.fullmatch(key):
            message = "instrument.key must contain lowercase letters and digits separated by single hyphens"
            raise ValueError(message)
        return key

    @field_validator("name")
    @classmethod
    def _validate_name(cls, name: str) -> str:
        stripped = name.strip()
        if not stripped:
            message = "instrument.name must contain non-whitespace characters"
            raise ValueError(message)
        return stripped

    @field_validator("aliases")
    @classmethod
    def _validate_aliases(cls, aliases: list[str]) -> list[str]:
        stripped = [alias.strip() for alias in aliases]
        if any(not alias for alias in stripped):
            message = "instrument.aliases entries must contain non-whitespace characters"
            raise ValueError(message)
        if len(set(stripped)) != len(stripped):
            message = "instrument.aliases entries must be unique"
            raise ValueError(message)
        return stripped

    @field_validator("filename")
    @classmethod
    def _validate_filename(cls, filename: Path) -> Path:
        if filename.is_absolute() or len(filename.parts) != 1 or filename.name in {"", ".", ".."}:
            message = "source.filename must be a filename, not an absolute or nested path"
            raise ValueError(message)
        return filename

    @field_validator("celex_id")
    @classmethod
    def _validate_celex_id(cls, celex_id: str) -> str:
        if not _CELEX_PATTERN.fullmatch(celex_id):
            message = "instrument.celex_id must be a CELEX identifier"
            raise ValueError(message)
        return celex_id

    @model_validator(mode="after")
    def _validate_effective_dates(self) -> Self:
        if (
            self.effective_from is not None
            and self.effective_to is not None
            and self.effective_to < self.effective_from
        ):
            message = "instrument.effective_to must not precede effective_from"
            raise ValueError(message)
        return self


class InstrumentRelationConfig(BaseModel):
    """A directed legal relationship declared between two corpus instruments.

    Attributes:
        source: Key of the instrument creating the legal effect.
        target: Key of the instrument affected by that effect.
        type: Legal semantics of the directed relationship.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    type: InstrumentRelationType

    @field_validator("source", "target")
    @classmethod
    def _validate_endpoint(cls, endpoint: str) -> str:
        if not _KEY_PATTERN.fullmatch(endpoint):
            message = "instrument relation endpoints must be valid instrument keys"
            raise ValueError(message)
        return endpoint


class ZoteroSourceConfig(BaseModel):
    """One or more Zotero collections in a single group supplying documents for a corpus.

    Attributes:
        group_id: Numeric Zotero group identifier shared by every listed collection.
        collection_keys: Eight-character Zotero collection keys to synchronize.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    group_id: int = Field(gt=0, description="Numeric Zotero group identifier.")
    collection_keys: list[str] = Field(
        min_length=1,
        description="Eight-character Zotero collection keys to synchronize.",
    )

    @field_validator("collection_keys")
    @classmethod
    def _validate_collection_keys(cls, collection_keys: list[str]) -> list[str]:
        pattern = re.compile(r"^[A-Z0-9]{8}$")
        for collection_key in collection_keys:
            if not pattern.fullmatch(collection_key):
                message = f"collection_keys entries must be 8 uppercase letters or digits, got {collection_key!r}"
                raise ValueError(message)
        return collection_keys


class CorpusConfig(BaseModel):
    """Validated contents of a per-corpus ``corpus.toml`` manifest.

    Attributes:
        name: Human-readable legal-family name used in prompts and citations.
        instruments: Authoritative legal instruments available within the corpus.
        instrument_relations: Directed legal relationships between instruments.
        guidelines: Zotero collections supplying corpus-local guidelines.
        rulings: Zotero collections supplying corpus-local rulings, or ``None`` when the corpus has no
            separate ruling collection. A collection may feed guidelines or rulings, never both, so no
            document is ingested twice.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, description="Human-readable legal-family name.")
    instruments: list[InstrumentConfig] = Field(min_length=1)
    instrument_relations: list[InstrumentRelationConfig] = Field(default_factory=list)
    guidelines: ZoteroSourceConfig
    rulings: ZoteroSourceConfig | None = None

    @field_validator("name")
    @classmethod
    def _validate_name(cls, name: str) -> str:
        stripped = name.strip()
        if not stripped:
            message = "name must contain non-whitespace characters"
            raise ValueError(message)
        return stripped

    @model_validator(mode="after")
    def _validate_instruments(self) -> Self:
        keys = [instrument.key for instrument in self.instruments]
        if len(set(keys)) != len(keys):
            message = "instrument keys must be unique within a corpus"
            raise ValueError(message)

        celex_ids = [instrument.celex_id for instrument in self.instruments]
        if len(set(celex_ids)) != len(celex_ids):
            message = "instrument CELEX identifiers must be unique within a corpus"
            raise ValueError(message)

        base_count = sum(instrument.role is InstrumentRole.BASE for instrument in self.instruments)
        if base_count != 1:
            message = "a corpus must declare exactly one base instrument"
            raise ValueError(message)

        known_keys = set(keys)
        relation_keys: set[tuple[str, str, InstrumentRelationType]] = set()
        for relation in self.instrument_relations:
            if relation.source not in known_keys or relation.target not in known_keys:
                message = "instrument relation endpoints must reference instruments in the same corpus"
                raise ValueError(message)
            if relation.source == relation.target:
                message = "an instrument cannot have a legal relationship with itself"
                raise ValueError(message)
            relation_key = (relation.source, relation.target, relation.type)
            if relation_key in relation_keys:
                message = "instrument relations must be unique within a corpus"
                raise ValueError(message)
            relation_keys.add(relation_key)

        if self.rulings is not None and self.rulings.group_id == self.guidelines.group_id:
            shared = sorted(set(self.guidelines.collection_keys) & set(self.rulings.collection_keys))
            if shared:
                message = f"Zotero collections cannot supply both guidelines and rulings, got {shared}"
                raise ValueError(message)

        return self
