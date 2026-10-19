from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from wbpoa.core.corpus.corpus_config import InstrumentRelationType
from wbpoa.core.model.qualified_citation import QualifiedCitation


class OperativeOperation(StrEnum):
    """The kind of textual effect a source provision has on a target provision."""

    MODIFIES = "modifies"


class OperativeRelation(BaseModel):
    """A directed, provision-level relationship derived from a declared instrument relation.

    Parsed deterministically from a source instrument's text against its
    declared target instrument (see `ProvisionReferenceParser.parse_operative_relations`).
    `match_span` and `matched_text` are relative to `normalize(source
    provision.text)` (NFKC-folded, whitespace-collapsed), not the raw
    HTML-extracted provision text, so offsets drift if compared directly
    against `Provision.text`.

    Attributes:
        source_provision_id: Identifier of the provision containing the reference.
        source_citation: Qualified citation of the source provision.
        target_provision_id: Identifier of the affected provision.
        target_citation: Qualified citation of the target provision.
        relation_type: Legal relationship declared between the two instruments.
        operation: Generic textual effect of the relation; a placeholder for
            future, more granular operations.
        match_span: Character offsets of the matched reference within the
            normalized source provision text.
        matched_text: The matched reference text itself.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_provision_id: UUID
    source_citation: QualifiedCitation
    target_provision_id: UUID
    target_citation: QualifiedCitation
    relation_type: InstrumentRelationType
    operation: OperativeOperation = OperativeOperation.MODIFIES
    match_span: tuple[int, int]
    matched_text: str
