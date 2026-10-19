from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, Field

from wbpoa.core.inference import RetrievalResult
from wbpoa.core.model.qualified_citation import QualifiedCitation


class CandidateNode(BaseModel):
    """One node of U, Algorithm 3's candidate pool: a shared wrapper over RelevantElement/ExpansionNode."""

    id: UUID
    name: str
    type: str
    text: str
    summary: str
    origin: Literal["relevant", "element_expansion", "document_expansion"] = Field(
        description="Which of R, N_elem or N_doc this node came from."
    )
    qualified_citation: QualifiedCitation | None = Field(
        default=None, description="The node's instrument-qualified identity; None for guidelines/rulings."
    )


class RedactedNode(BaseModel):
    """One node of S, Algorithm 3's grounding context: id/name/type plus the redaction agent's kept text."""

    id: UUID
    name: str
    type: str
    text: str = Field(
        description="The node's full text or summary, as shown to the redaction agent, copied through unchanged for kept nodes."
    )
    qualified_citation: QualifiedCitation | None = Field(
        default=None, description="The node's instrument-qualified identity; None for guidelines/rulings."
    )


class CitedProvision(BaseModel):
    """A provision cited by one or more returned guidelines/rulings: a lead for further research."""

    id: UUID
    name: str
    type: str
    qualified_citation: QualifiedCitation | None = Field(
        default=None, description="The provision's instrument-qualified identity."
    )
    cited_by: list[str] = Field(
        default_factory=list, description="Names of the returned guidelines/rulings citing it, in first-seen order."
    )


class SearchResult(BaseModel):
    """One search tool call: reformulation, Algorithm 2 retrieval, and redaction for one question."""

    question: str = Field(description="The question the single-corpus agent asked the search tool.")
    reformulated_query: str = Field(
        description="q', the one query handed to Algorithm 2; the question itself when reformulation is disabled."
    )
    retrieval: RetrievalResult = Field(
        default_factory=RetrievalResult, description="Algorithm 2's result for q': the call's only search."
    )
    candidate_pool: list[CandidateNode] = Field(default_factory=list)
    passages: list[RedactedNode] = Field(
        default_factory=list, description="S for this call: the passages returned to the single-corpus agent."
    )
    cited_provisions: list[CitedProvision] = Field(
        default_factory=list,
        description=(
            "Provisions cited by the guideline/ruling passages but not themselves among the passages, most-cited first."
        ),
    )


class SingleCorpusResult(BaseModel):
    """The single-corpus agent's answer, plus every search it made, for logging/inspection."""

    query: str
    answer: str
    searches: list[SearchResult] = Field(default_factory=list)
    looked_up: list[RedactedNode] = Field(
        default_factory=list, description="Provisions fetched with the lookup_provision tool, in call order."
    )
    grounding_context: list[RedactedNode] = Field(
        default_factory=list,
        description=(
            "Union of every search's passages and every looked-up provision, deduplicated by node "
            "id in first-seen order: the provisions the answer may cite."
        ),
    )


class AnswerResult(Protocol):
    """What callers read off any answering pipeline's result: the answer and the nodes it rests on."""

    @property
    def answer(self) -> str:
        """The final answer text."""
        ...

    @property
    def grounding_context(self) -> Sequence[RedactedNode]:
        """The nodes the answer is grounded in; their qualified citations are the cited provisions."""
        ...


class QuestionAnswerer(Protocol):
    """
    Any pipeline that answers one question end to end.

    Satisfied by `SingleCorpusAgent` (one corpus) and by the multi-corpus agent.
    """

    def run(self, query: str, /) -> AnswerResult:
        """
        Answer `query`.

        Args:
            query: The raw user question.

        Returns:
            The answer together with its grounding context.

        """
        ...


@runtime_checkable
class ProvidesRunMetadata(Protocol):
    """An answering pipeline that contributes its own groups to a run's metadata."""

    def run_metadata(self) -> dict[str, object]:
        """
        Describe the pipeline-specific configuration a run was produced with.

        Returns:
            Top-level metadata groups, merged into the run metadata.

        """
        ...
