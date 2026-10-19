from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from wbpoa.core.corpus.corpus_config import InstrumentRelationType
from wbpoa.core.model.operative_relation import OperativeOperation
from wbpoa.core.model.qualified_citation import QualifiedCitation


class RelevantElement(BaseModel):
    """A ranked element (Article/Recital/Annex) returned by Algorithm 2 step 8."""

    id: UUID
    name: str
    type: str
    text: str
    summary: str
    score: float = Field(description="score(x) = alpha*esim + (1-alpha)*eauth.")
    alpha: float = Field(description="Mixing weight between element similarity and citation authority used for score.")
    similarity: float = Field(description="esim(q, x): the hybrid semantic-lexical similarity of x to the query.")
    semantic: float = Field(description="sem(q, x): cosine similarity on this element's summary embedding.")
    lexical: float = Field(description="lex_E(q, x): BM25 score of x's summary, min-max normalised within E.")
    gamma: float = Field(description="Mixing weight between semantic and lexical similarity used for esim.")
    citation_weight: float = Field(description="cw(x): sum of lambda(u) over the nodes u citing this element.")
    authority: float = Field(description="eauth(x, E): log-normalised citation weight within x's element type.")
    rerank_raw: float | None = Field(
        default=None, description="rr(q, x): raw cross-encoder logit, set only when x was in the reranker's shortlist."
    )
    rerank: float | None = Field(
        default=None, description="score_rr(x) after the rho blend; None when the reranker didn't run or shortlist x."
    )
    rho: float = Field(
        default=0.0, description="Blend weight used to produce `rerank`; 0.0 when the reranker did not run."
    )
    qualified_citation: QualifiedCitation | None = Field(
        default=None, description="The element's instrument-qualified identity."
    )


class ExpansionNode(BaseModel):
    """A citation-neighbourhood node (cited element, guideline or ruling) returned by Algorithm 2 step 8."""

    id: UUID
    name: str
    type: str
    text: str
    summary: str
    similarity: float = Field(description="xsim(q, u): the hybrid semantic-lexical similarity of u to the query.")
    semantic: float = Field(description="sem(q, u): cosine similarity on this node's summary embedding.")
    lexical: float = Field(description="lex_N(q, u): BM25 score of u's summary, min-max normalised within its pool.")
    pool: Literal["element", "document"] = Field(
        description="Which pool u was normalised and ranked within; scores are not comparable across pools."
    )
    depth: int = Field(default=0, description="Minimum citation hops from the relevant element(s) that reached u.")
    decay: float = Field(default=1.0, description="delta**depth; xsim = similarity * decay.")
    reached_from: list[UUID] = Field(
        default_factory=list,
        description="Ids of the relevant elements of R from which this node was reached (Remark 8).",
    )
    rerank_raw: float | None = Field(
        default=None, description="rr(q, u): raw cross-encoder logit, set only when u was in the reranker's shortlist."
    )
    rerank: float | None = Field(
        default=None, description="xsim_rr(q, u) after the rho blend and decay; None when u wasn't reranked."
    )
    rho: float = Field(
        default=0.0, description="Blend weight used to produce `rerank`; 0.0 when the reranker did not run."
    )
    qualified_citation: QualifiedCitation | None = Field(
        default=None, description="The node's instrument-qualified identity; None for guidelines/rulings."
    )
    relation_type: InstrumentRelationType | None = Field(
        default=None, description="The OPERATIVE_RELATION's legal relationship type; None for citation expansion."
    )
    operation: OperativeOperation | None = Field(
        default=None, description="The OPERATIVE_RELATION's textual effect; None for citation expansion."
    )


class RetrievalResult(BaseModel):
    """The output of Algorithm 2: the ranked relevant elements and their ranked expansion nodes."""

    relevant_elements: list[RelevantElement] = Field(default_factory=list)
    expansion_nodes: list[ExpansionNode] = Field(default_factory=list)
