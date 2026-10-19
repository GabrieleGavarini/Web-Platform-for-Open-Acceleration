from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Final, Protocol
from uuid import UUID

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from structlog import get_logger

from wbpoa.core.corpus import CorpusContext
from wbpoa.core.generation.chat_model import budget_texts, build_chat_model, count_tokens, max_input_tokens
from wbpoa.core.generation.models import CandidateNode, RedactedNode
from wbpoa.core.settings import CoreSettings, get_core_settings

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

log = get_logger()

# Estimated tokens added per candidate by its "[id] Type Name\n" header and block separators, on
# top of its own text -- padding so the real prompt never exceeds the computed budget. Dominated
# by the bracketed UUID itself (~21 cl100k_base tokens, measured), much larger than a plain index.
_PER_CANDIDATE_OVERHEAD_TOKENS: Final = 40


class _RedactionSelection(BaseModel):
    """Structured response schema for the redaction agent: kept candidate ids, best first."""

    ids: list[UUID] = Field(
        description=(
            "The ids (shown in brackets) of every candidate to keep, copied exactly as shown, in "
            "decreasing order of relevance (most relevant first). Empty if none of the candidates "
            "are relevant."
        )
    )


class _StructuredRunnable(Protocol):
    """The one method this agent calls on the runnable `with_structured_output` returns."""

    def invoke(
        self, messages: list[BaseMessage], /, config: RunnableConfig | None = None
    ) -> dict[str, Any] | BaseModel: ...


class _StructuredChatModel(Protocol):
    """Structural protocol for the one method this agent needs beyond `chat_model.ChatModel`."""

    def with_structured_output(self, schema: type[_RedactionSelection]) -> _StructuredRunnable: ...


class RedactionError(RuntimeError):
    """Raised when the redaction model call itself fails, or returns an unparseable response."""


class RedactionAgent:
    """
    Algorithm 3 step 4: select S subseteq U by id; each kept node keeps its text unchanged.

    Unlike the other two Algorithm 3 agents (Reformulation, Generation), which get free text
    back from the model, this agent binds a structured-output schema (`_RedactionSelection`) via
    LangChain's `with_structured_output`: the task reduces to picking which candidates to keep,
    so the model only ever needs to return a list of ids, and structured output is the more
    robust way to get exactly that back -- at the cost of being a first-of-its-kind pattern in
    this codebase, unproven against this Ollama-served endpoint (no other agent here establishes
    a precedent for it).

    Attributes:
        llm: The chat model this agent binds structured output against to select candidates.

    """

    REDACTION_PROMPT: Final = """You filter retrieved passages for a legal question-answering system \
    covering the EU AI Act, its articles, recitals and annexes, official guidelines, and rulings.

    You are given a question and a list of candidate passages, each labeled with its id in brackets.
    Decide which candidates to keep.

    Keep a candidate if it contributes to answering the question: it states part of the answer,
    supplies a definition or scope condition the answer depends on, or gives the legal basis a claim
    would cite. This does not means that the candidate provide the answer directly or in full, but that
    it helpes generating the answer. Keep a candidate even if it is not the most relevant, as long as it contributes
    to the answer. Discard a candidate if it is irrelevant, off-topic, or does not contribute to answering the question.

    Do not be too strict with what you are removing. It is better to keep a candidate that is even remotely relevant than
    to remove a candidate that is actually relevant. If you are unsure, keep it.

    Return the ids of every candidate you keep, copied exactly as shown, ordered by how directly each
    answers the question, most direct first. Return an empty list if no candidate qualifies. Never
    return an id that is not in the candidate list.
    """

    def __init__(self, settings: CoreSettings | None = None, corpus_context: CorpusContext | None = None) -> None:
        """
        Args:
            settings: Overrides for CoreSettings; defaults to the process-wide settings.
            corpus_context: Selected corpus identity used to render the prompt.

        """
        self._settings = settings if settings is not None else get_core_settings()
        self._corpus_context = corpus_context or CorpusContext(key="legacy", name="AI Act")
        self._prompt = self.REDACTION_PROMPT.replace("EU AI Act", self._corpus_context.name)
        self._redaction_field = self._settings.generation.redaction_field
        self._max_input_tokens = max_input_tokens(self._settings.llm)
        self.llm: _StructuredChatModel = build_chat_model(
            self._settings.llm,
            temperature=self._settings.generation.redaction_temperature,
            enable_thinking=self._settings.generation.redaction_enable_thinking,
        )

    def _candidate_text(self, candidate: CandidateNode) -> str:
        """Return the text of `candidate` to show the model, per GenerationSettings.redaction_field."""
        return candidate.text if self._redaction_field == "text" else candidate.summary

    def _instrument_name(self, candidate: CandidateNode) -> str | None:
        """Return the display name of `candidate`'s instrument, or None for guidelines/rulings."""
        if candidate.qualified_citation is None:
            return None
        return self._corpus_context.instrument_names.get(candidate.qualified_citation.instrument_key)

    def redact(
        self, query: str, candidates: list[CandidateNode], *, config: RunnableConfig | None = None
    ) -> list[RedactedNode]:
        """
        Select the nodes of `candidates` relevant to `query`.

        Args:
            query: The original (non-reformulated) user query.
            candidates: The candidate pool U.
            config: Optional LangChain run config (e.g. carrying a Langfuse callback handler).

        Returns:
            The redaction S, ordered best-first as the model returned it; empty when `candidates`
            is empty, or when the model selects nothing (Remark 15 -- not an error).

        Raises:
            RedactionError: If the model call itself fails, or returns an unparseable response.

        """
        if not candidates:
            return []

        raw_texts = [self._candidate_text(node) for node in candidates]
        overhead = count_tokens(self._prompt) + count_tokens(query) + len(candidates) * _PER_CANDIDATE_OVERHEAD_TOKENS
        budget = max(self._max_input_tokens - overhead, 0)
        texts = budget_texts(raw_texts, budget)
        if texts != raw_texts:
            log.warning(
                "redaction_candidates_truncated",
                num_candidates=len(candidates),
                per_candidate_budget=budget // len(candidates) if candidates else 0,
            )

        blocks = "\n\n".join(
            f"[{node.id}] {node.type} {node.name}"
            + (f" ({instrument_name})" if (instrument_name := self._instrument_name(node)) else "")
            + f"\n{text}"
            for node, text in zip(candidates, texts, strict=True)
        )
        user_content = f"<question>\n{query.strip()}\n</question>\n\n<candidates>\n{blocks}\n</candidates>"
        messages = [SystemMessage(content=self._prompt), HumanMessage(content=user_content)]

        log.info("redaction_started", query=query, num_candidates=len(candidates))
        start = time.perf_counter()
        try:
            response = self.llm.with_structured_output(_RedactionSelection).invoke(messages, config=config)
        except Exception as e:
            log.exception("redaction_invoke_failed", duration_seconds=round(time.perf_counter() - start, 3))
            msg = "Redaction request failed."
            raise RedactionError(msg) from e
        duration = time.perf_counter() - start

        if not isinstance(response, _RedactionSelection):
            msg = "Redaction model returned an unparseable structured response."
            raise RedactionError(msg)

        by_id = {candidate.id: candidate for candidate in candidates}
        selected: list[RedactedNode] = []
        for node_id in response.ids:
            candidate = by_id.get(node_id)
            if candidate is None:
                log.warning("redaction_invalid_id", id=str(node_id))
                continue
            selected.append(
                RedactedNode(
                    id=candidate.id,
                    name=candidate.name,
                    type=candidate.type,
                    text=self._candidate_text(candidate),
                    qualified_citation=candidate.qualified_citation,
                )
            )

        if not selected:
            log.info("redaction_selected_none", duration_seconds=round(duration, 3))
        else:
            log.info("redaction_completed", num_selected=len(selected), duration_seconds=round(duration, 3))
        return selected
