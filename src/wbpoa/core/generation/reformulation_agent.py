from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Final, Protocol

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from structlog import get_logger

from wbpoa.core.corpus import CorpusContext
from wbpoa.core.generation.chat_model import build_chat_model
from wbpoa.core.settings import CoreSettings, get_core_settings

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

log = get_logger()


class ReformulationError(RuntimeError):
    """Raised when the reformulation model call fails or returns no query."""


class _ReformulationResult(BaseModel):
    """Structured response schema for the reformulation agent: exactly one search question."""

    query: str = Field(
        ...,
        description=(
            "The single search question: the input question rewritten as one focused, self-contained "
            "question in the corpus's own legal terms."
        ),
    )


class _StructuredRunnable(Protocol):
    """The one method this agent calls on the runnable `with_structured_output` returns."""

    def invoke(
        self, messages: list[BaseMessage], /, config: RunnableConfig | None = None
    ) -> dict[str, Any] | BaseModel: ...


class _StructuredChatModel(Protocol):
    """Structural protocol for the one method this agent needs beyond `chat_model.ChatModel`."""

    def with_structured_output(self, schema: type[_ReformulationResult]) -> _StructuredRunnable: ...


class ReformulationAgent:
    """
    Algorithm 3 step 1: rewrite one research question into exactly one focused, self-contained
    search question for Algorithm 2's hybrid retrieval, naming the concepts, actors, and
    obligations the passages should cover. Decomposing a question into several is the
    single-corpus agent's job, not this agent's: each search runs exactly one search.

    Like the redaction agent, this agent binds a structured-output schema
    (`_ReformulationResult`) via LangChain's `with_structured_output`, rather than parsing free
    text: the task reduces to producing one query, so structured output is the more robust way to
    get exactly that back.

    Attributes:
        llm: The chat model this agent binds structured output against to reformulate a query.

    """

    REFORMULATION_PROMPT: Final = """You rewrite a legal research question into one search question for a \
retriever over the AI Act corpus: its articles, recitals and annexes, official guidelines, and rulings. The \
retriever finds passages by meaning and by shared legal terms, so the search question must say exactly what \
the passages it should find are about.

Return exactly one search question. Never split the question into several: it has already been broken down,
and if it covers more than one point, keep them together in one sentence.

Rules:
1. Keep every content word of the original question verbatim: nouns, named entities, technical terms,
acronyms, article numbers. You may add words around them; you may not replace them with synonyms.
2. Name what the passages should contain: the concepts, the actors (e.g. provider, deployer, importer,
notified body, market surveillance authority, Commission), the obligations, rights, conditions, exceptions,
deadlines, or penalties the answer turns on.
3. Make the question self-contained: resolve pronouns and implicit references, and name the AI Act as the
instrument when the question concerns it.
4. Use the legal terminology of the AI Act rather than colloquial wording, when the original leaves no doubt
about which legal term is meant.
5. Resolve the question toward a single, literal reading -- never hedge across readings.
6. Add nothing the original question does not ask about: no adjacent topics, no answer, no assumptions.
7. One sentence, under 40 words.

Question: AI Act penalties
Search question: What administrative fines does the AI Act impose for infringements, and what maximum amount
applies to each tier of infringement?

Question: Do SMEs get any relief from the compliance burden?
Search question: What relief from AI Act compliance obligations, such as simplified requirements or reduced
conformity assessment fees, is available to SMEs?

Question: Can a deployer rely on it being CE-marked?
Search question: May a deployer of a high-risk AI system rely on the CE marking affixed by the provider to meet
its own obligations under the AI Act?
"""

    def __init__(self, settings: CoreSettings | None = None, corpus_context: CorpusContext | None = None) -> None:
        """
        Args:
            settings: Overrides for CoreSettings; defaults to the process-wide settings.
            corpus_context: Selected corpus identity used to render the prompt.

        """
        self._settings = settings if settings is not None else get_core_settings()
        self._corpus_context = corpus_context or CorpusContext(key="legacy", name="AI Act")
        prompt = self.REFORMULATION_PROMPT.replace("AI Act", self._corpus_context.name)
        self.llm: _StructuredChatModel = build_chat_model(
            self._settings.llm, enable_thinking=self._settings.generation.reformulation_enable_thinking
        )
        self._prompt = prompt

    def reformulate(self, query: str, *, config: RunnableConfig | None = None) -> str:
        """
        Rewrite `query` into the one search question Algorithm 2 runs on.

        Args:
            query: The research question.
            config: Optional LangChain run config (e.g. carrying a Langfuse callback handler).

        Returns:
            The search question, stripped of surrounding whitespace.

        Raises:
            ValueError: If `query` is empty or whitespace only.
            ReformulationError: If the model call fails, returns an unparseable response, or
                returns a blank query.

        """
        if not query or not query.strip():
            msg = "Cannot reformulate an empty query."
            raise ValueError(msg)

        messages = [SystemMessage(content=self._prompt), HumanMessage(content=query)]

        log.info("reformulation_started", query=query)
        start = time.perf_counter()
        try:
            response = self.llm.with_structured_output(_ReformulationResult).invoke(messages, config=config)
        except Exception as e:
            log.exception("reformulation_invoke_failed", duration_seconds=round(time.perf_counter() - start, 3))
            msg = "Reformulation request failed."
            raise ReformulationError(msg) from e
        duration = time.perf_counter() - start

        if not isinstance(response, _ReformulationResult):
            msg = "Reformulation model returned an unparseable structured response."
            raise ReformulationError(msg)

        reformulated = response.query.strip()
        if not reformulated:
            log.error("reformulation_empty_response", duration_seconds=round(duration, 3))
            msg = "The model returned a blank reformulated query."
            raise ReformulationError(msg)

        log.info("reformulation_completed", reformulated_query=reformulated, duration_seconds=round(duration, 3))
        return reformulated
