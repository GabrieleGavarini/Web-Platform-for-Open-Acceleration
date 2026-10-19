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


class ResearchPlanError(RuntimeError):
    """Raised when the planning model call fails or returns no usable question."""


class _ResearchPlan(BaseModel):
    """Structured response schema for the research planner: one broad question per legal issue."""

    questions: list[str] = Field(
        description=(
            "One broad, self-contained research question per legal issue the answer must settle, most "
            "important first. A single question when the question raises only one issue."
        )
    )


class _StructuredRunnable(Protocol):
    """The one method this planner calls on the runnable `with_structured_output` returns."""

    def invoke(
        self, messages: list[BaseMessage], /, config: RunnableConfig | None = None
    ) -> dict[str, Any] | BaseModel: ...


class _StructuredChatModel(Protocol):
    """Structural protocol for the one method this planner needs from the chat model."""

    def with_structured_output(self, schema: type[_ResearchPlan]) -> _StructuredRunnable: ...


class ResearchPlanner:
    """
    Break a question down into the legal issues it raises, as one broad research question each.

    The single-corpus agent runs these questions as its first, broad research round, before it
    reviews the passages and refines the issues they leave unsettled. Like the reformulation
    agent, the planner binds a structured-output schema (`_ResearchPlan`) rather than parsing
    free text.

    Attributes:
        llm: The chat model this planner binds structured output against.

    """

    PLAN_PROMPT: Final = """You plan the legal research for a question about the {corpus}: its articles, \
recitals and annexes, official guidelines, and rulings. A search tool will search the corpus with each \
question you write, and the answer will be written from what it finds.

Break the question down into the separate legal issues a complete answer must settle. Consider, and keep
only those the question raises or that its answer necessarily turns on:
- the definitions and scope the answer depends on;
- the actors concerned (e.g. provider, deployer, importer, notified body, authority, Commission);
- the obligations, rights, or prohibitions that apply;
- the conditions, exceptions, thresholds, and deadlines;
- who decides, supervises, or enforces, and the procedure followed.
Most questions need two or three research questions; a genuinely narrow one (e.g. what one defined term
means) needs only one. Fewer, well-separated questions are better than many overlapping ones.

For each issue write one broad research question:
1. Self-contained: it reads on its own, with every pronoun and implicit reference resolved and the {corpus}
named as the instrument.
2. Distinct: each question must target a different part of the law. If the same provisions would likely
answer two issues (e.g. what support exists and who grants it), write one question covering both instead of
two near-identical questions. Never write two questions that differ only in wording.
3. Broad: describe the topic of the issue in the corpus's own legal terms, so the search surfaces every
provision that may bear on it. Do not guess article numbers.
4. Keep every content word of the original question that belongs to the issue: nouns, named entities,
technical terms, acronyms.
5. One sentence, under 30 words. Never answer the question.

Write at most {max_questions} questions, most important first.

Question: What must GPAI providers do about systemic risk, and who checks it?
Questions:
- When is a general-purpose AI model classified as a GPAI model with systemic risk under the {corpus}?
- What obligations do providers of GPAI models with systemic risk have, and how is compliance with them
supervised and enforced under the {corpus}?

Question: What does "deployer" mean?
Questions:
- How does the {corpus} define a deployer of an AI system?
"""

    def __init__(self, settings: CoreSettings | None = None, corpus_context: CorpusContext | None = None) -> None:
        """
        Args:
            settings: Overrides for CoreSettings; defaults to the process-wide settings.
            corpus_context: Corpus identity used to render the prompt.

        """
        self._settings = settings if settings is not None else get_core_settings()
        self._corpus_context = corpus_context or CorpusContext(key="legacy", name="AI Act")
        self._prompt = self.PLAN_PROMPT.replace("{corpus}", self._corpus_context.name)
        self.llm: _StructuredChatModel = build_chat_model(
            self._settings.llm, enable_thinking=self._settings.generation.generation_enable_thinking
        )

    def plan(self, query: str, max_questions: int, *, config: RunnableConfig | None = None) -> list[str]:
        """
        Break `query` down into broad research questions, one per legal issue.

        Args:
            query: The raw user question.
            max_questions: The most questions to return; extra questions the model writes are dropped.
            config: Optional LangChain run config (e.g. carrying a Langfuse callback handler).

        Returns:
            Between 1 and `max_questions` research questions, most important first.

        Raises:
            ResearchPlanError: If the model call fails, returns an unparseable response, or returns
                no non-blank question.

        """
        messages = [
            SystemMessage(content=self._prompt.replace("{max_questions}", str(max_questions))),
            HumanMessage(content=query.strip()),
        ]
        log.info("research_plan_started", query=query)
        start = time.perf_counter()
        try:
            response = self.llm.with_structured_output(_ResearchPlan).invoke(messages, config=config)
        except Exception as error:
            message = "Research planning request failed."
            raise ResearchPlanError(message) from error

        if not isinstance(response, _ResearchPlan):
            message = "Research planner returned an unparseable structured response."
            raise ResearchPlanError(message)

        questions = [question.strip() for question in response.questions if question.strip()][:max_questions]
        if not questions:
            message = "Research planner returned no questions."
            raise ResearchPlanError(message)

        log.info("research_plan_completed", questions=questions, duration_seconds=round(time.perf_counter() - start, 3))
        return questions
