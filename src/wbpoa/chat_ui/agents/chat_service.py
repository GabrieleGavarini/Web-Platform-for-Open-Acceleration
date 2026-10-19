from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import TYPE_CHECKING

from neo4j.exceptions import DriverError, Neo4jError
from structlog import get_logger

from wbpoa.chat_ui.agents.chat_target import ChatTargetKind
from wbpoa.chat_ui.app.routes import Routes
from wbpoa.chat_ui.sources.models import ChatReply, FoundPassage, ResearchStep
from wbpoa.chat_ui.sources.source_resolver import SourceResolver
from wbpoa.core.generation import SingleCorpusAgentError
from wbpoa.core.generation.models import RedactedNode, SingleCorpusResult
from wbpoa.core.generation.research_observer import (
    AnswerListener,
    ResearchEvent,
    ResearchEventPhase,
    ResearchObserver,
)
from wbpoa.multi_corpora import MultiCorporaResult, MultiCorpusAgentError

if TYPE_CHECKING:
    from uuid import UUID

    from wbpoa.chat_ui.agents.chat_target import ChatTarget
    from wbpoa.chat_ui.agents.types import AgentSource

log = get_logger()

type StepListener = Callable[[int, ResearchStep], None]


class ChatService:
    """
    Answer one chat question with the target's agent and shape the result for the chat.

    While the agent runs, every search, lookup, and sub-agent call it reports through
    `ResearchObserver` is turned into a `ResearchStep` and handed to the caller's listener with its
    position, so the chat can show the research live: a step is shown running as soon as it starts,
    then replaced in place once it finishes. The same steps become the reply's research trail, and
    the answer itself is handed to an answer listener while it is written. Agent and database
    failures become an error reply rather than an exception, so the chat can show them; they are
    logged with their traceback.

    Attributes:
        pool: Provides each target's agent.
    """

    def __init__(self, pool: AgentSource) -> None:
        """
        Args:
            pool: Provides each target's agent, normally the `AgentPool`.
        """
        self.pool = pool

    def ask(
        self,
        target: ChatTarget,
        question: str,
        on_step: StepListener | None = None,
        on_answer: AnswerListener | None = None,
    ) -> ChatReply:
        """
        Answer a question; blocking, so call it off the event loop.

        Args:
            target: The corpus or pairing to answer from.
            question: The user's question.
            on_step: Called on the agent's thread with each research step and its position in the
                trail: a new position when a step starts, the same one again when it finishes.
            on_answer: Called on the agent's thread with the answer written so far, before its
                citations become footnotes; ``""`` clears a draft that was not the answer.

        Returns:
            The footnoted answer with its cited sources and research trail, or an error reply.
        """
        started = perf_counter()
        steps: list[ResearchStep] = []
        positions: dict[UUID, int] = {}

        def record(event: ResearchEvent) -> None:
            step = self.step(target, event)
            index = positions.get(event.id)
            if index is None:
                index = len(steps)
                steps.append(step)
                if event.phase == ResearchEventPhase.STARTED:
                    positions[event.id] = index
            else:
                steps[index] = step
            if on_step is not None:
                on_step(index, step)

        try:
            with ResearchObserver.listen(record, on_answer):
                result = self.pool.agent(target).run(question)
        except (SingleCorpusAgentError, MultiCorpusAgentError) as error:
            log.exception("chat_ui_agent_failed", target=target.id)
            return self._error(target, question, f"The agent could not answer: {error}", steps, started)
        except (DriverError, Neo4jError) as error:
            log.exception("chat_ui_database_failed", target=target.id)
            return self._error(target, question, f"The graph database is unreachable: {error}", steps, started)

        answer_html, sources = SourceResolver(target.corpora).resolve(result.answer, self._grounding(target, result))
        return ChatReply(
            question=question,
            target_name=target.name,
            answer_html=answer_html,
            sources=sources,
            research=steps,
            seconds=perf_counter() - started,
        )

    @staticmethod
    def step(target: ChatTarget, event: ResearchEvent) -> ResearchStep:
        """
        Turn a reported research event into a step the chat can show, its passages linked to the graph.

        Args:
            target: The target being answered, whose manifests name each provision's instrument.
            event: The event the agent reported.

        Returns:
            The step, its passages as graph links; running while the event only says it started. In a
            pairing, the step names the sub agent it belongs to (by its corpus) and that corpus's
            position, so the chat can tag and colour each agent's work.
        """
        agent, agent_slot = None, 0
        if target.kind == ChatTargetKind.MULTI:
            for slot, corpus in enumerate(target.corpora):
                if corpus.key == event.corpus_key:
                    agent, agent_slot = corpus.config.name, slot
        names = {
            instrument.key: instrument.name for corpus in target.corpora for instrument in corpus.config.instruments
        }
        found = [
            FoundPassage(
                name=node.name,
                type=node.type,
                instrument_name=names.get(node.qualified_citation.instrument_key) if node.qualified_citation else None,
                url=Routes.node_url(event.corpus_key, node.id),
            )
            for node in event.passages
        ]
        return ResearchStep(
            kind=event.kind,
            corpus_key=event.corpus_key,
            text=event.text,
            found=found,
            error=event.error,
            running=event.phase == ResearchEventPhase.STARTED,
            agent=agent,
            agent_slot=agent_slot,
        )

    @staticmethod
    def _error(target: ChatTarget, question: str, message: str, steps: list[ResearchStep], started: float) -> ChatReply:
        """An error reply, keeping the research made before the failure; steps cut short no longer run."""
        return ChatReply(
            question=question,
            target_name=target.name,
            error=message,
            research=[step.model_copy(update={"running": False}) for step in steps],
            seconds=perf_counter() - started,
        )

    @staticmethod
    def _grounding(target: ChatTarget, result: object) -> list[tuple[str, RedactedNode]]:
        """Pair every grounding node with the corpus partition it was retrieved from."""
        if isinstance(result, MultiCorporaResult):
            return [
                (call.corpus_key, node)
                for call in result.sub_agent_calls
                if call.result is not None
                for node in call.result.grounding_context
            ]
        if isinstance(result, SingleCorpusResult):
            return [(target.key, node) for node in result.grounding_context]
        return []
