from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import StrEnum
from typing import ClassVar
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from wbpoa.core.generation.models import RedactedNode


class ResearchEventKind(StrEnum):
    """What an agent just did while researching a question."""

    SEARCH = "search"
    LOOKUP = "lookup"
    DELEGATION = "delegation"


class ResearchEventPhase(StrEnum):
    """Whether a research step has just begun or is done."""

    STARTED = "started"
    FINISHED = "finished"


class ResearchEvent(BaseModel):
    """One research step, reported while the agent is still running."""

    id: UUID = Field(
        default_factory=uuid4, description="Shared by a step's STARTED and FINISHED events, so they can be paired."
    )
    kind: ResearchEventKind
    phase: ResearchEventPhase = ResearchEventPhase.FINISHED
    corpus_key: str = Field(description="The corpus searched, looked up, or delegated to.")
    text: str = Field(description="The question searched or delegated, or the provision looked up.")
    passages: list[RedactedNode] = Field(
        default_factory=list, description="What a search returned, or the provision a lookup fetched."
    )
    error: str | None = Field(default=None, description="Why a finished step failed; None when it succeeded.")


type ResearchListener = Callable[[ResearchEvent], None]
type AnswerListener = Callable[[str], None]


class ResearchObserver:
    """
    Report the research steps and the answer being written by the agents running in the current context, as they happen.

    The listeners live in context variables, so they follow the thread (and the context) that
    runs the agent: two questions answered at once by the same cached agent each reach only
    their own listeners. Without a listener, reporting does nothing.
    """

    _listener: ClassVar[ContextVar[ResearchListener | None]] = ContextVar("wbpoa_research_listener", default=None)
    _answer_listener: ClassVar[ContextVar[AnswerListener | None]] = ContextVar("wbpoa_answer_listener", default=None)

    @classmethod
    @contextmanager
    def listen(cls, listener: ResearchListener, on_answer: AnswerListener | None = None) -> Iterator[None]:
        """
        Send every step reported inside the block to `listener`, and the answer being written to `on_answer`.

        Args:
            listener: Called with each step, on the thread running the agent.
            on_answer: Called with the visible answer so far each time it grows; ``""`` clears a draft
                that turned out not to be the answer. None leaves the answer unstreamed.

        Yields:
            Nothing; both listeners are detached when the block exits.
        """
        token = cls._listener.set(listener)
        answer_token = cls._answer_listener.set(on_answer)
        try:
            yield
        finally:
            cls._answer_listener.reset(answer_token)
            cls._listener.reset(token)

    @classmethod
    @contextmanager
    def quiet_answers(cls) -> Iterator[None]:
        """
        Detach the answer listener inside the block, keeping the step listener.

        A manager wraps its sub agents' runs in it, so only its own answer streams.

        Yields:
            Nothing; the answer listener is restored when the block exits.
        """
        token = cls._answer_listener.set(None)
        try:
            yield
        finally:
            cls._answer_listener.reset(token)

    @classmethod
    def report(cls, event: ResearchEvent) -> None:
        """
        Report one step to the current listener, if any.

        Args:
            event: The step just started or finished.
        """
        listener = cls._listener.get()
        if listener is not None:
            listener(event)

    @classmethod
    def streams_answer(cls) -> bool:
        """
        Tell whether anyone listens to the answer being written.

        Returns:
            True when an answer listener is attached in the current context.
        """
        return cls._answer_listener.get() is not None

    @classmethod
    def report_answer(cls, text: str) -> None:
        """
        Report the answer written so far to the current answer listener, if any.

        Args:
            text: The visible answer so far, in full; ``""`` clears a draft that was not the answer.
        """
        listener = cls._answer_listener.get()
        if listener is not None:
            listener(text)
