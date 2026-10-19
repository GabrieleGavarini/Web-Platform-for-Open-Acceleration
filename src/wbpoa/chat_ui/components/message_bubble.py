from __future__ import annotations

from typing import TYPE_CHECKING

from nicegui import ui

from wbpoa.chat_ui.components.research_trail import ResearchTrail
from wbpoa.chat_ui.components.source_list import SourceList
from wbpoa.chat_ui.sources.source_resolver import SourceResolver

if TYPE_CHECKING:
    from collections.abc import Callable

    from nicegui.element import Element
    from nicegui.elements.html import Html
    from nicegui.elements.label import Label

    from wbpoa.chat_ui.sources.models import ChatReply, ResearchStep


class MessageBubble:
    """
    One exchange of the thread: the question, then the footnoted answer, an error, or the waiting state.

    The answer and its footnotes sit side by side, the footnotes in a column on the right; on a
    narrow window the stylesheet stacks them.
    """

    @staticmethod
    def render(reply: ChatReply, on_retry: Callable[[str], None]) -> None:
        """
        Draw a finished exchange.

        Args:
            reply: The question and what came back for it.
            on_retry: Called with the question when the user retries a failed exchange.
        """
        with ui.element("article"):
            ui.label(reply.question).classes("wbpoa-question")
            if reply.error is not None:
                ResearchTrail.render(reply.research)
                with ui.element("div").classes("wbpoa-error").props('role="alert"'):
                    ui.icon("error_outline").classes("text-xl")
                    with ui.column().classes("gap-2"):
                        ui.label(reply.error)
                        ui.button("Ask again", on_click=lambda: on_retry(reply.question), color=None).props(
                            "flat dense no-caps"
                        ).classes("wbpoa-secondary self-start")
                return
            with ui.element("div").classes("wbpoa-reply-body"):
                with ui.element("div").classes("wbpoa-reply-main"):
                    with ui.element("div").classes("wbpoa-reply-meta"):
                        ui.label(f"Answered from {reply.target_name}")
                        ui.label(f"in {MessageBubble.duration(reply.seconds)}")
                    ResearchTrail.render(reply.research)
                    ui.html(reply.answer_html).classes("wbpoa-answer")
                with ui.element("aside").classes("wbpoa-reply-sources").props('aria-label="Sources"'):
                    ui.html('<h3 class="wbpoa-sources-title">Sources</h3>', sanitize=False)
                    SourceList.render(reply.sources)

    @staticmethod
    def pending(question: str, target_name: str) -> tuple[Label, Element]:
        """
        Draw an exchange still being researched: a status line, the live research and answer, and the sources' placeholder.

        Args:
            question: The question asked.
            target_name: The corpus or pairing researched.

        Returns:
            The status label, for the caller to tick with the elapsed time, and the empty container
            the caller fills with `live` as the research and the answer arrive.
        """
        with ui.element("article").props('aria-busy="true"'):
            ui.label(question).classes("wbpoa-question")
            with ui.element("div").classes("wbpoa-reply-body"):
                with ui.element("div").classes("wbpoa-reply-main"):
                    status = (
                        ui.label(f"Researching {target_name}")
                        .classes("wbpoa-pending-status")
                        .props('role="status" aria-live="polite"')
                    )
                    live = ui.element("div").classes("wbpoa-live").props('aria-label="Research so far"')
                with ui.element("aside").classes("wbpoa-reply-sources").props('aria-hidden="true"'):
                    ui.html('<h3 class="wbpoa-sources-title">Sources</h3>', sanitize=False)
                    MessageBubble._skeleton(3)
        return status, live

    @staticmethod
    def live(steps: list[ResearchStep], answer: str) -> Html | None:
        """
        Draw the research made so far, newest last, then the answer as far as it is shown.

        Args:
            steps: The steps reported so far; nothing is drawn for them while there are none.
            answer: The answer shown so far, citations still in brackets, rendered from its
                Markdown; while empty, a placeholder stands in for it.

        Returns:
            The element showing the answer, for the caller to update with
            `SourceResolver.draft_html` as it grows; None while the placeholder is shown.
        """
        ResearchTrail.render(steps)
        if not answer:
            MessageBubble._skeleton(4)
            return None
        return ui.html(SourceResolver.draft_html(answer), sanitize=False).classes(
            "wbpoa-answer wbpoa-answer--streaming"
        )

    @staticmethod
    def _skeleton(lines: int) -> None:
        """Placeholder lines for text still to come."""
        with ui.element("div").classes("wbpoa-pending-answer").props('aria-hidden="true"'):
            for _ in range(lines):
                ui.element("div").classes("wbpoa-skeleton")

    @staticmethod
    def duration(seconds: float) -> str:
        """
        Say how long something took.

        Args:
            seconds: The duration.

        Returns:
            E.g. ``48 s`` or ``2 min 05 s``.
        """
        whole = round(seconds)
        minutes, rest = divmod(whole, 60)
        return f"{minutes} min {rest:02d} s" if minutes else f"{rest} s"
