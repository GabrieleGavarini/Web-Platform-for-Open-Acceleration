from __future__ import annotations

import asyncio
import time
from html import escape
from itertools import count
from typing import TYPE_CHECKING, Final

from nicegui import app, run, ui

from wbpoa.chat_ui.agents.chat_target import ChatTargetKind
from wbpoa.chat_ui.app import routes
from wbpoa.chat_ui.app.routes import Routes
from wbpoa.chat_ui.components.message_bubble import MessageBubble
from wbpoa.chat_ui.components.page_frame import PageFrame
from wbpoa.chat_ui.sources.models import ChatReply, ResearchStep
from wbpoa.chat_ui.sources.source_resolver import SourceResolver

if TYPE_CHECKING:
    from nicegui.client import Client
    from nicegui.elements.button import Button
    from nicegui.elements.html import Html
    from nicegui.elements.label import Label
    from nicegui.elements.select import Select
    from nicegui.elements.textarea import Textarea
    from nicegui.page_arguments import PageArguments

    from wbpoa.chat_ui.agents.chat_service import ChatService
    from wbpoa.chat_ui.agents.chat_target import ChatTarget
    from wbpoa.chat_ui.agents.types import TargetCatalog

TARGET_KEY: Final = "wbpoa_chat_target"
REPLIES_KEY: Final = "wbpoa_chat_replies"
PENDING_KEY: Final = "wbpoa_chat_pending"
POLL_SECONDS: Final = 0.25
# The answer is typed out at a steady pace rather than shown in the bursts the model writes it in:
# each tick reveals a share of what is waiting, between a few characters and a dozen (75 to 300 a
# second), so even an answer that arrives in one burst reads as written.
TYPE_SECONDS: Final = 0.04
MIN_CHARS_PER_TICK: Final = 3
MAX_CHARS_PER_TICK: Final = 12
CATCH_UP_DIVISOR: Final = 30
# Once answered, how long the reply waits for a typing that no longer advances (the page is not shown).
TYPING_STALL_SECONDS: Final = 0.5
# While the answer is typed, the page follows it if the reader is within this distance of the bottom.
FOLLOW_SLACK_PX: Final = 240


class ChatPage:
    """
    The chat: pick a corpus or pairing, ask, and read footnoted answers whose footnotes open the citation graph.

    One instance is built per page render. The conversation lives in the browser tab's client
    storage, so it survives moving to the graph and back; picking another target starts a new
    one. A question runs in a worker thread that records its research steps and the answer being
    written in the pending entry; the page polls the storage four times a second, which shows each
    step as soon as it starts, and also picks up an answer that finished while the tester was
    reading the graph. The answer being written is typed out by a faster timer at a steady pace,
    its Markdown rendered as it goes; how far it got is kept in the pending entry, so a redrawn
    page resumes there, and the finished reply waits for the typing to catch up.

    Attributes:
        catalog: Lists the targets.
        service: Answers questions.
    """

    def __init__(self, catalog: TargetCatalog, service: ChatService) -> None:
        """
        Args:
            catalog: Lists the targets.
            service: Answers questions.
        """
        self.catalog = catalog
        self.service = service
        self._store: dict = {}
        self._replies: list[dict[str, object]] = []
        self._shown: tuple[int, bool] = (0, False)
        self._status: Label | None = None
        self._live_revision: object = None
        self._answer_html: Html | None = None
        self._answer_shown = ""
        self._composer: Textarea | None = None
        self._send: Button | None = None
        self._picker: Select | None = None
        self._new_conversation: Button | None = None
        self._args: PageArguments | None = None
        self._client: Client | None = None

    def render(self, args: PageArguments) -> None:
        """
        Build the chat page for the current client.

        Args:
            args: The sub page's arguments, used to re-render it when the target changes.
        """
        ui.page_title("Chat · WBPOA")
        targets = self.catalog.targets()
        if not targets:
            self._no_targets()
            return
        self._args = args
        self._client = ui.context.client
        self._store = app.storage.client
        self._replies = self._store.setdefault(REPLIES_KEY, [])
        self._shown = self._state()
        target = self._target()
        with (
            PageFrame.render(routes.CHAT, [target.name, *target.instrument_names]).classes("wbpoa-page--app"),
            ui.element("div").classes("wbpoa-split"),
        ):
            self._aside(targets, target)
            with ui.element("section").classes("wbpoa-conversation").props('aria-label="Conversation"'):
                self._thread()
                self._composer_box()
        self._enable(enabled=self._store.get(PENDING_KEY) is None)
        ui.timer(POLL_SECONDS, self._sync)
        ui.timer(TYPE_SECONDS, self._type)

    def _target(self) -> ChatTarget:
        """The target chosen in this tab, or the first one."""
        return self.catalog.target(self._store.get(TARGET_KEY, "")) or self.catalog.targets()[0]

    def _state(self) -> tuple[int, bool]:
        """How many replies there are, and whether a question is being researched."""
        return len(self._replies), self._store.get(PENDING_KEY) is not None

    @ui.refreshable_method
    def _thread(self) -> None:
        """The conversation, or the opening screen when it is empty."""
        pending = self._store.get(PENDING_KEY)
        self._answer_html = None
        if not self._replies and pending is None:
            self._empty_state(self._target())
            return
        with ui.element("div").classes("wbpoa-thread"):
            for reply in self._replies:
                MessageBubble.render(ChatReply.model_validate(reply), self._retry)
            if pending is not None:
                self._status, live = MessageBubble.pending(str(pending["question"]), str(pending["target_name"]))
                with live:
                    self._live()

    @ui.refreshable_method
    def _live(self) -> None:
        """The research of the question being answered, as far as it has got, then its answer as far as it is typed."""
        pending = self._store.get(PENDING_KEY)
        self._live_revision = pending.get("revision") if pending is not None else None
        self._answer_shown = self._typed(pending)
        self._answer_html = MessageBubble.live(self._live_steps(), self._answer_shown)

    def _live_steps(self) -> list[ResearchStep]:
        """The research steps reported so far for the pending question."""
        pending = self._store.get(PENDING_KEY)
        research = pending.get("research", []) if pending is not None else []
        return [ResearchStep.model_validate(step) for step in research]

    async def ask(self, text: str | None) -> None:
        """
        Research a question in a worker thread and record the reply in the tab's storage.

        Each research step the agent reports is put in the pending entry on the event loop, at its
        position in the trail (a step reported again when it finishes replaces its running self),
        and so is the answer as far as it is written, so the page can show both while the agent
        is still working. Once answered, the reply waits for the page to finish typing the answer
        out, unless the typing stalls because the page is not shown.

        Args:
            text: The question; blank questions, and questions sent while another is running, are ignored.
        """
        question = (text or "").strip()
        if not question or self._store.get(PENDING_KEY) is not None:
            return
        target = self._target()
        if self._composer is not None:
            self._composer.value = ""
        self._store[PENDING_KEY] = {
            "question": question,
            "target_name": target.name,
            "started": time.time(),
            "research": [],
            "revision": 0,
            "answer": "",
            "typed": 0,
        }
        # Client storage wraps what it is given in observable copies: write to those, which the page reads.
        pending = self._store[PENDING_KEY]
        research = pending["research"]
        revisions = count(1)
        loop = asyncio.get_running_loop()

        def put_step(index: int, step: dict[str, object]) -> None:
            if index < len(research):
                research[index] = step
            else:
                research.append(step)
            pending["revision"] = next(revisions)

        def on_step(index: int, step: ResearchStep) -> None:
            loop.call_soon_threadsafe(put_step, index, step.model_dump(mode="json"))

        def on_answer(answer: str) -> None:
            loop.call_soon_threadsafe(pending.__setitem__, "answer", answer)

        self._sync()
        reply = await run.io_bound(self.service.ask, target, question, on_step, on_answer)
        if reply is not None and reply.error is None:
            await self._finish_typing(pending)
        self._store.pop(PENDING_KEY, None)
        if reply is not None:
            self._replies.append(reply.model_dump(mode="json"))
        self._sync()

    @staticmethod
    async def _finish_typing(pending: dict[str, object]) -> None:
        """Wait until the answer is typed out in full, or until the typing stops advancing."""
        last, stalled = -1, 0.0
        while (typed := len(ChatPage._typed(pending))) < len(ChatPage._draft(pending)):
            stalled = stalled + TYPE_SECONDS if typed == last else 0.0
            if stalled >= TYPING_STALL_SECONDS:
                return
            last = typed
            await asyncio.sleep(TYPE_SECONDS)

    def _type(self) -> None:
        """
        Type the answer being written out a few characters further, rendering its Markdown.

        Reveals a share of what is waiting, within a fixed range, so the answer flows at a steady
        pace and catches up gently, never in bursts, when the model writes faster than it is
        shown; a citation is revealed whole, once it is written. A draft that was cleared or
        replaced starts over.
        """
        pending = self._store.get(PENDING_KEY)
        if pending is None:
            return
        draft = self._draft(pending)
        typed = len(self._typed(pending))
        if pending.get("typed") != typed:
            # The draft was cleared or replaced by a shorter one: start over from what is left.
            pending["typed"] = typed
        if typed < len(draft):
            step = min(MAX_CHARS_PER_TICK, max(MIN_CHARS_PER_TICK, (len(draft) - typed) // CATCH_UP_DIVISOR))
            pending["typed"] = self._past_citation(draft, min(len(draft), typed + step))
        shown = self._typed(pending)
        if bool(shown) != (self._answer_html is not None):
            self._live.refresh()
        elif self._answer_html is not None and shown != self._answer_shown:
            self._answer_html.content = SourceResolver.draft_html(shown)
            self._answer_shown = shown
            self._follow()

    def _follow(self) -> None:
        """Keep the end of the answer being typed in view, unless the reader has scrolled up to read."""
        if self._client is not None:
            self._client.run_javascript(
                "const page = document.scrollingElement;"
                f"if (page.scrollHeight - page.clientHeight - page.scrollTop < {FOLLOW_SLACK_PX})"
                " page.scrollTop = page.scrollHeight;"
            )

    def _retry(self, question: str) -> None:
        """Put a failed question back in the box."""
        if self._composer is not None:
            self._composer.value = question
            self._composer.run_method("focus")

    def _sync(self) -> None:
        """
        Keep the page in step with the tab's storage.

        Redraws the thread when a question started (bringing it into view) or a reply arrived. A
        reply whose answer was typed out in view keeps the reader where they are, since it takes
        the place of the same text; one that was not brings the start of the reply into view, not
        the end of the page. While a question is being answered, ticks the waiting line and
        redraws the research when a step started or finished; typing the answer out is left to
        `_type`.
        """
        state = self._state()
        if state != self._shown:
            landed = state[0] > self._shown[0]
            typed_in_view = self._answer_html is not None
            self._shown = state
            self._thread.refresh()
            self._enable(enabled=not state[1])
            if landed and not typed_in_view:
                self._scroll_to_latest()
            elif state[1]:
                self._scroll_to_end()
            return
        pending = self._store.get(PENDING_KEY)
        if pending is None:
            return
        if self._status is not None:
            self._status.text = self._progress(pending)
        if pending.get("revision") != self._live_revision:
            self._live.refresh()
            if not self._draft(pending):
                self._scroll_to_end()

    def _scroll_to_end(self) -> None:
        """
        Bring the newest part of the thread into view.

        Goes through the page's client, not the current slot: by the time an answer lands, the element
        that started the question (an example button) may have been redrawn away.
        """
        if self._client is not None:
            self._client.run_javascript("window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'})")

    def _scroll_to_latest(self) -> None:
        """Bring the start of the newest exchange to the top of the view, so its answer reads from the beginning."""
        if self._client is not None:
            self._client.run_javascript(
                "document.querySelector('.wbpoa-thread > article:last-of-type')"
                "?.scrollIntoView({block: 'start', behavior: 'smooth'})"
            )

    def _enable(self, *, enabled: bool) -> None:
        """Enable or disable what must wait for the running question: Ask, the target picker, and the reset."""
        for control in (self._send, self._picker, self._new_conversation):
            if control is not None:
                control.set_enabled(enabled)

    def _choose(self, target_id: str) -> None:
        """Switch the tab to another target, starting a new conversation, and redraw the page."""
        if target_id == self._target().id:
            return
        self._replies.clear()
        self._store[TARGET_KEY] = target_id
        if self._args is not None:
            self._args.frame.refresh()

    def _clear(self) -> None:
        """Start a new conversation."""
        self._replies.clear()
        self._shown = self._state()
        self._thread.refresh()

    def _aside(self, targets: list[ChatTarget], target: ChatTarget) -> None:
        """The aside: the target picker, its instruments and graphs, and a reset."""
        with ui.element("aside").classes("wbpoa-aside wbpoa-app-aside wbpoa-chat-aside").props('aria-label="Corpus"'):
            self._picker = (
                ui.select(
                    {item.id: self._option(item) for item in targets},
                    value=target.id,
                    label="Ask about",
                    on_change=lambda event: self._choose(str(event.value)),
                )
                .props("outlined dense options-dense")
                .classes("w-full")
            )
            self._instruments(target)
            self._new_conversation = (
                ui.button("New conversation", icon="restart_alt", on_click=self._clear, color=None)
                .props('flat no-caps aria-label="New conversation"')
                .classes("wbpoa-secondary self-start")
            )

    def _composer_box(self) -> None:
        """The question box, sent with Enter or the Ask button."""
        with ui.element("div").classes("wbpoa-composer"):
            ui.html('<span class="wbpoa-label" aria-hidden="true">Your question</span>', sanitize=False)
            with ui.element("div").classes("wbpoa-composer-row"):
                composer = ui.textarea().props('outlined autogrow dense rows=1 aria-label="Your question"')
                composer.classes("w-full").on("keydown.enter.exact.prevent", lambda: self.ask(composer.value))
                self._send = (
                    ui.button("Ask", on_click=lambda: self.ask(composer.value))
                    .props("unelevated no-caps")
                    .classes("wbpoa-primary")
                )
            ui.label("Enter sends the question. Shift and Enter starts a new line.").classes("wbpoa-hint")
        self._composer = composer

    def _empty_state(self, target: ChatTarget) -> None:
        """The opening screen: what the chat does, and the target's example questions."""
        with ui.element("div").classes("wbpoa-empty"):
            ui.html(f"<h1>Ask a question about {escape(target.name)}</h1>", sanitize=False)
            ui.html(
                "<p>The agent researches the corpus, then answers citing the provisions it relies on. "
                "Each footnote opens that provision in the citation graph.</p>",
                sanitize=False,
            )
            if target.examples:
                ui.html('<h2 class="wbpoa-section-title">Example questions</h2>', sanitize=False)
                with ui.element("div").classes("wbpoa-examples"):
                    for example in target.examples:
                        ui.button(example, on_click=lambda _, question=example: self.ask(question), color=None).props(
                            "flat no-caps align=left"
                        ).classes("wbpoa-example")

    @staticmethod
    def _no_targets() -> None:
        """The page shown when no corpus manifest was found."""
        with PageFrame.render(routes.CHAT, ["Chat"]):
            ui.label(
                "No corpus manifest was found. Start the app with --corpora-dir pointing at data/corpora."
            ).classes("wbpoa-error")

    @staticmethod
    def _draft(pending: dict[str, object] | None) -> str:
        """The answer written so far for the pending question; empty while it is still being researched."""
        answer = pending.get("answer") if pending is not None else None
        return answer if isinstance(answer, str) else ""

    @staticmethod
    def _past_citation(draft: str, cut: int) -> int:
        """Move a cut that falls inside a written citation to its end, so the citation appears whole."""
        opening = draft.rfind("[", 0, cut)
        if opening > draft.rfind("]", 0, cut) and (closing := draft.find("]", opening)) != -1:
            return closing + 1
        return cut

    @staticmethod
    def _typed(pending: dict[str, object] | None) -> str:
        """The part of the pending answer typed out so far."""
        typed = pending.get("typed") if pending is not None else None
        return ChatPage._draft(pending)[: typed if isinstance(typed, int) else 0]

    @staticmethod
    def _progress(pending: dict[str, object]) -> str:
        """The waiting line: what is being done (researching, then writing the answer) and for how long."""
        started = pending.get("started")
        elapsed = MessageBubble.duration(time.time() - started if isinstance(started, float) else 0.0)
        if ChatPage._draft(pending):
            return f"Writing the answer: {elapsed}"
        return f"Researching {pending.get('target_name')}: {elapsed}"

    @staticmethod
    def _option(target: ChatTarget) -> str:
        """The select label of a target: pairings say so."""
        return f"{target.name} (pairing)" if target.kind == ChatTargetKind.MULTI else target.name

    @staticmethod
    def _instruments(target: ChatTarget) -> None:
        """List the target's instruments and link each corpus's citation graph."""
        with ui.element("div").classes("wbpoa-aside-extra"):
            ui.html('<h2 class="wbpoa-section-title">Instruments</h2>', sanitize=False)
            ui.html(
                '<ul class="wbpoa-meta-list">'
                + "".join(f"<li>{escape(name)}</li>" for name in target.instrument_names)
                + "</ul>",
                sanitize=False,
            )
        with ui.element("div").classes("wbpoa-aside-extra"):
            ui.html('<h2 class="wbpoa-section-title">Citation graph</h2>', sanitize=False)
            with ui.element("ul").classes("wbpoa-meta-list"):
                for corpus in target.corpora:
                    with ui.element("li"):
                        ui.link(f"Browse {corpus.config.name}", Routes.corpus_url(corpus.key))
