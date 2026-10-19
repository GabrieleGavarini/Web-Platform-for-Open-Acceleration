from __future__ import annotations

import html
from collections import Counter
from typing import TYPE_CHECKING, Final

from nicegui import ui

from wbpoa.core.generation.research_observer import ResearchEventKind

if TYPE_CHECKING:
    from wbpoa.chat_ui.sources.models import FoundPassage, ResearchStep

_NOUNS: Final = {
    ResearchEventKind.SEARCH: ("search", "searches"),
    ResearchEventKind.LOOKUP: ("lookup", "lookups"),
    ResearchEventKind.DELEGATION: ("sub-agent call", "sub-agent calls"),
}
_LABELS: Final = {
    ResearchEventKind.SEARCH: "Searched",
    ResearchEventKind.LOOKUP: "Looked up",
    ResearchEventKind.DELEGATION: "Delegated to",
}
_RUNNING_LABELS: Final = {
    ResearchEventKind.SEARCH: "Searching",
    ResearchEventKind.LOOKUP: "Looking up",
    ResearchEventKind.DELEGATION: "Delegating to",
}
_PROVISIONS: Final = ("Article", "Recital", "Annex")
# The stylesheet defines this many agent colours; further agents reuse them in order.
_AGENT_COLOURS: Final = 4


class ResearchTrail:
    """What the agent did to reach the answer: its searches with what they found, its lookups, its sub-agent calls."""

    @staticmethod
    def summary(steps: list[ResearchStep]) -> str:
        """
        Describe the research in a few words.

        Args:
            steps: The research steps.

        Returns:
            E.g. ``4 searches, 2 lookups``; ``No research recorded`` when there are none.
        """
        counts = Counter(step.kind for step in steps)
        parts = [
            f"{counts[kind]} {_NOUNS[kind][0] if counts[kind] == 1 else _NOUNS[kind][1]}"
            for kind in ResearchEventKind
            if counts[kind]
        ]
        return ", ".join(parts) if parts else "No research recorded"

    @staticmethod
    def steps_html(steps: list[ResearchStep]) -> str:
        """
        Render the steps as an ordered list, each search followed by what it found.

        Args:
            steps: The research steps, in the order they were made.

        Returns:
            An escaped ``<ol>``: provisions found are links to the graph; guidelines and rulings
            are counted, since they are research aids rather than sources. A step still running
            says what is being done (e.g. ``Searching``) and has nothing found yet. A delegation
            names the sub agent it was given to in a tag of that agent's colour, and the searches
            and lookups that agent then made are nested under it, in a list ruled in its colour. A
            sub agent's step with no delegation before it stays at the top level, tagged instead.
        """
        groups: list[tuple[ResearchStep, list[ResearchStep]]] = []
        delegations: dict[str, list[ResearchStep]] = {}
        for step in steps:
            if step.kind == ResearchEventKind.DELEGATION and step.agent is not None:
                groups.append((step, []))
                delegations[step.agent] = groups[-1][1]
            elif step.agent is not None and step.agent in delegations:
                delegations[step.agent].append(step)
            else:
                groups.append((step, []))
        items = "".join(ResearchTrail._step_html(step, children) for step, children in groups)
        return f'<ol class="wbpoa-steps">{items}</ol>'

    @staticmethod
    def render(steps: list[ResearchStep]) -> None:
        """
        Draw the research trail above an answer, open, so the reader can fold it away.

        The live research and the finished reply draw it alike, so the answer below does not
        move when the reply lands.

        Args:
            steps: The research steps, in the order they were made; nothing is drawn without any.
        """
        if not steps:
            return
        with (
            ui.expansion(f"Research: {ResearchTrail.summary(steps)}", value=True).classes("wbpoa-trail").props("dense")
        ):
            ui.html(ResearchTrail.steps_html(steps), sanitize=False)

    @staticmethod
    def _step_html(step: ResearchStep, children: list[ResearchStep], *, nested: bool = False) -> str:
        """One step: who did it in a pairing, what was (or is being) done, what it found, then its sub steps."""
        label = _RUNNING_LABELS[step.kind] if step.running else _LABELS[step.kind]
        classes = ["step-running"] if step.running else []
        agent = html.escape(step.agent or step.corpus_key)
        if step.kind == ResearchEventKind.DELEGATION:
            classes += ["step-delegation", ResearchTrail._agent_class(step)]
            head = f'<span class="step-kind">{label}</span> <span class="step-agent">{agent} agent</span>'
        elif step.agent is not None:
            classes.append(ResearchTrail._agent_class(step))
            tag = "" if nested else f'<span class="step-agent step-agent--sub">{agent}</span> '
            head = f'{tag}<span class="step-kind">{label}</span>'
        else:
            head = f'<span class="step-kind">{label}</span>'
        head += f" {html.escape(step.text)}"
        opening = f'<li class="{" ".join(classes)}">' if classes else "<li>"
        found = "" if step.running else ResearchTrail._found_html(step)
        sub = ""
        if children:
            items = "".join(ResearchTrail._step_html(child, [], nested=True) for child in children)
            sub = f'<ol class="step-subtasks" aria-label="Research by the {agent} agent">{items}</ol>'
        error = f'<span class="step-error">Failed: {html.escape(step.error)}</span>' if step.error else ""
        return f"{opening}{head}{found}{sub}{error}</li>"

    @staticmethod
    def _agent_class(step: ResearchStep) -> str:
        """The class giving a step its agent's colour."""
        return f"step-agent-{step.agent_slot % _AGENT_COLOURS + 1}"

    @staticmethod
    def _found_html(step: ResearchStep) -> str:
        """The passages a step returned: provisions as links, documents counted; nothing for a failed step."""
        if step.kind == ResearchEventKind.DELEGATION or step.error:
            return ""
        if not step.found:
            return '<span class="step-found step-none">Nothing relevant found.</span>'
        provisions = [passage for passage in step.found if passage.type in _PROVISIONS]
        documents = len(step.found) - len(provisions)
        links = ", ".join(ResearchTrail._link(passage) for passage in provisions)
        if documents:
            noun = "guideline or ruling" if documents == 1 else "guidelines and rulings"
            links = f"{links}{'; ' if links else ''}{documents} {noun}"
        return f'<span class="step-found">{links}</span>'

    @staticmethod
    def _link(passage: FoundPassage) -> str:
        """A provision found, linked to its graph page."""
        title = html.escape(f"{passage.name}, {passage.instrument_name}" if passage.instrument_name else passage.name)
        return f'<a href="{html.escape(passage.url, quote=True)}" title="{title}">{html.escape(passage.name)}</a>'
