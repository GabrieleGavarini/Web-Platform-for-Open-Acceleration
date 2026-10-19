from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING, Final

from nicegui import run, ui

from wbpoa.chat_ui.agents.chat_target import ChatTargetKind
from wbpoa.chat_ui.app.routes import Routes
from wbpoa.chat_ui.components.neighbour_list import PLURALS
from wbpoa.chat_ui.graph.citation_graph import TYPE_ORDER

if TYPE_CHECKING:
    from wbpoa.chat_ui.agents.types import TargetCatalog
    from wbpoa.chat_ui.graph.models import NodeSummary
    from wbpoa.core.corpus import CorpusWorkspace

NAV_ORDER: Final = ("Recital", "Article", "Annex", "Guideline", "Ruling")
"""The groups of the pane, in the order the Official Journal sets a regulation: recitals first."""

NAV_LIMIT: Final = 20_000
"""Upper bound on the nodes listed for one corpus; every corpus holds far fewer."""


class CorpusNav:
    """
    The citation graph's left pane: a corpus picker over the corpus's contents, one folded group per node type.

    In a corpus of several instruments, each provision group folds its instruments too. The pane
    is drawn once per corpus and stays while the reader moves between the corpus's nodes;
    ``reading.js`` marks the node being read, opens the groups holding it, scrolls the pane to
    it, and keeps the groups a reader opened open. Each corpus's listing is read once and kept,
    since the graph does not change while the app runs.

    Attributes:
        catalog: Lists the corpora and opens their graphs.
    """

    def __init__(self, catalog: TargetCatalog) -> None:
        """
        Args:
            catalog: Lists the corpora and opens their graphs.
        """
        self.catalog = catalog
        self._listings: dict[str, list[NodeSummary]] = {}

    def corpora(self) -> list[CorpusWorkspace]:
        """
        The browsable corpora.

        Returns:
            Every single corpus, in key order.
        """
        return [target.corpora[0] for target in self.catalog.targets() if target.kind == ChatTargetKind.SINGLE]

    async def listing(self, workspace: CorpusWorkspace) -> list[NodeSummary]:
        """
        Every node of the corpus the pane lists, read once per corpus in a worker thread.

        Args:
            workspace: The corpus.

        Returns:
            Its Articles, Recitals, Annexes, guidelines, and rulings, grouped by type and naturally
            ordered; empty, and not kept, when the server is shutting down.

        Raises:
            DriverError: The graph database is unreachable; nothing is kept.
            Neo4jError: The graph database failed the read; nothing is kept.
        """
        cached = self._listings.get(workspace.key)
        if cached is not None:
            return cached
        nodes = await run.io_bound(self._read, workspace)
        if nodes is None:
            return []
        self._listings[workspace.key] = nodes
        return nodes

    def render(self, workspace: CorpusWorkspace, nodes: list[NodeSummary] | None) -> None:
        """
        Draw the pane.

        Args:
            workspace: The corpus shown.
            nodes: Its listing, or None when it could not be read (the pane then says so).
        """
        with ui.element("aside").classes("wbpoa-aside wbpoa-app-aside wbpoa-graph-aside").props('aria-label="Corpus"'):
            ui.select(
                {corpus.key: corpus.config.name for corpus in self.corpora()},
                value=workspace.key,
                label="Corpus",
                on_change=lambda event: ui.navigate.to(Routes.corpus_url(str(event.value))),
            ).props("outlined dense options-dense").classes("w-full")
            if nodes is None:
                ui.label("The graph database is unreachable, so the contents cannot be listed.").classes("wbpoa-error")
                return
            instruments = [instrument.name for instrument in workspace.config.instruments]
            ui.html(self._contents(workspace.key, nodes, instruments), sanitize=False)

    def _read(self, workspace: CorpusWorkspace) -> list[NodeSummary]:
        """Open the corpus's graph and list its nodes; run in a worker thread."""
        return self.catalog.graph(workspace).search("", TYPE_ORDER, NAV_LIMIT)

    @staticmethod
    def _contents(corpus_key: str, nodes: list[NodeSummary], instruments: list[str]) -> str:
        """
        The folded groups as one HTML block; each fold carries a ``data-key`` that reading.js remembers it by.

        With several instruments, a provision group folds one group per instrument, in the manifest's
        order; the first instrument's fold is open, so opening the group shows its numbers at once.
        """
        groups = []
        for node_type in NAV_ORDER:
            group = [node for node in nodes if node.type == node_type]
            if not group:
                continue
            if len(instruments) > 1 and group[0].is_provision:
                folds = []
                for name in instruments:
                    listed = [node for node in group if node.instrument_name == name]
                    if listed:
                        summary = CorpusNav._summary(name, len(listed))
                        links = CorpusNav._links(listed, corpus_key)
                        key = f"{node_type}/{name}"
                        folds.append(
                            CorpusNav._fold(key, "wbpoa-contents-instrument", summary, links, opened=not folds)
                        )
                body = "".join(folds)
            else:
                body = CorpusNav._links(group, corpus_key)
            summary = CorpusNav._summary(PLURALS[node_type], len(group))
            groups.append(CorpusNav._fold(node_type, "wbpoa-contents-group", summary, body))
        return (
            f'<nav class="wbpoa-contents" aria-label="Contents" data-corpus="{escape(corpus_key, quote=True)}">'
            f"{''.join(groups)}</nav>"
        )

    @staticmethod
    def _fold(key: str, kind: str, summary: str, body: str, *, opened: bool = False) -> str:
        """One folded group, its summary over its body."""
        return (
            f'<details class="{kind}" data-key="{escape(key, quote=True)}"{" open" if opened else ""}>'
            f"<summary>{summary}</summary>{body}</details>"
        )

    @staticmethod
    def _summary(label: str, count: int) -> str:
        """A fold's label and count, laid out in a span so the summary keeps its native display, and so its role."""
        return (
            f'<span class="wbpoa-contents-summary"><span class="wbpoa-contents-label">{escape(label)}</span>'
            f'<span class="wbpoa-count">{count}</span></span>'
        )

    @staticmethod
    def _links(nodes: list[NodeSummary], corpus_key: str) -> str:
        """
        One group's links.

        Provisions are set as a grid of their numbers (``6``, ``XIII``), each link named in full for
        screen readers and on hover; guidelines and rulings as a list of names, their file-name
        underscores read as spaces.
        """
        provisions = bool(nodes) and nodes[0].is_provision
        items = []
        for node in nodes:
            name = escape(node.name, quote=True)
            if provisions:
                label = escape(CorpusNav._number(node))
                named = f' aria-label="{name}" title="{name}"'
            else:
                label = escape(node.name.replace("_", " "))
                named = f' title="{name}"'
            href = escape(Routes.node_url(corpus_key, node.id), quote=True)
            items.append(f'<li><a href="{href}"{named}>{label}</a></li>')
        kind = "wbpoa-contents-grid" if provisions else "wbpoa-contents-docs"
        return f'<ul class="{kind}">{"".join(items)}</ul>'

    @staticmethod
    def _number(node: NodeSummary) -> str:
        """A provision's number as its name gives it, e.g. ``6`` for ``Article 6``; the whole name if it has none."""
        head, _, number = node.name.rpartition(" ")
        return number if head.casefold() == node.type.casefold() and number else node.name
