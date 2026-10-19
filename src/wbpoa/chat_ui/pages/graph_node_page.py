from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING
from uuid import UUID

from neo4j.exceptions import DriverError, Neo4jError
from nicegui import run, ui
from structlog import get_logger

from wbpoa.chat_ui.app.routes import Routes
from wbpoa.chat_ui.components.community_list import CommunityList
from wbpoa.chat_ui.components.neighbour_list import NeighbourList
from wbpoa.chat_ui.components.provision_text import ProvisionText
from wbpoa.chat_ui.components.theme import GRAPH_PALETTE
from wbpoa.chat_ui.graph.ego_graph_chart import EgoGraphChart

if TYPE_CHECKING:
    from nicegui.events import EChartPointClickEventArguments

    from wbpoa.chat_ui.agents.types import TargetCatalog
    from wbpoa.chat_ui.graph.models import GraphNodeView
    from wbpoa.core.corpus import CorpusWorkspace

log = get_logger()


class GraphNodePage:
    """
    One node of the citation graph, set like an article of the Official Journal, with its citations beside it.

    It is drawn in the corpus's shell (`GraphCorpusPage`), beside the navigation pane, and is the
    only part redrawn when the reader moves to another node of the corpus. The aside on the right
    draws the provisions cited together with the node and lists what cites it and what it cites;
    every neighbour, in the drawing and in the lists, links to its own page.

    Attributes:
        catalog: Opens the corpus graphs.
    """

    def __init__(self, catalog: TargetCatalog) -> None:
        """
        Args:
            catalog: Opens the corpus graphs.
        """
        self.catalog = catalog

    async def render(self, workspace: CorpusWorkspace, node_id: str) -> None:
        """
        Draw one node beside the corpus's pane.

        Args:
            workspace: The corpus, from the shell.
            node_id: The node id from the path.
        """
        try:
            parsed = UUID(node_id)
        except ValueError:
            self._missing(workspace)
            return
        try:
            view = await run.io_bound(self._read, workspace, parsed)
        except (DriverError, Neo4jError):
            log.exception("chat_ui_graph_node_failed", corpus=workspace.key, node_id=node_id)
            with ui.element("div").classes("wbpoa-corpus"):
                ui.label("The graph database is unreachable, so this node cannot be read.").classes("wbpoa-error")
            return
        if view is None:
            self._missing(workspace)
            return
        ui.page_title(f"{view.node.name} · {view.corpus_name} · WBPOA")
        self._reading(view, several=len(workspace.config.instruments) > 1)

    def _reading(self, view: GraphNodeView, *, several: bool) -> None:
        """The node's text, with its communities beneath, and its citations beside it."""
        with ui.element("div").classes("wbpoa-reading wbpoa-reading--app"):
            with ui.element("div").classes("wbpoa-reading-main"):
                with ui.element("article").classes("wbpoa-article"):
                    self._head(view)
                    self._body(view)
                CommunityList.render(view.communities, view.corpus_key, view.node.id)
            with ui.element("aside").classes("wbpoa-reading-side").props('aria-label="Citations"'):
                self._ego_graph(view)
                NeighbourList.render(
                    "Cited by",
                    view.cited_by,
                    view.corpus_key,
                    empty="Nothing in the corpus cites it.",
                    show_instrument=several,
                )
                NeighbourList.render(
                    "Cites",
                    view.cites,
                    view.corpus_key,
                    empty="It cites nothing in the corpus.",
                    show_instrument=several,
                )

    def _read(self, workspace: CorpusWorkspace, node_id: UUID) -> GraphNodeView | None:
        """Open the corpus's graph and read the node; run in a worker thread, as both may wait on I/O."""
        return self.catalog.graph(workspace).node(node_id)

    @staticmethod
    def _head(view: GraphNodeView) -> None:
        """The heading, OJ style: the provision's name in italics over its title, and the instrument beneath."""
        lines = []
        if view.title is not None:
            lines.append(f'<p class="kind">{escape(view.node.name)}</p><h1>{escape(view.title)}</h1>')
        else:
            lines.append(f"<h1>{escape(view.node.name)}</h1>")
        if view.node.instrument_name is not None:
            source = escape(view.node.instrument_name)
            if view.celex_id:
                url = escape(Routes.eur_lex_url(view.celex_id), quote=True)
                source = f'<a href="{url}" target="_blank" rel="noopener">{source}</a>'
            lines.append(f'<p class="instrument">{source}</p>')
        else:
            lines.append(f'<p class="instrument">{escape(view.node.type)}</p>')
        lines.append(GraphNodePage._counts(view))
        ui.html("".join(lines), sanitize=False).classes("wbpoa-article-head")

    @staticmethod
    def _counts(view: GraphNodeView) -> str:
        """A line under the heading counting the node's citations and communities."""
        counts = [f"Cited by {view.cited_by_count}", f"Cites {view.cites_count}"]
        if view.communities:
            count = len(view.communities)
            counts.append(f"{count} {'community' if count == 1 else 'communities'}")
        return f'<p class="wbpoa-counts">{"".join(f"<span>{count}</span>" for count in counts)}</p>'

    @staticmethod
    def _body(view: GraphNodeView) -> None:
        """A provision's full text, set in paragraphs; a document's summary, its full text folded below."""
        if view.node.is_provision:
            ui.html(ProvisionText.html(view.text or view.summary), sanitize=False).classes("wbpoa-article-text")
            return
        ui.label(view.summary).classes("wbpoa-document-summary")
        if view.text:
            with ui.expansion("Full text").props("dense"):
                ui.label(view.text).classes("wbpoa-document-text")

    @staticmethod
    def _ego_graph(view: GraphNodeView) -> None:
        """The provisions cited together with the node, drawn, each tie as thick as it is heavy; clicking one opens its page."""
        graph = EgoGraphChart(GRAPH_PALETTE).build(view)
        with ui.element("section").classes("wbpoa-ego-block"):
            ui.html('<h2 class="wbpoa-section-title">Co-citations</h2>', sanitize=False)
            if not any(graph.links):
                ui.label("No provision is cited together with it.").classes("wbpoa-empty-note")
                return

            def open_node(event: EChartPointClickEventArguments) -> None:
                link = graph.link_at(event.data_type, event.data_index)
                if link is not None:
                    ui.navigate.to(link)

            ui.echart(graph.options, on_point_click=open_node).classes("wbpoa-ego w-full").props(
                'aria-label="Drawing of the provisions cited together with it, each line as thick as the tie is strong"'
            )

    @staticmethod
    def _missing(workspace: CorpusWorkspace) -> None:
        """What stands beside the pane for a node that is not in the corpus."""
        ui.page_title("Not found · WBPOA")
        with ui.element("section").classes("wbpoa-corpus"), ui.element("div").classes("wbpoa-doc"):
            ui.html("<h1>Not in the graph</h1>", sanitize=False)
            ui.html(
                f"<p>This link does not point to an Article, Recital, Annex, guideline, or ruling of "
                f"{escape(workspace.config.name)}. Pick one from the contents on the left, or "
                f'<a href="{escape(Routes.corpus_url(workspace.key), quote=True)}">see the corpus</a>.</p>',
                sanitize=False,
            )
