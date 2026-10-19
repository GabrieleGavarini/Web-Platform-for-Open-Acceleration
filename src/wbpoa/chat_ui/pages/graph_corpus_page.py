from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

from neo4j.exceptions import DriverError, Neo4jError
from nicegui import ui
from structlog import get_logger

from wbpoa.chat_ui.app import routes
from wbpoa.chat_ui.app.routes import Routes
from wbpoa.chat_ui.components.corpus_nav import NAV_ORDER
from wbpoa.chat_ui.components.neighbour_list import PLURALS
from wbpoa.chat_ui.components.page_frame import PageFrame

if TYPE_CHECKING:
    from wbpoa.chat_ui.components.corpus_nav import CorpusNav
    from wbpoa.chat_ui.graph.models import NodeSummary
    from wbpoa.chat_ui.pages.graph_node_page import GraphNodePage
    from wbpoa.core.corpus import CorpusWorkspace

log = get_logger()


class GraphCorpusPage:
    """
    A corpus of the citation graph: the navigation pane beside an account of the corpus or one of its nodes.

    The page is the graph's shell: the masthead, the running head, and the pane are drawn once per
    corpus, and the corpus's nodes open in nested sub pages beside them, so moving from node to
    node redraws only the node. The graph has no page of its own: ``/graph`` shows the first corpus.

    Attributes:
        nav: The left pane, shared with the node pages.
        node_page: Draws a node beside the pane.
    """

    def __init__(self, nav: CorpusNav, node_page: GraphNodePage) -> None:
        """
        Args:
            nav: The left pane, shared with the node pages.
            node_page: Draws a node beside the pane.
        """
        self.nav = nav
        self.node_page = node_page

    async def render(self, corpus: str | None = None) -> None:
        """
        Build one corpus's shell, with its account or the node the rest of the path names.

        Args:
            corpus: The corpus key from the path; None for ``/graph``, which shows the first corpus.
        """
        corpora = self.nav.corpora()
        workspace = corpora[0] if corpus is None and corpora else self.nav.catalog.corpus(corpus or "")
        if workspace is None:
            self._not_found(corpus or "")
            return
        try:
            nodes = await self.nav.listing(workspace)
        except (DriverError, Neo4jError):
            log.exception("chat_ui_graph_listing_failed", corpus=workspace.key)
            nodes = None
        instruments = [instrument.name for instrument in workspace.config.instruments]
        with (
            PageFrame.render(routes.GRAPH, [workspace.config.name, *instruments]).classes("wbpoa-page--app"),
            ui.element("div").classes("wbpoa-split"),
        ):
            self.nav.render(workspace, nodes)
            with ui.element("div").classes("wbpoa-graph-main"):
                if corpus is None:
                    self.overview(workspace, nodes)
                else:
                    ui.sub_pages(
                        {"/": self.overview, "/{node_id}": self.node_page.render},
                        data={"workspace": workspace, "nodes": nodes},
                    )

    def overview(self, workspace: CorpusWorkspace, nodes: list[NodeSummary] | None) -> None:
        """
        Draw the account of the corpus beside the pane.

        Args:
            workspace: The corpus.
            nodes: Its listing, or None when it could not be read.
        """
        ui.page_title(f"{workspace.config.name} citation graph · WBPOA")
        with ui.element("section").classes("wbpoa-corpus").props('aria-label="Overview"'):
            ui.html(self._overview(workspace, nodes), sanitize=False).classes("wbpoa-doc")

    @staticmethod
    def _overview(workspace: CorpusWorkspace, nodes: list[NodeSummary] | None) -> str:
        """The corpus's name, its instruments with their official texts, and how many nodes of each type it holds."""
        instruments = "".join(
            f"<tr><td>{escape(instrument.name)}</td><td>{escape(instrument.role.value.capitalize())}</td>"
            f'<td><a href="{escape(Routes.eur_lex_url(instrument.celex_id), quote=True)}" target="_blank" '
            f'rel="noopener">{escape(instrument.celex_id)}</a></td></tr>'
            for instrument in workspace.config.instruments
        )
        parts = [
            f"<h1>{escape(workspace.config.name)}</h1>",
            (
                '<p class="lede">Open a group of the contents to list its provisions or documents. Each opens '
                "on its own page, with what it cites and what cites it.</p>"
            ),
            "<h2>Instruments</h2>",
            (
                "<table><thead><tr><th>Instrument</th><th>Role</th><th>CELEX</th></tr></thead>"
                f"<tbody>{instruments}</tbody></table>"
            ),
        ]
        if nodes:
            counts = "".join(
                f'<tr><td>{PLURALS[node_type]}</td><td class="num">{count}</td></tr>'
                for node_type in NAV_ORDER
                if (count := sum(node.type == node_type for node in nodes))
            )
            parts.append(
                '<h2>In the graph</h2><table class="wbpoa-counts"><thead><tr><th>Type</th><th class="num">Nodes</th>'
                f"</tr></thead><tbody>{counts}</tbody></table>"
            )
        return "".join(parts)

    @staticmethod
    def _not_found(corpus: str) -> None:
        """The page for an unknown corpus key."""
        ui.page_title("Corpus not found · WBPOA")
        with PageFrame.render(routes.GRAPH, ["Citation graph"]), ui.element("div").classes("wbpoa-doc"):
            ui.html("<h1>Corpus not found</h1>", sanitize=False)
            ui.html(
                f"<p>No corpus has the key <code>{escape(corpus)}</code>. "
                f'<a href="{routes.GRAPH}">Browse the citation graph</a>.</p>',
                sanitize=False,
            )
