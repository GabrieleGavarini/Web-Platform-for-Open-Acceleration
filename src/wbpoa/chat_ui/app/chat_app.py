# No `from __future__ import annotations` here: NiceGUI injects `PageArguments` into a sub page
# builder by comparing the parameter's evaluated annotation with the class itself.
from collections.abc import Callable

from nicegui import app, ui
from nicegui.page_arguments import PageArguments

from wbpoa.chat_ui.agents.chat_service import ChatService
from wbpoa.chat_ui.agents.types import AgentSource, TargetCatalog
from wbpoa.chat_ui.app import routes
from wbpoa.chat_ui.components.corpus_nav import CorpusNav
from wbpoa.chat_ui.components.page_frame import PageFrame
from wbpoa.chat_ui.components.theme import STATIC_DIR, Theme
from wbpoa.chat_ui.pages.about_page import AboutPage
from wbpoa.chat_ui.pages.chat_page import ChatPage
from wbpoa.chat_ui.pages.graph_corpus_page import GraphCorpusPage
from wbpoa.chat_ui.pages.graph_node_page import GraphNodePage


class ChatApp:
    """
    The NiceGUI application: one root page that routes, client side, to the chat, the graph, and the about page.

    Client-side routing keeps the tab's NiceGUI client alive while moving between pages, so the
    conversation held in its client storage survives a trip into the citation graph; the masthead
    and running head are drawn once, above the routed pages, and stay while moving between them.

    Attributes:
        catalog: Lists targets and opens graphs.
    """

    def __init__(
        self, catalog: TargetCatalog, agents: AgentSource, *, on_shutdown: Callable[[], None] | None = None
    ) -> None:
        """
        Args:
            catalog: Lists targets and opens graphs, normally the `AgentPool`.
            agents: Provides each target's agent, normally the same `AgentPool`.
            on_shutdown: Called when the server stops, e.g. to close the pool's drivers.
        """
        self.catalog = catalog
        self._service = ChatService(agents)
        self._graph_corpus = GraphCorpusPage(CorpusNav(catalog), GraphNodePage(catalog))
        self._about = AboutPage(catalog)
        self._on_shutdown = on_shutdown

    def root(self) -> None:
        """Build the root page every URL is served from: the theme and the frame, then the routed page."""
        Theme.apply()
        PageFrame.mount()
        ui.sub_pages(
            {
                routes.CHAT: self._render_chat,
                routes.GRAPH: self._graph_corpus.render,
                routes.GRAPH_CORPUS: self._graph_corpus.render,
                routes.ABOUT: self._about.render,
            }
        )

    def _render_chat(self, args: PageArguments) -> None:
        """Route the chat to a fresh page instance, which holds this render's client state."""
        ChatPage(self.catalog, self._service).render(args)

    def run(self, *, host: str, port: int) -> None:
        """
        Serve the app until interrupted.

        Args:
            host: Interface to bind.
            port: Port to listen on.
        """
        Theme.serve_static()
        if self._on_shutdown is not None:
            app.on_shutdown(self._on_shutdown)
        ui.run(
            self.root,
            host=host,
            port=port,
            title="WBPOA",
            favicon=STATIC_DIR / "logo-sssa.svg",
            reload=False,
            show=False,
            storage_secret=None,
        )
