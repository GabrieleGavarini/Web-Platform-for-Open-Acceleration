from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final
from weakref import WeakKeyDictionary

from nicegui import ui

from wbpoa.chat_ui.app import routes
from wbpoa.chat_ui.components.theme import LOGO_URL

if TYPE_CHECKING:
    from nicegui.client import Client
    from nicegui.element import Element
    from nicegui.elements.label import Label
    from nicegui.elements.link import Link

NAV: Final = (("Chat", routes.CHAT), ("Citation graph", routes.GRAPH), ("About", routes.ABOUT))


class PageFrame:
    """
    The masthead with logo and navigation, the running head under it, and each page's content column.

    The masthead and the running head are drawn once per browser tab, above the routed pages, and
    stay while the reader moves between the chat, the graph, and the about page; each page only
    marks its navigation link and sets the running head's facts.
    """

    _frames: ClassVar[WeakKeyDictionary[Client, PageFrame]] = WeakKeyDictionary()

    def __init__(self) -> None:
        """Draw the skip link, the masthead, and an empty running head into the current slot."""
        ui.link("Skip to content", "#wbpoa-main").classes("wbpoa-skip")
        self._links: dict[str, Link] = {}
        with ui.element("header").classes("wbpoa-masthead"):
            with ui.link(target=routes.CHAT).classes("wbpoa-brand").props('aria-label="WBPOA home"'):
                ui.html(
                    f'<img src="{LOGO_URL}" alt="Scuola Superiore Sant&#39;Anna" width="140" height="38">',
                    sanitize=False,
                )
                ui.html("WBPOA<small>Answers over EU law, grounded in its citations</small>", sanitize=False).classes(
                    "wbpoa-brand-name"
                )
            with ui.element("nav").classes("wbpoa-nav").props('aria-label="Main"'):
                for label, target in NAV:
                    self._links[target] = ui.link(label, target)
        with ui.element("div").classes("wbpoa-running-head").props('role="note"'):
            self._facts = ui.element("div").classes("flex gap-x-6 gap-y-1 flex-wrap")
            self._trailing: Label = ui.label("").classes("wbpoa-running-trailing")

    @classmethod
    def mount(cls) -> PageFrame:
        """
        Draw the frame for the current browser tab, above where its pages are routed.

        Returns:
            The frame, which `render` finds again for the tab's pages.
        """
        frame = cls()
        cls._frames[ui.context.client] = frame
        return frame

    @staticmethod
    def render(active: str, running_head: list[str], *, trailing: str = "EN") -> Element:
        """
        Set the frame for one page and open its content column.

        Args:
            active: The nav route of the current page, e.g. ``routes.GRAPH``.
            running_head: Short facts set in the running head, left to right, like the Official
                Journal's: the corpus, the instrument, the CELEX number.
            trailing: Text at the right end of the running head.

        Returns:
            The content column, for the page to fill.

        Raises:
            KeyError: The tab's frame was not mounted.
        """
        PageFrame._frames[ui.context.client].show(active, running_head, trailing=trailing)
        return ui.element("main").classes("wbpoa-page").props('id="wbpoa-main" tabindex="-1"')

    def show(self, active: str, running_head: list[str], *, trailing: str = "EN") -> None:
        """
        Mark the current page's navigation link and set the running head.

        Args:
            active: The nav route of the current page.
            running_head: The running head's facts, left to right; the first is set in bold.
            trailing: Text at the right end of the running head.
        """
        for target, link in self._links.items():
            if target == active:
                link.props('aria-current="page"')
            else:
                link.props(remove="aria-current")
        self._facts.clear()
        with self._facts:
            for index, fact in enumerate(running_head):
                ui.label(fact).classes("wbpoa-running-lead" if index == 0 else "")
        self._trailing.text = trailing
