from __future__ import annotations

import html
from typing import TYPE_CHECKING, Final

from nicegui import ui

from wbpoa.chat_ui.app.routes import Routes

if TYPE_CHECKING:
    from wbpoa.chat_ui.graph.models import NeighbourGroup, NodeSummary

SHOWN: Final = 8
"""Neighbours listed per type before the rest fold away."""

PLURALS: Final = {
    "Article": "Articles",
    "Recital": "Recitals",
    "Annex": "Annexes",
    "Guideline": "Guidelines",
    "Ruling": "Rulings",
}


class NeighbourList:
    """One side of a node's neighbourhood ("Cites" or "Cited by"), grouped by type, every entry a link to its page."""

    @staticmethod
    def render(title: str, groups: list[NeighbourGroup], corpus_key: str, *, empty: str, show_instrument: bool) -> None:
        """
        Draw one side of the neighbourhood.

        Args:
            title: The side's heading, e.g. ``Cited by``.
            groups: The neighbours, grouped by type.
            corpus_key: The corpus the links point into.
            empty: What to say when there are no neighbours.
            show_instrument: Whether to name each provision's instrument, for corpora with several.
        """
        total = sum(len(group.nodes) for group in groups)
        with ui.element("section").classes("wbpoa-neighbour-side"):
            with ui.element("div").classes("wbpoa-group-head"):
                ui.html(f"<h3>{html.escape(title)}</h3>", sanitize=False)
                ui.label(str(total)).classes("wbpoa-count")
            if not groups:
                ui.label(empty).classes("wbpoa-empty-note")
                return
            with ui.element("div").classes("wbpoa-neighbours"):
                for group in groups:
                    NeighbourList._group(group, corpus_key, show_instrument=show_instrument)

    @staticmethod
    def _group(group: NeighbourGroup, corpus_key: str, *, show_instrument: bool) -> None:
        """Draw one type's neighbours, the first few listed and the rest folded."""
        with ui.element("div").classes("wbpoa-neighbour-group"):
            ui.html(
                f"<h4><span>{PLURALS.get(group.type, group.type)}</span>"
                f'<span class="wbpoa-count">{len(group.nodes)}</span></h4>',
                sanitize=False,
            )
            shown, rest = group.nodes[:SHOWN], group.nodes[SHOWN:]
            ui.html(NeighbourList.links(shown, corpus_key, show_instrument=show_instrument), sanitize=False)
            if rest:
                with ui.expansion(f"Show {len(rest)} more").props("dense"):
                    ui.html(NeighbourList.links(rest, corpus_key, show_instrument=show_instrument), sanitize=False)

    @staticmethod
    def links(nodes: list[NodeSummary], corpus_key: str, *, show_instrument: bool = False) -> str:
        """
        Render nodes as a list of links to their pages.

        Args:
            nodes: The nodes to list.
            corpus_key: The corpus the links point into.
            show_instrument: Whether to name each provision's instrument after its link.

        Returns:
            A ``<ul>`` of escaped links.
        """
        items = "".join(
            f'<li><a href="{html.escape(Routes.node_url(corpus_key, node.id), quote=True)}">{html.escape(node.name)}</a>'
            + (
                f' <span class="wbpoa-meta">{html.escape(node.instrument_name)}</span>'
                if show_instrument and node.instrument_name
                else ""
            )
            + "</li>"
            for node in nodes
        )
        return f"<ul>{items}</ul>"
