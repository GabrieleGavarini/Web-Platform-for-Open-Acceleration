from __future__ import annotations

import html
import re
from typing import TYPE_CHECKING, Final

from nicegui import ui

from wbpoa.chat_ui.app.routes import Routes

if TYPE_CHECKING:
    from uuid import UUID

    from wbpoa.chat_ui.graph.models import CommunityView

SUMMARY_EXCERPT: Final = 320
"""Characters of a community's summary shown before the rest folds away."""

_WHITESPACE: Final = re.compile(r"\s+")


class CommunityList:
    """The communities a provision belongs to, finest first: each one's title, summary, and provisions."""

    @staticmethod
    def render(communities: list[CommunityView], corpus_key: str, current: UUID) -> None:
        """
        Draw the provision's communities; nothing is drawn when it has none.

        Args:
            communities: The communities, finest first.
            corpus_key: The corpus the member links point into.
            current: The provision the page is about, marked rather than linked among the members.
        """
        if not communities:
            return
        with ui.element("section").classes("wbpoa-communities").props('aria-label="Communities"'):
            with ui.element("div").classes("wbpoa-group-head"):
                ui.html("<h2>Communities</h2>", sanitize=False)
                ui.label(str(len(communities))).classes("wbpoa-count")
            ui.label(
                "Topical groups the co-citation clustering placed this provision in, from the closest to the broadest."
            ).classes("wbpoa-meta")
            for community in communities:
                CommunityList._community(community, corpus_key, current)

    @staticmethod
    def _community(community: CommunityView, corpus_key: str, current: UUID) -> None:
        """Draw one community: its level and size, title, summary excerpt, and the rest folded."""
        size = len(community.members)
        with ui.element("article").classes("wbpoa-community"):
            ui.html(
                f'<p class="wbpoa-community-level">Level {community.layer} · {size} provisions</p>'
                f"<h3>{html.escape(community.name)}</h3>"
                f'<p class="wbpoa-community-summary">{html.escape(CommunityList.excerpt(community.summary))}</p>',
                sanitize=False,
            )
            with ui.expansion(f"Full summary and the {size} provisions").props("dense"):
                ui.label(community.summary).classes("wbpoa-community-full")
                ui.html(CommunityList.members(community, corpus_key, current), sanitize=False)

    @staticmethod
    def excerpt(summary: str, limit: int = SUMMARY_EXCERPT) -> str:
        """
        Shorten a summary to its opening, cut at a word boundary.

        Args:
            summary: The community's summary.
            limit: Characters to keep at most, before the ellipsis.

        Returns:
            The summary on one line, whole when it fits, otherwise cut and ended with an ellipsis.
        """
        text = _WHITESPACE.sub(" ", summary).strip()
        if len(text) <= limit:
            return text
        return text[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "…"

    @staticmethod
    def members(community: CommunityView, corpus_key: str, current: UUID) -> str:
        """
        Render a community's provisions as links to their pages.

        Args:
            community: The community.
            corpus_key: The corpus the links point into.
            current: The provision the page is about, set in bold and not linked.

        Returns:
            A ``<ul>`` of escaped links.
        """
        items = "".join(
            f'<li aria-current="page"><strong>{html.escape(member.name)}</strong></li>'
            if member.id == current
            else f'<li><a href="{html.escape(Routes.node_url(corpus_key, member.id), quote=True)}">'
            f"{html.escape(member.name)}</a></li>"
            for member in community.members
        )
        return f'<ul class="wbpoa-link-columns">{items}</ul>'
