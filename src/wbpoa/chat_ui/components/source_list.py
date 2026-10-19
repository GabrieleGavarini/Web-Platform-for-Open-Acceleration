from __future__ import annotations

import html
from typing import TYPE_CHECKING

from nicegui import ui

if TYPE_CHECKING:
    from wbpoa.chat_ui.sources.models import UsedSource


class SourceList:
    """The footnotes under an answer: the provisions it cites, each the target of its citations and a link to the graph."""

    @staticmethod
    def render(sources: list[UsedSource]) -> None:
        """
        Draw the footnotes.

        Args:
            sources: The cited provisions, in footnote order; nothing is drawn but a note when empty.
        """
        if not sources:
            ui.label("This answer cites no provision of the corpus.").classes("wbpoa-no-sources")
            return
        items = "".join(
            f'<li class="wbpoa-footnote" id="{html.escape(source.anchor, quote=True)}">'
            f'<span class="wbpoa-footnote-number">({source.number})</span>'
            "<span>"
            f'<a href="{html.escape(source.url, quote=True)}">{html.escape(source.provision)}</a> '
            f'<span class="instrument">{html.escape(source.instrument_name)}</span>'
            f'<span class="celex">CELEX {html.escape(source.celex_id)}</span>'
            f'<span class="excerpt">{html.escape(source.excerpt)}</span>'
            "</span></li>"
            for source in sources
        )
        ui.html(f'<ol aria-label="Sources cited">{items}</ol>', sanitize=False).classes("wbpoa-footnotes")
