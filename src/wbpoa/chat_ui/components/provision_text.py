from __future__ import annotations

import html
import re
from typing import Final

DASH: Final = re.compile(r"^[\N{EM DASH}\N{EN DASH}-]$")
"""A line holding only the indent dash of an item, as the Official Journal's text arrives."""

POINT: Final = re.compile(r"^(?:\(\d+\)|\d+\.)\s")
"""A numbered point or paragraph: ``(1) …`` or ``1. …``."""

SUB: Final = re.compile(r"^\([a-z]{1,4}\)\s")
"""A lettered or roman sub-point: ``(a) …``, ``(iv) …``."""


class ProvisionText:
    """The text of an Article, Recital, or Annex, set as paragraphs: numbered points spaced apart, items indented."""

    @staticmethod
    def html(text: str) -> str:
        """
        Render a provision's plain text, one line per block, as escaped paragraphs.

        A line that is only a dash is joined to the line after it, the item it introduces.

        Args:
            text: The provision's text, its blocks separated by newlines.

        Returns:
            ``<p>`` elements: ``point`` for numbered points, ``sub`` for lettered ones, ``dash`` for
            dashed items, plain otherwise; an empty string for empty text.
        """
        lines = [line.strip() for line in text.splitlines()]
        blocks: list[str] = []
        dash: str | None = None
        for line in lines:
            if not line:
                continue
            if DASH.match(line):
                if dash is not None:
                    blocks.append(ProvisionText._block("dash", dash))
                dash = line
                continue
            if dash is not None:
                blocks.append(ProvisionText._block("dash", f"{dash} {line}"))
                dash = None
                continue
            kind = "point" if POINT.match(line) else "sub" if SUB.match(line) else None
            blocks.append(ProvisionText._block(kind, line))
        if dash is not None:
            blocks.append(ProvisionText._block("dash", dash))
        return "".join(blocks)

    @staticmethod
    def _block(kind: str | None, line: str) -> str:
        """One escaped paragraph, classed by its kind."""
        opening = f'<p class="{kind}">' if kind else "<p>"
        return f"{opening}{html.escape(line)}</p>"
