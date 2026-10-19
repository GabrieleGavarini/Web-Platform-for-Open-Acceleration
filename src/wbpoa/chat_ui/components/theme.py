from pathlib import Path
from typing import Final

from nicegui import app, ui

from wbpoa.chat_ui.graph.ego_graph_chart import EgoGraphPalette

STATIC_DIR: Final = Path(__file__).resolve().parent.parent / "static"
STATIC_URL: Final = "/wbpoa-static"
LOGO_URL: Final = f"{STATIC_URL}/logo-sssa.svg"

_FONTS: Final = (
    "https://fonts.googleapis.com/css2?"
    "family=Source+Code+Pro:wght@400&"
    "family=Source+Sans+3:wght@400;600&"
    "family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;0,8..60,700;1,8..60,400&display=swap"
)

GRAPH_PALETTE: Final = EgoGraphPalette(focus="#c8202c", provision="#5a6e9f", edge="#8a8f9c", label="#7d818c")
"""Mid-tone colours that read on both the light and the dark paper, since the chart is drawn server side."""


class Theme:
    """The UI's look: fonts, colour tokens (light and dark, following the system), and Quasar brand colours."""

    @staticmethod
    def serve_static() -> None:
        """Serve the stylesheet, page script, and logo; call once at app start."""
        app.add_static_files(STATIC_URL, STATIC_DIR)

    @staticmethod
    def stylesheet_url() -> str:
        """
        The stylesheet's URL, versioned by its modification time so browsers never keep a stale copy.

        Returns:
            E.g. ``/wbpoa-static/theme.css?v=1760000000``.
        """
        return Theme._versioned("theme.css")

    @staticmethod
    def script_url() -> str:
        """
        The page script's URL (a node page's scroll sync), versioned like the stylesheet.

        Returns:
            E.g. ``/wbpoa-static/reading.js?v=1760000000``.
        """
        return Theme._versioned("reading.js")

    @staticmethod
    def _versioned(name: str) -> str:
        """A static file's URL with its modification time as the version."""
        return f"{STATIC_URL}/{name}?v={(STATIC_DIR / name).stat().st_mtime_ns}"

    @staticmethod
    def apply() -> None:
        """Load the fonts, stylesheet, and page script into the current page and follow the system's light or dark mode."""
        ui.add_head_html(
            '<link rel="preconnect" href="https://fonts.googleapis.com">'
            '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
            f'<link rel="stylesheet" href="{_FONTS}">'
            f'<link rel="stylesheet" href="{Theme.stylesheet_url()}">'
            f'<script src="{Theme.script_url()}" defer></script>'
        )
        ui.colors(primary="#b40010", secondary="#1d2f57", accent="#b40010", negative="#7c1d12")
        ui.dark_mode(None)
