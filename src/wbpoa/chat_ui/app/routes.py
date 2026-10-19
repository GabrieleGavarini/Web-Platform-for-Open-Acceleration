from typing import Final
from urllib.parse import quote
from uuid import UUID

CHAT: Final = "/"
GRAPH: Final = "/graph"
GRAPH_CORPUS: Final = "/graph/{corpus}"
"""A corpus's shell; its nodes, ``/graph/{corpus}/{node_id}``, are sub pages nested in it."""
ABOUT: Final = "/about"
EUR_LEX: Final = "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}"


class Routes:
    """URL builders for the chat UI's pages, so pages and components never format paths by hand."""

    @staticmethod
    def corpus_url(corpus_key: str) -> str:
        """
        Build the URL of a corpus's citation-graph page.

        Args:
            corpus_key: Canonical corpus key.

        Returns:
            ``/graph/<corpus>``.
        """
        return f"{GRAPH}/{quote(corpus_key)}"

    @staticmethod
    def eur_lex_url(celex_id: str) -> str:
        """
        Build the EUR-Lex URL of an instrument's official text.

        Args:
            celex_id: The instrument's CELEX number, e.g. ``32016R0679``.

        Returns:
            The EUR-Lex page of the English text.
        """
        return EUR_LEX.format(celex=quote(celex_id))

    @staticmethod
    def node_url(corpus_key: str, node_id: UUID) -> str:
        """
        Build the URL of one node's citation-graph page.

        Args:
            corpus_key: Canonical key of the corpus partition holding the node.
            node_id: The node's id.

        Returns:
            ``/graph/<corpus>/<node id>``.
        """
        return f"{GRAPH}/{quote(corpus_key)}/{node_id}"
