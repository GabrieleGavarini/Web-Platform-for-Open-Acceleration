from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Final, Protocol

from wbpoa.chat_ui.graph.models import CoCitation, CommunityView, GraphNodeView, NeighbourGroup, NodeSummary

if TYPE_CHECKING:
    from uuid import UUID

    from wbpoa.core.corpus import CorpusWorkspace
    from wbpoa.core.model.graph.node import Node

TYPE_ORDER: Final = ("Article", "Recital", "Annex", "Guideline", "Ruling")
"""Node types the browser shows, provisions first."""

_TITLE_MAX_LENGTH: Final = 160
_MIN_COMMUNITY_SIZE: Final = 2
"""A community of one provision only repeats that provision, so the page leaves it out."""
_NUMBER: Final = re.compile(r"\b(\d+)([a-z]?)\b")
# A line holding only a point's label, e.g. "(a)", "(iv)", "2." or "b)": the Official Journal sets it
# on the same line as the point's text, which the extracted text puts on the next line.
_POINT_LABEL: Final = re.compile(r"^(\([0-9a-z]{1,6}\)|[0-9]{1,3}[a-z]?\.|[a-z]\))$")
_WHITESPACE: Final = re.compile(r"\s+")
_HEADING_TRIM: Final = " `\t\N{NO-BREAK SPACE}"
"""Characters trimmed off an article's heading line: spaces, and the stray backticks of some extractions."""
_ROMAN: Final = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


class GraphReader(Protocol):
    """The read side of `Neo4jDriver` the browser needs."""

    def get_node_by_id(self, node_id: UUID) -> Node | None:
        """
        Read one node with its text.

        Args:
            node_id: The node's id.

        Returns:
            The node, or None when it is not in the corpus.
        """
        ...

    def citation_neighbourhood(self, node_id: UUID) -> tuple[list[Node], list[Node]]:
        """
        Read what a node cites and what cites it.

        Args:
            node_id: The node's id.

        Returns:
            ``(cited, citing)`` nodes, without text.
        """
        ...

    def co_citation_neighbourhood(self, node_id: UUID) -> list[tuple[Node, float]]:
        """
        Read the nodes cited together with a node.

        Args:
            node_id: The node's id.

        Returns:
            ``(neighbour, weight)`` pairs over the node's co-citation arches, without text.
        """
        ...

    def find_nodes(self, term: str, *, types: Sequence[str], limit: int) -> list[Node]:
        """
        Find nodes by name.

        Args:
            term: Substring of the name.
            types: Node types to search.
            limit: Maximum number of nodes.

        Returns:
            The matching nodes, without text.
        """
        ...

    def community_ancestry(self, node_id: UUID) -> list[tuple[Node, list[Node]]]:
        """
        Read the communities above a provision.

        Args:
            node_id: The provision's id.

        Returns:
            ``(community, members)`` pairs, finest first; members without text.
        """
        ...


class CitationGraph:
    """
    Read-only browser over one corpus's citation graph: node pages and name search.

    Attributes:
        corpus: The corpus browsed.
    """

    def __init__(self, corpus: CorpusWorkspace, reader: GraphReader) -> None:
        """
        Args:
            corpus: The corpus browsed; its manifest names each provision's instrument.
            reader: Reads the corpus's partition, normally its `Neo4jDriver`.
        """
        self.corpus = corpus
        self._reader = reader
        self._instrument_names = {instrument.key: instrument.name for instrument in corpus.config.instruments}

    def node(self, node_id: UUID) -> GraphNodeView | None:
        """
        Read one node and its citation neighbourhood.

        Args:
            node_id: The node's id.

        Returns:
            The node's view, or None when the node is not in this corpus or is not an
            Article, Recital, Annex, guideline, or ruling. A provision's view carries its
            communities, finest first, leaving out any that groups it alone or repeats the
            provisions of a finer one, and the provisions cited together with it, heaviest
            first.
        """
        node = self._reader.get_node_by_id(node_id)
        if node is None or node.type not in TYPE_ORDER:
            return None
        cited, citing = self._reader.citation_neighbourhood(node_id)
        title, text = self._split_title(node)
        summary = self._summary(node)
        return GraphNodeView(
            corpus_key=self.corpus.key,
            corpus_name=self.corpus.config.name,
            node=summary,
            title=title,
            celex_id=node.celex_id,
            text=text,
            summary=node.summary,
            cites=self._group(cited),
            cited_by=self._group(citing),
            co_cited=self._co_cited(node_id) if summary.is_provision else [],
            communities=self._communities(node_id) if summary.is_provision else [],
        )

    def search(self, term: str, types: Sequence[str] = TYPE_ORDER, limit: int = 60) -> list[NodeSummary]:
        """
        Find nodes by name.

        Args:
            term: Substring of the node name, e.g. ``Article 6`` or ``EDPB``; blank lists every node.
            types: Node types to search; unknown types are ignored.
            limit: Maximum number of nodes.

        Returns:
            Matching nodes, grouped in `TYPE_ORDER` and naturally ordered within a type.
        """
        wanted = [node_type for node_type in TYPE_ORDER if node_type in types]
        if not wanted:
            return []
        nodes = self._reader.find_nodes(term.strip(), types=wanted, limit=limit)
        return [node for group in self._group(nodes) for node in group.nodes]

    def _co_cited(self, node_id: UUID) -> list[CoCitation]:
        """The provisions cited together with the node, heaviest first, then in natural order."""
        weights: dict[UUID, float] = {}
        nodes: dict[UUID, NodeSummary] = {}
        for neighbour, weight in self._reader.co_citation_neighbourhood(node_id):
            if neighbour.id == node_id or neighbour.type not in TYPE_ORDER:
                continue
            summary = self._summary(neighbour)
            if summary.is_provision:
                nodes[neighbour.id] = summary
                weights[neighbour.id] = weights.get(neighbour.id, 0.0) + weight
        ranked = sorted(nodes.values(), key=lambda node: (-weights[node.id], _natural_key(node)))
        return [CoCitation(node=node, weight=weights[node.id]) for node in ranked]

    def _communities(self, node_id: UUID) -> list[CommunityView]:
        """The provision's communities, finest first, without singletons or repeats of a finer one."""
        communities: list[CommunityView] = []
        seen: set[frozenset[UUID]] = set()
        for community, members in sorted(self._reader.community_ancestry(node_id), key=lambda pair: pair[0].layer):
            listed = [member for group in self._group(members) for member in group.nodes]
            key = frozenset(member.id for member in listed)
            if len(listed) < _MIN_COMMUNITY_SIZE or key in seen:
                continue
            seen.add(key)
            communities.append(
                CommunityView(
                    id=community.id,
                    name=community.name,
                    layer=community.layer,
                    summary=community.summary,
                    members=listed,
                )
            )
        return communities

    def _summary(self, node: Node) -> NodeSummary:
        """Label one node with its instrument's name."""
        instrument = self._instrument_names.get(node.instrument_key) if node.instrument_key else None
        return NodeSummary(id=node.id, name=node.name, type=node.type, instrument_name=instrument)

    def _group(self, nodes: Iterable[Node]) -> list[NeighbourGroup]:
        """Group nodes by type in `TYPE_ORDER`, sorted naturally within each type, dropping other types."""
        groups = {node_type: NeighbourGroup(type=node_type) for node_type in TYPE_ORDER}
        for node in nodes:
            group = groups.get(node.type)
            if group is not None and all(listed.id != node.id for listed in group.nodes):
                group.nodes.append(self._summary(node))
        for group in groups.values():
            group.nodes.sort(key=_natural_key)
        return [group for group in groups.values() if group.nodes]

    @staticmethod
    def _split_title(node: Node) -> tuple[str | None, str]:
        """
        Split an article's heading off its text.

        Article texts open with their name and heading lines, e.g. ``Article 6`` then
        ``Lawfulness of processing``; the page already shows the name, so both leave the body.
        The name matches whatever spaces it is written with (EUR-Lex uses non-breaking ones),
        stray backticks around the heading are dropped, and a point's label standing alone on
        its line, such as ``(a)``, is joined to the point's text.

        Args:
            node: The node read with its text.

        Returns:
            ``(heading, body)``; the heading is None when the text has none.
        """
        lines = node.text.strip().splitlines()
        if lines and _squash(lines[0]) == _squash(node.name):
            lines = lines[1:]
        if node.type == "Article" and len(lines) > 1:
            heading = lines[0].strip(_HEADING_TRIM)
            if heading and len(heading) <= _TITLE_MAX_LENGTH and not heading[0].isdigit():
                return heading, _join_point_labels(lines[1:])
        return None, _join_point_labels(lines)


def _squash(text: str) -> str:
    """Case-folded text with every run of whitespace, non-breaking spaces included, made one space."""
    return _WHITESPACE.sub(" ", text).strip().casefold()


def _join_point_labels(lines: list[str]) -> str:
    """The body text, each point label that stands alone on its line joined to the line after it."""
    joined: list[str] = []
    label: str | None = None
    for line in lines:
        stripped = line.strip()
        if label is not None:
            joined.append(f"{label} {stripped}" if stripped else label)
            label = None
        elif _POINT_LABEL.match(stripped):
            label = stripped
        else:
            joined.append(line.rstrip())
    if label is not None:
        joined.append(label)
    return "\n".join(joined).strip()


def _natural_key(node: NodeSummary) -> tuple[str, int, str, str]:
    """Order ``Article 2`` before ``Article 10`` and ``Annex II`` before ``Annex X``; documents by name."""
    if node.is_provision:
        match = _NUMBER.search(node.name)
        if match is not None:
            return node.instrument_name or "", int(match[1]), match[2], node.name
        numeral = node.name.rsplit(" ", 1)[-1].upper()
        if numeral and all(char in _ROMAN for char in numeral):
            return node.instrument_name or "", _roman(numeral), "", node.name
    return node.instrument_name or "", 0, node.name.casefold(), node.name


def _roman(numeral: str) -> int:
    """Value of a Roman numeral such as ``XIV``."""
    values = [_ROMAN[char] for char in numeral]
    return sum(
        -value if value < following else value for value, following in zip(values, [*values[1:], 0], strict=True)
    )
