"""Okapi BM25 lexical index over node full text, the lexical channel of Algorithm 2."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from uuid import UUID

from rank_bm25 import BM25Okapi

from wbpoa.core.model.graph import Node

# Keep digits: article and annex numbers are precisely the tokens the lexical channel exists to match.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split ``text`` on non-alphanumerics, keeping digit runs."""
    return _TOKEN_RE.findall(text.lower())


class Bm25Index:
    """A BM25 index over a reference set of node full text, queried to score lexical relevance."""

    def __init__(self, ids: Sequence[UUID], corpus: Sequence[Sequence[str]]) -> None:
        self._ids = list(ids)
        self._bm25 = BM25Okapi(list(corpus)) if corpus else None

    @classmethod
    def from_nodes(cls, nodes: Iterable[Node]) -> Bm25Index:
        """Build an index over the full text of ``nodes``, keyed by node id."""
        ids: list[UUID] = []
        corpus: list[list[str]] = []
        for node in nodes:
            ids.append(node.id)
            corpus.append(tokenize(node.text))
        return cls(ids, corpus)

    @classmethod
    def from_texts(cls, texts: Iterable[tuple[UUID, str]]) -> Bm25Index:
        """Build an index over ``(id, text)`` pairs, avoiding loading full nodes."""
        ids: list[UUID] = []
        corpus: list[list[str]] = []
        for node_id, text in texts:
            ids.append(node_id)
            corpus.append(tokenize(text))
        return cls(ids, corpus)

    def scores(self, query: str) -> dict[UUID, float]:
        """Return the raw BM25 score of ``query`` against each indexed text, keyed by node id."""
        if self._bm25 is None:
            return dict.fromkeys(self._ids, 0.0)
        raw = self._bm25.get_scores(tokenize(query))
        return {node_id: float(score) for node_id, score in zip(self._ids, raw, strict=True)}
