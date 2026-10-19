"""Similarity helpers used throughout Algorithm 2's scoring formulas."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from uuid import UUID

from scipy.spatial.distance import cosine as _scipy_cosine


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Compute cos(a, b), returning 0.0 for a zero-length vector instead of raising.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        The cosine similarity in ``[-1, 1]``, or ``0.0`` if either vector has zero norm.

    """
    if not a or not b or not any(a) or not any(b):
        return 0.0
    # scipy returns the cosine *distance* (1 - similarity).
    distance = float(_scipy_cosine(a, b))
    if math.isnan(distance):  # defensive: a zero-norm vector still slipped through.
        return 0.0
    return 1.0 - distance


def min_max_normalize(values: Mapping[UUID, float]) -> dict[UUID, float]:
    """
    Min-max normalise ``values`` into ``[0, 1]``, returning all-zeros when max equals min.

    Implements the ``lex_S`` normalisation of Algorithm 2, including the convention that a
    vanishing denominator (every value equal) collapses the score to ``0.0``.

    Args:
        values: A mapping from node id to a raw score (e.g. BM25).

    Returns:
        A mapping from node id to its normalised score in ``[0, 1]``.

    """
    if not values:
        return {}
    lowest = min(values.values())
    highest = max(values.values())
    spread = highest - lowest
    if spread == 0.0:
        return dict.fromkeys(values, 0.0)
    return {key: (value - lowest) / spread for key, value in values.items()}


def hybrid_similarity(semantic: float, lexical: float, gamma: float) -> float:
    """
    Combine a semantic and a lexical score, ``hs = gamma*sem + (1-gamma)*lex`` (Algorithm 2).

    Args:
        semantic: The cosine semantic score ``sem(q, v)``.
        lexical: The min-max normalised lexical score ``lex_S(q, v)``.
        gamma: Mixing weight in ``[0, 1]``; ``gamma == 1`` recovers the purely semantic similarity.

    Returns:
        The hybrid similarity.

    """
    return gamma * semantic + (1.0 - gamma) * lexical
