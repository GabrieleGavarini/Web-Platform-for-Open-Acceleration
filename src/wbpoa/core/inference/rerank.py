"""Cross-encoder reranker for Algorithm 2's cascade stages (steps 4(ii) and 7)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import UUID

from structlog import get_logger

from wbpoa.core.inference.similarity import min_max_normalize
from wbpoa.core.ingestion.utils.chunking import split_into_chunks

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

log = get_logger()

# Reserved for the model's special tokens (e.g. [CLS]/[SEP]) plus a margin for tokenizer
# estimate error, mirroring the fixed-margin half of Summarizer's context-window budgeting.
_SPECIAL_TOKEN_MARGIN = 16
# Some tokenizers report a sentinel (e.g. 1_000_000_000_000) for `model_max_length` when the
# model config doesn't set one; anything past this is treated as "unset" and falls back below.
_UNREASONABLE_MAX_TOKENS = 100_000
_DEFAULT_MAX_TOKENS = 512


@runtime_checkable
class Reranker(Protocol):
    """A cross-encoder rr(q, u), scored over a bounded shortlist. Mirrors Bm25Index.scores' shape."""

    def scores(self, query: str, docs: Sequence[tuple[UUID, str]]) -> dict[UUID, float]:
        """
        Score ``query`` against each of ``docs`` in a single joint forward pass per pair.

        Args:
            query: The natural-language query.
            docs: ``(id, text)`` pairs to score against ``query``.

        Returns:
            The raw reranker score ``rr(q, u)`` for each id in ``docs``.

        """
        ...


class NullReranker:
    """No-op reranker installed when reranking is disabled; the disabled path exercises the same code."""

    def scores(self, query: str, docs: Sequence[tuple[UUID, str]]) -> dict[UUID, float]:  # noqa: ARG002
        """Return an empty mapping regardless of input, so callers fall back to the first-pass score."""
        return {}


class CrossEncoderReranker:
    """Wraps a `sentence_transformers.CrossEncoder`, chunking long text on paragraph boundaries (Remark 12)."""

    def __init__(
        self,
        model: str,
        *,
        device: str | None = None,
        max_tokens: int | None = None,
        token: str | None = None,
    ) -> None:
        """
        Args:
            model: A `sentence_transformers.CrossEncoder`-compatible model name or path.
            device: Device to load the model on; defaults to the model's own default resolution.
            max_tokens: Token budget per (query, chunk) pair; resolved from the model's own
                tokenizer on first use when not given explicitly.
            token: Optional Hugging Face access token for gated models.

        """
        self._model_name = model
        self._device = device
        self._max_tokens = max_tokens
        self._token = token
        self._cross_encoder = None

    @property
    def _model(self) -> CrossEncoder:
        """Lazily load the underlying `CrossEncoder`, so constructing a reranker costs nothing when disabled."""
        if self._cross_encoder is None:
            # Deferred: avoid the sentence_transformers/torch import cost when reranking is never used.
            from sentence_transformers import CrossEncoder  # noqa: PLC0415

            log.info("cross_encoder_reranker_loading", model=self._model_name, device=self._device)
            self._cross_encoder = CrossEncoder(self._model_name, device=self._device, token=self._token)
        return self._cross_encoder

    def _resolved_max_tokens(self) -> int:
        if self._max_tokens is not None:
            return self._max_tokens
        reported = getattr(self._model.tokenizer, "model_max_length", None)
        if not reported or reported > _UNREASONABLE_MAX_TOKENS:
            return _DEFAULT_MAX_TOKENS
        return max(int(reported) - _SPECIAL_TOKEN_MARGIN, 1)

    def scores(self, query: str, docs: Sequence[tuple[UUID, str]]) -> dict[UUID, float]:
        """
        Score ``query`` against each of ``docs``, chunking any text exceeding the model's window.

        Every chunk of every doc is scored in a single batched `predict()` call; a doc's final
        score is the maximum over its chunks (Remark 12), so a long document isn't silently
        truncated to its least discriminative (opening) part.

        Args:
            query: The natural-language query.
            docs: ``(id, text)`` pairs to score against ``query``.

        Returns:
            The raw reranker score for each id in ``docs``.

        """
        if not docs:
            return {}

        tokenizer = self._model.tokenizer
        budget = max(self._resolved_max_tokens() - len(tokenizer.encode(query, add_special_tokens=False)), 1)

        pairs: list[tuple[str, str]] = []
        chunk_ids: list[UUID] = []
        for doc_id, text in docs:
            chunks = split_into_chunks(
                text,
                budget,
                count_tokens=lambda t: len(tokenizer.encode(t, add_special_tokens=False)),
                encode=lambda t: tokenizer.encode(t, add_special_tokens=False),
                decode=lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
            )
            for chunk in chunks or [text]:
                pairs.append((query, chunk))
                chunk_ids.append(doc_id)

        raw_scores = self._model.predict(pairs, convert_to_numpy=True)

        scores: dict[UUID, float] = {}
        for doc_id, raw_score in zip(chunk_ids, raw_scores, strict=True):
            chunk_score = float(raw_score)
            if doc_id not in scores or chunk_score > scores[doc_id]:
                scores[doc_id] = chunk_score
        return scores


def rerank_blend(rerank: Mapping[UUID, float], first_pass: Mapping[UUID, float], rho: float) -> dict[UUID, float]:
    """
    Blend a reranker's raw scores with a first-pass score, over the reranker's own shortlist.

    Implements blend_P(u) = rho * rsim_P(q, u) + (1 - rho) * s(u), where rsim_P is the
    min-max normalisation of ``rerank`` within its own keys (the shortlist P) -- so rho == 0
    recovers ``first_pass`` exactly and rho == 1 lets the cross-encoder alone decide.

    Args:
        rerank: Raw reranker scores rr(q, u), keyed by id; its keys define the shortlist P.
        first_pass: The first-pass score s(u) for every id in ``rerank``.
        rho: Blend weight in [0, 1].

    Returns:
        The blended score for every id in ``rerank`` (only the shortlist -- callers must fall
        back to ``first_pass`` for anything not in this result, e.g. when ``rerank`` is empty).

    """
    rsim = min_max_normalize(rerank)
    return {doc_id: rho * rsim[doc_id] + (1 - rho) * first_pass[doc_id] for doc_id in rerank}
