"""Tokenizer-agnostic paragraph-boundary chunking, shared by any component that must fit text into a token budget."""

from __future__ import annotations

from collections.abc import Callable, Sequence


def split_into_chunks(
    text: str,
    max_tokens: int,
    *,
    count_tokens: Callable[[str], int],
    encode: Callable[[str], Sequence[int]],
    decode: Callable[[Sequence[int]], str],
) -> list[str]:
    """
    Split ``text`` into chunks of at most ``max_tokens`` tokens each, under a caller-supplied tokenizer.

    Splits on paragraph boundaries (blank lines) where possible; a single paragraph longer than
    ``max_tokens`` is hard-split on raw token boundaries as a last resort. Parameterized over
    ``count_tokens``/``encode``/``decode`` rather than a fixed tokenizer, since different callers
    (an LLM's tiktoken encoding, a cross-encoder's own tokenizer) count and split text differently.

    Args:
        text: The text to split.
        max_tokens: Maximum estimated tokens per chunk, under the caller's tokenizer.
        count_tokens: Estimates the token count of a string.
        encode: Encodes a string into token ids, under the same tokenizer as ``count_tokens``.
        decode: Decodes token ids back into a string, the inverse of ``encode``.

    Returns:
        The chunks, in the original order.

    """
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    # "\n\n".join() adds separator tokens between paragraphs; must be counted too, or a
    # joined chunk can silently exceed max_tokens and loop forever when re-split later.
    separator_tokens = count_tokens("\n\n")

    for paragraph in (p for p in text.split("\n\n") if p.strip()):
        paragraph_tokens = count_tokens(paragraph)

        if paragraph_tokens > max_tokens:
            if current:
                chunks.append("\n\n".join(current))
                current, current_tokens = [], 0
            encoded = encode(paragraph)
            chunks.extend(decode(encoded[i : i + max_tokens]) for i in range(0, len(encoded), max_tokens))
            continue

        added_tokens = paragraph_tokens + separator_tokens if current else paragraph_tokens
        if current and current_tokens + added_tokens > max_tokens:
            chunks.append("\n\n".join(current))
            current, current_tokens = [], 0
            added_tokens = paragraph_tokens

        current.append(paragraph)
        current_tokens += added_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks
