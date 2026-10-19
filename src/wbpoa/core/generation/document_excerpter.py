from __future__ import annotations

from typing import Final
from uuid import UUID, uuid5

import tiktoken

from wbpoa.core.generation.chat_model import count_tokens
from wbpoa.core.inference.lexical import Bm25Index, tokenize
from wbpoa.core.ingestion.utils.chunking import split_into_chunks

_encoding = tiktoken.get_encoding("cl100k_base")
_CHUNK_NAMESPACE: Final = UUID("5f0c8a4e-7a53-4d2f-9f0e-2b8d7f1c3a61")
EXCERPT_SEPARATOR: Final = "\n[…]\n"


class DocumentExcerpter:
    """
    Pick the passages of a long guideline or ruling that bear on a research question.

    Guidelines and rulings are stored as whole documents, far too long to show the agent, while
    their summaries only say what a document is about. The excerpter splits a document into
    paragraph-aligned chunks, scores each chunk against the question with BM25, and returns the
    best-scoring chunks in document order, so the agent sees what the document actually says on
    the point it asked about.

    Attributes:
        excerpt_tokens: Maximum estimated tokens per excerpt chunk.
        excerpts: Maximum number of chunks returned per document.

    """

    def __init__(self, excerpt_tokens: int, excerpts: int) -> None:
        """
        Args:
            excerpt_tokens: Maximum estimated tokens per excerpt chunk.
            excerpts: Maximum number of chunks returned per document.

        """
        self.excerpt_tokens = excerpt_tokens
        self.excerpts = excerpts

    def excerpt(self, text: str, question: str) -> str | None:
        """
        Return the chunks of `text` most relevant to `question`.

        Args:
            text: The document's full text.
            question: The research question the excerpt should answer.

        Returns:
            Up to `excerpts` chunks sharing a term with `question`, the best BM25-scored first
            chosen, returned in document order and joined by `EXCERPT_SEPARATOR`; None when no
            chunk shares a term with the question.

        """
        chunks = split_into_chunks(
            text,
            self.excerpt_tokens,
            count_tokens=count_tokens,
            encode=_encoding.encode,
            decode=_encoding.decode,
        )
        terms = set(tokenize(question))
        # BM25 IDF turns non-positive for a term in half the chunks or more (always, for a
        # one-chunk document), so relevance is gated on sharing a term, and BM25 only ranks.
        matching = [index for index, chunk in enumerate(chunks) if terms & set(tokenize(chunk))]
        ids = [uuid5(_CHUNK_NAMESPACE, str(index)) for index in range(len(chunks))]
        scores = Bm25Index.from_texts(zip(ids, chunks, strict=True)).scores(question)
        ranked = sorted(matching, key=lambda index: (scores[ids[index]], -index), reverse=True)
        best = sorted(ranked[: self.excerpts])
        if not best:
            return None
        return EXCERPT_SEPARATOR.join(chunks[index].strip() for index in best)
