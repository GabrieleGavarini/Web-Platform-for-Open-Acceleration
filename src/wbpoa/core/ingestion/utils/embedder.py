"""Dense text embeddings backed by a local sentence-transformers model."""

from __future__ import annotations

import time
from collections.abc import Sequence

from langchain_huggingface import HuggingFaceEmbeddings
from structlog import get_logger

from wbpoa.core.settings import get_core_settings

log = get_logger()


class EmbeddingError(RuntimeError):
    """Raised when the embedding model fails to encode its input."""


class Embedder:
    """Encode text into dense vectors for indexing and retrieval."""

    def __init__(self) -> None:
        settings = get_core_settings()

        self.embed_model = settings.embed.model
        self.embed_device = settings.embed.device
        self.embed_normalize = settings.embed.normalize
        self.hf_token = settings.hf_token.get_secret_value() if settings.hf_token is not None else None

        self._dimension: int | None = None
        self.embeddings = self._build_embeddings()

    def _build_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Load the sentence-transformers model onto the configured device.

        Returns:
            the :class:`HuggingFaceEmbeddings` instance.

        """
        log.info(
            "embeddings_initialized",
            model=self.embed_model,
            device=self.embed_device,
            normalize=self.embed_normalize,
        )
        model_kwargs = {"device": self.embed_device}
        if self.hf_token is not None:
            model_kwargs["token"] = self.hf_token
        return HuggingFaceEmbeddings(
            model_name=self.embed_model,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": self.embed_normalize},
        )

    @property
    def dimension(self) -> int:
        """
        Vector width of the configured model, resolved on first access.

        Returns:
            the number of components in an embedding.

        """
        if self._dimension is None:
            self._dimension = len(self.create_embedding("dimension probe"))
            log.info("embedding_dimension_resolved", dimension=self._dimension)
        return self._dimension

    def create_embedding(self, text: str) -> list[float]:
        """
        Create an embedding for the specified text.

        Args:
            text: the text to encode.

        Returns:
            the embedding as a list of floats.

        Raises:
            ValueError: if ``text`` is empty or whitespace only.
            EmbeddingError: if the model fails to encode the text.

        """
        if not text or not text.strip():
            msg = "Cannot embed empty text."
            raise ValueError(msg)

        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            log.exception("embedding_failed", chars=len(text))
            msg = "Failed to create the embedding."
            raise EmbeddingError(msg) from e

    def create_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Create embeddings for several texts in one batched pass.

        Args:
            texts: the texts to encode.

        Returns:
            the embeddings, in the same order as ``texts``.

        Raises:
            ValueError: if any element of ``texts`` is empty or whitespace only.
            EmbeddingError: if the model fails to encode the batch.

        """
        if any(not text or not text.strip() for text in texts):
            msg = "Cannot embed empty text."
            raise ValueError(msg)

        log.info(
            "Batch Embedding started",
            count=len(texts),
        )

        start = time.perf_counter()
        try:
            embeddings = self.embeddings.embed_documents(list(texts))
            duration = round(time.perf_counter() - start, 3)
            log.info(
                "Batch Embedding completed",
                count=len(texts),
                duration_seconds=duration,
            )
        except Exception as e:
            log.exception(
                "Batch Embedding failed",
                count=len(texts),
                duration_seconds=round(time.perf_counter() - start, 3),
            )
            msg = "Failed to create the embeddings."
            raise EmbeddingError(msg) from e

        return embeddings
