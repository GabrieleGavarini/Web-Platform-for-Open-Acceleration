from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Protocol

import structlog

from wbpoa.core.model.element.element import Element
from wbpoa.single_corpus.ingestion.utils import PDFConverter, ProgressLogger
from wbpoa.single_corpus.settings import IngestionSettings

log = structlog.get_logger()


class _PdfConverter(Protocol):
    def get_page_count(self, pdf_path: str | Path) -> int:
        """Return the number of pages in a PDF."""
        ...

    def convert_pdf_to_text(
        self,
        pdf_path: str | Path,
        *,
        progress: ProgressLogger | None = None,
        current_document: int | None = None,
    ) -> str:
        """Convert a PDF directly to text."""
        ...

    def convert_pdf_to_markdown(
        self,
        pdf_path: str | Path,
        output_dir: str | Path | None = None,
        *,
        progress: ProgressLogger | None = None,
        current_document: int | None = None,
    ) -> Path:
        """Convert a PDF to a persisted Markdown file."""
        ...


class BaseDocumentIngestion[T: Element](ABC):
    """
    Shared PDF ingestion flow: page-limit check, PDF-to-Markdown conversion, model construction.
    """

    kind: ClassVar[str]
    """Document kind, logged as a field on the 'document_skipped' event, e.g. 'guideline' or 'ruling'."""

    def __init__(
        self,
        markdown_output_dir: Path | None = None,
        *,
        pdf_converter: _PdfConverter | None = None,
        settings: IngestionSettings | None = None,
    ) -> None:
        """Initialize document ingestion.

        Args:
            markdown_output_dir: Corpus-local directory for reusable Markdown output. When omitted,
                conversion remains in memory for backward compatibility.
            pdf_converter: Optional converter supplied by callers.
            settings: Page-count settings. Defaults to environment-backed ingestion settings.
        """
        self.pdf_converter = pdf_converter if pdf_converter is not None else PDFConverter()
        self.settings = settings if settings is not None else IngestionSettings()
        self.markdown_output_dir = markdown_output_dir

    @abstractmethod
    def _max_pages(self) -> int | None:
        """Configured page-count limit for this document kind, or None if unlimited."""

    @abstractmethod
    def _build(self, name: str, text: str) -> T:
        """Construct the ingested element from its name and extracted text."""

    def ingest(
        self,
        path: str | Path,
        name: str,
        *,
        progress: ProgressLogger | None = None,
        current_document: int | None = None,
    ) -> T | None:
        """
        Ingests a document from the specified path.

        Args:
            path (str | Path): The path to the document to be ingested.
            name (str): The name of the document to be ingested.
            progress: Optional batch progress tracker.
            current_document: 1-indexed position of this document in the batch.

        Returns:
            T | None: The ingested element, or None if it exceeds the configured page limit.

        """
        max_pages = self._max_pages()
        if max_pages is not None:
            page_count = self.pdf_converter.get_page_count(path)
            if page_count > max_pages:
                log.info("document_skipped", kind=self.kind, path=str(path), page_count=page_count, max_pages=max_pages)
                return None

        if self.markdown_output_dir is None:
            md_text = self.pdf_converter.convert_pdf_to_text(
                path,
                progress=progress,
                current_document=current_document,
            )
        else:
            markdown_path = self.pdf_converter.convert_pdf_to_markdown(
                path,
                output_dir=self.markdown_output_dir,
                progress=progress,
                current_document=current_document,
            )
            md_text = markdown_path.read_text(encoding="utf-8")

        return self._build(name, md_text)
