from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pypdfium2
import structlog
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from pydantic import AnyUrl

from wbpoa.single_corpus.ingestion.utils.progress_logger import ProgressLogger
from wbpoa.single_corpus.settings import get_single_corpus_settings

log = structlog.get_logger()

_PARSER_TIERS: tuple[Literal["pypdfium2", "docling", "vlm"], ...] = ("pypdfium2", "docling", "vlm")


class PDFConverter:
    """
    Converts PDF files to text through a 3-tier cascade: pypdfium2, then docling, then a remote VLM.

    Each tier is cheaper but weaker than the next: a plain text-layer read (pypdfium2), a docling
    pass with OCR/table-structure disabled, and finally a remote VLM (Vision-Language Model)
    pipeline for scanned/image-only PDFs. ``settings.vlm.default_pdf_parser`` selects which tier the
    cascade starts at; it still falls through to later tiers when the selected one yields no text.
    It provides methods to convert individual PDF files as well as entire directories of PDFs.
    """

    def __init__(self):
        self.settings = get_single_corpus_settings()

        self.native_converter: DocumentConverter = self._build_native_converter()
        self.vlm_converter: DocumentConverter = self._build_vlm_converter()

    def _build_native_converter(self) -> DocumentConverter:
        """
        Builds a DocumentConverter that extracts only the PDF's embedded/programmatic text layer.

        No OCR, table-structure model, or VLM are involved. This backs the cascade's "docling"
        tier, used when the cheaper pypdfium2 tier finds no text.

        Returns:
            DocumentConverter: A configured DocumentConverter instance.

        """
        pipeline_options = PdfPipelineOptions(do_ocr=False, do_table_structure=False)
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            },
        )

    def _build_vlm_converter(self) -> DocumentConverter:
        """
        Builds a DocumentConverter configured to use the remote VLM pipeline for PDF to Markdown conversion.

        Returns:
            DocumentConverter: A configured DocumentConverter instance.

        """
        vlm_options = ApiVlmOptions(
            url=AnyUrl(f"{self.settings.llm.base_url}/chat/completions"),
            params={
                "model": self.settings.llm.model,
                "chat_template_kwargs": {"enable_thinking": self.settings.vlm.enable_thinking},
            },
            headers={"Authorization": f"Bearer {self.settings.llm.api_key}"},
            timeout=self.settings.vlm.timeout,
            scale=self.settings.vlm.scale,
            temperature=0,
            concurrency=self.settings.vlm.concurrency,
            response_format=ResponseFormat.MARKDOWN,
            prompt=self.settings.vlm.prompt,
        )

        pipeline_options = VlmPipelineOptions(
            vlm_options=vlm_options,
            enable_remote_services=True,
        )
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=DoclingParseDocumentBackend,
                    pipeline_cls=VlmPipeline,
                ),
            },
        )

    def get_page_count(self, pdf_path: str | Path) -> int:
        """
        Return the number of pages in a PDF without running the (expensive) VLM pipeline.

        Args:
            pdf_path (str | Path): The path to the PDF file.

        Returns:
            int: The number of pages in the PDF.

        """
        with pypdfium2.PdfDocument(str(pdf_path)) as pdf:
            return len(pdf)

    def _extract_pypdfium2_text(self, pdf_path: str | Path) -> str:
        """
        Extract a PDF's embedded text layer directly, without any docling/layout model.

        Args:
            pdf_path (str | Path): The path to the PDF file.

        Returns:
            str: The concatenated per-page text, stripped.

        """
        with pypdfium2.PdfDocument(str(pdf_path)) as pdf:
            pages = [page.get_textpage().get_text_range() for page in pdf]
        return "\n\n".join(pages).strip()

    def _extract_docling_text(self, pdf_path: str | Path) -> str:
        """
        Extract a PDF's text via docling's native pass (OCR and table-structure disabled).

        Args:
            pdf_path (str | Path): The path to the PDF file.

        Returns:
            str: The extracted Markdown text, stripped.

        """
        return self.native_converter.convert(source=str(pdf_path)).document.export_to_markdown().strip()

    def _extract_vlm_text(self, pdf_path: str | Path) -> str:
        """
        Extract a PDF's text via the remote VLM pipeline.

        Args:
            pdf_path (str | Path): The path to the PDF file.

        Returns:
            str: The extracted Markdown text, stripped.

        """
        return self.vlm_converter.convert(source=str(pdf_path)).document.export_to_markdown().strip()

    def convert_pdf_to_text(
        self,
        pdf_path: str | Path,
        *,
        progress: ProgressLogger | None = None,
        current_document: int | None = None,
    ) -> str:
        """
        Convert a PDF file to text through the pypdfium2 -> docling -> VLM cascade.

        ``settings.vlm.default_pdf_parser`` selects which tier the cascade starts at; whichever
        tier is tried first, the cascade still falls through to the remaining, more expensive
        tiers if a tier yields no text at all, e.g. for scanned/image-only PDFs.

        Args:
            pdf_path (str | Path): The path to the PDF file to be converted.
            progress: Optional batch progress tracker.
            current_document: 1-indexed position of this PDF in the batch.

        Returns:
            str: The extracted text from the PDF.

        Raises:
            RuntimeError: If every tried tier fails to produce content.

        """
        filename = Path(pdf_path).name
        pdf_size_kb = round(Path(pdf_path).stat().st_size / 1024, 1)
        progress_fields = (
            progress.snapshot(current_document, completed_documents=current_document - 1)
            if progress is not None and current_document is not None
            else {}
        )
        log.info("pdf_converting", filename=filename, pdf_size_kb=pdf_size_kb, **progress_fields)

        tiers: dict[Literal["pypdfium2", "docling", "vlm"], Callable[[str | Path], str]] = {
            "pypdfium2": self._extract_pypdfium2_text,
            "docling": self._extract_docling_text,
            "vlm": self._extract_vlm_text,
        }
        start_index = _PARSER_TIERS.index(self.settings.vlm.default_pdf_parser)

        start = time.perf_counter()
        for tier in _PARSER_TIERS[start_index:]:
            text = tiers[tier](pdf_path)
            if text:
                duration_seconds = round(time.perf_counter() - start, 3)
                log.info(
                    "pdf_converted",
                    filename=filename,
                    pipeline=tier,
                    md_size_chars=len(text),
                    pdf_ingestion_seconds=duration_seconds,
                    **(
                        progress.snapshot(current_document, completed_documents=current_document)
                        if progress is not None and current_document is not None
                        else {}
                    ),
                )
                return text
            log.info("pdf_tier_no_text", filename=filename, pipeline=tier)

        duration_seconds = round(time.perf_counter() - start, 3)
        log.error(
            "pdf_conversion_failed",
            filename=filename,
            reason="empty_output",
            pdf_ingestion_seconds=duration_seconds,
        )
        raise RuntimeError(f"PDF conversion failed: no content generated for {pdf_path}")

    def convert_pdf_to_markdown(
        self,
        pdf_path: str | Path,
        output_dir: str | Path | None = None,
        *,
        progress: ProgressLogger | None = None,
        current_document: int | None = None,
    ) -> Path:
        """
        Convert a PDF to Markdown through the remote VLM pipeline.

        If the target Markdown already exists, the conversion is skipped and the
        existing path is returned.

        Args:
            pdf_path: path of the PDF file to convert.
            output_dir: destination folder for the Markdown; defaults to ``self.settings.md_dir``.
            progress: Optional batch progress tracker.
            current_document: 1-indexed position of this PDF in the batch.

        Returns:
            the path of the generated (or already existing) Markdown file.

        """
        target_dir = Path(output_dir) if output_dir is not None else Path(self.settings.md_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        out_path = target_dir / f"{Path(pdf_path).stem}.md"
        if out_path.exists():
            log.info(
                "pdf_skipped",
                filename=Path(pdf_path).name,
                reason="already_exists",
                pdf_ingestion_seconds=0.0,
                **(progress.record(current_document) if progress is not None and current_document is not None else {}),
            )
            return out_path

        md_text = self.convert_pdf_to_text(
            pdf_path,
            progress=progress,
            current_document=current_document,
        )

        # Only write the file if conversion was successful and produced content
        if md_text.strip():
            out_path.write_text(md_text, encoding="utf-8")
            return out_path
        log.error("pdf_conversion_failed", filename=Path(pdf_path).name, reason="empty_output")
        out_path.unlink(missing_ok=True)  # Remove the empty file if it was created
        raise RuntimeError(f"PDF conversion failed: no content generated for {pdf_path}")

    def convert_pdf_dir(
        self,
        pattern: str = "**/*.pdf",
        output_dir: str | Path | None = None,
    ) -> list[Path]:
        """
        Convert to Markdown all the PDFs in ``self.settings.pdf_dir``.

        PDFs whose target Markdown already exists are skipped.

        Args:
            pattern: glob relative to the PDF folder.
            output_dir: destination folder for the Markdown files; defaults to ``self.settings.md_dir``.

        Returns:
            the list of Markdown paths (generated or already existing).

        """
        target_dir = Path(output_dir) if output_dir is not None else Path(self.settings.md_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        # Collect both lower- and upper-case extension variants so the function
        # works on case-sensitive Linux filesystems (e.g. ".pdf" and ".PDF").
        stem, _, ext = pattern.rpartition(".")
        pdf_paths = sorted(
            {
                *Path(self.settings.pdf_dir).glob(pattern),
                *Path(self.settings.pdf_dir).glob(f"{stem}.{ext.upper()}"),
            }
        )

        log.info("pdf_dir_start", pdf_dir=str(self.settings.pdf_dir), total_pdfs=len(pdf_paths))

        outputs: list[Path] = []
        skipped = 0
        total = len(pdf_paths)
        progress = ProgressLogger(total=total)
        for idx, pdf_path in enumerate(pdf_paths, start=1):
            out_path = target_dir / f"{pdf_path.stem}.md"
            if out_path.exists():
                log.info(
                    "pdf_skipped",
                    filename=pdf_path.name,
                    reason="already_exists",
                    pdf_ingestion_seconds=0.0,
                    **progress.record(idx),
                )
                outputs.append(out_path)
                skipped += 1
                continue
            md_text = self.convert_pdf_to_text(pdf_path, progress=progress, current_document=idx)

            # Only write the file if conversion was successful and produced content
            if md_text.strip():
                out_path.write_text(md_text, encoding="utf-8")
                outputs.append(out_path)
            else:
                log.error(
                    "pdf_conversion_failed", filename=pdf_path.name, reason="empty_output", **progress.record(idx)
                )
                out_path.unlink(missing_ok=True)  # Remove the empty file if it was created
                raise RuntimeError(f"PDF conversion failed: no content generated for {pdf_path}")
        log.info("pdf_dir_complete", converted=len(outputs) - skipped, skipped=skipped, total=total)
        return outputs
