from __future__ import annotations

from typing import ClassVar

from wbpoa.core.model.element import Guideline
from wbpoa.single_corpus.ingestion.base_document_ingestion import BaseDocumentIngestion


class GuidelineIngestion(BaseDocumentIngestion[Guideline]):
    """
    Ingests a guideline document and turns it into a :class:`Guideline`.
    """

    kind: ClassVar[str] = "guideline"

    def _max_pages(self) -> int | None:
        return self.settings.guideline_max_pages

    def _build(self, name: str, text: str) -> Guideline:
        return Guideline(name=name, text=text)
