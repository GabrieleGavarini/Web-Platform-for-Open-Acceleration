from __future__ import annotations

from typing import ClassVar

from wbpoa.core.model.element import Ruling
from wbpoa.single_corpus.ingestion.base_document_ingestion import BaseDocumentIngestion


class RulingIngestion(BaseDocumentIngestion[Ruling]):
    """
    Ingests a ruling document and turns it into a :class:`Ruling`.
    """

    kind: ClassVar[str] = "ruling"

    def _max_pages(self) -> int | None:
        return self.settings.ruling_max_pages

    def _build(self, name: str, text: str) -> Ruling:
        return Ruling(name=name, text=text)
