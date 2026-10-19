from __future__ import annotations

import re
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

import structlog
from bs4 import BeautifulSoup, Tag

from wbpoa.core.corpus import InstrumentConfig
from wbpoa.core.model.element import Annex, Article, Instrument, Recital, ValidProvision

log = structlog.get_logger()


class InstrumentIngestion:
    """Turn one configured legal instrument into qualified top-level provisions."""

    _CITABLE_ID = re.compile(r"^(art|rct|anx)_([^\.]+)$")
    _CHAPTER_ID = re.compile(r"^cpt_([^\.]+)$")
    _SECTION_ID = re.compile(r"^cpt_[^\.]+\.sct_(.+)$")

    def ingest(self, source_path: str | Path, corpus_id: str, config: InstrumentConfig) -> Instrument:
        """Ingest one instrument source declared by a corpus manifest.

        Args:
            source_path: Local CELLAR HTML source path.
            corpus_id: Legal-family corpus key owning this instrument copy.
            config: Validated instrument metadata from ``corpus.toml``.

        Returns:
            The instrument and its qualified top-level provisions.

        Raises:
            FileNotFoundError: If ``source_path`` does not exist.
        """
        path = Path(source_path)
        soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
        provisions = self._extract_provisions(soup, corpus_id, config)

        instrument = Instrument(
            key=config.key,
            name=config.name,
            celex_id=config.celex_id,
            role=config.role,
            aliases=config.aliases,
            status=config.status,
            effective_from=config.effective_from,
            effective_to=config.effective_to,
            provisions=provisions,
        )
        log.info(
            "instrument_ingested",
            corpus_id=corpus_id,
            instrument_key=config.key,
            path=str(path),
            provisions=len(instrument.provisions),
        )
        return instrument

    def _extract_provisions(
        self,
        soup: BeautifulSoup,
        corpus_id: str,
        config: InstrumentConfig,
    ) -> list[ValidProvision]:
        """Build one provision for each top-level CELLAR citable container.

        Args:
            soup: Parsed CELLAR document.
            corpus_id: Legal-family corpus key owning the provisions.
            config: Source instrument metadata.

        Returns:
            Citable provisions in document order.
        """
        provisions: list[ValidProvision] = []
        seen_ids: set[str] = set()

        for body in soup.find_all(id=self._CITABLE_ID):
            identifier = str(body.get("id", ""))
            if identifier in seen_ids or self._has_citable_ancestor(body):
                continue
            seen_ids.add(identifier)

            match = self._CITABLE_ID.fullmatch(identifier)
            if match is None:
                continue
            kind, number = match.groups()
            name = self._provision_name(body, kind, number)
            if name is None:
                log.warning("provision_name_not_found", instrument_key=config.key, identifier=identifier)
                continue

            text = self._body_text(body)
            if not text:
                log.warning("provision_text_not_found", instrument_key=config.key, identifier=identifier)
                continue

            provision_path = self._provision_path(kind, number)
            provision_id = self._provision_id(corpus_id, config.key, provision_path)
            identity = {
                "id": provision_id,
                "corpus_id": corpus_id,
                "instrument_key": config.key,
                "celex_id": config.celex_id,
                "provision_path": provision_path,
                "name": name,
                "text": text,
            }
            if kind == "art":
                provisions.append(
                    Article(
                        **identity,
                        chapter=self._ancestor_label(body, self._CHAPTER_ID, "chapter"),
                        section=self._ancestor_label(body, self._SECTION_ID, "section"),
                    )
                )
            elif kind == "rct":
                provisions.append(Recital(**identity))
            else:
                provisions.append(Annex(**identity))

        return provisions

    @staticmethod
    def _provision_id(corpus_id: str, instrument_key: str, provision_path: str) -> UUID:
        """Return a deterministic UUID for one corpus-owned provision copy."""
        return uuid5(NAMESPACE_URL, f"wbpoa:{corpus_id}:{instrument_key}:{provision_path}")

    @staticmethod
    def _provision_path(kind: str, number: str) -> str:
        """Return a canonical path derived from a CELLAR container identifier."""
        labels = {"art": "Article", "rct": "Recital", "anx": "Annex"}
        return f"{labels[kind]} {number}"

    def _has_citable_ancestor(self, body: Tag) -> bool:
        """Return whether a citable container encloses ``body``."""
        return any(self._CITABLE_ID.fullmatch(str(ancestor.get("id", ""))) for ancestor in body.parents)

    @staticmethod
    def _provision_name(body: Tag, kind: str, number: str) -> str | None:
        """Extract the display name from a CELLAR provision heading."""
        if kind == "art":
            heading = body.find("p", class_="oj-ti-art")
            prefix = "Article"
        elif kind == "anx":
            heading = next(
                (
                    paragraph
                    for paragraph in body.find_all("p", class_="oj-doc-ti")
                    if paragraph.get_text(" ", strip=True).lower().startswith("annex")
                ),
                None,
            )
            prefix = "Annex"
        else:
            heading = None
            prefix = "Recital"

        if heading is not None:
            text = " ".join(heading.get_text(" ", strip=True).split())
            if text:
                return text.replace("\xa0", " ")
        return f"{prefix} {number}" if number else None

    @staticmethod
    def _ancestor_label(body: Tag, pattern: re.Pattern[str], prefix: str) -> str | None:
        """Return a normalized structural label from the nearest matching ancestor."""
        for ancestor in body.parents:
            match = pattern.fullmatch(str(ancestor.get("id", "")))
            if match is not None:
                return f"{prefix} {match.group(1)}"
        return None

    @staticmethod
    def _body_text(body: Tag) -> str:
        """Extract provision text while preserving paragraph boundaries."""
        paragraphs = [text for paragraph in body.find_all("p") if (text := paragraph.get_text(" ", strip=True))]
        return "\n".join(paragraphs) if paragraphs else body.get_text(" ", strip=True)
