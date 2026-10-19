from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final, Protocol

from structlog import get_logger

from wbpoa.core.corpus.corpus_config import InstrumentRole
from wbpoa.core.generation.models import RedactedNode

if TYPE_CHECKING:
    from wbpoa.core.corpus import CorpusContext
    from wbpoa.core.model.graph import Node

log = get_logger()

# "Article 62", "art. 62(2)(a)", "ANNEX III", "Recital 28" -> kind and number; paragraph and
# point suffixes are ignored, since provisions are stored at top level.
_PROVISION: Final = re.compile(
    r"^\s*(?P<kind>art(?:icle)?\.?|recital|annex)\s+(?P<number>[0-9]+[a-z]?|[ivxlc]+)\b", re.IGNORECASE
)
_KINDS: Final = {"art": "Article", "art.": "Article", "article": "Article", "recital": "Recital", "annex": "Annex"}


class ProvisionLookupError(ValueError):
    """Raised when a lookup names an unknown instrument or an unparseable provision."""


class _ProvisionDriver(Protocol):
    """The one driver method the lookup needs. `Neo4jDriver` satisfies it."""

    def qualified_provision(self, instrument_key: str, celex_id: str, provision_path: str) -> Node | None: ...


class ProvisionLookup:
    """
    Fetch one specific provision of one specific instrument, by its number: no search involved.

    Used by the single-corpus agent's `lookup_provision` tool to read provisions that research
    has surfaced -- leads, provisions a passage refers to -- directly, instead of hoping a search
    retrieves them.

    Attributes:
        corpus_context: Corpus identity supplying instrument names, roles, and CELEX identifiers.

    """

    def __init__(self, driver: _ProvisionDriver, corpus_context: CorpusContext) -> None:
        """
        Args:
            driver: Graph access, scoped to the corpus partition.
            corpus_context: Corpus identity supplying instrument names, roles, and CELEX identifiers.

        """
        self._driver = driver
        self.corpus_context = corpus_context

    def instrument_names(self) -> list[str]:
        """
        List the instruments a lookup may name, as the display names shown in passage headers.

        Returns:
            The corpus's instrument display names, in manifest order.

        """
        return list(self.corpus_context.instrument_names.values())

    def lookup(self, instrument: str, provision: str) -> RedactedNode | None:
        """
        Return the provision `provision` of `instrument`, or None when the instrument has no such provision.

        Args:
            instrument: The instrument's display name or key (case-insensitive); the corpus name
                resolves to the corpus's base instrument when there is exactly one.
            provision: The provision, e.g. ``Article 62``, ``Article 62(2)`` (read as Article 62),
                ``Annex III``, or ``Recital 28``.

        Returns:
            The provision as a passage carrying its full text and qualified citation, or None.

        Raises:
            ProvisionLookupError: If `instrument` matches no instrument of the corpus, or
                `provision` is not an article, recital, or annex reference.

        """
        instrument_key = self._instrument_key(instrument)
        provision_path = self._provision_path(provision)
        celex_id = self.corpus_context.instrument_celex_ids.get(instrument_key, "")
        node = self._driver.qualified_provision(instrument_key, celex_id, provision_path)
        log.info("provision_lookup", instrument=instrument_key, provision=provision_path, found=node is not None)
        if node is None:
            return None
        return RedactedNode(
            id=node.id, name=node.name, type=node.type, text=node.text, qualified_citation=node.qualified_citation
        )

    def _instrument_key(self, instrument: str) -> str:
        """
        Resolve an instrument display name, key, or the corpus name to an instrument key.

        Args:
            instrument: The instrument as the model wrote it.

        Returns:
            The instrument key.

        Raises:
            ProvisionLookupError: If nothing matches.

        """
        wanted = instrument.strip().casefold()
        for key, name in self.corpus_context.instrument_names.items():
            if wanted in {key.casefold(), name.casefold()}:
                return key
        base_keys = [key for key, role in self.corpus_context.instrument_roles.items() if role is InstrumentRole.BASE]
        if wanted == self.corpus_context.name.casefold() and len(base_keys) == 1:
            return base_keys[0]
        message = f"Unknown instrument {instrument!r}. Use one of: {', '.join(self.instrument_names())}."
        raise ProvisionLookupError(message)

    @staticmethod
    def _provision_path(provision: str) -> str:
        """
        Normalise a provision reference to its canonical top-level path, e.g. ``Article 62``.

        Args:
            provision: The provision as the model wrote it.

        Returns:
            ``Article <n>``, ``Recital <n>``, or ``Annex <ROMAN>``.

        Raises:
            ProvisionLookupError: If `provision` is not an article, recital, or annex reference.

        """
        match = _PROVISION.match(provision)
        if match is None:
            message = (
                f"Cannot read {provision!r} as a provision: write e.g. 'Article 62', 'Recital 28', or 'Annex III'."
            )
            raise ProvisionLookupError(message)
        kind = _KINDS[match["kind"].casefold()]
        number = match["number"].upper() if kind == "Annex" else match["number"].lower()
        return f"{kind} {number}"
