from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING, Final

from wbpoa.core.model.citation import Citation

if TYPE_CHECKING:
    from wbpoa.core.corpus.corpus_config import InstrumentConfig

_BRACKET: Final = re.compile(r"\[([^\[\]]+)\]")
# "Article 9(1)", "ANNEX III", "Recital 28", or a bare "Annex"; paragraph and point suffixes
# are ignored, since provisions are stored at top level. A dotted number ("Section 4.2.1") is
# an annex point, not a provision number.
_PROVISION: Final = re.compile(
    r"\b(?P<kind>article|recital|annex)\b(?:\s+(?P<number>[0-9]+[a-z]?|[ivxlc]+)\b(?![.\d]))?", re.IGNORECASE
)
# Headers of guidelines and rulings, as rendered by `ContextFormatter.header`.
_DOCUMENT_TYPES: Final = ("guideline", "ruling")


class AnswerCitationParser:
    """Resolve the bracketed citations written in an answer to qualified provisions.

    Answers cite provisions by copying the passage headers they were shown, e.g.
    ``[Article 9(1) Regulation (EU) 2016/679]``, sometimes with a short instrument
    name instead (``[Article 5 AI Act]``), with several provisions in one bracket,
    or with no instrument at all. A bracket's instrument is the longest instrument
    name or alias it ends with. Brackets naming no instrument, and a bare ``Annex``
    without a number, are resolved against the item's grounding context: they count
    only when exactly one context provision matches.

    Attributes:
        names: Instrument display names and aliases, casefolded, mapped to instrument keys,
            longest first.
    """

    def __init__(self, instruments: Iterable[InstrumentConfig]) -> None:
        """
        Index the names under which answers may cite each instrument.

        Args:
            instruments: Instruments of every corpus whose answers will be parsed.
        """
        pairs = {
            name.casefold(): instrument.key
            for instrument in instruments
            for name in (instrument.name, *instrument.aliases)
        }
        self.names: list[tuple[str, str]] = sorted(pairs.items(), key=lambda pair: -len(pair[0]))

    def parse(self, answer: str, context: Iterable[Citation] = ()) -> frozenset[Citation]:
        """
        Collect the provisions an answer cites.

        Args:
            answer: Answer text.
            context: Provisions of the item's grounding context, used to resolve brackets
                without an instrument and bare annexes.

        Returns:
            The distinct cited provisions; guideline and ruling brackets, and brackets
            that cannot be resolved to one provision, contribute nothing.
        """
        shown = frozenset(context)
        cited: set[Citation] = set()
        for bracket in _BRACKET.findall(answer):
            cited.update(self.parse_bracket(bracket, shown))
        return frozenset(cited)

    def parse_bracket(self, bracket: str, context: Iterable[Citation] = ()) -> list[Citation]:
        """
        Resolve the provisions cited inside one bracket.

        Args:
            bracket: The bracket's text, without the square brackets, e.g. ``Article 9(1) GDPR``.
            context: Provisions of the item's grounding context, used to resolve a bracket
                without an instrument and a bare annex.

        Returns:
            The distinct cited provisions in the order they appear in the bracket; empty for a
            guideline or ruling bracket, or one that resolves to no provision.
        """
        text = bracket.strip()
        if text.casefold().startswith(_DOCUMENT_TYPES):
            return []
        shown = frozenset(context)
        instrument_key, provisions = self._split(text)
        cited: list[Citation] = []
        for match in _PROVISION.finditer(provisions):
            citation = self._resolve(match, instrument_key, shown)
            if citation is not None and citation not in cited:
                cited.append(citation)
        return cited

    def _split(self, text: str) -> tuple[str | None, str]:
        """Split a bracket into its instrument key (if it ends with a known name) and the rest."""
        folded = text.casefold()
        for name, key in self.names:
            if folded.endswith(name):
                rest = text[: len(text) - len(name)]
                if not rest or not rest[-1].isalnum():
                    return key, rest
        return None, text

    @staticmethod
    def _resolve(match: re.Match[str], instrument_key: str | None, shown: frozenset[Citation]) -> Citation | None:
        """Turn one provision match into a citation, falling back to the context when underspecified."""
        kind = match["kind"].capitalize()
        number = match["number"]
        if number is not None:
            path = f"{kind} {number.upper() if kind == 'Annex' and not number.isdigit() else number}"
            if instrument_key is not None:
                return Citation(instrument_key, path)
            candidates = [citation for citation in shown if citation.provision_path == path]
        else:
            if kind != "Annex" or instrument_key is None:
                return None
            candidates = [
                citation
                for citation in shown
                if citation.instrument_key == instrument_key and citation.provision_type == "Annex"
            ]
        return candidates[0] if len(candidates) == 1 else None
