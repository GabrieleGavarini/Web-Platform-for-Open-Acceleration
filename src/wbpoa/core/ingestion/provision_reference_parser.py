from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from uuid import UUID

import structlog

from wbpoa.core.corpus.corpus_config import InstrumentRelationType
from wbpoa.core.model.element.corpus.corpus import Corpus
from wbpoa.core.model.element.corpus.instrument import Instrument
from wbpoa.core.model.element.corpus.provision import Provision
from wbpoa.core.model.operative_relation import OperativeOperation, OperativeRelation

_WHITESPACE = re.compile(r"\s+")
log = structlog.get_logger()

# Abbreviations accepted in place of a leading label. Scholarship and case
# law abbreviate where the official text does not, so "Art. 6(2) AI Act" and
# "Article 6(2) of the AI Act" must resolve alike. Keys are lowercased labels.
_LABEL_ABBREVIATIONS: dict[str, tuple[str, ...]] = {
    "article": ("art", "artt"),
    "annex": ("ann",),
    "chapter": ("chap", "ch"),
    "paragraph": ("para", "par"),
    "recital": ("rec",),
    "section": ("sect",),
    "title": ("tit",),
}

# A subdivision below the level a citation name addresses, as in the "(g)" of
# "Article 1(g)". Values are parenthesized, numeric or roman; requiring one of
# those shapes keeps a stray "point of departure" from reading as a
# subdivision. Atomic because the three shapes are distinguished by their
# leading character, so there is never a legitimate reason to backtrack into
# a different one once a shape has matched.
_SUBDIVISION_VALUE = r"(?>\([^()\s]{1,12}\)|\d{1,4}|[ivxlc]{1,7})"
_SUBDIVISION_ORDINAL = r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last|final)"
_SUBDIVISION_LABEL = r"(?:sub-?)?(?:paragraph|point|indent|letter|sentence|subsection|clause|sub)s?\.?"
# One subdivision, in any of the three notations that occur in practice:
# "(g)" and "(2)(g)"; ", point (g)" and ", second subparagraph"; "§ 2".
# Atomic for the same reason as _SUBDIVISION_VALUE: the three notations are
# distinguished by their leading token (an opening paren, a comma/space, or
# "§"), so once one matches there is no ambiguity left to backtrack through.
_SUBDIVISION_STEP = (
    r"(?>"
    rf"\ ?\([^()\s]{{1,12}}\)"
    rf"|,?\ (?:the\ )?(?:{_SUBDIVISION_ORDINAL}\ )?{_SUBDIVISION_LABEL}"
    rf"(?:\ {_SUBDIVISION_VALUE})?"
    rf"|\ §{{1,2}}\ ?{_SUBDIVISION_VALUE}"
    r")"
)
# Subdivisions may themselves be enumerated: "Article 1(g) and (h)",
# "Article 1, points (a) to (c)".
_SUBDIVISION_ENUMERATION = rf"(?:\ (?:and|or|to|through)\ {_SUBDIVISION_VALUE})*"
# The repetition is bounded because each step is an alternation, and an
# unbounded one would let a failing branch explore too many ways to divide the
# same text. Six covers "Article 6(2), point (a), second indent".
_SUBDIVISION_TAIL = rf"(?:{_SUBDIVISION_STEP}{_SUBDIVISION_ENUMERATION}){{0,6}}"

# Sibling citation names sharing one leading label, as in "Articles 8 to 15
# GDPR" or "Arts. 1, 5 AI Act". Only the first sibling resolves to an element;
# the rest are consumed so that the reference is recognized as
# corpus-qualified as a whole, rather than leaving its first item to match
# bare. Expanding a range into its members is a separate concern.
_SIBLING_LIST = (
    r"(?:(?:,|\ and|\ or|\ to|\ through|\ ?[\u2013\u2014-])"
    r"\ ?(?:\d{1,4}(?:\.\d{1,3})*|[ivxlc]{1,7}))*"
)

# Joins a citation name to a trailing corpus name. All of "Article 1 AI Act",
# "Article 1 of AI Act", "Article 1 of the AI Act" and "Article 1, point (g),
# of the AI Act" occur; "the" is only accepted after "of", since "Article 1
# the AI Act" is not a form anyone writes.
_CORPUS_CONNECTOR = r",?\ (?:of\ (?:the\ )?)?"

# A standalone acronym such as "GDPR" or "TFEU": two to six uppercase letters
# not embedded in a longer word. Case-sensitive (unlike the rest of this
# module's matching) because "looks like an acronym" is exactly the signal
# wanted here; folding case would fire on ordinary capitalized words.
#
# `_find_foreign_qualifier` additionally excludes two shapes that empirical
# testing found otherwise collide with genuine self-citations:
#   - Short roman numerals (II, III, IV, ... XX): "Tit. II, Art. 5" (a Title
#     number) and "Annex II, IV, V or VI" (a sibling list of annex numbers)
#     both contain roman-numeral tokens indistinguishable in shape from a real
#     acronym, and both are extremely common in EU legal citation style.
#   - A small set of common English words that appear in all-caps headings
#     ("...IS ARTICLE 48 APPLICABLE?") or as enumeration connectors ("Annex IX
#     (complete) OR - Annex X"), which read as valid 2-6 letter acronyms once
#     case-folded to upper case. This list is deliberately bounded to common
#     function words, not arbitrary capitalized document text.
_BARE_ACRONYM = r"(?<!\w)[A-Z]{2,6}(?!\w)"

_ROMAN_NUMERAL_TOKENS = frozenset(
    {
        "II",
        "III",
        "IV",
        "VI",
        "VII",
        "VIII",
        "IX",
        "XI",
        "XII",
        "XIII",
        "XIV",
        "XV",
        "XVI",
        "XVII",
        "XVIII",
        "XIX",
        "XX",
    }
)
_COMMON_WORD_TOKENS = frozenset(
    {
        "OF",
        "IS",
        "IN",
        "TO",
        "OR",
        "BY",
        "AT",
        "AS",
        "ON",
        "IT",
        "BE",
        "AN",
        "IF",
        "NO",
        "SO",
        "AND",
        "ARE",
        "FOR",
        "NOT",
        "ALL",
        "ANY",
        "CAN",
        "MAY",
        "WAS",
        "HAS",
        "HAD",
        "THE",
        "BUT",
        "OUT",
        "PER",
        "VIA",
        "NOR",
        "TOO",
        "YET",
        "HOW",
        "WHY",
        "WHO",
        "ITS",
        "WITH",
        "FROM",
        "THIS",
        "THAT",
        "WILL",
        "WHEN",
        "WERE",
        "BEEN",
        "HAVE",
        "THAN",
        "THEN",
        "SUCH",
        "EACH",
        "INTO",
        "ONLY",
        "ALSO",
        "MOST",
        "SOME",
        "MUST",
        "EVEN",
        "WHAT",
        "YOUR",
        "OVER",
        "VERY",
        "JUST",
        "MORE",
        "BOTH",
        "WHICH",
        "WHERE",
        "THEIR",
        "THESE",
        "THOSE",
        "WOULD",
        "COULD",
        "SHALL",
        "UNDER",
        "ABOUT",
        "AFTER",
        "BEING",
        "WHOSE",
        "WHILE",
        "USING",
        "BASED",
        "GIVEN",
        "EVERY",
        "OTHER",
        "SHOULD",
        "DURING",
        "BEFORE",
        "BECOME",
        "THROUGH",
    }
)
_EXCLUDED_ACRONYM_TOKENS = _ROMAN_NUMERAL_TOKENS | _COMMON_WORD_TOKENS

# A spelled-out instrument name ending in a common legal-instrument noun, as in
# "Clinical Trials Regulation" or "Berne Convention". Bounded at five leading
# capitalized words to keep backtracking cheap.
_LEGAL_INSTRUMENT_NOUN = r"(?:Directive|Regulation|Convention|Treaty|Protocol|Code|Charter|Act)"
_SPELLED_OUT_INSTRUMENT_NAME = rf"(?:[A-Z][\w'\u2019-]*\ ){{1,5}}{_LEGAL_INSTRUMENT_NOUN}"

# Either signal: a bare acronym or a spelled-out name. Captured in a named
# group so the acronym-exclusion check can be applied to just the marker text,
# not to any subdivision/sibling/connector text matched alongside it.
_FOREIGN_MARKER = rf"(?:{_BARE_ACRONYM}|{_SPELLED_OUT_INSTRUMENT_NAME})"

# Width of the lookbehind window checked for a prefixed marker ("GDPR Article
# 25", "Clinical Trials Regulation, Article 5"). Generous enough to contain
# the longest spelled-out-name shape (5 capitalized words + noun + comma +
# space) with margin.
_FOREIGN_PREFIX_WINDOW = 90

# Matched via .search() against a short trailing window ending exactly at a
# bare match's start (\Z anchors to that boundary). Tolerates an optional
# comma before the mandatory space, matching how such names are actually
# cited ("MDR, Article 2(2)", "Clinical Trials Regulation, Article 5").
_FOREIGN_PREFIX = re.compile(rf"(?P<marker>{_FOREIGN_MARKER}),?\ \Z")

# Matched via .match(text, pos=<bare match end>); reuses the exact connector
# grammar a *recognized* qualified suffix already accepts (subdivision tail,
# sibling list, corpus connector), scoped case-insensitive, so this heuristic
# never fires on less surrounding text than a real qualifier would need.
_FOREIGN_SUFFIX = re.compile(
    rf"(?i:{_SUBDIVISION_TAIL}{_SIBLING_LIST}{_CORPUS_CONNECTOR})(?P<marker>{_FOREIGN_MARKER})"
)

# The informal name of the pre-GDPR advisory body, frequently mistaken for a
# citation to "Article 29" itself. Never a genuine self-citation: no family
# instrument's Article 29 content is ever actually about a working party.
_NAMED_BODY_SUFFIX = re.compile(r"(?i:\ (?:the\ )?(?:data\ protection\ )?working\ part(?:y|ies))")


def normalize(text: str) -> str:
    """
    Fold a string into a form safe for literal comparison.

    Applies NFKC normalization, which maps the non-breaking spaces and
    typographic variants found in EUR-Lex exports onto plain ASCII spaces,
    then collapses runs of whitespace so that a citation split across a line
    break still matches its single-spaced form.

    Args:
        text: The string to normalize.

    Returns:
        The normalized string, stripped of leading and trailing whitespace.

    """
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


# Reserved group names for the shared corpus-qualifier decorations, disjoint
# from the per-element "g<n>" groups in `elements_by_group`.
_QUALIFIED_PREFIX_GROUP = "qualified_prefix"
_QUALIFIED_SUFFIX_GROUP = "qualified_suffix"


@dataclass(frozen=True)
class ProvisionMatcher:
    """
    A compiled alternation over every citation name in a corpus.

    Attributes:
        pattern: A single regex whose named groups each correspond to one
            citation name, decorated with at most one shared, optional
            corpus-qualifier prefix or suffix ("AI Act Article 51", "Article
            51 of the AI Act"). Matching all names at once, and sharing the
            qualifier decorations across all of them, keeps parsing linear in
            the length of the text rather than in the size of the corpus.
        elements_by_group: Maps each per-element group name to the element it
            denotes. Several groups may map to the same element when that
            element has multiple citation names.

    """

    pattern: re.Pattern[str]
    provisions_by_group: dict[str, Provision]


class ProvisionReferenceParser:
    """Find references from text to the provisions of one instrument."""

    def __init__(self) -> None:
        self._matchers: dict[tuple[str, bool, bool], ProvisionMatcher] = {}

    def build_matcher(
        self,
        instrument: Instrument,
        *,
        allow_plural_labels: bool = True,
        allow_abbreviated_labels: bool = True,
    ) -> ProvisionMatcher:
        """
        Compile a matcher for a corpus, reusing a cached one when possible.

        The matcher always recognizes every form of a citation name: bare
        ("Article 51"), corpus-prefixed ("AI Act Article 51") and
        corpus-suffixed ("Article 51 of the AI Act"), the last of these
        tolerating subdivisions between the name and the corpus name
        ("Article 51(1), point (b), of the AI Act"). Which forms a caller
        cares about is decided at parse time, so a single compiled pattern
        serves both readings.

        The cache is keyed on the corpus id and the options, so repeatedly
        parsing every element of a corpus against that same corpus compiles
        the pattern once. The cache assumes element citation names are stable
        for the lifetime of a corpus id; mutating them requires a new parser.

        Args:
            instrument: The instrument whose provisions may be referenced.
            allow_plural_labels: Whether to accept a plural leading label, so
                that "Articles 51 and 52" matches the first of them rather
                than neither.
            allow_abbreviated_labels: Whether to accept the abbreviations in
                `_LABEL_ABBREVIATIONS` in place of a leading label, as
                scholarship and case law use them freely.

        Returns:
            The compiled matcher.

        Raises:
            ValueError: If no element yields a usable citation name, which
                would otherwise compile to a pattern matching everything.

        """
        key = (instrument.key, allow_plural_labels, allow_abbreviated_labels)
        if key in self._matchers:
            return self._matchers[key]

        instrument_names = [normalize(instrument.name), *(normalize(alias) for alias in instrument.aliases)]
        escaped_names = sorted({re.escape(name) for name in instrument_names if name}, key=len, reverse=True)

        provisions_by_group: dict[str, Provision] = {}
        # (citation name, core branch, group), one per name.
        cores: list[tuple[str, str, str]] = []

        for element in instrument.provisions:
            for name in sorted({normalize(name) for name in element.to_citation_names}):
                if not name:
                    continue

                group = f"g{len(provisions_by_group)}"
                provisions_by_group[group] = element
                core = self._to_branch(
                    name,
                    allow_plural_labels=allow_plural_labels,
                    allow_abbreviated_labels=allow_abbreviated_labels,
                )
                cores.append((name, core, group))

        if not cores:
            raise ValueError(f"Instrument {instrument.name!r} yields no citation names.")

        # Most specific citation name first, since Python's alternation returns
        # the leftmost branch that matches rather than the longest: "Article
        # 5.1" must win over "Article 5" at the same position. Ties break on
        # the core itself, to keep the compiled pattern independent of set
        # iteration order.
        cores.sort(key=lambda item: (len(item[0]), item[1]), reverse=True)

        core_alternation = "|".join(f"(?P<{group}>{core})" for _, core, group in cores)

        # The corpus-qualifier prefix ("AI Act Article 51") and suffix
        # ("Article 51, point (b), of the AI Act") are shared across every
        # citation name rather than compiled once per name: the prior design
        # multiplied the expensive, backtracking-prone subdivision-tail regex
        # by every citation name in the corpus, which dominated parse time on
        # long documents. Sharing it once keeps that cost independent of
        # corpus size. Both remain optional and greedy, so a qualified
        # reading is preferred over a bare one, matching the prior behavior.
        if escaped_names:
            escaped_corpus = f"(?:{'|'.join(escaped_names)})"
            # ",?" mirrors the suffix's own comma tolerance below, so that
            # "AI Act, Article 6(1)" and "MDR, Article 2(2)" are recognized as
            # qualified rather than falling through to bare matching.
            prefix = rf"(?:(?P<{_QUALIFIED_PREFIX_GROUP}>{escaped_corpus},?\ ))?"
            suffix = rf"(?:(?P<{_QUALIFIED_SUFFIX_GROUP}>{_SUBDIVISION_TAIL}{_SIBLING_LIST}{_CORPUS_CONNECTOR}{escaped_corpus}))?"
        else:
            prefix = ""
            suffix = ""

        pattern = re.compile(
            rf"(?<!\w){prefix}(?:{core_alternation}){suffix}(?!\w)",
            re.IGNORECASE,
        )

        matcher = ProvisionMatcher(pattern=pattern, provisions_by_group=provisions_by_group)
        self._matchers[key] = matcher
        return matcher

    @staticmethod
    def _to_branch(
        name: str,
        *,
        allow_plural_labels: bool,
        allow_abbreviated_labels: bool,
    ) -> str:
        """
        Turn a citation name into a regex branch.

        Args:
            name: A normalized citation name, such as "Article 51".
            allow_plural_labels: Whether the leading label may be pluralized.
            allow_abbreviated_labels: Whether the leading label may be
                abbreviated.

        Returns:
            The escaped name, with its first token widened to the accepted
            spellings of that label.

        """
        label, separator, rest = name.partition(" ")
        if not separator:
            return re.escape(name)

        spellings = [re.escape(label)]
        if allow_abbreviated_labels:
            spellings += [re.escape(abbreviation) for abbreviation in _LABEL_ABBREVIATIONS.get(label.lower(), ())]

        leading = f"(?:{'|'.join(spellings)})" if len(spellings) > 1 else spellings[0]

        # "e?s" so that "Annexes" pluralizes "Annex" as well as "Articles"
        # does "Article".
        plural = "(?:e?s)?" if allow_plural_labels else ""
        # The optional period covers "Art." and "Arts."; it is harmless after
        # a label spelled out in full.
        stop = r"\.?" if allow_abbreviated_labels else ""
        return rf"{leading}{plural}{stop}\ {re.escape(rest)}"

    @staticmethod
    def _is_qualified(matched: dict[str, str | None]) -> bool:
        """
        Report whether a match carried a corpus-qualifier prefix or suffix.

        Args:
            matched: The named-group dict of a match against a compiled
                `ProvisionMatcher.pattern`.

        Returns:
            Whether the corpus-qualifier prefix or suffix group fired.

        """
        return matched.get(_QUALIFIED_PREFIX_GROUP) is not None or matched.get(_QUALIFIED_SUFFIX_GROUP) is not None

    @staticmethod
    def _spans_overlap(first: tuple[int, int], second: tuple[int, int]) -> bool:
        """
        Report whether two half-open character spans overlap.

        Args:
            first: A `(start, end)` span.
            second: A `(start, end)` span.

        Returns:
            Whether the spans share at least one character position.

        """
        return first[0] < second[1] and second[0] < first[1]

    @staticmethod
    def _find_foreign_qualifier(text: str, span: tuple[int, int]) -> str | None:
        """
        Look for an adjacent marker suggesting a bare match names another instrument.

        Checks, using the same connector grammar a recognized corpus qualifier
        would use, for either a short adjacent acronym ("GDPR", "TFEU") or a
        spelled-out name ending in a common legal-instrument noun ("of the
        Berne Convention", "Clinical Trials Regulation, "). Excludes
        roman-numeral and common-English-word tokens, which otherwise collide
        with structural markers like "Tit. II" or enumeration connectors like
        "... OR ...". Also checks for "Article 29 (Data Protection) Working
        Party", the name of a pre-GDPR advisory body frequently mistaken for a
        citation to that article number.

        Text a *family* instrument's own name or alias would recognize never
        reaches this check: it would already have produced an overlapping
        qualified span and shadowed this bare match before
        `parse_family_references` groups matches by span.

        Args:
            text: The normalized text the span was matched against.
            span: The `(start, end)` span of the bare citation match.

        Returns:
            The adjacent text that looks like a foreign qualifier, or `None`.

        """
        start, end = span
        window = text[max(0, start - _FOREIGN_PREFIX_WINDOW) : start]
        prefix_match = _FOREIGN_PREFIX.search(window)
        if prefix_match is not None and prefix_match.group("marker").upper() not in _EXCLUDED_ACRONYM_TOKENS:
            return prefix_match.group()
        suffix_match = _FOREIGN_SUFFIX.match(text, end)
        if suffix_match is not None and suffix_match.group("marker").upper() not in _EXCLUDED_ACRONYM_TOKENS:
            return suffix_match.group()
        named_body_match = _NAMED_BODY_SUFFIX.match(text, end)
        return named_body_match.group() if named_body_match is not None else None

    def parse_for_reference(
        self,
        text: str,
        instrument: Instrument,
        *,
        include_corpus_name: bool = True,
        source: Provision | None = None,
        **matcher_options: bool,
    ) -> list[Provision]:
        """
        Find the provisions referenced by a text.

        A reference may name the corpus, either before the citation name
        ("AI Act Article 51") or after it ("Article 51 AI Act", "Article 51 of
        AI Act", "Article 51 of the AI Act", "Art. 51(1), point (b), of the AI
        Act"), or leave it implicit ("Article 51"), the latter being the usual
        form for self-citation within a single instrument. Matching is
        case-insensitive so that a heading rendered as "ANNEX XII" is
        recognized as the same reference as an inline "Annex XII".

        Args:
            text: The text to parse.
            instrument: The instrument that can be referenced.
            include_corpus_name: Whether corpus-qualified references count.
                When false, only bare references are returned and a qualified
                one is skipped in full, so "Article 51, point (b), of the AI
                Act" yields nothing rather than a bare reference to Article 51.
            source: The element the text belongs to, when parsing a corpus
                against itself. It is excluded from the results, since an
                element's own heading almost always appears in its own text and
                would otherwise register as a reference to itself.
            **matcher_options: Further options forwarded to `build_matcher`.

        Returns:
            The referenced elements, deduplicated and ordered by first
            appearance in the text. Empty when nothing is referenced.

        """
        matcher = self.build_matcher(instrument, **matcher_options)

        references: dict[UUID, Provision] = {}
        for match in matcher.pattern.finditer(normalize(text)):
            matched = match.groupdict()
            group = next((g for g in matcher.provisions_by_group if matched.get(g) is not None), None)
            if group is None:
                continue
            if not include_corpus_name and self._is_qualified(matched):
                continue
            element = matcher.provisions_by_group[group]
            if source is not None and element.id == source.id:
                continue
            references.setdefault(element.id, element)

        return list(references.values())

    def parse_family_references(self, text: str, corpus: Corpus) -> list[Provision]:
        """Resolve family-document citations only when qualified or globally unique.

        Bare matches adjacent to what looks like a reference to a different,
        non-family instrument (an unrecognized acronym, a spelled-out name
        ending in a common legal-instrument noun, or the "Article 29 Working
        Party" body name) are excluded rather than resolved, since they are
        unlikely to actually be about a family instrument.

        Args:
            text: Guideline or ruling text that may cite any family instrument.
            corpus: Legal-family corpus containing every candidate instrument.

        Returns:
            Qualified or unambiguous provisions, deduplicated by identifier.
        """
        normalized_text = normalize(text)
        qualified: dict[UUID, Provision] = {}
        qualified_spans: list[tuple[int, int]] = []
        bare_matches: list[tuple[tuple[int, int], Provision]] = []

        for instrument in corpus.instruments:
            matcher = self.build_matcher(instrument)
            for match in matcher.pattern.finditer(normalized_text):
                matched = match.groupdict()
                group = next((name for name in matcher.provisions_by_group if matched.get(name) is not None), None)
                if group is None:
                    continue
                provision = matcher.provisions_by_group[group]
                if self._is_qualified(matched):
                    qualified.setdefault(provision.id, provision)
                    qualified_spans.append(match.span())
                else:
                    bare_matches.append((match.span(), provision))

        # A qualified match's span always strictly contains the span of a
        # bare match at the same core position (the qualifier prefix/suffix
        # only ever adds characters around the core citation name). A bare
        # match overlapping a qualified one is therefore a matcher for the
        # wrong instrument backtracking past a qualifier it doesn't
        # recognize, not a genuine second reading, and must be discarded
        # rather than left to resolve on its own in an uncontested span
        # bucket.
        bare_by_span: dict[tuple[int, int], list[Provision]] = {}
        for span, provision in bare_matches:
            if any(self._spans_overlap(span, qualified_span) for qualified_span in qualified_spans):
                continue
            bare_by_span.setdefault(span, []).append(provision)

        resolved = dict(qualified)
        for span, candidates in bare_by_span.items():
            foreign_qualifier = self._find_foreign_qualifier(normalized_text, span)
            if foreign_qualifier is not None:
                log.warning(
                    "foreign_qualified_reference_skipped",
                    citation=normalized_text[slice(*span)],
                    adjacent_text=foreign_qualifier,
                    candidates=[candidate.qualified_citation.model_dump(mode="json") for candidate in candidates],
                )
                continue

            unique = {candidate.qualified_citation: candidate for candidate in candidates}
            if len(unique) == 1:
                provision = next(iter(unique.values()))
                resolved.setdefault(provision.id, provision)
            else:
                log.warning(
                    "ambiguous_family_provision_reference",
                    citation=normalized_text[slice(*span)],
                    candidates=[candidate.qualified_citation.model_dump(mode="json") for candidate in candidates],
                )

        return list(resolved.values())

    def parse_operative_relations(
        self,
        source_instrument: Instrument,
        target_instrument: Instrument,
        relation_type: InstrumentRelationType,
    ) -> list[OperativeRelation]:
        """
        Derive operative provision relations from a declared instrument relation.

        Parses every provision of `source_instrument` for explicit,
        corpus-qualified references to `target_instrument` (e.g. "Article 5
        of Regulation (EU) 2024/1689"). Bare references are ignored: within a
        source instrument's own text, a bare citation is already claimed as a
        self-reference by the ordinary self-citation resolution pass, so
        treating it as also possibly operative would be redundant, or wrong
        when the source instrument happens to share a provision number with
        the target.

        Args:
            source_instrument: The instrument whose text is parsed, e.g. an
                amending act.
            target_instrument: The instrument its qualified references are
                resolved against, e.g. the base act it amends.
            relation_type: The declared legal relationship between the two
                instruments, stamped onto every resolved relation.

        Returns:
            One `OperativeRelation` per distinct `(source provision, target
            provision)` pair resolved, in document and match order. Empty
            when the source instrument names the target but no qualified
            reference is found.

        """
        matcher = self.build_matcher(target_instrument)
        relations: dict[tuple[UUID, UUID], OperativeRelation] = {}

        for provision in source_instrument.provisions:
            normalized_text = normalize(provision.text)
            for match in matcher.pattern.finditer(normalized_text):
                matched = match.groupdict()
                group = next((name for name in matcher.provisions_by_group if matched.get(name) is not None), None)
                if group is None or not self._is_qualified(matched):
                    continue
                target = matcher.provisions_by_group[group]
                key = (provision.id, target.id)
                relations.setdefault(
                    key,
                    OperativeRelation(
                        source_provision_id=provision.id,
                        source_citation=provision.qualified_citation,
                        target_provision_id=target.id,
                        target_citation=target.qualified_citation,
                        relation_type=relation_type,
                        operation=OperativeOperation.MODIFIES,
                        match_span=match.span(),
                        matched_text=normalized_text[slice(*match.span())],
                    ),
                )

        return list(relations.values())
