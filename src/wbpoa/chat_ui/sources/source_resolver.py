from __future__ import annotations

import html
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final
from uuid import uuid4

import markdown2

from wbpoa.chat_ui.app.routes import Routes
from wbpoa.chat_ui.sources.models import UsedSource
from wbpoa.core.generation.answer_citation_parser import AnswerCitationParser
from wbpoa.core.model.citation import Citation

if TYPE_CHECKING:
    from wbpoa.core.corpus import CorpusWorkspace
    from wbpoa.core.generation.models import RedactedNode

EXCERPT_LENGTH: Final = 180

_BRACKET: Final = re.compile(r"\[([^\[\]]+)\]")
_WHITESPACE: Final = re.compile(r"\s+")
# Stands in for a citation link while the Markdown is rendered: letters only, so Markdown leaves it alone.
_PLACEHOLDER: Final = "WBPOAFOOTNOTE{index}X"
_MARKDOWN_EXTRAS: Final = ("break-on-newline", "cuddled-lists", "tables", "strike")
# A draft's unclosed bracket longer than this is a stray bracket, not a citation being written.
_CITATION_MAX_LENGTH: Final = 200


class SourceResolver:
    """
    Link an answer's bracketed citations to its footnotes, and list the sources it actually cites.

    A bracket is resolved with `AnswerCitationParser` against the answer's grounding context;
    only provisions present in that context become sources, so a citation the agent invented
    stays plain text. A resolved bracket keeps its text, e.g. ``[Article 6 GDPR]``, and links to
    the footnote of the first provision it cites; each footnote links to the provision's page.
    Sources are numbered in the order the answer first cites them.

    Attributes:
        corpora: The corpora the answer was grounded in.
    """

    def __init__(self, corpora: Sequence[CorpusWorkspace]) -> None:
        """
        Args:
            corpora: The corpora the answer was grounded in; their instruments' names and aliases
                are the names a bracket may cite.
        """
        self.corpora = tuple(corpora)
        instruments = [instrument for corpus in corpora for instrument in corpus.config.instruments]
        self._parser = AnswerCitationParser(instruments)
        self._instruments = {instrument.key: instrument for instrument in instruments}

    def resolve(self, answer: str, grounding: Sequence[tuple[str, RedactedNode]]) -> tuple[str, list[UsedSource]]:
        """
        Footnote an answer.

        Args:
            answer: The agent's answer text.
            grounding: Every grounding node, paired with the key of the corpus partition holding it.

        Returns:
            ``(answer_html, sources)``: the answer's Markdown rendered as HTML (raw HTML in it
            escaped) whose resolved brackets became links to their footnotes, and the cited
            sources in footnote order, each with a footnote anchor unique to this answer.
        """
        if not answer.strip():
            return "", []
        anchor_prefix = f"fn-{uuid4().hex[:8]}"
        context: dict[Citation, tuple[str, RedactedNode]] = {}
        for corpus_key, node in grounding:
            citation = node.qualified_citation
            if citation is not None:
                context.setdefault(Citation(citation.instrument_key, citation.provision_path), (corpus_key, node))

        sources: dict[Citation, UsedSource] = {}
        markers: list[str] = []
        pieces: list[str] = []
        position = 0
        for match in _BRACKET.finditer(answer):
            pieces.append(answer[position : match.start()])
            position = match.end()
            cited = [citation for citation in self._parser.parse_bracket(match[1], context) if citation in context]
            if not cited:
                # Escaped, so Markdown keeps the brackets as text rather than reading a link.
                pieces.append(f"\\[{match[1]}\\]")
                continue
            cited_sources = [self._source(citation, context[citation], sources, anchor_prefix) for citation in cited]
            pieces.append(_PLACEHOLDER.format(index=len(markers)))
            markers.append(self._citation_link(match[1], cited_sources))
        pieces.append(answer[position:])

        rendered = markdown2.markdown("".join(pieces), safe_mode="escape", extras=list(_MARKDOWN_EXTRAS))
        for index, marker in enumerate(markers):
            rendered = rendered.replace(_PLACEHOLDER.format(index=index), marker)
        return str(rendered).strip(), list(sources.values())

    @staticmethod
    def draft_html(answer: str) -> str:
        """
        Render an answer still being written, so it reads like the finished one while it grows.

        Args:
            answer: The answer so far, possibly cut mid-word or mid-markup.

        Returns:
            The Markdown rendered as HTML (raw HTML in it escaped). Each complete bracket is styled
            as a citation, so the text does not reflow when it becomes a footnote link once the
            answer is complete; a bracket still being written is held back until it closes. An
            opening ``**`` or backtick whose closing one is not written yet is left out, so the
            half-written span shows as plain text rather than as stray markup.
        """
        opening = answer.rfind("[")
        if opening > answer.rfind("]") and len(answer) - opening <= _CITATION_MAX_LENGTH:
            answer = answer[:opening].rstrip()
        for marker in ("**", "`"):
            if answer.count(marker) % 2:
                head, _, tail = answer.rpartition(marker)
                answer = head + tail
        brackets = [match[1] for match in _BRACKET.finditer(answer)]
        placeholders = iter(range(len(brackets)))
        marked = _BRACKET.sub(lambda _: _PLACEHOLDER.format(index=next(placeholders)), answer)
        rendered = str(markdown2.markdown(marked, safe_mode="escape", extras=list(_MARKDOWN_EXTRAS)))
        for index, bracket in enumerate(brackets):
            cite = f'<span class="wbpoa-cite">[{html.escape(bracket, quote=False)}]</span>'
            rendered = rendered.replace(_PLACEHOLDER.format(index=index), cite)
        return rendered.strip()

    def _source(
        self,
        citation: Citation,
        located: tuple[str, RedactedNode],
        sources: dict[Citation, UsedSource],
        anchor_prefix: str,
    ) -> UsedSource:
        """Return the citation's numbered source, numbering it when first cited."""
        if citation not in sources:
            corpus_key, node = located
            instrument = self._instruments[citation.instrument_key]
            number = len(sources) + 1
            sources[citation] = UsedSource(
                number=number,
                corpus_key=corpus_key,
                node_id=node.id,
                provision=citation.provision_path,
                instrument_name=instrument.name,
                celex_id=instrument.celex_id,
                excerpt=self._excerpt(node),
                url=Routes.node_url(corpus_key, node.id),
                anchor=f"{anchor_prefix}-{number}",
            )
        return sources[citation]

    @staticmethod
    def _citation_link(bracket: str, cited: list[UsedSource]) -> str:
        """The bracket, as written, linking to the footnote of the first provision it cites."""
        label = html.escape(
            "; ".join(f"({source.number}) {source.provision}, {source.instrument_name}" for source in cited)
        )
        return (
            f'<a class="wbpoa-cite" href="#{cited[0].anchor}" title="{label}">[{html.escape(bracket, quote=False)}]</a>'
        )

    @staticmethod
    def _excerpt(node: RedactedNode) -> str:
        """The opening of a provision's text, without its leading name line, cut at a word boundary."""
        text = node.text.strip()
        first, _, rest = text.partition("\n")
        if first.strip().casefold() == node.name.casefold():
            text = rest
        text = _WHITESPACE.sub(" ", text).strip()
        if len(text) <= EXCERPT_LENGTH:
            return text
        return text[:EXCERPT_LENGTH].rsplit(" ", 1)[0].rstrip(",;:") + "…"
