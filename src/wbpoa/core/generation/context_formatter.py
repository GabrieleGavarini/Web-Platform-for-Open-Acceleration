from __future__ import annotations

from collections.abc import Set as AbstractSet
from typing import Final
from uuid import UUID

from structlog import get_logger

from wbpoa.core.corpus import CorpusContext
from wbpoa.core.corpus.corpus_config import InstrumentRole
from wbpoa.core.generation.chat_model import budget_texts
from wbpoa.core.generation.models import CitedProvision, RedactedNode

log = get_logger()

# Estimated tokens added per context node by its "[Type Name]\n" header and block separators,
# on top of its own text -- padding so the real prompt never exceeds the computed budget.
_PER_NODE_OVERHEAD_TOKENS: Final = 20

EMPTY_CONTEXT: Final = "(no context retrieved)"
ALREADY_SHOWN: Final = "(already returned by an earlier call; see above)"

LEADS_HEADER: Final = (
    "Leads -- provisions the guidelines and rulings above cite that you have not been shown yet. "
    "They are not passages: never cite them; look up any that bears on the question."
)


class ContextFormatter:
    """
    Render grounding passages as the citation-headed blocks the single-corpus agent reads.

    Each passage becomes one block introduced by the bracketed citation the answer must copy,
    e.g. ``[Article 6 AI Act]``, plus an out-of-band role note for non-base instruments.

    Attributes:
        corpus_context: Corpus identity supplying each instrument's display name and role.

    """

    def __init__(self, corpus_context: CorpusContext) -> None:
        """
        Args:
            corpus_context: Corpus identity supplying each instrument's display name and role.

        """
        self.corpus_context = corpus_context

    def _instrument_name(self, node: RedactedNode) -> str | None:
        """Return the display name of `node`'s instrument, or None for guidelines/rulings."""
        if node.qualified_citation is None:
            return None
        return self.corpus_context.instrument_names.get(node.qualified_citation.instrument_key)

    def _instrument_role(self, node: RedactedNode) -> InstrumentRole | None:
        """Return the retrieval role of `node`'s instrument, or None for guidelines/rulings."""
        if node.qualified_citation is None:
            return None
        return self.corpus_context.instrument_roles.get(node.qualified_citation.instrument_key)

    def header(self, node: RedactedNode) -> str:
        """
        Render one passage's header: the citation form the answer copies, plus a role note.

        The role note sits outside the brackets for non-base instruments, so the model never
        echoes it into a citation.

        Args:
            node: The passage to render the header of.

        Returns:
            ``[<name> <instrument>]`` for provisions of a known instrument, ``[<type> <name>]``
            otherwise, followed by ``(<role> instrument)`` for non-base instruments.

        """
        instrument_name = self._instrument_name(node)
        bracket = f"[{node.name} {instrument_name}]" if instrument_name else f"[{node.type} {node.name}]"
        role = self._instrument_role(node)
        if role is not None and role is not InstrumentRole.BASE:
            return f"{bracket} ({role.value} instrument)"
        return bracket

    def format(self, nodes: list[RedactedNode], budget: int, already_shown: AbstractSet[UUID] = frozenset()) -> str:
        """
        Render `nodes` as one header-and-text block per passage, within a token budget.

        Passages in `already_shown` are collapsed to their header plus `ALREADY_SHOWN`, so a
        repeated passage costs no budget and the new ones stand out. If the remaining texts do
        not fit, each is truncated to an equal share of what is left of `budget` once the
        per-passage header overhead is reserved.

        Args:
            nodes: The passages, in the order they should appear.
            budget: Token budget for the whole rendered block.
            already_shown: Ids of passages the reader has already been shown in full.

        Returns:
            The blocks separated by blank lines, or `EMPTY_CONTEXT` when `nodes` is empty.

        """
        if not nodes:
            return EMPTY_CONTEXT
        fresh = [node for node in nodes if node.id not in already_shown]
        text_budget = max(budget - len(nodes) * _PER_NODE_OVERHEAD_TOKENS, 0)
        raw_texts = [node.text for node in fresh]
        texts = budget_texts(raw_texts, text_budget)
        if texts != raw_texts:
            log.warning("context_truncated", num_context_nodes=len(fresh), per_node_budget=text_budget // len(fresh))
        text_by_id = {node.id: text for node, text in zip(fresh, texts, strict=True)}
        return "\n\n".join(f"{self.header(node)}\n{text_by_id.get(node.id, ALREADY_SHOWN)}" for node in nodes)

    def leads(self, provisions: list[CitedProvision]) -> str:
        """
        Render cited-but-unretrieved provisions as a leads footer for a research result.

        Leads carry no brackets, so they cannot be mistaken for citable passage headers.

        Args:
            provisions: The leads, already filtered and ordered as they should appear.

        Returns:
            `LEADS_HEADER` followed by one ``- <name> (<instrument>) -- cited by N documents``
            line per lead, or an empty string when there are none.

        """
        if not provisions:
            return ""
        lines = [LEADS_HEADER]
        for provision in provisions:
            instrument = (
                self.corpus_context.instrument_names.get(provision.qualified_citation.instrument_key)
                if provision.qualified_citation is not None
                else None
            )
            label = f"{provision.name} ({instrument})" if instrument else f"{provision.type} {provision.name}"
            count = len(provision.cited_by)
            lines.append(f"- {label} -- cited by {count} document{'s' if count != 1 else ''}")
        return "\n".join(lines)
