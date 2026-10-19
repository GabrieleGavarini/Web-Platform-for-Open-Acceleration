from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from wbpoa.chat_ui.agents.chat_target import ChatTarget
    from wbpoa.chat_ui.graph.citation_graph import CitationGraph
    from wbpoa.core.corpus import CorpusWorkspace
    from wbpoa.core.generation.models import QuestionAnswerer


class AgentSource(Protocol):
    """Provides the agent answering a chat target; `AgentPool` in production."""

    def agent(self, target: ChatTarget) -> QuestionAnswerer:
        """
        Return the target's agent.

        Args:
            target: The target to answer.

        Returns:
            The agent.
        """
        ...


class TargetCatalog(Protocol):
    """Lists chat targets and opens corpus graphs for the pages; `AgentPool` in production."""

    def targets(self) -> list[ChatTarget]:
        """
        Return every chat target.

        Returns:
            Single corpora, then pairings.
        """
        ...

    def target(self, target_id: str) -> ChatTarget | None:
        """
        Find a target by id.

        Args:
            target_id: A `ChatTarget.id`.

        Returns:
            The target, or None.
        """
        ...

    def corpus(self, corpus_key: str) -> CorpusWorkspace | None:
        """
        Find a single corpus by key.

        Args:
            corpus_key: Canonical corpus key.

        Returns:
            The corpus, or None.
        """
        ...

    def graph(self, corpus: CorpusWorkspace) -> CitationGraph:
        """
        Return a corpus's citation-graph browser.

        Args:
            corpus: The corpus to browse.

        Returns:
            The browser.
        """
        ...
