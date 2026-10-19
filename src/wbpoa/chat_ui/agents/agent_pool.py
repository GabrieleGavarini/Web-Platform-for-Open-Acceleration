from __future__ import annotations

import csv
from contextlib import ExitStack
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Final, Protocol

from structlog import get_logger

from wbpoa.chat_ui.agents.chat_target import ChatTarget, ChatTargetKind
from wbpoa.chat_ui.graph.citation_graph import CitationGraph, GraphReader

if TYPE_CHECKING:
    from wbpoa.core.corpus import CorpusRegistry, CorpusWorkspace
    from wbpoa.core.generation.models import QuestionAnswerer
    from wbpoa.multi_corpora import MultiCorporaRegistry, MultiCorporaWorkspace

log = get_logger()

EXAMPLE_COUNT: Final = 3


class ClosableGraphReader(GraphReader, Protocol):
    """A graph reader holding a connection the pool must release."""

    def close(self) -> None:
        """Release the reader's connection."""
        ...


class AgentFactory[D: ClosableGraphReader](Protocol):
    """Builds the drivers and agents an `AgentPool` caches; `DefaultAgentFactory` in production."""

    def driver(self, corpus: CorpusWorkspace) -> D:
        """
        Open a driver on one corpus's partition.

        Args:
            corpus: The corpus whose partition to open.

        Returns:
            An open driver; the pool closes it.
        """
        ...

    def single_agent(self, corpus: CorpusWorkspace, driver: D) -> QuestionAnswerer:
        """
        Build the single-corpus agent of one corpus over an open driver.

        Args:
            corpus: The corpus answered from.
            driver: The driver `driver` opened on that corpus.

        Returns:
            The agent.
        """
        ...

    def multi_agent(self, pairing: MultiCorporaWorkspace, stack: ExitStack) -> QuestionAnswerer:
        """
        Build the multi-corpus agent of a pairing.

        Args:
            pairing: The pairing answered from.
            stack: Exit stack that releases whatever the agent opens.

        Returns:
            The agent.
        """
        ...


class AgentPool[D: ClosableGraphReader]:
    """
    Discover chat targets and lazily build, cache, and finally release their agents and drivers.

    One driver is opened per corpus and shared by that corpus's single-corpus agent and its
    citation-graph browser. Multi-corpus agents open their own sub-agent drivers. Agents are
    built from the worker threads answering questions, one at a time under their own lock;
    drivers and graph browsers are opened under a second lock held only briefly, so a graph page
    never waits for an agent's models to load.

    Attributes:
        corpus_registry: Registry of single corpora.
        multi_corpora_registry: Registry of multi-corpora pairings.
        dataset_dir: Directory of question CSVs the example questions come from.
    """

    def __init__(
        self,
        corpus_registry: CorpusRegistry,
        multi_corpora_registry: MultiCorporaRegistry,
        factory: AgentFactory[D],
        *,
        dataset_dir: Path,
    ) -> None:
        """
        Args:
            corpus_registry: Registry of single corpora.
            multi_corpora_registry: Registry of multi-corpora pairings.
            factory: Builds drivers and agents.
            dataset_dir: Directory holding ``<corpus>.csv`` and ``multi-corpora/<pairing>.csv``
                question CSVs; their first questions become the chat's examples.
        """
        self.corpus_registry = corpus_registry
        self.multi_corpora_registry = multi_corpora_registry
        self.dataset_dir = dataset_dir
        self._factory = factory
        self._lock = Lock()
        self._build_lock = Lock()
        self._stack = ExitStack()
        self._targets: list[ChatTarget] | None = None
        self._drivers: dict[str, D] = {}
        self._agents: dict[str, QuestionAnswerer] = {}
        self._graphs: dict[str, CitationGraph] = {}

    def targets(self) -> list[ChatTarget]:
        """
        Return every chat target: each single corpus, then each pairing, in key order.

        Returns:
            The discovered targets, read once and cached.
        """
        if self._targets is None:
            singles = [
                ChatTarget(
                    kind=ChatTargetKind.SINGLE,
                    key=corpus.key,
                    name=corpus.config.name,
                    corpora=(corpus,),
                    examples=self._examples(self.dataset_dir / f"{corpus.key}.csv"),
                )
                for corpus in self.corpus_registry.discover()
            ]
            pairings = [
                ChatTarget(
                    kind=ChatTargetKind.MULTI,
                    key=pairing.key,
                    name=pairing.config.name,
                    corpora=pairing.corpora,
                    examples=self._examples(self.dataset_dir / "multi-corpora" / f"{pairing.key}.csv"),
                )
                for pairing in self.multi_corpora_registry.discover()
            ]
            self._targets = singles + pairings
        return self._targets

    def target(self, target_id: str) -> ChatTarget | None:
        """
        Find a target by its id.

        Args:
            target_id: A `ChatTarget.id`, e.g. ``single:gdpr``.

        Returns:
            The target, or None when no target has that id.
        """
        return next((target for target in self.targets() if target.id == target_id), None)

    def corpus(self, corpus_key: str) -> CorpusWorkspace | None:
        """
        Find a single corpus by key.

        Args:
            corpus_key: Canonical corpus key.

        Returns:
            The corpus workspace, or None when no single corpus has that key.
        """
        return next(
            (
                target.corpora[0]
                for target in self.targets()
                if target.kind == ChatTargetKind.SINGLE and target.key == corpus_key
            ),
            None,
        )

    def agent(self, target: ChatTarget) -> QuestionAnswerer:
        """
        Return the target's agent, building it on first use.

        Args:
            target: The target to answer.

        Returns:
            The cached agent.
        """
        with self._build_lock:
            if target.id not in self._agents:
                log.info("chat_ui_agent_building", target=target.id)
                if target.kind == ChatTargetKind.SINGLE:
                    corpus = target.corpora[0]
                    with self._lock:
                        driver = self._driver(corpus)
                    agent = self._factory.single_agent(corpus, driver)
                else:
                    pairing = self.multi_corpora_registry.load(target.key)
                    with ExitStack() as opened:
                        agent = self._factory.multi_agent(pairing, opened)
                        with self._lock:
                            self._stack.enter_context(opened.pop_all())
                self._agents[target.id] = agent
            return self._agents[target.id]

    def graph(self, corpus: CorpusWorkspace) -> CitationGraph:
        """
        Return the citation-graph browser of one corpus, opening its driver on first use.

        Args:
            corpus: The corpus to browse.

        Returns:
            The cached browser.
        """
        with self._lock:
            if corpus.key not in self._graphs:
                self._graphs[corpus.key] = CitationGraph(corpus, self._driver(corpus))
            return self._graphs[corpus.key]

    def close(self) -> None:
        """Close every driver and agent resource opened so far; the pool can be reused afterwards."""
        with self._build_lock, self._lock:
            self._stack.close()
            self._stack = ExitStack()
            self._drivers.clear()
            self._agents.clear()
            self._graphs.clear()

    def _driver(self, corpus: CorpusWorkspace) -> D:
        """Return the corpus's shared driver, opening it on first use; the caller holds the lock."""
        if corpus.key not in self._drivers:
            driver = self._factory.driver(corpus)
            self._stack.callback(driver.close)
            self._drivers[corpus.key] = driver
        return self._drivers[corpus.key]

    @staticmethod
    def _examples(path: Path) -> tuple[str, ...]:
        """Return the first questions of a dataset CSV with a ``question`` column, or none when it is absent."""
        if not path.is_file():
            return ()
        with path.open(encoding="utf-8", newline="") as handle:
            questions = (row.get("question", "").strip() for row in csv.DictReader(handle))
            return tuple(question for question, _ in zip(filter(None, questions), range(EXAMPLE_COUNT), strict=False))
