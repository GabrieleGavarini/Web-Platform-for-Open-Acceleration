from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from typing import TYPE_CHECKING

from structlog import get_logger

from wbpoa.core.corpus.context import CorpusContext
from wbpoa.core.driver.neo4j_driver import Neo4jDriver
from wbpoa.core.generation.single_corpus_agent import SingleCorpusAgent
from wbpoa.core.inference import Retriever

if TYPE_CHECKING:
    from types import TracebackType

    from wbpoa.core.corpus.corpus_registry import CorpusWorkspace
    from wbpoa.core.inference.rerank import Reranker
    from wbpoa.core.ingestion.utils import Embedder
    from wbpoa.core.model.graph.edge import Edge
    from wbpoa.multi_corpora.corpus.multi_corpora_registry import MultiCorporaWorkspace
    from wbpoa.multi_corpora.settings import MultiCorporaSettings

log = get_logger()

type DriverFactory = Callable[[MultiCorporaSettings, str], Neo4jDriver[Edge]]


def _default_driver_factory(settings: MultiCorporaSettings, corpus_id: str) -> Neo4jDriver[Edge]:
    """Open a Neo4j driver scoped to `corpus_id`'s graph partition."""
    return Neo4jDriver(settings, corpus_id=corpus_id)


class MultiCorporaSubAgents:
    """
    Context manager building the two per-corpus sub agents of a multi-corpora pairing.

    For each sub-corpus it opens a Neo4j driver scoped to that corpus's partition, ensures its
    schema, and builds an unmodified `Retriever` and `SingleCorpusAgent` exactly as the
    single-corpus commands do -- with that corpus's own settings copy and `CorpusContext`. The
    embedder and reranker are shared, since both are corpus-independent and costly to load.
    Every driver opened is closed on exit, including when building the second sub agent fails.

    Attributes:
        workspace: The pairing whose sub-corpora are opened.

    """

    def __init__(
        self,
        workspace: MultiCorporaWorkspace,
        *,
        settings: MultiCorporaSettings,
        embedder: Embedder,
        reranker: Reranker | None = None,
        driver_factory: DriverFactory | None = None,
    ) -> None:
        """
        Args:
            workspace: The pairing whose two sub-corpora become sub agents.
            settings: MultiCorporaSettings shared by both sub agents; each gets a deep copy with its own
                `act_name`, so per-corpus overrides never leak across the pair.
            embedder: Query embedder shared by both retrievers.
            reranker: Cross-encoder shared by both retrievers; None keeps the retriever's no-op default.
            driver_factory: Opens the driver for one corpus key; defaults to `Neo4jDriver`.

        """
        self.workspace = workspace
        self._settings = settings
        self._embedder = embedder
        self._reranker = reranker
        self._driver_factory = driver_factory or _default_driver_factory
        self._stack: ExitStack | None = None

    def __enter__(self) -> tuple[SingleCorpusAgent, SingleCorpusAgent]:
        """
        Open both sub-corpora and build their single-corpus agents.

        Returns:
            The two single-corpus agents, in the pairing's manifest order.

        """
        with ExitStack() as stack:
            first, second = self.workspace.corpora
            agents = (self._build(first, stack), self._build(second, stack))
            self._stack = stack.pop_all()
        return agents

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close every driver opened by `__enter__`."""
        if self._stack is not None:
            self._stack.close()
            self._stack = None

    def _build(self, corpus: CorpusWorkspace, stack: ExitStack) -> SingleCorpusAgent:
        """
        Open one sub-corpus's driver and build its single-corpus agent.

        Args:
            corpus: The sub-corpus to open.
            stack: Exit stack the driver is registered on, so it is closed with the pair.

        Returns:
            The sub-corpus's single-corpus agent, bound to its own partition and `CorpusContext`.

        """
        settings = self._settings.model_copy(deep=True)
        settings.act_name = corpus.config.name
        driver = stack.enter_context(self._driver_factory(settings, corpus.key))
        driver.ensure_schema(dimensions=self._embedder.dimension)
        retriever = Retriever(driver, self._embedder, reranker=self._reranker, settings=settings)
        log.info("multi_corpora_sub_agent_opened", multi_corpora=self.workspace.key, corpus=corpus.key)
        return SingleCorpusAgent(retriever, settings=settings, corpus_context=CorpusContext.from_workspace(corpus))
