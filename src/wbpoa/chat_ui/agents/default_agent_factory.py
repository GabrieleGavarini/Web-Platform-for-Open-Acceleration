from __future__ import annotations

from contextlib import ExitStack
from functools import cached_property

from wbpoa.core.corpus import CorpusContext, CorpusWorkspace
from wbpoa.core.driver.neo4j_driver import Neo4jDriver
from wbpoa.core.generation import SingleCorpusAgent
from wbpoa.core.inference import CrossEncoderReranker, Retriever
from wbpoa.core.ingestion.utils import Embedder
from wbpoa.core.model.graph.edge import Edge
from wbpoa.multi_corpora import MultiCorporaContext, MultiCorporaSubAgents, MultiCorporaWorkspace, MultiCorpusAgent
from wbpoa.multi_corpora.settings import MultiCorporaSettings


class DefaultAgentFactory:
    """
    Build Neo4j drivers and agents exactly as the answer commands do, sharing one embedder and reranker.

    The embedder and the cross-encoder are loaded on first use and shared by every agent, since
    both are corpus-independent and costly to load. Each corpus gets its own deep copy of the
    settings with its own ``act_name``.

    Attributes:
        settings: Settings every driver and agent is built from.
    """

    def __init__(self, settings: MultiCorporaSettings) -> None:
        """
        Args:
            settings: Settings every driver and agent is built from; copied per corpus, never mutated.
        """
        self.settings = settings

    @cached_property
    def embedder(self) -> Embedder:
        """The query embedder shared by every retriever, loaded on first use."""
        return Embedder()

    @cached_property
    def reranker(self) -> CrossEncoderReranker:
        """The cross-encoder shared by every retriever, loaded lazily by the reranker itself."""
        inference = self.settings.inference
        token = self.settings.hf_token
        return CrossEncoderReranker(
            inference.rerank_model,
            device=inference.rerank_device,
            token=token.get_secret_value() if token is not None else None,
        )

    def _corpus_settings(self, corpus: CorpusWorkspace) -> MultiCorporaSettings:
        """Return a deep copy of the settings naming `corpus` as the act."""
        settings = self.settings.model_copy(deep=True)
        settings.act_name = corpus.config.name
        return settings

    def driver(self, corpus: CorpusWorkspace) -> Neo4jDriver[Edge]:
        """
        Open a driver on one corpus's partition.

        Args:
            corpus: The corpus whose partition to open.

        Returns:
            An open driver; the caller closes it.
        """
        return Neo4jDriver(self._corpus_settings(corpus), corpus_id=corpus.key)

    def single_agent(self, corpus: CorpusWorkspace, driver: Neo4jDriver[Edge]) -> SingleCorpusAgent:
        """
        Build the single-corpus agent of one corpus over an open driver.

        Args:
            corpus: The corpus answered from.
            driver: An open driver on that corpus's partition.

        Returns:
            The corpus's single-corpus agent.
        """
        settings = self._corpus_settings(corpus)
        driver.ensure_schema(dimensions=self.embedder.dimension)
        retriever = Retriever(driver, self.embedder, reranker=self.reranker, settings=settings)
        return SingleCorpusAgent(retriever, settings=settings, corpus_context=CorpusContext.from_workspace(corpus))

    def multi_agent(self, pairing: MultiCorporaWorkspace, stack: ExitStack) -> MultiCorpusAgent:
        """
        Build the multi-corpus agent of a pairing, its sub agents' drivers registered on `stack`.

        Args:
            pairing: The pairing answered from.
            stack: Exit stack that closes the sub agents' drivers.

        Returns:
            The pairing's multi-corpus agent.
        """
        sub_agents = stack.enter_context(
            MultiCorporaSubAgents(pairing, settings=self.settings, embedder=self.embedder, reranker=self.reranker)
        )
        return MultiCorpusAgent(sub_agents, context=MultiCorporaContext.from_workspace(pairing), settings=self.settings)
