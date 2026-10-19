import time
from abc import ABC, abstractmethod
from collections.abc import Hashable, Mapping, Sequence
from uuid import UUID

from pydantic import RootModel
from structlog import get_logger

from wbpoa.core.model.element import Element, Instrument, Provision, ValidProvision
from wbpoa.core.model.graph import Edge, Node
from wbpoa.single_corpus.graph_builder.base_graph_builder import NODES_PATH_NOT_PROVIDED, BaseGraphBuilder

log = get_logger()


class GenericGraphBuilder[E: Edge, K: Hashable](BaseGraphBuilder[E], ABC):
    """
    Generic graph builder that creates nodes and edges from elements in a corpus, purely in-memory. This class is
    abstract and must be subclassed to implement _create_or_merge_edge. Use sync_to_driver() to write the result
    to Neo4j once nodes and edges are ready.
    """

    @staticmethod
    def _owning_instrument(element: Element, instruments_by_key: Mapping[str, Instrument] | None) -> Instrument | None:
        """Look up the owning `Instrument` for a `Provision` element, if a lookup was supplied."""
        if not isinstance(element, Provision) or instruments_by_key is None:
            return None
        return instruments_by_key.get(element.instrument_key)

    def _create_node(self, element: Element, instruments_by_key: Mapping[str, Instrument] | None = None) -> Node:
        """
        Create a Node from an Element (a provision or a Guideline/Paper). This does not include the arch. Uses the summarizer and the embedder to create summaries and embeddings.

        Args:
            element (Element): The element to create the node from.
            instruments_by_key (Mapping[str, Instrument] | None): Lookup of the corpus's instruments
                by key, used to stamp a provision node with its owning instrument's role, status,
                and effective dates. None (the default) leaves those fields unset.

        Returns:
            Node: The created node.

        """
        summary = self.summarizer.create_summary(element.text)

        summary_embedding = self.embedder.create_embedding(summary)

        instrument = self._owning_instrument(element, instruments_by_key)
        node = Node(
            corpus_id=self.corpus_id,
            instrument_key=element.instrument_key if isinstance(element, Provision) else None,
            celex_id=element.celex_id if isinstance(element, Provision) else None,
            provision_path=element.provision_path if isinstance(element, Provision) else None,
            instrument_role=instrument.role if instrument is not None else None,
            instrument_status=instrument.status if instrument is not None else None,
            effective_from=instrument.effective_from if instrument is not None else None,
            effective_to=instrument.effective_to if instrument is not None else None,
            id=element.id,
            name=element.name,
            type=type(element).__name__,
            text=element.text,
            summary=summary,
            summary_embedding=summary_embedding,
            layer=0,
        )

        log.debug(
            "Initialized node",
            node_id=node.id,
            node_name=node.name,
            node_type=node.type,
        )
        return node

    def _create_nodes_batch(
        self,
        elements: Sequence[Element],
        batch_size: int = 64,
        *,
        avoid_summarization: bool = False,
        instruments_by_key: Mapping[str, Instrument] | None = None,
    ) -> set[Node | None]:
        """
        Create a batch of Nodes from a sequence of Elements (provisions, guidelines, or rulings). This does
        not include the arch. Uses the summarizer and the embedder to create summaries and embeddings.

        Args:
            elements (Sequence[Element]): The elements to create the nodes from.
            batch_size (int): The batch size to use for summarization and embedding. Defaults to 64.
            avoid_summarization (bool): Whether to avoid summarization and use the original text as the summary. Defaults to False.
            instruments_by_key (Mapping[str, Instrument] | None): Lookup of the corpus's instruments
                by key, used to stamp provision nodes with their owning instrument's role, status,
                and effective dates. None (the default) leaves those fields unset.

        Returns:
            set[Node | None]: The created nodes.

        """
        nodes: set[Node | None] = set()

        for i in range(0, len(elements), batch_size):
            log.info(
                "Processing batch of nodes",
                batch_size=min(batch_size, len(elements) - i),
                total_nodes=len(nodes),
                batch_number=i // batch_size + 1,
            )
            batch = elements[i : i + batch_size]
            texts = [element.text for element in batch]

            if avoid_summarization:
                summaries = texts
                summarization_time = 0.0
            else:
                start = time.perf_counter()
                summaries = self.summarizer.create_summaries(texts)
                summarization_time = time.perf_counter() - start

            start = time.perf_counter()
            summary_embeddings = self.embedder.create_embeddings(summaries)
            summary_embedding_time = time.perf_counter() - start

            for element, summary, summary_embedding in zip(
                batch,
                summaries,
                summary_embeddings,
                strict=True,
            ):
                instrument = self._owning_instrument(element, instruments_by_key)
                node = Node(
                    corpus_id=self.corpus_id,
                    instrument_key=element.instrument_key if isinstance(element, Provision) else None,
                    celex_id=element.celex_id if isinstance(element, Provision) else None,
                    provision_path=element.provision_path if isinstance(element, Provision) else None,
                    instrument_role=instrument.role if instrument is not None else None,
                    instrument_status=instrument.status if instrument is not None else None,
                    effective_from=instrument.effective_from if instrument is not None else None,
                    effective_to=instrument.effective_to if instrument is not None else None,
                    id=element.id,
                    name=element.name,
                    type=type(element).__name__,
                    text=element.text,
                    summary=summary,
                    summary_embedding=summary_embedding,
                    layer=0,
                )
                nodes.add(node)

            log.info(
                "Processed batch of nodes",
                batch_size=len(batch),
                total_nodes=len(nodes),
                batch_number=i // batch_size + 1,
                summarization_time=round(summarization_time, 3),
                summary_embedding_time=round(summary_embedding_time, 3),
            )

        return nodes

    def adopt_element_nodes(self, nodes: set[Node | None]) -> None:
        """
        Reuse another builder's already-computed element nodes instead of recomputing their
        summaries/embeddings, so the two graphs stay consistent for shared elements.

        Args:
            nodes (set[Node | None]): The nodes to adopt, e.g. another builder's self.nodes.

        """
        self.nodes = set(nodes)

    def load_nodes(self) -> None:
        """
        Load self.nodes from nodes_path. Shared across concrete subclasses since Node is not parameterized by E.
        """
        nodes_json = self._read_cache_text(
            self.nodes_path,
            cache_exists=self.has_node_cache(),
            path_not_provided_msg=NODES_PATH_NOT_PROVIDED,
            kind="Node",
        )
        self.nodes = RootModel[set[Node | None]].model_validate_json(nodes_json).root
        self._validate_node_corpus(self.nodes)

    def _node_by_id(self) -> dict[UUID, Node]:
        """
        Build a lookup of the current in-memory nodes by id, for edge construction.

        Returns:
            dict[UUID, Node]: The nodes currently in self.nodes, keyed by id.

        """
        return {node.id: node for node in self.nodes if node is not None}

    def create_nodes(
        self,
        elements: list[ValidProvision],
        *,
        enable_batch_processing: bool = True,
        avoid_summarization: bool = False,
        instruments_by_key: Mapping[str, Instrument] | None = None,
    ) -> set[Node | None]:
        """
        Create a Node for each element in the corpus and store them in self.nodes. Uses the summarizer and the
        embedder, so this is the expensive step worth caching independently of edges.

        Args:
            elements (list[ValidProvision]): The provisions to create the nodes from.
            enable_batch_processing (bool): Whether to use batched summarization/embedding. Defaults to True.
            avoid_summarization (bool): Whether to avoid summarization and use the original text as the summary. Defaults to False.
            instruments_by_key (Mapping[str, Instrument] | None): Lookup of the corpus's instruments
                by key, used to stamp provision nodes with their owning instrument's role, status,
                and effective dates. None (the default) leaves those fields unset.

        Returns:
            set[Node | None]: The created nodes.

        """
        log.info("Creating nodes", num_elements=len(elements))

        if enable_batch_processing:
            self.nodes = self._create_nodes_batch(
                elements, avoid_summarization=avoid_summarization, instruments_by_key=instruments_by_key
            )
        else:
            self.nodes = {self._create_node(element, instruments_by_key) for element in elements}

        log.info("Created nodes", nodes=len(self.nodes))
        return self.nodes

    @abstractmethod
    def _build_edge_index(self) -> dict[K, E]:
        """
        Index self.edges by the subclass's own endpoint-id key, for O(1) existence checks.

        Must be implemented by concrete subclasses (a `frozenset` of endpoint ids for undirected
        edges, an ordered `tuple` for directed ones). Rebuilt fresh on every call rather than
        cached on the instance, so it stays correct regardless of how self.edges was populated.

        Returns:
            dict[K, E]: Existing edges keyed by the subclass's endpoint-id key.

        """
        raise NotImplementedError

    @abstractmethod
    def _create_or_merge_edge(
        self,
        element: ValidProvision,
        node_by_id: dict[UUID, Node],
        edge_index: dict[K, E],
    ) -> set[E | None]:
        """
        Create or merge the edge(s) for one element, purely in-memory. Must be implemented by concrete implementations.

        Args:
            element (ValidProvision): The provision to create the edge from.
            node_by_id (dict[UUID, Node]): Lookup of the in-memory nodes by id.
            edge_index (dict[K, E]): Existing edges keyed by the subclass's own endpoint-id
                key (e.g. a `frozenset` for undirected edges, a `tuple` for directed ones), shared
                across every element in the same `create_edges` call so a later element sees edges
                an earlier one already created. Implementations must update it in place.

        Returns:
            set[E | None]: The created or updated edges.

        """
        raise NotImplementedError

    def create_edges(self, elements: list[ValidProvision]) -> set[E | None]:
        """
        Create edges for each element in the corpus, purely in-memory, and store them in self.edges. Requires
        self.nodes to already be populated (e.g. via create_nodes or load_nodes).

        Args:
            elements (list[ValidProvision]): The provisions to create the edges from.

        Returns:
            set[E | None]: The created edges.

        """
        log.info("Creating edges", num_elements=len(elements))

        node_by_id = self._node_by_id()
        edge_index = self._build_edge_index()
        for element in elements:
            edges = self._create_or_merge_edge(element, node_by_id, edge_index)
            self.edges.update(edges)

        log.info("Created edges", edges=len(self.edges))
        return self.edges

    def sync_to_driver(self, *, clear: bool = True) -> None:
        """
        Write the current in-memory nodes and edges to Neo4j.

        Args:
            clear (bool): Whether to clear the whole graph first. Defaults to True; pass False when
                another builder's graph (e.g. co-citation) must be preserved, since Neo4j MERGE
                already reconciles nodes/edges shared between builders.

        """
        log.info("Syncing graph to Neo4j", nodes=len(self.nodes), edges=len(self.edges), clear=clear)
        if clear:
            self.driver.clear_graph()
        self.driver.create_graph(self.nodes, self.edges)
