from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from pydantic import RootModel
from structlog import get_logger

from wbpoa.core.driver.neo4j_driver import Neo4jDriver
from wbpoa.core.ingestion.utils import Embedder
from wbpoa.core.model.graph import Edge, Node
from wbpoa.core.model.graph.edge_directed import EdgeDirected
from wbpoa.core.model.graph.edge_undirected import EdgeUndirected
from wbpoa.single_corpus.ingestion.utils import Summarizer

log = get_logger()

NODES_PATH_NOT_PROVIDED = "Nodes path not provided, cannot access the node cache."
EDGES_PATH_NOT_PROVIDED = "Edges path not provided, cannot access the edge cache."


@dataclass(frozen=True, slots=True)
class GraphBuilderContext[E: Edge]:
    """Shared runtime dependencies and corpus identity for graph builders."""

    driver: Neo4jDriver[E]
    corpus_id: str


class BaseGraphBuilder[E: Edge](ABC):
    """
    Base class for graph builders. Owns the in-memory nodes/edges, the Neo4j driver, and independent
    JSON caches for nodes and edges, so the expensive node summaries/embeddings can be reused without
    forcing a rebuild of the (cheap) edges, and vice versa.
    """

    driver: Neo4jDriver[E]
    summarizer: Summarizer
    embedder: Embedder

    nodes: set[Node | None]
    edges: set[E | None]

    def __init__(
        self,
        nodes_path: Path | None = None,
        edges_path: Path | None = None,
        *,
        summarizer: Summarizer,
        embedder: Embedder,
        context: GraphBuilderContext[E] | None = None,
    ) -> None:
        self.driver = context.driver if context is not None else Neo4jDriver[E]()
        self.corpus_id = context.corpus_id if context is not None else "legacy"
        self.summarizer = summarizer
        self.embedder = embedder

        self.nodes = set()
        self.edges = set()

        self.nodes_path = nodes_path
        self.edges_path = edges_path

    def has_node_cache(self) -> bool:
        """
        Check if the node cache JSON file exists.

        Returns:
            bool: True if the file exists, False otherwise.

        """
        if not self.nodes_path:
            log.warning(NODES_PATH_NOT_PROVIDED)
            return False

        return self.nodes_path.exists()

    def has_edge_cache(self) -> bool:
        """
        Check if the edge cache JSON file exists.

        Returns:
            bool: True if the file exists, False otherwise.

        """
        if not self.edges_path:
            log.warning(EDGES_PATH_NOT_PROVIDED)
            return False

        return self.edges_path.exists()

    def _read_cache_text(self, path: Path | None, *, cache_exists: bool, path_not_provided_msg: str, kind: str) -> str:
        """
        Validate a cache path and read its contents, factoring out the checks shared by load_nodes/load_edges.

        Args:
            path (Path | None): The cache file path.
            cache_exists (bool): Whether the cache file exists (e.g. from has_node_cache()/has_edge_cache()).
            path_not_provided_msg (str): The message to log/raise if path is not set.
            kind (str): A human-readable label for the cache ("Node" or "Edge"), used in error messages.

        Raises:
            ValueError: If path is not set.
            FileNotFoundError: If the cache file does not exist.

        Returns:
            str: The raw JSON text read from the cache file.

        """
        if not path:
            log.error(path_not_provided_msg)
            raise ValueError(path_not_provided_msg)

        if not cache_exists:
            log.error("Cache does not exist", kind=kind, path=path)
            raise FileNotFoundError(f"{kind} cache does not exist: {path}")

        return path.read_text(encoding="utf-8")

    def _validate_node_corpus(self, nodes: set[Node | None]) -> None:
        """Reject a node cache created for a different corpus partition."""
        mismatched = {node.corpus_id for node in nodes if node is not None and node.corpus_id != self.corpus_id}
        if mismatched:
            raise ValueError(
                f"Node cache contains corpus partitions {sorted(mismatched)!r}; expected {self.corpus_id!r}."
            )

    def _validate_edge_corpus(self, edges: set[E | None]) -> None:
        """Reject an edge cache whose endpoints belong to another corpus."""
        for edge in edges:
            if edge is None:
                continue
            endpoints: list[Node] = []
            if isinstance(edge, EdgeDirected):
                endpoints.extend((edge.source, edge.target))
            elif isinstance(edge, EdgeUndirected):
                endpoints.extend(edge.nodes)
            if any(node.corpus_id != self.corpus_id for node in endpoints):
                raise ValueError(f"Edge cache contains an endpoint outside corpus {self.corpus_id!r}.")

    def save_nodes(self) -> None:
        """
        Save self.nodes, including their summaries and embeddings, to nodes_path.
        """
        if not self.nodes_path:
            log.error(NODES_PATH_NOT_PROVIDED)
            raise ValueError(NODES_PATH_NOT_PROVIDED)

        if self.has_node_cache():
            log.warning("Node cache already exists. Overwriting.", nodes_path=self.nodes_path)

        self.nodes_path.parent.mkdir(parents=True, exist_ok=True)
        model = RootModel[set[Node | None]].model_validate(self.nodes)
        self.nodes_path.write_text(model.model_dump_json(indent=2), encoding="utf-8")

    @abstractmethod
    def load_nodes(self) -> None:
        """
        Load self.nodes from nodes_path. Must be implemented by concrete implementations.
        """
        raise NotImplementedError

    def save_edges(self) -> None:
        """
        Save self.edges to edges_path.
        """
        if not self.edges_path:
            log.error(EDGES_PATH_NOT_PROVIDED)
            raise ValueError(EDGES_PATH_NOT_PROVIDED)

        if self.has_edge_cache():
            log.warning("Edge cache already exists. Overwriting.", edges_path=self.edges_path)

        self.edges_path.parent.mkdir(parents=True, exist_ok=True)
        model = RootModel[set[E | None]].model_validate(self.edges)
        self.edges_path.write_text(model.model_dump_json(indent=2), encoding="utf-8")

    @abstractmethod
    def load_edges(self) -> None:
        """
        Load self.edges from edges_path. Must be implemented by concrete implementations, since E is erased
        at runtime and RootModel[set[E | None]] would otherwise deserialize into the abstract Edge base class.
        """
        raise NotImplementedError
