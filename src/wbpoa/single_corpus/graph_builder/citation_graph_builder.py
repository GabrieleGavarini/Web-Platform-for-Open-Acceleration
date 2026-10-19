from collections.abc import Sequence
from uuid import UUID

from pydantic import RootModel
from structlog import get_logger

from wbpoa.core.model.element import Guideline, Ruling, ValidProvision
from wbpoa.core.model.graph import EdgeDirected, Node
from wbpoa.single_corpus.graph_builder.base_graph_builder import EDGES_PATH_NOT_PROVIDED
from wbpoa.single_corpus.graph_builder.generic_graph_builder import GenericGraphBuilder

log = get_logger()

CITATION_EDGE_TYPE = "cites"


class CitationGraphBuilder(GenericGraphBuilder[EdgeDirected, tuple[UUID, UUID]]):
    """
    Build the citation graph: a directed, unweighted graph recording which document cites which
    element. Nodes are provisions, guidelines, and rulings.
    """

    def load_edges(self) -> None:
        """
        Load self.edges from edges_path as EdgeDirected instances.
        """
        edges_json = self._read_cache_text(
            self.edges_path,
            cache_exists=self.has_edge_cache(),
            path_not_provided_msg=EDGES_PATH_NOT_PROVIDED,
            kind="Edge",
        )
        self.edges = RootModel[set[EdgeDirected | None]].model_validate_json(edges_json).root
        self._validate_edge_corpus(self.edges)

    def _build_edge_index(self) -> dict[tuple[UUID, UUID], EdgeDirected]:
        """
        Index self.edges by (source_id, target_id), for O(1) existence checks.

        Rebuilt fresh on every call (not cached on the instance) so callers stay correct
        regardless of how self.edges was populated, at the cost of one O(|edges|) pass per
        call instead of one per pair.

        Returns:
            dict[tuple[UUID, UUID], EdgeDirected]: Existing directed edges keyed by (source, target).

        """
        return {(edge.source.id, edge.target.id): edge for edge in self.edges if edge is not None}

    def _create_or_merge_edge(
        self,
        element: ValidProvision,
        node_by_id: dict[UUID, Node],
        edge_index: dict[tuple[UUID, UUID], EdgeDirected],
    ) -> set[EdgeDirected | None]:
        """
        Create a directed edge element -> reference for each of element.provision_references, purely in memory.
        Raise an exception if a provision has no corresponding node. If an edge already exists between the
        nodes, leave it as-is (the citation graph is unweighted), otherwise create a new one.

        Args:
            element (ValidProvision): The element to create the edge from.
            node_by_id (dict[UUID, Node]): Lookup of the in-memory nodes by id.
            edge_index (dict[tuple[UUID, UUID], EdgeDirected]): Existing edges keyed by
                (source, target) id, shared across every element in the same `create_edges` call;
                updated in place as new edges are created.

        Raises:
            ValueError: If a reference node does not exist for a given reference in element.provision_references

        Returns:
            set[EdgeDirected | None]: The edges for this element (existing or newly created).

        """
        if not element.provision_references:
            return set()

        source_node = node_by_id.get(element.id)
        if not source_node:
            raise ValueError(f"Source node with id {element.id} does not exist.")

        edges: set[EdgeDirected | None] = set()
        for reference_id in element.provision_references:
            reference_node = node_by_id.get(reference_id)
            if not reference_node:
                raise ValueError(f"Reference node with id {reference_id} does not exist.")

            key = (element.id, reference_id)
            edge = edge_index.get(key)
            if not edge:
                edge = EdgeDirected(source=source_node, target=reference_node, type=CITATION_EDGE_TYPE, weight=1.0)
                edge_index[key] = edge

            edges.add(edge)

        return edges

    def add_node(self, element: Guideline | Ruling) -> None:
        """
        Create a node for a new guideline or ruling and store it in self.nodes, purely in memory.

        Args:
            element (Guideline | Ruling): The guideline or ruling to create the node from.

        """
        self.nodes.add(self._create_node(element))

    def add_nodes(self, elements: Sequence[Guideline] | Sequence[Ruling], batch_size: int = 64) -> None:
        """
        Create nodes for a batch of guidelines or rulings and store them in self.nodes, purely in memory.
        Unlike add_node, this batches summarization/embedding calls, so prefer it when adding several
        guidelines or rulings at once.

        Args:
            elements (Sequence[Guideline] | Sequence[Ruling]): The guidelines or rulings to create the nodes from.
            batch_size (int): The batch size to use for summarization and embedding. Defaults to 64.

        """
        if not elements:
            return
        self.nodes.update(self._create_nodes_batch(elements, batch_size=batch_size))

    def add_arch(self, element: Guideline | Ruling) -> None:
        """
        Create a directed edge guideline/ruling -> element for each of element.provision_references, purely in memory.
        If an edge already exists between the nodes, leave it as-is (unweighted), otherwise create a new one.

        Args:
            element (Guideline | Ruling): The guideline or ruling to create the edges from.

        Raises:
            ValueError: If the guideline's own node or a reference node does not exist.

        """
        if not element.provision_references:
            return

        node_by_id = self._node_by_id()
        edge_index = self._build_edge_index()

        source_node = node_by_id.get(element.id)
        if not source_node:
            raise ValueError(f"Source node with id {element.id} does not exist.")

        for reference_id in element.provision_references:
            reference_node = node_by_id.get(reference_id)
            if not reference_node:
                raise ValueError(f"Reference node with id {reference_id} does not exist.")

            key = (element.id, reference_id)
            if key in edge_index:
                continue

            edge = EdgeDirected(source=source_node, target=reference_node, type=CITATION_EDGE_TYPE, weight=1.0)
            self.edges.add(edge)
            edge_index[key] = edge

            log.debug(
                "Added edge between nodes",
                source_node=source_node.id,
                reference_node=reference_node.id,
            )
