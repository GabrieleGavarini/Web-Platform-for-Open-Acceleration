from collections.abc import Mapping, Sequence
from typing import cast
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import RootModel
from structlog import get_logger

from wbpoa.core.corpus.corpus_config import InstrumentRelationConfig, InstrumentRelationType
from wbpoa.core.model.element import Instrument
from wbpoa.core.model.graph import Edge, EdgeDirected, EdgeUndirected, Node
from wbpoa.core.model.operative_relation import OperativeRelation
from wbpoa.single_corpus.graph_builder.base_graph_builder import BaseGraphBuilder

log = get_logger()

INSTRUMENT_TYPE = "Instrument"
INSTRUMENT_LAYER = -1
OPERATIVE_RELATION_TYPE = "operative_relation"
OPERATIVE_AFFINITY_TYPE = "operative_affinity"


class InstrumentGraphBuilder(BaseGraphBuilder[Edge]):
    """
    Build the instrument-level legal graph: one node per instrument, directed instrument relation
    edges (AMENDS/IMPLEMENTS/SUPPLEMENTS/REPEALS/REPLACES), directed OPERATIVE_RELATION provision
    edges, and derived weighted undirected OPERATIVE_AFFINITY provision edges.

    Instrument nodes are written at layer -1 -- outside every layer 0/1+ query -- so they never
    leak into vector search or Leiden as accidental isolated singletons. Operative edges reference
    provision nodes this builder does not itself create; callers pass in the already-built provision
    nodes (from CoCitationGraphBuilder/CitationGraphBuilder) instead of having them recomputed.
    """

    def load_nodes(self) -> None:
        nodes_json = self._read_cache_text(
            self.nodes_path,
            cache_exists=self.has_node_cache(),
            path_not_provided_msg="Nodes path not provided, cannot access the node cache.",
            kind="Node",
        )
        self.nodes = RootModel[set[Node | None]].model_validate_json(nodes_json).root
        self._validate_node_corpus(self.nodes)

    def load_edges(self) -> None:
        edges_json = self._read_cache_text(
            self.edges_path,
            cache_exists=self.has_edge_cache(),
            path_not_provided_msg="Edges path not provided, cannot access the edge cache.",
            kind="Edge",
        )
        self.edges = cast(
            "set[Edge | None]",
            RootModel[set[EdgeDirected | EdgeUndirected | None]].model_validate_json(edges_json).root,
        )
        self._validate_edge_corpus(self.edges)

    def save_nodes(self) -> None:
        if self.nodes_path is None:
            message = "Nodes path not provided, cannot access the node cache."
            raise ValueError(message)
        self.nodes_path.parent.mkdir(parents=True, exist_ok=True)
        super().save_nodes()

    def save_edges(self) -> None:
        if self.edges_path is None:
            message = "Edges path not provided, cannot access the edge cache."
            raise ValueError(message)
        self.edges_path.parent.mkdir(parents=True, exist_ok=True)
        model = RootModel[set[EdgeDirected | EdgeUndirected | None]].model_validate(self.edges)
        self.edges_path.write_text(model.model_dump_json(indent=2), encoding="utf-8")

    def sync_to_driver(self, *, clear: bool = True) -> None:
        if clear:
            self.driver.clear_graph()
        self.driver.create_graph(self.nodes, self.edges)

    def create_instrument_nodes(self, instruments: Sequence[Instrument]) -> set[Node]:
        """
        Create one non-vector Node per instrument and add them to self.nodes.

        Args:
            instruments: The corpus's instruments.

        Returns:
            The created instrument nodes.

        """
        nodes = {
            Node(
                corpus_id=self.corpus_id,
                instrument_key=instrument.key,
                celex_id=instrument.celex_id,
                instrument_role=instrument.role,
                instrument_status=instrument.status,
                effective_from=instrument.effective_from,
                effective_to=instrument.effective_to,
                id=self._instrument_node_id(instrument),
                name=instrument.name,
                type=INSTRUMENT_TYPE,
                text=instrument.name,
                summary=instrument.name,
                summary_embedding=[],
                layer=INSTRUMENT_LAYER,
            )
            for instrument in instruments
        }
        self.nodes.update(nodes)
        log.info("instrument_graph_nodes_created", instrument_count=len(nodes))
        return nodes

    def _instrument_node_id(self, instrument: Instrument) -> UUID:
        """Derive a deterministic node id, mirroring `InstrumentIngestion`'s provision id derivation."""
        return uuid5(NAMESPACE_URL, f"wbpoa:{self.corpus_id}:{instrument.key}:instrument")

    def _instrument_nodes_by_key(self) -> dict[str, Node]:
        return {
            node.instrument_key: node
            for node in self.nodes
            if node is not None and node.type == INSTRUMENT_TYPE and node.instrument_key is not None
        }

    def create_instrument_relation_edges(self, relations: Sequence[InstrumentRelationConfig]) -> set[EdgeDirected]:
        """
        Create one unweighted directed edge per declared instrument relation and add them to self.edges.

        Requires the source and target instrument nodes to already be in self.nodes (see
        `create_instrument_nodes`).

        Args:
            relations: Declared instrument relations, e.g. `workspace.instrument_relations()`.

        Raises:
            ValueError: If a relation's source or target instrument node does not exist.

        Returns:
            The created edges.

        """
        instrument_nodes = self._instrument_nodes_by_key()
        edges: set[EdgeDirected] = set()
        for relation in relations:
            source_node = instrument_nodes.get(relation.source)
            target_node = instrument_nodes.get(relation.target)
            if source_node is None or target_node is None:
                raise ValueError(
                    f"Instrument node for relation {relation.source!r} -> {relation.target!r} does not exist."
                )
            edges.add(
                EdgeDirected(
                    source=source_node,
                    target=target_node,
                    type=relation.type.value,
                    weight=1.0,
                    relation_type=relation.type,
                )
            )
        self.edges.update(edges)
        log.info("instrument_graph_relation_edges_created", edge_count=len(edges))
        return edges

    def create_operative_edges(
        self,
        operative_relations: Sequence[OperativeRelation],
        provision_nodes: Mapping[UUID, Node],
        affinity_weights: Mapping[InstrumentRelationType, float | None],
    ) -> None:
        """
        Create directed OPERATIVE_RELATION edges and derived undirected OPERATIVE_AFFINITY edges.

        One OPERATIVE_RELATION edge is created per distinct (source, target) provision pair --
        repeated relations between the same pair (e.g. two separate textual references to the same
        target) are left as-is once the first is created, the same unweighted dedup
        `CitationGraphBuilder` uses for CITES. An OPERATIVE_AFFINITY edge is additionally created or
        merged for the same pair, weighted by `affinity_weights[relation.relation_type]`; a relation
        type mapped to None does not contribute an affinity edge. Repeated pairs accumulate affinity
        weight additively, tracked in `weight_contributions` keyed by the contributing relation's
        source provision id -- the same merge pattern `CoCitationGraphBuilder` uses for CO_CITATION.

        Args:
            operative_relations: The corpus's resolved operative provision relations.
            provision_nodes: Lookup of already-built provision nodes by id (this builder does not
                create provision nodes itself).
            affinity_weights: Per-relation-type OPERATIVE_AFFINITY weight; None disables that type.

        Raises:
            ValueError: If a relation's source or target provision node does not exist.

        """
        operative_index: dict[tuple[UUID, UUID], EdgeDirected] = {
            (edge.source.id, edge.target.id): edge
            for edge in self.edges
            if isinstance(edge, EdgeDirected) and edge.type == OPERATIVE_RELATION_TYPE
        }
        affinity_index: dict[frozenset[UUID], EdgeUndirected] = {
            frozenset(node.id for node in edge.nodes): edge
            for edge in self.edges
            if isinstance(edge, EdgeUndirected) and edge.type == OPERATIVE_AFFINITY_TYPE
        }

        for relation in operative_relations:
            source_node = provision_nodes.get(relation.source_provision_id)
            target_node = provision_nodes.get(relation.target_provision_id)
            if source_node is None or target_node is None:
                raise ValueError(
                    f"Provision node for operative relation {relation.source_provision_id} -> "
                    f"{relation.target_provision_id} does not exist."
                )

            operative_key = (relation.source_provision_id, relation.target_provision_id)
            if operative_key not in operative_index:
                operative_edge = EdgeDirected(
                    source=source_node,
                    target=target_node,
                    type=OPERATIVE_RELATION_TYPE,
                    weight=1.0,
                    relation_type=relation.relation_type,
                    operation=relation.operation,
                )
                self.edges.add(operative_edge)
                operative_index[operative_key] = operative_edge

            weight = affinity_weights.get(relation.relation_type)
            if weight is None:
                continue

            affinity_key = frozenset({relation.source_provision_id, relation.target_provision_id})
            affinity_edge = affinity_index.get(affinity_key)
            if affinity_edge:
                affinity_edge.weight += weight
                affinity_edge.weight_contributions[relation.source_provision_id] = (
                    affinity_edge.weight_contributions.get(relation.source_provision_id, 0.0) + weight
                )
            else:
                affinity_edge = EdgeUndirected(
                    nodes=(source_node, target_node),
                    weight=weight,
                    type=OPERATIVE_AFFINITY_TYPE,
                    relation_type=relation.relation_type,
                    weight_contributions={relation.source_provision_id: weight},
                )
                self.edges.add(affinity_edge)
                affinity_index[affinity_key] = affinity_edge

        log.info(
            "instrument_graph_operative_edges_created",
            operative_relation_edges=len(operative_index),
            operative_affinity_edges=len(affinity_index),
        )
