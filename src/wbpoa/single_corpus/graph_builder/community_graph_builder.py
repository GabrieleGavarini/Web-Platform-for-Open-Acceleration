from __future__ import annotations

import time
from collections import Counter
from typing import cast
from uuid import UUID, uuid4

from pydantic import RootModel
from structlog import get_logger

from wbpoa.core.corpus.corpus_config import InstrumentRole
from wbpoa.core.model.graph import Edge, EdgeDirected, EdgeUndirected, Node
from wbpoa.single_corpus.graph_builder.base_graph_builder import BaseGraphBuilder
from wbpoa.single_corpus.ingestion.utils import Summarizer

log = get_logger()


class CommunityGraphBuilder(BaseGraphBuilder[Edge]):
    """
    Build and cache the hierarchical community graph described in Algorithm 1.

    Layer 0 is the already-built co-citation graph. ``build`` projects that layer into Neo4j GDS,
    runs Leiden, then creates one aggregate community layer at a time with community nodes,
    community-to-child membership edges, and weighted inter-community co-citation edges.
    """

    COMMUNITY_TYPE = "Community"
    MEMBERSHIP_TYPE = "member"
    AGGREGATE_TYPE = "community_co_citation"

    def load_nodes(self) -> None:
        nodes_json = self._read_cache_text(
            self.nodes_path,
            cache_exists=self.has_node_cache(),
            path_not_provided_msg="Nodes path not provided, cannot access the node cache.",
            kind="Node",
        )
        self.nodes = RootModel[set[Node | None]].model_validate_json(nodes_json).root

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

    @staticmethod
    def _assignment_key(value: int | str) -> str:
        return str(value)

    @staticmethod
    def _member_instrument_info(children: list[Node]) -> tuple[list[str], list[InstrumentRole]]:
        """
        Aggregate the distinct instrument keys/roles among a community's descendant provisions.

        Each child contributes its own ``member_instrument_keys``/``member_instrument_roles`` if it
        is itself a community (propagating the union upward through the hierarchy), or its own
        ``instrument_key``/``instrument_role`` if it is a provision; anything else (a Guideline or
        Ruling child, for instance) contributes nothing.

        Returns:
            Sorted, deduplicated instrument keys and instrument roles.

        """
        keys: set[str] = set()
        roles: set[InstrumentRole] = set()
        for child in children:
            if child.type == CommunityGraphBuilder.COMMUNITY_TYPE:
                keys.update(child.member_instrument_keys)
                roles.update(child.member_instrument_roles)
            elif child.instrument_key is not None:
                keys.add(child.instrument_key)
                if child.instrument_role is not None:
                    roles.add(child.instrument_role)
        return sorted(keys), sorted(roles, key=lambda role: role.value)

    def _make_community_node(self, layer: int, _community_id: int | str, children: list[Node]) -> Node:
        """
        Create one community node for Algorithm 1 step 17.

        The node summary is produced from the summaries of the children in the layer below, its
        name is a short LLM-generated title of that summary, and the summary embedding is
        computed immediately so the community can later be retrieved by semantic similarity. A single-member community instead reuses that member's name,
        summary, and summary embedding as-is, since summarizing a group of one is redundant.
        """
        member_text = "\n\n".join(f"{child.name}:\n{child.summary}" for child in children)
        if len(children) == 1:
            child = children[0]
            name, summary, summary_embedding = child.name, child.summary, child.summary_embedding
        else:
            summary = self.summarizer.create_summary(
                member_text, system_prompt=Summarizer.AGGREGATE_SUMMARIZATION_PROMPT
            )
            name = self.summarizer.create_name(summary)
            summary_embedding = self.embedder.create_embedding(summary)
        member_instrument_keys, member_instrument_roles = self._member_instrument_info(children)
        return Node(
            corpus_id=self.corpus_id,
            id=uuid4(),
            name=name,
            type=self.COMMUNITY_TYPE,
            text=member_text,
            summary=summary,
            summary_embedding=summary_embedding,
            layer=layer,
            member_instrument_keys=member_instrument_keys,
            member_instrument_roles=member_instrument_roles,
        )

    def _make_community_nodes_batch(self, layer: int, groups: dict[str, list[Node]]) -> dict[str, Node]:
        """
        Create community nodes for one layer with batched summaries and embeddings.

        This is the batched equivalent of ``_make_community_node``. It keeps the same
        summary-on-summary semantics from Algorithm 1 step 17, but summarizes all
        multi-member community texts for the current layer in one LLM batch and embeds the
        resulting summaries in one embedding batch. Single-member communities skip the
        summarizer/embedder entirely and reuse their member's name, summary, and summary
        embedding as-is.
        """
        community_items = list(groups.items())
        member_texts = {
            community_id: "\n\n".join(f"{child.name}:\n{child.summary}" for child in children)
            for community_id, children in community_items
        }
        singleton_items = [(community_id, children) for community_id, children in community_items if len(children) == 1]
        multi_items = [(community_id, children) for community_id, children in community_items if len(children) > 1]

        instrument_info = {
            community_id: self._member_instrument_info(children) for community_id, children in community_items
        }

        nodes: dict[str, Node] = {
            community_id: Node(
                corpus_id=self.corpus_id,
                id=uuid4(),
                name=children[0].name,
                type=self.COMMUNITY_TYPE,
                text=member_texts[community_id],
                summary=children[0].summary,
                summary_embedding=children[0].summary_embedding,
                layer=layer,
                member_instrument_keys=instrument_info[community_id][0],
                member_instrument_roles=instrument_info[community_id][1],
            )
            for community_id, children in singleton_items
        }

        if multi_items:
            multi_texts = [member_texts[community_id] for community_id, _ in multi_items]
            summaries = self.summarizer.create_summaries(
                multi_texts, system_prompt=Summarizer.AGGREGATE_SUMMARIZATION_PROMPT
            )
            names = self.summarizer.create_names(summaries)
            summary_embeddings = self.embedder.create_embeddings(summaries)
            nodes.update(
                {
                    community_id: Node(
                        corpus_id=self.corpus_id,
                        id=uuid4(),
                        name=name,
                        type=self.COMMUNITY_TYPE,
                        text=member_text,
                        summary=summary,
                        summary_embedding=summary_embedding,
                        layer=layer,
                        member_instrument_keys=instrument_info[community_id][0],
                        member_instrument_roles=instrument_info[community_id][1],
                    )
                    for (community_id, _), member_text, summary, name, summary_embedding in zip(
                        multi_items,
                        multi_texts,
                        summaries,
                        names,
                        summary_embeddings,
                        strict=True,
                    )
                }
            )

        log.info(
            "community_graph_batch_summaries_created",
            layer=layer,
            singleton_count=len(singleton_items),
            batched_count=len(multi_items),
        )
        return nodes

    def _groups_for_layer(
        self,
        layer: int,
        children: list[Node],
        child_source_ids: dict[UUID, set[UUID]],
        assignments: dict[UUID, list[int]],
    ) -> dict[str, list[Node]]:
        """Group the previous layer's children by their Leiden assignment for ``layer``."""
        groups: dict[str, list[Node]] = {}
        for child in children:
            source_ids = child_source_ids[child.id]
            source_id = next(iter(source_ids))
            levels = assignments.get(source_id, [])
            if not levels:
                continue
            if len(levels) < layer:
                continue
            community_id = levels[layer - 1]
            groups.setdefault(self._assignment_key(community_id), []).append(child)
        return groups

    def _create_membership_edges(
        self,
        groups: dict[str, list[Node]],
        communities: dict[str, Node],
    ) -> set[EdgeDirected | None]:
        """Create Algorithm 1 step 15 membership edges for one community layer."""
        return {
            EdgeDirected(
                source=communities[community_id],
                target=member,
                type=self.MEMBERSHIP_TYPE,
                weight=1.0,
            )
            for community_id, members in groups.items()
            for member in members
        }

    def _create_aggregate_edges(self, layer: int, communities: dict[str, Node]) -> set[EdgeUndirected]:
        """Create Algorithm 1 step 16 aggregate co-citation edges for one community layer."""
        aggregate_edges: set[EdgeUndirected] = set()
        for first_id, second_id, weight in self.driver.aggregate_weights(layer):
            first = next((node for node in communities.values() if str(node.id) == first_id), None)
            second = next((node for node in communities.values() if str(node.id) == second_id), None)
            if first is not None and second is not None and weight > 0:
                aggregate_edges.add(EdgeUndirected(nodes=(first, second), weight=weight, type=self.AGGREGATE_TYPE))
        return aggregate_edges

    @staticmethod
    def _log_layer_grouped(layer: int, groups: dict[str, list[Node]], child_count: int) -> None:
        """Log the number and size range of communities created for one layer, plus its singleton count."""
        group_sizes = [len(members) for members in groups.values()]
        size_histogram = dict(sorted(Counter(group_sizes).items(), key=lambda x: int(x[0])))
        log.info(
            "community_graph_layer_grouped",
            layer=layer,
            community_count=len(groups),
            child_count=child_count,
            min_community_size=min(group_sizes),
            max_community_size=max(group_sizes),
            singleton_count=size_histogram.get(1, 0),
            size_histogram=size_histogram,
        )

    @staticmethod
    def _log_build_started(
        max_levels: int,
        leiden_gamma: float,
        children: list[Node],
        node_count: int,
        edge_count: int,
    ) -> None:
        """Log the starting point for Algorithm 1 steps 13-18."""
        log.info(
            "community_graph_build_started",
            max_levels=max_levels,
            leiden_gamma=leiden_gamma,
            layer_0_node_count=len(children),
            cached_node_count=node_count,
            cached_edge_count=edge_count,
        )

    def build(  # noqa: PLR0915
        self,
        max_levels: int = 3,
        leiden_gamma: float = 1.0,
        *,
        include_backbone: bool = True,
        include_operative_affinity: bool = True,
    ) -> None:
        """
        Build Algorithm 1 steps 13-18.

        The co-citation graph is treated as community layer 0. For each requested layer, Leiden
        assignments group the children from the previous layer into community nodes, membership
        edges point from each community to its children, and aggregate undirected edges store the
        sum of crossing co-citation weights between communities. Existing persisted communities
        are removed before the rebuild, while layer-0 nodes and edges are preserved.

        Args:
            max_levels: Maximum number of community layers to create above layer 0.
            leiden_gamma: Leiden resolution parameter; higher values generally yield
                smaller, more numerous communities.
            include_backbone: Whether the layer-0 projection includes semantic backbone edges
                alongside co-citation ones; False reproduces the pre-backbone projection, for A/B.
            include_operative_affinity: Whether the layer-0 projection includes cross-instrument
                operative affinity edges alongside co-citation ones.

        Raises:
            ValueError: If ``max_levels`` is negative.

        """
        if max_levels < 0:
            message = "max_levels must be non-negative"
            raise ValueError(message)
        if leiden_gamma <= 0:
            message = "leiden_gamma must be strictly positive"
            raise ValueError(message)
        if max_levels == 0:
            log.info("community_graph_build_skipped", reason="max_levels_is_zero")
            return

        start = time.perf_counter()
        self.driver.drop_communities()
        children = [node for node in self.nodes if node is not None and node.layer == 0]
        graph_name = "cocitation"
        self._log_build_started(max_levels, leiden_gamma, children, len(self.nodes), len(self.edges))
        self.driver.project_cocitation(
            graph_name, include_backbone=include_backbone, include_operative_affinity=include_operative_affinity
        )
        log.info(
            "community_graph_projection_created",
            graph_name=graph_name,
            layer=0,
            include_backbone=include_backbone,
            include_operative_affinity=include_operative_affinity,
        )
        try:
            assignments = self.driver.run_leiden(
                graph_name,
                max_levels=max_levels,
                leiden_gamma=leiden_gamma,
            )
        finally:
            self.driver.drop_projection(graph_name)
        log.info(
            "community_graph_leiden_completed",
            graph_name=graph_name,
            requested_levels=max_levels,
            leiden_gamma=leiden_gamma,
            assigned_source_count=len(assignments),
        )
        child_source_ids = {child.id: {child.id} for child in children}
        built_layers = 0
        total_communities = 0
        total_membership_edges = 0
        total_aggregate_edges = 0

        for layer in range(1, max_levels + 1):
            if len(children) <= 1:
                log.info(
                    "community_graph_build_stopped",
                    layer=layer,
                    reason="not_enough_children",
                    child_count=len(children),
                )
                break
            groups = self._groups_for_layer(layer, children, child_source_ids, assignments)

            if not groups:
                log.info("community_graph_build_stopped", layer=layer, reason="no_leiden_groups")
                break
            self._log_layer_grouped(layer, groups, len(children))
            communities = self._make_community_nodes_batch(layer, groups)
            membership_edges = self._create_membership_edges(groups, communities)
            self.nodes.update(communities.values())
            self.edges.update(membership_edges)
            total_communities += len(communities)
            total_membership_edges += len(membership_edges)
            self.driver.create_graph(
                set(communities.values()),
                cast("set[Edge | None]", membership_edges),
            )
            log.info(
                "community_graph_membership_edges_created",
                layer=layer,
                community_count=len(communities),
                membership_edge_count=len(membership_edges),
            )

            aggregate_edges = self._create_aggregate_edges(layer, communities)
            self.edges.update(aggregate_edges)
            total_aggregate_edges += len(aggregate_edges)
            self.driver.create_graph(
                set(communities.values()),
                cast("set[Edge | None]", aggregate_edges),
            )
            log.info(
                "community_graph_aggregate_edges_created",
                layer=layer,
                aggregate_edge_count=len(aggregate_edges),
                aggregate_weight=sum(edge.weight for edge in aggregate_edges),
            )
            child_source_ids = {
                community.id: set().union(*(child_source_ids[member.id] for member in members))
                for community_id, members in groups.items()
                for community in (communities[community_id],)
            }
            children = list(communities.values())
            built_layers = layer
            if len(children) <= 1:
                log.info(
                    "community_graph_build_stopped",
                    layer=layer,
                    reason="single_top_level_community",
                    community_count=len(children),
                )
                break

        self.driver.drop_projection(graph_name)
        log.info(
            "community_graph_build_completed",
            requested_levels=max_levels,
            built_layers=built_layers,
            total_communities=total_communities,
            total_membership_edges=total_membership_edges,
            total_aggregate_edges=total_aggregate_edges,
            duration=round(time.perf_counter() - start, 3),
        )
