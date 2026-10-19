import math
from collections import Counter, defaultdict
from itertools import combinations
from typing import ClassVar
from uuid import UUID

import numpy as np
from pydantic import RootModel
from structlog import get_logger

from wbpoa.core.model.element import Annex, Article, Guideline, Ruling, ValidProvision
from wbpoa.core.model.graph import EdgeUndirected, Node
from wbpoa.single_corpus.graph_builder.base_graph_builder import EDGES_PATH_NOT_PROVIDED
from wbpoa.single_corpus.graph_builder.generic_graph_builder import GenericGraphBuilder

log = get_logger()


class CoCitationGraphBuilder(GenericGraphBuilder[EdgeUndirected, frozenset[UUID]]):
    """
    Build a co-citation graph from the elements in the database. Each node represents a provision, and an edge is created between two nodes if they are cited together in a guideline.
    The weight of the edge is determined by the number of times the two elements are cited together.
    """

    MIN_REFERENCES_FOR_PAIR = 2
    PROVISION_TYPES: ClassVar[tuple[str, ...]] = (Article.__name__, Annex.__name__)

    def load_edges(self) -> None:
        """
        Load self.edges from edges_path as EdgeUndirected instances.
        """
        edges_json = self._read_cache_text(
            self.edges_path,
            cache_exists=self.has_edge_cache(),
            path_not_provided_msg=EDGES_PATH_NOT_PROVIDED,
            kind="Edge",
        )
        self.edges = RootModel[set[EdgeUndirected | None]].model_validate_json(edges_json).root
        self._validate_edge_corpus(self.edges)

    def _build_edge_index(self) -> dict[frozenset[UUID], EdgeUndirected]:
        """
        Index self.edges by their endpoint-id pair, for O(1) existence checks.

        Rebuilt fresh on every call (not cached on the instance) so callers stay correct
        regardless of how self.edges was populated (created here, loaded from cache, or set
        directly), at the cost of one O(|edges|) pass per call instead of one per pair.

        Returns:
            dict[frozenset[UUID], EdgeUndirected]: Existing edges keyed by their two endpoint ids.

        """
        return {frozenset(node.id for node in edge.nodes): edge for edge in self.edges if edge is not None}

    def _create_or_merge_edge(
        self,
        element: ValidProvision,
        node_by_id: dict[UUID, Node],
        edge_index: dict[frozenset[UUID], EdgeUndirected],
    ) -> set[EdgeUndirected | None]:
        """
        Create an edge between the node corresponding to the element and all the nodes corresponding to the provision_references elements. Raise an exception if a
        provision has no corresponding nodes. If an edge already exists between the nodes, update the weights. Otherwise create a new one.

        Args:
            element (ValidProvision): The element to create the edge from.
            node_by_id (dict[UUID, Node]): Lookup of the in-memory nodes by id.
            edge_index (dict[frozenset[UUID], EdgeUndirected]): Existing edges keyed by endpoint-id
                pair, shared across every element in the same `create_edges` call; updated in place
                as new edges are created so a later element sees an earlier one's edges.

        Raises:
            ValueError: If a reference node does not exist for a given reference in element.provision_references

        Returns:
            set[EdgeUndirected | None]: The created edges, or an empty set if no edges were created (e.g., if the element has no provision references).

        """
        if not element.provision_references:
            return set()

        edges: set[EdgeUndirected | None] = set()

        log.debug(
            "Creating or merging edges for element",
            element_id=element.id,
            element_name=element.name,
            edge_type="co_citation",
            provision_references=element.provision_references,
        )

        source_node = node_by_id.get(element.id)
        if not source_node:
            raise ValueError(f"Source node with id {element.id} does not exist.")

        for reference_id in element.provision_references:
            reference_node = node_by_id.get(reference_id)
            if not reference_node:
                raise ValueError(f"Reference node with id {reference_id} does not exist.")

            key = frozenset({element.id, reference_id})
            edge = edge_index.get(key)
            if edge:
                log.debug(
                    "Edge already exists between nodes, updating weight",
                    source_node=source_node.id,
                    reference_node=reference_node.id,
                    edge_type=edge.type,
                    old_weight=edge.weight,
                    new_weight=edge.weight + element.citation_weight,
                )
                edge.weight += element.citation_weight
                edge.weight_contributions[element.id] = (
                    edge.weight_contributions.get(element.id, 0.0) + element.citation_weight
                )
            else:
                edge = EdgeUndirected(
                    nodes=(source_node, reference_node),
                    weight=element.citation_weight,
                    type="co_citation",
                    weight_contributions={element.id: element.citation_weight},
                )
                edge_index[key] = edge

            edges.add(edge)

            log.debug(
                "Created or updated edge",
                source_node=source_node.id,
                reference_node=reference_node.id,
                edge_weight=edge.weight,
            )
        return edges

    def add_arch(self, element: Guideline | Ruling) -> None:
        """
        Create or update the arches based on the new elements, purely in-memory. An edge is created between each
        distinct pair of element in element.provision_references, with fractional counting: each pair receives
        citation_weight / (n - 1), where n is the number of elements element cites, so a source's total
        contribution is linear (citation_weight * n / 2) rather than quadratic in how much it cites (steps 7/11).
        If the edge already exists, its weight is updated, otherwise a new one is created.

        Args:
            element (Guideline | Ruling): The guideline or ruling to create the edges from.

        Raises:
            ValueError: If a reference node does not exist for a given reference in element.provision_references

        """
        n = len(element.provision_references)
        if n < self.MIN_REFERENCES_FOR_PAIR:
            return

        per_pair_weight = element.citation_weight / (n - 1)
        node_by_id = self._node_by_id()
        edge_index = self._build_edge_index()

        for node_1_id, node_2_id in sorted(map(sorted, combinations(set(element.provision_references), 2))):
            node_1 = node_by_id.get(node_1_id)
            node_2 = node_by_id.get(node_2_id)
            if not node_1 or not node_2:
                raise ValueError(f"One or both reference nodes with ids {node_1_id} and {node_2_id} do not exist.")

            key = frozenset({node_1_id, node_2_id})
            edge = edge_index.get(key)
            if edge:
                edge.weight += per_pair_weight
                edge.weight_contributions[element.id] = edge.weight_contributions.get(element.id, 0.0) + per_pair_weight
            else:
                edge = EdgeUndirected(
                    nodes=(node_1, node_2),
                    weight=per_pair_weight,
                    type="co_citation",
                    weight_contributions={element.id: per_pair_weight},
                )
                self.edges.add(edge)
                edge_index[key] = edge

            log.debug(
                "Added edge between nodes",
                node_1=node_1.id,
                node_2=node_2.id,
                edge_weight=edge.weight,
            )

    def add_structure_arch(
        self,
        elements: list[ValidProvision],
        *,
        section_weight: float | None = 2.0,
        chapter_weight: float | None = 1.0,
    ) -> None:
        """
        Add co-citation edges between articles that share a chapter or a section.

        Every pair of articles sharing a section receives an edge of weight
        section_weight / sqrt(|S| - 1), |S| the section's article count; every pair sharing a
        chapter receives an edge of weight chapter_weight / sqrt(|C| - 1), |C| the chapter's
        article count. Since a section is nested inside its chapter, two articles in the same
        section receive both edges, merged additively into one, exactly like add_arch merges
        multiple citing sources into one edge. Section labels (e.g. "section 1") repeat across
        different chapters, so section groups are keyed by (chapter, section), never by section
        alone. Articles with no chapter (or no section) simply don't participate in that grouping.
        Passing None for either weight disables that whole edge category.

        Args:
            elements: The provisions to scan for chapter/section membership; only Article
                instances with a chapter/section set contribute.
            section_weight: Weight budget w_s distributed as one edge weight per section group.
                None disables section edges entirely.
            chapter_weight: Weight budget w_c distributed as one edge weight per chapter group.
                None disables chapter edges entirely.

        Raises:
            ValueError: If a node for one of the grouped articles does not exist.

        """
        articles = [element for element in elements if isinstance(element, Article)]
        node_by_id = self._node_by_id()
        edge_index = self._build_edge_index()

        chapter_groups: dict[tuple[str, str], list[Article]] = defaultdict(list)
        section_groups: dict[tuple[str, str, str], list[Article]] = defaultdict(list)
        for article in articles:
            if article.chapter is not None:
                chapter_groups[(article.instrument_key, article.chapter)].append(article)
                if article.section is not None:
                    section_groups[(article.instrument_key, article.chapter, article.section)].append(article)

        chapter_edges = (
            self._add_group_edges(list(chapter_groups.values()), chapter_weight, node_by_id, edge_index)
            if chapter_weight is not None
            else 0
        )
        section_edges = (
            self._add_group_edges(list(section_groups.values()), section_weight, node_by_id, edge_index)
            if section_weight is not None
            else 0
        )
        log.info(
            "Added structure edges",
            chapter_groups=len(chapter_groups),
            section_groups=len(section_groups),
            chapter_edges=chapter_edges,
            section_edges=section_edges,
            chapter_weight=chapter_weight,
            section_weight=section_weight,
        )

    def _add_group_edges(
        self,
        groups: list[list[Article]],
        weight: float,
        node_by_id: dict[UUID, Node],
        edge_index: dict[frozenset[UUID], EdgeUndirected],
    ) -> int:
        """Add an edge of weight / sqrt(|group| - 1) between every pair of each group of size >= 2."""
        edge_count = 0
        for group in groups:
            n = len(group)
            if n < self.MIN_REFERENCES_FOR_PAIR:
                continue

            edge_weight = weight / math.sqrt(n - 1)
            for article_1, article_2 in combinations(sorted(group, key=lambda article: article.id), 2):
                node_1 = node_by_id.get(article_1.id)
                node_2 = node_by_id.get(article_2.id)
                if not node_1 or not node_2:
                    raise ValueError(f"One or both nodes with ids {article_1.id} and {article_2.id} do not exist.")

                key = frozenset({article_1.id, article_2.id})
                edge = edge_index.get(key)
                if edge:
                    edge.weight += edge_weight
                else:
                    edge = EdgeUndirected(
                        nodes=(node_1, node_2),
                        weight=edge_weight,
                        type="co_citation",
                        weight_contributions={},
                    )
                    self.edges.add(edge)
                    edge_index[key] = edge
                edge_count += 1
        return edge_count

    def add_semantic_backbone(
        self,
        k: int = 1,
        weight: float = 0.25,
        degree_threshold: int = 1,
        *,
        mutual: bool = True,
    ) -> None:
        """
        Add semantic_backbone edges so low-degree nodes aren't isolated from community detection.

        Degree is counted from self.edges only (no driver access, mirroring add_arch). Candidates
        are nodes whose co-citation degree is at or below degree_threshold -- including degree-0
        nodes, the population Leiden cannot merge on its own. Similarity is cosine over
        summary_embedding, computed as a single matrix product since the embedder normalises its
        vectors (A @ A.T). A candidate's own top-k neighbours are searched among PROVISION_TYPES
        nodes only (Article/Annex): the backbone exists to attach an orphan -- typically a recital
        that cites but is never cited -- to the provision it most plausibly explains, not to glue
        it to another orphan. The mutual reciprocity check below is deliberately left reading the
        unrestricted, whole-population top-k computed for every node: mutuality means "the
        provision also counts this node among its nearest neighbours in the whole population", not
        "among its nearest provisions" -- restricting both sides would make mutual=True accept
        zero edges for any non-provision candidate, since such a candidate could then never appear
        in anyone's (provision-restricted) top-k.

        Args:
            k: Number of nearest provision neighbours considered per candidate node.
            weight: Weight stored on every backbone edge (not the computed similarity). Deliberately
                decisive (matches Ruling.citation_weight, the level-2 weight) rather than negligible,
                since degree_threshold -- not weight -- gates whether a node merges at all.
            degree_threshold: Nodes with degree at or below this are backbone candidates.
            mutual: If True, keep an edge only when each endpoint is in the other's top-k; if
                False, keep all proposed edges but cap in-degree per target node at k.

        """
        nodes = [node for node in self.nodes if node is not None]
        min_nodes_for_backbone = 2
        if len(nodes) < min_nodes_for_backbone:
            return

        degree: Counter[UUID] = Counter()
        for edge in self.edges:
            if edge is None:
                continue
            degree[edge.nodes[0].id] += 1
            degree[edge.nodes[1].id] += 1

        embeddings = np.array([node.summary_embedding for node in nodes])
        similarity = embeddings @ embeddings.T
        np.fill_diagonal(similarity, -np.inf)
        # Top-k is computed for every node, not just candidates, so mutuality can be checked
        # against a neighbour that isn't itself a candidate. This stays over the whole
        # population -- only a candidate's own search (below) is provision-restricted.
        top_k = {i: set(np.argsort(-similarity[i])[:k].tolist()) for i in range(len(nodes))}

        candidates = [i for i, node in enumerate(nodes) if degree.get(node.id, 0) <= degree_threshold]

        provision_mask = np.array([node.type in self.PROVISION_TYPES for node in nodes])
        masked_similarity = np.where(provision_mask, similarity, -np.inf)
        # The != -inf filter matters: with fewer than k provisions available, argsort still
        # returns k indices, and the leftover slots point at masked-out columns whose
        # (unmasked) similarity is a real number, not a sentinel -- silently proposing an
        # edge to a non-provision node if left unfiltered.
        provision_top_k = {
            i: [j for j in np.argsort(-masked_similarity[i])[:k].tolist() if masked_similarity[i][j] != -np.inf]
            for i in candidates
        }
        proposed = [(float(similarity[i][j]), i, j) for i in candidates for j in provision_top_k[i]]

        if mutual:
            accepted = [(sim, i, j) for sim, i, j in proposed if i in top_k[j]]
        else:
            proposed.sort(key=lambda item: item[0], reverse=True)
            in_degree: Counter[int] = Counter()
            accepted = []
            for sim, i, j in proposed:
                if in_degree[j] >= k:
                    continue
                in_degree[j] += 1
                accepted.append((sim, i, j))

        new_edges: set[EdgeUndirected] = set()
        seen_pairs: set[frozenset[UUID]] = set()
        edge_index = self._build_edge_index()
        for _sim, i, j in accepted:
            node_i, node_j = nodes[i], nodes[j]
            pair = frozenset({node_i.id, node_j.id})
            if pair in seen_pairs or pair in edge_index:
                continue  # already backboned this pair, or a real co-citation edge already links it
            seen_pairs.add(pair)
            new_edges.add(
                EdgeUndirected(nodes=(node_i, node_j), weight=weight, type="semantic_backbone", weight_contributions={})
            )
        self.edges.update(new_edges)
        log.info(
            "Added semantic backbone edges",
            candidate_count=len(candidates),
            provision_count=int(provision_mask.sum()),
            edge_count=len(new_edges),
            k=k,
            weight=weight,
            mutual=mutual,
        )
