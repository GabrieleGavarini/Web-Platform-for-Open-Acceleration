from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date
from types import TracebackType
from typing import Any, Final, Literal, LiteralString, Self, cast
from uuid import UUID

from neo4j import Driver, GraphDatabase, ManagedTransaction, Session
from neo4j.time import Date as Neo4jDate
from structlog import get_logger

from wbpoa.core.corpus.corpus_config import InstrumentRelationType, InstrumentRole, InstrumentStatus
from wbpoa.core.model.graph.edge import Edge
from wbpoa.core.model.graph.edge_directed import EdgeDirected
from wbpoa.core.model.graph.edge_undirected import EdgeUndirected
from wbpoa.core.model.graph.node import Node
from wbpoa.core.model.operative_relation import OperativeOperation
from wbpoa.core.settings import CoreSettings, get_core_settings

log = get_logger()

NODE_LABEL: Final = "Entity"
SUMMARY_EMBEDDING_PROPERTY: Final = "summary_embedding"
ELEMENT_TYPES: Final = ("Article", "Recital", "Annex")
# Node properties read when listing nodes for browsing: everything but the text and the embedding.
_LISTING_PROJECTION: Final = (
    ".id, .corpus_id, .name, .type, .layer, .instrument_key, .celex_id, .provision_path, "
    ".instrument_role, .instrument_status"
)

_RELATIONSHIP_TYPE_PATTERN: Final = re.compile(r"^[A-Z_][A-Z0-9_]*$")


class Neo4jDriverError(RuntimeError):
    """Raised when a graph write cannot be carried out as requested."""


def _relationship_type(edge: Edge) -> str:
    """
    Normalise ``edge.type`` into a valid Neo4j relationship type.

    Spaces and hyphens become underscores and the result is upper-cased, so
    ``"cited by"`` becomes ``CITED_BY``. Anything that is still not a bare
    identifier afterwards is rejected rather than silently mangled -- this
    validation is also what makes it safe to interpolate the type into the
    query text, which Cypher requires since a relationship type cannot be a
    parameter.

    Args:
        edge: The edge whose type should be normalised.

    Returns:
        The relationship type to use in Cypher.

    Raises:
        Neo4jDriverError: If the normalised type is not a valid identifier.

    """
    normalised = edge.type.strip().replace(" ", "_").replace("-", "_").upper()
    if not _RELATIONSHIP_TYPE_PATTERN.match(normalised):
        raise Neo4jDriverError(
            f"Edge type {edge.type!r} is not a valid Neo4j relationship type: "
            "expected letters, digits and underscores, not starting with a digit."
        )
    return normalised


class Neo4jDriver[E: Edge]:
    """
    Write graphs built from the Pydantic model into a Neo4j database.

    Nodes are written with a single label, :data:`NODE_LABEL`, keeping ``Node.type``
    as a property; one stable label means one vector index per embedding instead of
    one per type. ``Edge.type`` becomes the Neo4j relationship type. Neo4j has no
    undirected relationship, so :class:`EdgeUndirected` is stored smaller-``id``
    first and flagged with an ``undirected`` property; match it without a direction
    (``MATCH (a:Entity)-[r]-(b:Entity)``).

    Every write is a ``MERGE``, so re-ingesting the same graph updates it instead of
    duplicating it.

    The underlying :class:`neo4j.Driver` owns a connection pool, so share one
    instance rather than creating one per call, and use it as a context manager
    (or call :meth:`close`) to release the pool.

    Args:
        settings: Configuration to use. Defaults to
            :func:`wbpoa.core.settings.get_core_settings`, i.e. the ``NEO4J_*``
            environment variables or ``.env`` entries.

    Example:
        >>> with Neo4jDriver() as driver:
        ...     driver.ensure_schema()
        ...     driver.create_graph(nodes, edges)

    """

    def __init__(self, settings: CoreSettings | None = None, *, corpus_id: str = "legacy") -> None:
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*|legacy", corpus_id):
            raise ValueError(f"Invalid corpus id: {corpus_id!r}")
        self._settings = settings if settings is not None else get_core_settings()
        self.corpus_id = corpus_id
        self._corpus_label = f"Corpus_{corpus_id.replace('-', '_')}"
        self._driver: Driver | None = None

    @property
    def driver(self) -> Driver:
        """The underlying Neo4j driver, opened on first access."""
        if self._driver is None:
            neo4j = self._settings.neo4j
            self._driver = GraphDatabase.driver(neo4j.url, auth=(neo4j.user, neo4j.password.get_secret_value()))
        return self._driver

    def close(self) -> None:
        """Close the driver and its connection pool. Safe to call twice."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def clear_graph(self, batch_size: int = 10_000) -> None:
        """
        Delete every node and relationship in the configured database.

        Runs in batched transactions so a large graph does not exhaust the
        heap. Not atomic: a failure part-way leaves a partially deleted
        graph. Schema (constraints and indexes) is left in place.
        """
        query = cast(
            "LiteralString",
            f"MATCH (n:{NODE_LABEL}) WHERE n.corpus_id = $corpus_id "
            f"CALL (n) {{ DETACH DELETE n }} IN TRANSACTIONS OF {batch_size} ROWS",
        )
        with self.driver.session(database=self._settings.neo4j.database) as session:
            session.run(query, corpus_id=self.corpus_id).consume()
        log.info(
            "Cleared graph in database",
            database=self._settings.neo4j.database,
            batch_size=batch_size,
        )

    def drop_communities(self) -> None:
        """Remove persisted community nodes while preserving layer-0 graph data."""
        query = cast(
            "LiteralString",
            f"MATCH (n:{NODE_LABEL}) WHERE n.corpus_id = $corpus_id AND n.type = 'Community' DETACH DELETE n",
        )
        with self.driver.session(database=self._settings.neo4j.database) as session:
            session.run(query, corpus_id=self.corpus_id).consume()

    def project_cocitation(
        self, graph_name: str, *, include_backbone: bool = True, include_operative_affinity: bool = True
    ) -> None:
        """
        Project layer-0 co-citation (and, unless disabled, semantic backbone and operative affinity)
        relationships into an undirected GDS graph.

        ``include_backbone``/``include_operative_affinity`` are A/B toggles adding the kNN semantic
        backbone and the cross-instrument operative affinity graph on top of the plain
        citation-derived co-citation graph: each edge type is matched by its own branch (a single
        projected row can only carry one relationship), unioned together so Leiden sees all of them
        as one undirected weighted graph. GDS rejects an ``undirectedRelationshipTypes`` entry for a
        type that no row actually materialised, so each toggle is additionally gated on at least one
        matching relationship really existing in the database -- otherwise (e.g. right after
        ``--invalidate-community-cache`` alone, which doesn't touch the co-citation edges) the
        projection would fail rather than silently degrade.

        Args:
            graph_name: Name to give the GDS projection.
            include_backbone: Whether to also project SEMANTIC_BACKBONE relationships, if any exist.
            include_operative_affinity: Whether to also project OPERATIVE_AFFINITY relationships, if
                any exist.

        """
        self._validate_graph_name(graph_name)
        self.drop_projection(graph_name)
        include_backbone = include_backbone and self._relationship_exists("SEMANTIC_BACKBONE")
        include_operative_affinity = include_operative_affinity and self._relationship_exists("OPERATIVE_AFFINITY")
        # Leiden requires an undirected projection; each relationship is matched in a single
        # direction because the projection materialises both orientations itself.
        branches = [
            (
                f"MATCH (a:{NODE_LABEL}) WHERE a.corpus_id = $corpus_id AND a.layer = 0\n"
                f"OPTIONAL MATCH (a)-[r:CO_CITATION]->(b:{NODE_LABEL})\n"
                "WHERE b.corpus_id = $corpus_id AND b.layer = 0 AND r.undirected = true\n"
                "RETURN a AS a, b AS b, r AS r, 'CO_CITATION' AS relationship_type"
            )
        ]
        undirected_types = ["CO_CITATION"]
        if include_backbone:
            branches.append(
                f"MATCH (a:{NODE_LABEL}) WHERE a.corpus_id = $corpus_id AND a.layer = 0\n"
                f"OPTIONAL MATCH (a)-[r:SEMANTIC_BACKBONE]->(b:{NODE_LABEL})\n"
                "WHERE b.corpus_id = $corpus_id AND b.layer = 0 AND r.undirected = true\n"
                "RETURN a AS a, b AS b, r AS r, 'SEMANTIC_BACKBONE' AS relationship_type"
            )
            undirected_types.append("SEMANTIC_BACKBONE")
        if include_operative_affinity:
            branches.append(
                f"MATCH (a:{NODE_LABEL}) WHERE a.corpus_id = $corpus_id AND a.layer = 0\n"
                f"OPTIONAL MATCH (a)-[r:OPERATIVE_AFFINITY]->(b:{NODE_LABEL})\n"
                "WHERE b.corpus_id = $corpus_id AND b.layer = 0 AND r.undirected = true\n"
                "RETURN a AS a, b AS b, r AS r, 'OPERATIVE_AFFINITY' AS relationship_type"
            )
            undirected_types.append("OPERATIVE_AFFINITY")
        undirected_types_literal = "[" + ", ".join(f"'{name}'" for name in undirected_types) + "]"
        query = cast(
            "LiteralString",
            "CALL {\n" + "\nUNION ALL\n".join(branches) + "\n}\n"
            "RETURN gds.graph.project(\n"
            "$graph_name,\n"
            "a,\n"
            "b,\n"
            "{relationshipType: relationship_type, relationshipProperties: r {.weight}},\n"
            f"{{undirectedRelationshipTypes: {undirected_types_literal}}}\n"
            ")",
        )
        with self.driver.session(database=self._settings.neo4j.database) as session:
            session.run(query, graph_name=self._scoped_graph_name(graph_name), corpus_id=self.corpus_id).consume()

    def _relationship_exists(self, relationship_type: str) -> bool:
        """Return whether at least one relationship of ``relationship_type`` exists in the database."""
        query = cast(
            "LiteralString",
            f"RETURN EXISTS {{ MATCH (a:{NODE_LABEL})-[:{relationship_type}]-(b:{NODE_LABEL}) "
            "WHERE a.corpus_id = $corpus_id AND b.corpus_id = $corpus_id } AS relationship_exists",
        )
        with self.driver.session(database=self._settings.neo4j.database) as session:
            record = session.run(query, corpus_id=self.corpus_id).single()
            return bool(record["relationship_exists"]) if record else False

    def drop_projection(self, graph_name: str) -> None:
        """Drop a named GDS projection if it exists."""
        self._validate_graph_name(graph_name)
        query = cast(
            "LiteralString",
            "CALL gds.graph.drop($graph_name, false) YIELD graphName RETURN graphName",
        )
        with self.driver.session(database=self._settings.neo4j.database) as session:
            session.run(query, graph_name=self._scoped_graph_name(graph_name)).consume()

    def run_leiden(self, graph_name: str, max_levels: int, leiden_gamma: float = 1.0) -> dict[UUID, list[int]]:
        """
        Return intermediate Leiden community IDs keyed by persisted node UUID.

        Args:
            graph_name: Name of the projected GDS graph.
            max_levels: Maximum hierarchy levels to return.
            leiden_gamma: Leiden resolution parameter (must be > 0).

        """
        self._validate_graph_name(graph_name)
        graph_name = self._scoped_graph_name(graph_name)
        if leiden_gamma <= 0:
            message = "leiden_gamma must be strictly positive"
            raise ValueError(message)
        query = cast(
            "LiteralString",
            "CALL gds.leiden.stream($graph_name, {"
            "relationshipWeightProperty: 'weight', "
            "includeIntermediateCommunities: true, "
            "maxLevels: $max_levels, "
            "gamma: $leiden_gamma"
            "}) YIELD nodeId, communityId, intermediateCommunityIds "
            "RETURN gds.util.asNode(nodeId).id AS node_id, communityId, intermediateCommunityIds",
        )
        assignments: dict[UUID, list[int]] = {}
        with self.driver.session(database=self._settings.neo4j.database) as session:
            result = session.run(
                query,
                graph_name=graph_name,
                max_levels=max_levels,
                leiden_gamma=leiden_gamma,
            )
            for record in result:
                node_id = UUID(record["node_id"])
                intermediate = record.get("intermediateCommunityIds")
                if intermediate is None:
                    intermediate = [record["communityId"]]
                assignments[node_id] = [int(community_id) for community_id in intermediate]
        return assignments

    def aggregate_weights(self, layer: int) -> list[tuple[str, str, float]]:
        """
        Sum lower-layer co-citation weights crossing current-layer communities.

        Community pairs are matched by ``id`` rather than ``name``: names may now be
        LLM-generated titles and are not guaranteed unique.

        Returns:
            Tuples of ``(community_a_id, community_b_id, weight)``, ids as strings.

        """
        if layer < 1:
            message = "layer must be at least 1"
            raise ValueError(message)
        relationship_type = "CO_CITATION" if layer == 1 else "COMMUNITY_CO_CITATION"
        query = cast(
            "LiteralString",
            f"MATCH (a:{NODE_LABEL})-[r:{relationship_type}]-(b:{NODE_LABEL}) "
            "WHERE a.corpus_id = $corpus_id AND b.corpus_id = $corpus_id "
            "AND a.layer = $lower_layer AND b.layer = $lower_layer "
            "AND r.undirected = true "
            f"MATCH (ca:{NODE_LABEL})-[:MEMBER]->(a) "
            f"MATCH (cb:{NODE_LABEL})-[:MEMBER]->(b) "
            "WHERE ca.corpus_id = $corpus_id AND cb.corpus_id = $corpus_id "
            "AND ca.layer = $layer AND cb.layer = $layer AND ca.id < cb.id "
            "RETURN ca.id AS community_a, cb.id AS community_b, sum(r.weight) AS weight",
        )
        with self.driver.session(database=self._settings.neo4j.database) as session:
            result = session.run(query, layer=layer, lower_layer=layer - 1, corpus_id=self.corpus_id)
            return [(record["community_a"], record["community_b"], float(record["weight"])) for record in result]

    def _scoped_graph_name(self, graph_name: str) -> str:
        """Return a deterministic, corpus-specific GDS graph name."""
        scoped = f"{graph_name}_{self.corpus_id.replace('-', '_')}"
        self._validate_graph_name(scoped)
        return scoped

    @staticmethod
    def _validate_graph_name(graph_name: str) -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", graph_name):
            raise Neo4jDriverError(f"Invalid GDS graph name: {graph_name!r}")

    def ensure_schema(
        self,
        dimensions: int = 1024,
        similarity: Literal["cosine", "euclidean"] = "cosine",
    ) -> None:
        """
        Create the constraint and index the graph relies on.

        Idempotent, so it can run on every startup: a uniqueness constraint on
        ``Entity.id`` (which also backs the endpoint lookups in
        :meth:`_add_edge`), a composite index on the qualified-citation
        properties (backing :meth:`qualified_provision`), and the summary
        embedding vector index. Unlike ``CREATE VECTOR INDEX ... IF NOT EXISTS``
        alone, an existing index whose stored ``vector.dimensions`` no longer
        matches ``dimensions`` (e.g. after switching ``EMBED_MODEL``) is dropped
        and recreated instead of left stale.

        Args:
            dimensions: Dimensionality of the stored embeddings; must match the
                model in ``EmbeddingSettings``. 1024 is ``BAAI/bge-m3``.
            similarity: Vector similarity function. Cosine is correct for
                normalized embeddings (``EMBED_NORMALIZE=true``).

        """
        with self.driver.session(database=self._settings.neo4j.database) as session:
            session.run(
                cast(
                    "LiteralString",
                    f"CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (n:{NODE_LABEL}) REQUIRE n.id IS UNIQUE",
                )
            )
            session.run(
                cast(
                    "LiteralString",
                    f"CREATE INDEX entity_corpus_id IF NOT EXISTS FOR (n:{NODE_LABEL}) ON (n.corpus_id)",
                )
            )
            session.run(
                cast(
                    "LiteralString",
                    f"CREATE INDEX entity_qualified_citation IF NOT EXISTS FOR (n:{NODE_LABEL}) "
                    "ON (n.instrument_key, n.celex_id, n.provision_path)",
                )
            )
            self._ensure_vector_index(session, SUMMARY_EMBEDDING_PROPERTY, dimensions, similarity)

    def _ensure_vector_index(
        self,
        session: Session,
        property_name: str,
        dimensions: int,
        similarity: Literal["cosine", "euclidean"],
    ) -> None:
        """
        Create the vector index for ``property_name``, recreating it if its dimensions changed.

        Args:
            session: Open session to run the ``SHOW INDEXES``/``DROP INDEX``/``CREATE VECTOR INDEX``
                statements on.
            property_name: Embedding property the index is built on.
            dimensions: Dimensionality the index must have.
            similarity: Vector similarity function.

        """
        index_name = f"entity_{self.corpus_id.replace('-', '_')}_{property_name}"
        existing = session.run(
            "SHOW INDEXES YIELD name, options WHERE name = $index_name",
            index_name=index_name,
        ).single()
        if existing is not None:
            current_dimensions = existing["options"].get("indexConfig", {}).get("vector.dimensions")
            if current_dimensions == dimensions:
                return
            log.info(
                "Recreating vector index: dimensions changed",
                index_name=index_name,
                previous_dimensions=current_dimensions,
                dimensions=dimensions,
            )
            # Index names cannot be parameterised; index_name is built from a fixed set of property names.
            session.run(cast("LiteralString", f"DROP INDEX {index_name}"))

        # Index options cannot be parameterised. The casts are safe: an int and
        # a two-value Literal cannot carry injected Cypher.
        session.run(
            cast(
                "LiteralString",
                f"CREATE VECTOR INDEX {index_name} IF NOT EXISTS "
                f"FOR (n:{self._corpus_label}) ON (n.{property_name}) "
                f"OPTIONS {{indexConfig: {{"
                f"`vector.dimensions`: {dimensions}, "
                f"`vector.similarity_function`: '{similarity}'}}}}",
            )
        )

    def _add_node(self, tx: ManagedTransaction, node: Node) -> None:
        """
        Create or update one node inside an open transaction.

        Merged on ``id``, so a repeated call updates rather than duplicates.
        The summary embedding goes through ``db.create.setNodeVectorProperty``
        so it is stored in Neo4j's vector encoding and usable by the vector
        index; an empty vector is skipped, since that procedure rejects it.

        Args:
            tx: Transaction supplied by :meth:`create_graph`.
            node: The node to persist.

        """
        if node.corpus_id != self.corpus_id:
            raise Neo4jDriverError(f"Node {node.id} belongs to corpus {node.corpus_id!r}, not {self.corpus_id!r}.")
        log.debug(
            "Adding node to Neo4j",
            node_id=node.id,
            node_name=node.name,
            node_type=node.type,
        )

        clauses: list[LiteralString] = [
            cast("LiteralString", f"MERGE (n:{node.type}:{NODE_LABEL}:{self._corpus_label} {{id: $id}})"),
            "SET n += $properties",
        ]
        parameters: dict[str, Any] = {
            "id": str(node.id),
            "properties": {
                "corpus_id": node.corpus_id,
                "name": node.name,
                "type": node.type,
                "text": node.text,
                "summary": node.summary,
                "layer": node.layer,
                "instrument_key": node.instrument_key,
                "celex_id": node.celex_id,
                "provision_path": node.provision_path,
                "instrument_role": node.instrument_role.value if node.instrument_role is not None else None,
                "instrument_status": node.instrument_status.value if node.instrument_status is not None else None,
                "effective_from": node.effective_from,
                "effective_to": node.effective_to,
                "member_instrument_keys": node.member_instrument_keys,
                "member_instrument_roles": [role.value for role in node.member_instrument_roles],
            },
        }

        if node.summary_embedding:
            clauses.append(
                f"WITH n CALL db.create.setNodeVectorProperty(n, '{SUMMARY_EMBEDDING_PROPERTY}', $"
                f"{SUMMARY_EMBEDDING_PROPERTY})"
            )
            parameters[SUMMARY_EMBEDDING_PROPERTY] = node.summary_embedding

        tx.run("\n".join(clauses), parameters)

    def _add_edge(self, tx: ManagedTransaction, edge: Edge) -> None:
        """
        Create or update one edge inside an open transaction.

        Both endpoints must already exist -- :meth:`create_graph` writes all
        nodes first. They are matched rather than merged so a malformed edge
        fails loudly instead of creating a stub node holding only an ``id``.
        Undirected edges are reordered smaller-``id`` first, which makes the
        ``MERGE`` idempotent whichever way round the pair was supplied.

        Args:
            tx: Transaction supplied by :meth:`create_graph`.
            edge: An :class:`EdgeDirected` or :class:`EdgeUndirected`.

        Raises:
            Neo4jDriverError: If the edge subclass is unsupported, its ``type``
                is invalid, or an endpoint is missing from the database.

        """
        if isinstance(edge, EdgeDirected):
            source, target, undirected = edge.source, edge.target, False
        elif isinstance(edge, EdgeUndirected):
            source, target = sorted(edge.nodes, key=lambda node: str(node.id))
            undirected = True
        else:
            raise Neo4jDriverError(
                f"Unsupported edge type {type(edge).__name__!r}: expected EdgeDirected or EdgeUndirected."
            )

        if source.corpus_id != self.corpus_id or target.corpus_id != self.corpus_id:
            raise Neo4jDriverError(
                f"Edge endpoints must belong to corpus {self.corpus_id!r}; "
                f"got {source.corpus_id!r} and {target.corpus_id!r}."
            )
        relationship_type = _relationship_type(edge)
        query = cast(
            "LiteralString",
            f"MATCH (s:{source.type} {{id: $source_id, corpus_id: $corpus_id}})\n"
            f"MATCH (t:{target.type} {{id: $target_id, corpus_id: $corpus_id}})\n"
            f"MERGE (s)-[r:{relationship_type}]->(t)\n"
            "SET r.weight = $weight, r.undirected = $undirected, "
            "r.contributor_ids = $contributor_ids, r.contributor_weights = $contributor_weights, "
            "r.corpus_id = $corpus_id, r.relation_type = $relation_type, r.operation = $operation\n"
            "RETURN elementId(r) AS edge_id",
        )
        # Relationship properties cannot hold a map, so the contributions dict is split into two parallel arrays.
        contributor_ids = [str(element_id) for element_id in edge.weight_contributions]
        contributor_weights = list(edge.weight_contributions.values())
        record = tx.run(
            query,
            source_id=str(source.id),
            target_id=str(target.id),
            weight=edge.weight,
            undirected=undirected,
            contributor_ids=contributor_ids,
            contributor_weights=contributor_weights,
            corpus_id=self.corpus_id,
            relation_type=edge.relation_type.value if edge.relation_type is not None else None,
            operation=edge.operation.value if edge.operation is not None else None,
        ).single()

        if record is None:
            raise Neo4jDriverError(
                f"Cannot create {relationship_type} edge between {source.id} and "
                f"{target.id}: one or both nodes are missing from the database."
            )

    def create_graph(self, nodes: set[Node | None], edges: set[E | None]) -> None:
        """
        Persist a whole graph: all nodes first, then all edges.

        Nodes go first so every endpoint exists by the time relationships are
        created. Everything runs in a single write transaction, so a failure
        rolls the whole graph back.

        Args:
            nodes: Nodes to create or update.
            edges: Edges to create or update. Their endpoints must be in
                ``nodes`` or already in the database.

        Raises:
            Neo4jDriverError: If an edge is unsupported, carries an invalid
                type, or refers to a node that does not exist.

        """

        def write(tx: ManagedTransaction) -> None:
            for node in nodes:
                if node is not None:
                    self._add_node(tx, node)
            for edge in edges:
                if edge is not None:
                    self._add_edge(tx, edge)

        with self.driver.session(database=self._settings.neo4j.database) as session:
            session.execute_write(write)

        log.info(
            "Wrote %d nodes and %d edges to Neo4j",
            len(nodes),
            len(edges),
        )

    @staticmethod
    def _to_date(value: Neo4jDate | date | None) -> date | None:
        """Convert a stored Neo4j Date value back into a native :class:`datetime.date`."""
        if value is None:
            return None
        if isinstance(value, Neo4jDate):
            return value.to_native()
        return value

    @staticmethod
    def _to_node(node_data: Any) -> Node:
        """Reconstruct a :class:`Node` from a Neo4j node record value."""
        instrument_role = node_data.get("instrument_role")
        instrument_status = node_data.get("instrument_status")
        return Node(
            corpus_id=node_data.get("corpus_id", "legacy"),
            id=node_data["id"],
            name=node_data.get("name", ""),
            type=node_data.get("type", ""),
            text=node_data.get("text", ""),
            summary=node_data.get("summary", ""),
            summary_embedding=node_data.get(SUMMARY_EMBEDDING_PROPERTY, []),
            layer=node_data.get("layer", 0),
            instrument_key=node_data.get("instrument_key"),
            celex_id=node_data.get("celex_id"),
            provision_path=node_data.get("provision_path"),
            instrument_role=InstrumentRole(instrument_role) if instrument_role is not None else None,
            instrument_status=InstrumentStatus(instrument_status) if instrument_status is not None else None,
            effective_from=Neo4jDriver._to_date(node_data.get("effective_from")),
            effective_to=Neo4jDriver._to_date(node_data.get("effective_to")),
            member_instrument_keys=node_data.get("member_instrument_keys", []),
            member_instrument_roles=[InstrumentRole(role) for role in node_data.get("member_instrument_roles", [])],
        )

    def get_node_by_id(self, node_id: UUID) -> Node | None:
        """
        Retrieve a node by its ID.

        Args:
            node_id (UUID): The ID of the node to retrieve.

        Returns:
            The Node if found, otherwise None.

        """
        query = cast(
            "LiteralString",
            f"MATCH (n:{NODE_LABEL} {{id: $node_id, corpus_id: $corpus_id}}) RETURN n",
        )
        with self.driver.session(database=self._settings.neo4j.database) as session:
            record = session.run(query, node_id=str(node_id), corpus_id=self.corpus_id).single()
            if record is None:
                return None
            return self._to_node(record["n"])

    def nodes_by_ids(self, node_ids: Sequence[UUID]) -> list[Node]:
        """Return the nodes with the given ids, in arbitrary order; missing ids are simply absent."""
        query = cast(
            "LiteralString",
            f"UNWIND $node_ids AS node_id\nMATCH (n:{NODE_LABEL} {{id: node_id, corpus_id: $corpus_id}})\nRETURN n",
        )
        with self.driver.session(database=self._settings.neo4j.database) as session:
            result = session.run(query, node_ids=[str(node_id) for node_id in node_ids], corpus_id=self.corpus_id)
            return [self._to_node(record["n"]) for record in result]

    def qualified_provision(self, instrument_key: str, celex_id: str, provision_path: str) -> Node | None:
        """
        Retrieve a provision node by its instrument-qualified citation.

        Backed by the ``entity_qualified_citation`` composite index created in
        :meth:`ensure_schema`.

        Args:
            instrument_key: Stable key of the provision's source instrument.
            celex_id: Official CELEX identifier of the source instrument.
            provision_path: Canonical top-level path within the instrument.

        Returns:
            The Node if found, otherwise None.

        """
        query = cast(
            "LiteralString",
            f"MATCH (n:{NODE_LABEL} {{instrument_key: $instrument_key, celex_id: $celex_id, "
            "provision_path: $provision_path, corpus_id: $corpus_id}) RETURN n",
        )
        with self.driver.session(database=self._settings.neo4j.database) as session:
            record = session.run(
                query,
                instrument_key=instrument_key,
                celex_id=celex_id,
                provision_path=provision_path,
                corpus_id=self.corpus_id,
            ).single()
            if record is None:
                return None
            return self._to_node(record["n"])

    def get_edge_between_nodes(self, source_id: UUID, target_id: UUID) -> EdgeDirected | EdgeUndirected | None:
        """
        Retrieve an edge between two nodes by their IDs.

        The match is undirected, since :meth:`_add_edge` stores an
        :class:`EdgeUndirected` smaller-``id`` first regardless of the order it
        was given in. For a directed edge the stored orientation wins: the
        returned ``source`` and ``target`` may therefore be the reverse of the
        arguments.

        Args:
            source_id: The ID of one endpoint.
            target_id: The ID of the other endpoint.

        Returns:
            The Edge if found, otherwise None.

        Raises:
            Neo4jDriverError: If a relationship is found but one of its
                endpoints cannot be read back.

        """
        query = cast(
            "LiteralString",
            f"MATCH (s:{NODE_LABEL} {{id: $source_id, corpus_id: $corpus_id}})-[r]-(t:{NODE_LABEL} {{id: $target_id, corpus_id: $corpus_id}})\n"
            "RETURN r, startNode(r).id AS start_id",
        )
        with self.driver.session(database=self._settings.neo4j.database) as session:
            record = session.run(
                query, source_id=str(source_id), target_id=str(target_id), corpus_id=self.corpus_id
            ).single()

        if record is None:
            return None

        edge_data = record["r"]

        log.debug(
            "Retrieved edge from Neo4j",
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_data.type,
            edge_properties=dict(edge_data),
        )

        source_node = self.get_node_by_id(source_id)
        target_node = self.get_node_by_id(target_id)

        if source_node is None or target_node is None:
            log.error(
                "One or both nodes do not exist for the retrieved edge",
                source_id=source_id,
                target_id=target_id,
            )
            raise Neo4jDriverError(f"One or both nodes with IDs {source_id} and {target_id} do not exist.")

        weight = edge_data.get("weight", 1)
        weight_contributions = dict(
            zip(
                (UUID(element_id) for element_id in edge_data.get("contributor_ids", [])),
                edge_data.get("contributor_weights", []),
                strict=True,
            )
        )
        relation_type = edge_data.get("relation_type")
        operation = edge_data.get("operation")
        relation_type_value = InstrumentRelationType(relation_type) if relation_type is not None else None
        operation_value = OperativeOperation(operation) if operation is not None else None

        if edge_data.get("undirected", False):
            return EdgeUndirected(
                nodes=(source_node, target_node),
                type=edge_data.type,
                weight=weight,
                weight_contributions=weight_contributions,
                relation_type=relation_type_value,
                operation=operation_value,
            )

        if record["start_id"] == str(source_id):
            return EdgeDirected(
                source=source_node,
                target=target_node,
                type=edge_data.type,
                weight=weight,
                weight_contributions=weight_contributions,
                relation_type=relation_type_value,
                operation=operation_value,
            )
        return EdgeDirected(
            source=target_node,
            target=source_node,
            type=edge_data.type,
            weight=weight,
            weight_contributions=weight_contributions,
            relation_type=relation_type_value,
            operation=operation_value,
        )

    def create_or_update_edge(self, edge: EdgeDirected | EdgeUndirected) -> None:
        """
        Create a new edge or update an existing edge in the database.

        Args:
            edge: The edge to create or update.

        Raises:
            Neo4jDriverError: If the edge does not exist or is of an unsupported type.

        """

        def write(tx: ManagedTransaction) -> None:
            if edge is not None:
                self._add_edge(tx, edge)

        with self.driver.session(database=self._settings.neo4j.database) as session:
            session.execute_write(write)

    def vector_search(  # noqa: PLR0913
        self,
        embedding: Sequence[float],
        k: int,
        *,
        layer: int | None = None,
        node_types: Sequence[str] | None = None,
        candidate_pool: int | None = None,
        instrument_roles: Sequence[InstrumentRole] | None = None,
    ) -> list[tuple[Node, float]]:
        """
        Find the k nearest nodes to ``embedding`` by the summary embedding vector index, optionally restricted to one layer.

        Used by Algorithm 2 to shortlist candidate communities (layer >= 1) and elements
        (layer 0); the returned score reflects the index's configured similarity function
        (:meth:`ensure_schema`), so callers that need the algorithm's cosine ``sim(q, v)`` regardless
        of that configuration should recompute it themselves rather than trust this score directly.

        Args:
            embedding: The query vector.
            k: Number of nodes to return.
            layer: If given, only nodes with this ``layer`` are returned; since the index itself
                cannot pre-filter, more candidates than ``k`` are pulled from it first.
            node_types: If given, only nodes whose ``type`` is in this set are returned. Needed
                because ``layer`` alone does not distinguish provisions from guidelines/rulings:
                both default to ``layer=0``, but only provisions belong to the community graph.
            candidate_pool: Number of candidates to pull from the index before filtering by layer.
                Required when ``layer`` is given, since a tighter default risks under-filling the
                filtered result; ignored otherwise (``k`` is used).
            instrument_roles: If given, only nodes whose own ``instrument_role`` (elements) or any
                of whose ``member_instrument_roles`` (communities) is in this set are returned;
                Phase 4's relation-category filter. Nodes with neither property (guidelines,
                rulings) are always excluded when this filter is active, since they belong to no
                instrument. Regardless of this filter, nodes with ``instrument_status = 'inactive'``
                are always excluded (guidelines/rulings/communities, which carry no
                ``instrument_status``, are unaffected).

        Returns:
            Up to ``k`` ``(Node, index_score)`` pairs, best first.

        """
        index_name = f"{NODE_LABEL.lower()}_{self.corpus_id.replace('-', '_')}_{SUMMARY_EMBEDDING_PROPERTY}"
        pool = candidate_pool if candidate_pool is not None else k
        clauses: list[LiteralString] = [
            "CALL db.index.vector.queryNodes($index_name, $pool, $embedding) YIELD node, score",
        ]
        filters: list[LiteralString] = []
        if layer is not None:
            filters.append("node.layer = $layer")
        if node_types is not None:
            filters.append("node.type IN $node_types")
        if instrument_roles is not None:
            filters.append(
                "(node.instrument_role IN $instrument_roles OR "
                "any(role IN node.member_instrument_roles WHERE role IN $instrument_roles))"
            )
        filters.append("(node.instrument_status = 'active' OR node.instrument_status IS NULL)")
        filters.append("node.corpus_id = $corpus_id")
        if filters:
            clauses.append("WHERE " + " AND ".join(filters))
        clauses.append("RETURN node, score ORDER BY score DESC LIMIT $k")
        query = cast("LiteralString", "\n".join(clauses))
        with self.driver.session(database=self._settings.neo4j.database) as session:
            result = session.run(
                query,
                index_name=index_name,
                pool=pool,
                embedding=list(embedding),
                layer=layer,
                node_types=list(node_types) if node_types is not None else None,
                instrument_roles=[role.value for role in instrument_roles] if instrument_roles is not None else None,
                corpus_id=self.corpus_id,
                k=k,
            )
            return [(self._to_node(record["node"]), float(record["score"])) for record in result]

    def community_neighbors(
        self, community_id: UUID, m: int, *, instrument_roles: Sequence[InstrumentRole] | None = None
    ) -> list[tuple[Node, float]]:
        """
        Return the m communities joined to ``community_id`` by the heaviest COMMUNITY_CO_CITATION arches.

        Used for Algorithm 2 step 1(ii), neighbour expansion.

        Args:
            community_id: The seed community to expand from.
            m: Number of neighbour communities to return.
            instrument_roles: If given, only neighbours with at least one matching
                ``member_instrument_role`` are returned; Phase 4's relation-category filter.

        """
        filters: list[LiteralString] = ["n.corpus_id = $corpus_id"]
        if instrument_roles is not None:
            filters.append("any(role IN n.member_instrument_roles WHERE role IN $instrument_roles)")
        filters.append("(n.instrument_status = 'active' OR n.instrument_status IS NULL)")
        query = cast(
            "LiteralString",
            f"MATCH (c:{NODE_LABEL} {{id: $community_id, corpus_id: $corpus_id}})-[r:COMMUNITY_CO_CITATION]-(n:{NODE_LABEL})\n"
            "WHERE " + " AND ".join(filters) + "\n"
            "RETURN n, r.weight AS weight ORDER BY r.weight DESC LIMIT $m",
        )
        with self.driver.session(database=self._settings.neo4j.database) as session:
            result = session.run(
                query,
                community_id=str(community_id),
                m=m,
                instrument_roles=[role.value for role in instrument_roles] if instrument_roles is not None else None,
                corpus_id=self.corpus_id,
            )
            return [(self._to_node(record["n"]), float(record["weight"])) for record in result]

    def ancestors_at_layer(self, element_ids: Sequence[UUID], layer: int) -> dict[UUID, Node]:
        """
        Follow MEMBER arches upward from each layer-0 element to its ancestor community at ``layer``.

        Used for Algorithm 2 step 1(iii), the bottom-up vote. Ids without an ancestor at
        that layer (e.g. the community graph has fewer than ``layer`` layers) are simply absent from
        the result rather than raising.

        Args:
            element_ids: Layer-0 element ids to find ancestors for.
            layer: The community layer to stop at (>= 1).

        Returns:
            A mapping from element id to its ancestor :class:`Node` at ``layer``.

        """
        if layer < 1:
            message = "layer must be at least 1"
            raise ValueError(message)
        if not element_ids:
            return {}

        hops = f"*{layer}..{layer}"
        query = cast(
            "LiteralString",
            f"UNWIND $element_ids AS element_id\n"
            f"MATCH (e:{NODE_LABEL} {{id: element_id, corpus_id: $corpus_id}})<-[:MEMBER{hops}]-(a:{NODE_LABEL} {{layer: $layer, corpus_id: $corpus_id}})\n"
            "RETURN element_id, head(collect(DISTINCT a)) AS a",
        )
        with self.driver.session(database=self._settings.neo4j.database) as session:
            result = session.run(
                query,
                element_ids=[str(element_id) for element_id in element_ids],
                layer=layer,
                corpus_id=self.corpus_id,
            )
            return {UUID(record["element_id"]): self._to_node(record["a"]) for record in result}

    def elements_under_communities(
        self, community_ids: Sequence[UUID], *, instrument_roles: Sequence[InstrumentRole] | None = None
    ) -> list[Node]:
        """
        Return the union of layer-0 elements reached from any of ``community_ids`` by MEMBER arches.

        Used for Algorithm 2 step 3, expansion of the selected communities into candidate elements.

        Args:
            community_ids: The selected community ids to expand.
            instrument_roles: If given, only elements whose ``instrument_role`` is in this set are
                returned; Phase 4's relation-category filter. Regardless of this filter, elements
                with ``instrument_status = 'inactive'`` are always excluded.

        """
        filters: list[LiteralString] = ["e.corpus_id = $corpus_id"]
        if instrument_roles is not None:
            filters.append("e.instrument_role IN $instrument_roles")
        filters.append("(e.instrument_status = 'active' OR e.instrument_status IS NULL)")
        query = cast(
            "LiteralString",
            "UNWIND $community_ids AS community_id\n"
            f"MATCH (c:{NODE_LABEL} {{id: community_id, corpus_id: $corpus_id}})-[:MEMBER*]->(e:{NODE_LABEL} {{layer: 0}})\n"
            "WHERE " + " AND ".join(filters) + "\n"
            "RETURN DISTINCT e",
        )
        with self.driver.session(database=self._settings.neo4j.database) as session:
            result = session.run(
                query,
                community_ids=[str(community_id) for community_id in community_ids],
                instrument_roles=[role.value for role in instrument_roles] if instrument_roles is not None else None,
                corpus_id=self.corpus_id,
            )
            return [self._to_node(record["e"]) for record in result]

    def citing_type_counts(self, element_ids: Sequence[UUID]) -> dict[UUID, dict[str, int]]:
        """
        Count incoming CITES arches per source node type, for each of ``element_ids``.

        Used for Algorithm 2 step 4 to compute the citation weight ``cw(x)``; the mapping
        from node type to citation-level weight lambda is the Retriever's responsibility, not the
        driver's. Ids with no incoming CITES arch are simply absent from the result.
        """
        query = cast(
            "LiteralString",
            "UNWIND $element_ids AS element_id\n"
            f"MATCH (u:{NODE_LABEL} {{corpus_id: $corpus_id}})-[:CITES]->(x:{NODE_LABEL} {{id: element_id, corpus_id: $corpus_id}})\n"
            "RETURN element_id, u.type AS type, count(*) AS cnt",
        )
        counts: dict[UUID, dict[str, int]] = {}
        with self.driver.session(database=self._settings.neo4j.database) as session:
            result = session.run(
                query, element_ids=[str(element_id) for element_id in element_ids], corpus_id=self.corpus_id
            )
            for record in result:
                element_id = UUID(record["element_id"])
                counts.setdefault(element_id, {})[record["type"]] = record["cnt"]
        return counts

    def citing_documents(self, element_ids: Sequence[UUID]) -> list[Node]:
        """
        Return the distinct guidelines and rulings with a CITES arch into any of ``element_ids``.

        Used for Algorithm 2 step 6, which collects the citing documents of the
        whole cited closure at once; filtering out rulings, if requested, is the Retriever's
        responsibility.
        """
        query = cast(
            "LiteralString",
            "UNWIND $element_ids AS element_id\n"
            f"MATCH (u:{NODE_LABEL} {{corpus_id: $corpus_id}})-[:CITES]->(x:{NODE_LABEL} {{id: element_id, corpus_id: $corpus_id}})\n"
            "WHERE u.type IN ['Guideline', 'Ruling']\n"
            "RETURN DISTINCT u",
        )
        with self.driver.session(database=self._settings.neo4j.database) as session:
            result = session.run(
                query, element_ids=[str(element_id) for element_id in element_ids], corpus_id=self.corpus_id
            )
            return [self._to_node(record["u"]) for record in result]

    def cited_provisions(self, document_ids: Sequence[UUID]) -> list[tuple[UUID, Node]]:
        """
        Return the provisions each of ``document_ids`` cites, over its outgoing CITES arches.

        The reverse of `citing_documents`: used by the search tool to surface, as leads,
        the Articles/Recitals/Annexes a retrieved guideline or ruling relies on.

        Args:
            document_ids: Ids of the guidelines/rulings whose citations to read.

        Returns:
            One ``(document id, cited provision)`` pair per distinct CITES arch into an
            Article, Recital, or Annex of this corpus.

        """
        query = cast(
            "LiteralString",
            "UNWIND $document_ids AS document_id\n"
            f"MATCH (d:{NODE_LABEL} {{id: document_id, corpus_id: $corpus_id}})-[:CITES]->(x:{NODE_LABEL} {{corpus_id: $corpus_id}})\n"
            "WHERE x.type IN ['Article', 'Recital', 'Annex']\n"
            "RETURN DISTINCT d.id AS document_id, x",
        )
        with self.driver.session(database=self._settings.neo4j.database) as session:
            result = session.run(
                query, document_ids=[str(document_id) for document_id in document_ids], corpus_id=self.corpus_id
            )
            return [(UUID(record["document_id"]), self._to_node(record["x"])) for record in result]

    def citation_neighbourhood(self, node_id: UUID) -> tuple[list[Node], list[Node]]:
        """
        Return the nodes ``node_id`` cites and the nodes citing it, over CITES arches in this corpus.

        A read primitive for browsing the citation graph. Neighbours come back without their
        text and embedding (both empty), since a browser lists hundreds of them by name.

        Args:
            node_id: Id of the node whose neighbourhood to read.

        Returns:
            ``(cited, citing)``: the distinct targets of the node's outgoing CITES arches and
            the distinct sources of its incoming ones; both empty when the node is unknown.

        """
        query = cast(
            "LiteralString",
            f"MATCH (n:{NODE_LABEL} {{id: $node_id, corpus_id: $corpus_id}})-[:CITES]->(x:{NODE_LABEL} {{corpus_id: $corpus_id}})\n"
            f"RETURN DISTINCT 'cited' AS direction, x {{{_LISTING_PROJECTION}}} AS x\n"
            "UNION ALL\n"
            f"MATCH (x:{NODE_LABEL} {{corpus_id: $corpus_id}})-[:CITES]->(n:{NODE_LABEL} {{id: $node_id, corpus_id: $corpus_id}})\n"
            f"RETURN DISTINCT 'citing' AS direction, x {{{_LISTING_PROJECTION}}} AS x",
        )
        cited: list[Node] = []
        citing: list[Node] = []
        with self.driver.session(database=self._settings.neo4j.database) as session:
            for record in session.run(query, node_id=str(node_id), corpus_id=self.corpus_id):
                (cited if record["direction"] == "cited" else citing).append(self._to_node(record["x"]))
        return cited, citing

    def co_citation_neighbourhood(self, node_id: UUID) -> list[tuple[Node, float]]:
        """
        Return the nodes joined to ``node_id`` by a CO_CITATION arch in this corpus, with its weight.

        A read primitive for browsing the co-citation graph. The match is undirected, since
        :meth:`_add_edge` stores an undirected arch once, smaller ``id`` first. Semantic backbone
        arches are not co-citations and are left out. Neighbours come back without their text
        and embedding (both empty).

        Args:
            node_id: Id of the node whose co-citation neighbourhood to read.

        Returns:
            ``(neighbour, weight)`` pairs, heaviest first; empty when the node is unknown or has
            no co-citation arch.

        """
        query = cast(
            "LiteralString",
            f"MATCH (n:{NODE_LABEL} {{id: $node_id, corpus_id: $corpus_id}})-[r:CO_CITATION]-(x:{NODE_LABEL} {{corpus_id: $corpus_id}})\n"
            "WHERE r.undirected = true\n"
            f"RETURN x {{{_LISTING_PROJECTION}}} AS x, r.weight AS weight\n"
            "ORDER BY weight DESC",
        )
        with self.driver.session(database=self._settings.neo4j.database) as session:
            result = session.run(query, node_id=str(node_id), corpus_id=self.corpus_id)
            return [(self._to_node(record["x"]), float(record["weight"])) for record in result]

    def find_nodes(self, term: str, *, types: Sequence[str], limit: int) -> list[Node]:
        """
        Return the nodes of ``types`` whose name contains ``term``, case-insensitively.

        A read primitive for browsing the citation graph. Matches come back without their text
        and embedding, shortest name first, so ``Article 2`` precedes ``Article 21``.

        Args:
            term: Substring to look for in node names; an empty term matches every node.
            types: Node types to search, e.g. ``("Article", "Guideline")``.
            limit: Maximum number of nodes to return.

        Returns:
            At most ``limit`` matching nodes, ordered by name length, then name.

        """
        query = cast(
            "LiteralString",
            f"MATCH (n:{NODE_LABEL} {{corpus_id: $corpus_id}})\n"
            "WHERE n.type IN $types AND toLower(n.name) CONTAINS toLower($term)\n"
            f"RETURN n {{{_LISTING_PROJECTION}}} AS n\n"
            "ORDER BY size(n.name), n.name\n"
            "LIMIT $limit",
        )
        with self.driver.session(database=self._settings.neo4j.database) as session:
            result = session.run(query, term=term, types=list(types), limit=limit, corpus_id=self.corpus_id)
            return [self._to_node(record["n"]) for record in result]

    def community_ancestry(self, node_id: UUID) -> list[tuple[Node, list[Node]]]:
        """
        Return every community above ``node_id``, finest first, with the provisions it groups.

        A read primitive for browsing the citation graph: follows MEMBER arches upward from a
        layer-0 element to each of its ancestor communities, and down again from each one to its
        layer-0 members. Communities come with their summary; members without text or embedding.

        Args:
            node_id: Id of the layer-0 element whose communities to read.

        Returns:
            ``(community, members)`` pairs ordered by layer, members including ``node_id`` itself;
            empty when the node is unknown or belongs to no community.

        """
        query = cast(
            "LiteralString",
            f"MATCH (n:{NODE_LABEL} {{id: $node_id, corpus_id: $corpus_id}})<-[:MEMBER*]-(c:{NODE_LABEL} {{corpus_id: $corpus_id}})\n"
            "WHERE c.layer >= 1\n"
            f"MATCH (c)-[:MEMBER*]->(e:{NODE_LABEL} {{layer: 0, corpus_id: $corpus_id}})\n"
            f"WITH DISTINCT c, e\n"
            f"RETURN c {{{_LISTING_PROJECTION}, .summary}} AS c, collect(e {{{_LISTING_PROJECTION}}}) AS members\n"
            "ORDER BY c.layer, c.name",
        )
        with self.driver.session(database=self._settings.neo4j.database) as session:
            result = session.run(query, node_id=str(node_id), corpus_id=self.corpus_id)
            return [
                (self._to_node(record["c"]), [self._to_node(member) for member in record["members"]])
                for record in result
            ]

    def operative_targets(self, source_ids: Sequence[UUID]) -> list[Node]:
        """
        Return the distinct provisions affected by an OPERATIVE_RELATION from any of ``source_ids``.

        A low-level read primitive: the amending/implementing provision -> affected provision
        direction. Phase 4's expansion algorithm decides how and when to call this.
        """
        query = cast(
            "LiteralString",
            "UNWIND $source_ids AS source_id\n"
            f"MATCH (s:{NODE_LABEL} {{id: source_id, corpus_id: $corpus_id}})-[:OPERATIVE_RELATION]->(t:{NODE_LABEL})\n"
            "WHERE t.corpus_id = $corpus_id\n"
            "RETURN DISTINCT t",
        )
        with self.driver.session(database=self._settings.neo4j.database) as session:
            result = session.run(
                query, source_ids=[str(source_id) for source_id in source_ids], corpus_id=self.corpus_id
            )
            return [self._to_node(record["t"]) for record in result]

    def operative_sources(
        self, target_ids: Sequence[UUID]
    ) -> list[tuple[Node, InstrumentRelationType, OperativeOperation]]:
        """
        Return the provisions with an OPERATIVE_RELATION into any of ``target_ids``, with the edge's legal semantics.

        A low-level read primitive: the affected provision -> amending/implementing provision
        direction, i.e. the base-to-newer expansion Phase 4 builds retrieval on top of. Unlike most
        other read primitives this is not deduplicated by node: a provision reachable via more than
        one OPERATIVE_RELATION edge (e.g. two different amending instruments) is returned once per
        edge, so callers see every distinct ``(relation_type, operation)`` pair.
        """
        query = cast(
            "LiteralString",
            "UNWIND $target_ids AS target_id\n"
            f"MATCH (s:{NODE_LABEL})-[r:OPERATIVE_RELATION]->(t:{NODE_LABEL} {{id: target_id, corpus_id: $corpus_id}})\n"
            "WHERE s.corpus_id = $corpus_id\n"
            "RETURN s, r.relation_type AS relation_type, r.operation AS operation",
        )
        with self.driver.session(database=self._settings.neo4j.database) as session:
            result = session.run(
                query, target_ids=[str(target_id) for target_id in target_ids], corpus_id=self.corpus_id
            )
            return [
                (
                    self._to_node(record["s"]),
                    InstrumentRelationType(record["relation_type"]),
                    OperativeOperation(record["operation"]),
                )
                for record in result
            ]

    def texts_at_layer(
        self,
        layer: int,
        node_types: Sequence[str] | None = None,
        *,
        instrument_roles: Sequence[InstrumentRole] | None = None,
    ) -> list[tuple[UUID, str]]:
        """
        Return ``(id, text)`` pairs for every node at ``layer``, optionally restricted by type.

        A lightweight projection (no embeddings loaded) used by Algorithm 2 to build the
        BM25 reference sets L_{h0} and L_0 exactly, rather than approximating them from a vector
        shortlist.

        Args:
            layer: The community graph layer to read from (0 for elements, h >= 1 for communities).
            node_types: If given, only nodes whose ``type`` is in this set are returned.
            instrument_roles: If given, only nodes whose own ``instrument_role`` (elements) or any
                of whose ``member_instrument_roles`` (communities) is in this set are returned;
                Phase 4's relation-category filter. Regardless of this filter, nodes with
                ``instrument_status = 'inactive'`` are always excluded.

        """
        clauses: list[LiteralString] = [f"MATCH (n:{NODE_LABEL}) WHERE n.corpus_id = $corpus_id AND n.layer = $layer"]
        if node_types is not None:
            clauses.append("AND n.type IN $node_types")
        if instrument_roles is not None:
            clauses.append(
                "AND (n.instrument_role IN $instrument_roles OR "
                "any(role IN n.member_instrument_roles WHERE role IN $instrument_roles))"
            )
        clauses.append("AND (n.instrument_status = 'active' OR n.instrument_status IS NULL)")
        clauses.append("RETURN n.id AS id, n.text AS text")
        query = cast("LiteralString", "\n".join(clauses))
        with self.driver.session(database=self._settings.neo4j.database) as session:
            result = session.run(
                query,
                layer=layer,
                node_types=list(node_types) if node_types is not None else None,
                instrument_roles=[role.value for role in instrument_roles] if instrument_roles is not None else None,
                corpus_id=self.corpus_id,
            )
            return [(UUID(record["id"]), record["text"]) for record in result]

    def cited_element_closure(self, element_ids: Sequence[UUID], max_hops: int = 3) -> list[tuple[Node, int]]:
        """
        Return the elements reachable from ``element_ids`` by following CITES arches between elements.

        Implements ref*(x) of Algorithm 2 step 5: a variable-length traversal in which every
        node on the path -- not just the endpoint -- is an element (Article/Recital/Annex), so a path
        may not tunnel through a guideline or ruling. ``max_hops`` bounds the traversal (Cypher's
        variable-length range must be a literal, not a query parameter, hence the interpolation
        below); Cypher's per-path relationship uniqueness already guarantees termination on cycles.
        The seed elements themselves are not returned (they are R, added by the caller). Each
        returned node is paired with the minimum number of hops it took to reach it, for Algorithm
        2's hop-decayed xsim.

        Args:
            element_ids: Seed element ids to traverse the citation graph from.
            max_hops: Maximum number of CITES hops to follow.

        """
        if max_hops < 1:
            message = "max_hops must be at least 1"
            raise ValueError(message)
        element_types = list(ELEMENT_TYPES)
        query = cast(
            "LiteralString",
            "UNWIND $element_ids AS element_id\n"
            f"MATCH path = (x:{NODE_LABEL} {{id: element_id, corpus_id: $corpus_id}})-[:CITES*1..{max_hops}]->(y:{NODE_LABEL})\n"
            "WHERE all(node IN nodes(path) WHERE node.corpus_id = $corpus_id AND node.type IN $element_types)\n"
            "RETURN y, min(length(path)) AS depth",
        )
        with self.driver.session(database=self._settings.neo4j.database) as session:
            result = session.run(
                query,
                element_ids=[str(element_id) for element_id in element_ids],
                element_types=element_types,
                corpus_id=self.corpus_id,
            )
            return [(self._to_node(record["y"]), int(record["depth"])) for record in result]

    def community_sizes(self, community_ids: Sequence[UUID]) -> dict[UUID, int]:
        """Number of members each community has at the layer below. Keyed by community id."""
        if not community_ids:
            return {}
        query = """
        UNWIND $ids AS cid
        MATCH (c:Community {id: cid, corpus_id: $corpus_id})-[:MEMBER]->(m {corpus_id: $corpus_id})
        RETURN cid AS id, count(m) AS size
        """
        with self.driver.session(database=self._settings.neo4j.database) as session:
            return {
                UUID(record["id"]): record["size"]
                for record in session.run(query, ids=[str(i) for i in community_ids], corpus_id=self.corpus_id)
            }
