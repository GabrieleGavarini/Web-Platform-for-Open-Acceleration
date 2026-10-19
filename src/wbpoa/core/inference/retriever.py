"""Algorithm 2: hybrid retrieval of relevant elements and their citation neighbourhood."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import ClassVar, Literal, NamedTuple
from uuid import UUID

from structlog import get_logger

from wbpoa.core.corpus.corpus_config import InstrumentRelationType, InstrumentRole
from wbpoa.core.driver.neo4j_driver import Neo4jDriver
from wbpoa.core.inference.lexical import Bm25Index
from wbpoa.core.inference.models import ExpansionNode, RelevantElement, RetrievalResult
from wbpoa.core.inference.rerank import NullReranker, Reranker, rerank_blend
from wbpoa.core.inference.similarity import cosine_similarity, hybrid_similarity, min_max_normalize
from wbpoa.core.ingestion.utils import Embedder
from wbpoa.core.model.element import Annex, Article, Guideline, Recital, Ruling
from wbpoa.core.model.graph import Edge, Node
from wbpoa.core.model.operative_relation import OperativeOperation
from wbpoa.core.settings import CoreSettings, get_core_settings

log = get_logger()


class _ElementScore(NamedTuple):
    """The components behind Algorithm 2 step 4's score(x), kept together so RelevantElement can report them."""

    node: Node
    score: float
    similarity: float
    semantic: float
    lexical: float
    citation_weight: float
    authority: float
    rerank_raw: float | None = None
    rerank: float | None = None
    rho: float = 0.0


class RetrieverError(RuntimeError):
    """Raised when a query cannot be answered as requested."""


class Retriever:
    """
    Retrieve the elements and their citation neighbourhood relevant to a natural-language query.

    Implements Algorithm 2 in three parts: selecting the communities of the
    (already built) community graph relevant to the query (steps 1-2), selecting the elements within
    them through the citation graph (steps 3-4), and expanding the relevant elements into their
    citation neighbourhood (steps 5-8). Every similarity is *hybrid*: a semantic (cosine) channel
    mixed with a lexical (BM25) channel by ``gamma``, the lexical score min-max normalised within the
    reference set of the stage that uses it. The driver only returns graph structure and lightweight
    summary projections; all algorithm math lives here so it can be unit-tested without a live
    database.
    """

    LAMBDA_BY_TYPE: ClassVar[dict[str, float]] = {
        cls.__name__: cls.citation_weight for cls in (Article, Recital, Annex, Guideline, Ruling)
    }
    # Guideline/Ruling nodes also default to layer=0, but only these types belong to the community graph.
    ELEMENT_TYPES: ClassVar[tuple[str, ...]] = (Article.__name__, Recital.__name__, Annex.__name__)

    def __init__(
        self,
        driver: Neo4jDriver[Edge],
        embedder: Embedder,
        reranker: Reranker | None = None,
        settings: CoreSettings | None = None,
    ) -> None:
        """
        Args:
            driver: Read-only Neo4j driver used for all graph and vector-index queries.
            embedder: Embedder used to compute the query embedding.
            reranker: Cross-encoder used by the cascade stages 4(ii)/7; defaults to a no-op
                :class:`~wbpoa.core.inference.rerank.NullReranker`, so reranking is a strict opt-in even
                when ``rerank_enabled`` is set.
            settings: Overrides for :class:`~wbpoa.core.settings.CoreSettings`; defaults to the process-wide settings.

        """
        self.driver = driver
        self.embedder = embedder
        self.reranker = reranker if reranker is not None else NullReranker()
        self._settings = settings if settings is not None else get_core_settings()

    def retrieve(  # noqa: PLR0913
        self,
        query: str,
        *,
        h0: int | None = None,
        k: int | None = None,
        m: int | None = None,
        n: int | None = None,
        vote_top_j: int | None = None,
        r: int | None = None,
        ell_elements: int | None = None,
        ell_docs: int | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
        delta: float | None = None,
        max_hops: int | None = None,
        include_rulings: bool | None = None,
        rerank_enabled: bool | None = None,
        rerank_pool: int | None = None,
        rho: float | None = None,
        rerank_expansion: bool | None = None,
        relation_category: Sequence[InstrumentRole] | None = None,
        operative_expansion_enabled: bool | None = None,
        operative_decay: float | None = None,
    ) -> RetrievalResult:
        """
        Run Algorithm 2 end to end for ``query``.

        Any parameter left as ``None`` falls back to :class:`~wbpoa.core.settings.InferenceSettings`.

        Args:
            query: The natural-language query.
            h0: Community graph layer to select candidate communities from.
            k: Number of seed communities, and number of finally selected communities.
            m: Number of neighbour communities expanded from each seed.
            n: Number of layer-0 elements used for the bottom-up vote.
            vote_top_j: Number of top per-community similarities averaged into the bottom-up vote.
            r: Number of top-ranked elements returned as relevant documents.
            ell_elements: Number of element expansion nodes returned from the citation neighbourhood.
            ell_docs: Number of document expansion nodes returned from the citation neighbourhood.
            alpha: Mixing weight between element similarity and citation authority.
            beta: Mixing weight between community similarity and bottom-up vote.
            gamma: Mixing weight between the semantic and lexical similarity channels.
            delta: Per-hop decay applied to expansion similarity; 1.0 disables decay.
            max_hops: Maximum citation hops followed by the cited element closure.
            include_rulings: Whether rulings, not just guidelines, are admitted into the expansion set.
            rerank_enabled: Master switch for the cascade stages 4(ii) and 7.
            rerank_pool: Shortlist size p handed to the reranker, per pool.
            rho: Rerank blend weight, rho*rsim + (1-rho)*first-pass score.
            rerank_expansion: Whether step 7 reranks its two pools as well as step 4.
            relation_category: If given, restricts every stage to nodes whose instrument role (or,
                for communities, member instrument roles) is in this set; None searches every
                active instrument. Regardless of this filter, inactive instruments are never searched.
            operative_expansion_enabled: Whether relevant base-instrument elements are expanded
                toward newer amending/implementing provisions via OPERATIVE_RELATION edges,
                alongside citation expansion.
            operative_decay: Flat decay applied to xsim for operative-expansion results.

        Returns:
            A :class:`RetrievalResult` with up to ``r`` ranked relevant elements and up to
            ``ell_elements`` + ``ell_docs`` ranked expansion nodes.

        Raises:
            RetrieverError: If no communities exist at layer ``h0``.

        """
        inference = self._settings.inference
        h0 = h0 if h0 is not None else inference.h0
        k = k if k is not None else inference.k
        m = m if m is not None else inference.m
        n = n if n is not None else inference.n
        vote_top_j = vote_top_j if vote_top_j is not None else inference.vote_top_j
        r = r if r is not None else inference.r
        ell_elements = ell_elements if ell_elements is not None else inference.ell_elements
        ell_docs = ell_docs if ell_docs is not None else inference.ell_docs
        alpha = alpha if alpha is not None else inference.alpha
        beta = beta if beta is not None else inference.beta
        gamma = gamma if gamma is not None else inference.gamma
        delta = delta if delta is not None else inference.delta
        max_hops = max_hops if max_hops is not None else inference.max_hops
        include_rulings = include_rulings if include_rulings is not None else inference.include_rulings
        rerank_enabled = rerank_enabled if rerank_enabled is not None else inference.rerank_enabled
        rerank_pool = rerank_pool if rerank_pool is not None else inference.rerank_pool
        rho = rho if rho is not None else inference.rho
        rerank_expansion = rerank_expansion if rerank_expansion is not None else inference.rerank_expansion
        relation_category = relation_category if relation_category is not None else inference.relation_category
        operative_expansion_enabled = (
            operative_expansion_enabled
            if operative_expansion_enabled is not None
            else inference.operative_expansion_enabled
        )
        operative_decay = operative_decay if operative_decay is not None else inference.operative_decay
        pool_multiplier = inference.candidate_pool_multiplier

        embedding = self.embedder.create_embedding(query)
        log.info("Step 1: computed query embedding", query=query)

        log.info("Step 2: selecting candidate communities", h0=h0, k=k, m=m, n=n, beta=beta, gamma=gamma)
        communities = self._select_communities(
            embedding,
            query,
            h0=h0,
            k=k,
            m=m,
            n=n,
            vote_top_j=vote_top_j,
            beta=beta,
            gamma=gamma,
            pool_multiplier=pool_multiplier,
            relation_category=relation_category,
        )

        elements = self.driver.elements_under_communities(
            [community.id for community in communities], instrument_roles=relation_category
        )
        log.info(
            "Step 3: expanded selected communities into candidate elements",
            communities=[community.name for community in communities],
            num_elements=len(elements),
            elements=[element.name for element in elements],
        )

        relevant = self._rank_elements(
            embedding,
            query,
            elements,
            alpha=alpha,
            gamma=gamma,
            r=r,
            rerank_enabled=rerank_enabled,
            rerank_pool=rerank_pool,
            rho=rho,
        )
        relevant_elements = [
            RelevantElement(
                id=item.node.id,
                name=item.node.name,
                type=item.node.type,
                text=item.node.text,
                summary=item.node.summary,
                score=item.score,
                alpha=alpha,
                similarity=item.similarity,
                semantic=item.semantic,
                lexical=item.lexical,
                gamma=gamma,
                citation_weight=item.citation_weight,
                authority=item.authority,
                rerank_raw=item.rerank_raw,
                rerank=item.rerank,
                rho=item.rho,
                qualified_citation=item.node.qualified_citation,
            )
            for item in relevant
        ]

        expansion_nodes = self._expand_nodes(
            embedding,
            query,
            relevant,
            ell_elements=ell_elements,
            ell_docs=ell_docs,
            gamma=gamma,
            delta=delta,
            max_hops=max_hops,
            include_rulings=include_rulings,
            rerank_enabled=rerank_enabled,
            rerank_pool=rerank_pool,
            rho=rho,
            rerank_expansion=rerank_expansion,
            relation_category=relation_category,
            operative_expansion_enabled=operative_expansion_enabled,
            operative_decay=operative_decay,
        )
        log.info(
            "Step 8: returning relevant elements and expansion nodes",
            num_relevant=len(relevant_elements),
            num_expansion=len(expansion_nodes),
        )
        return RetrievalResult(relevant_elements=relevant_elements, expansion_nodes=expansion_nodes)

    def _select_communities(  # noqa: PLR0913
        self,
        embedding: Sequence[float],
        query: str,
        *,
        h0: int,
        k: int,
        m: int,
        n: int,
        vote_top_j: int,
        beta: float,
        gamma: float,
        pool_multiplier: int,
        relation_category: Sequence[InstrumentRole] | None,
    ) -> list[Node]:
        """
        Algorithm 2 steps 1-2: seed, expand, vote, then score and keep the top-k communities.

        Args:
            embedding: The query embedding, used for the semantic channel.
            query: The natural-language query, used for the lexical (BM25) channel.
            h0: Community graph layer to select candidate communities from.
            k: Number of seed communities, and number of finally selected communities.
            m: Number of neighbour communities expanded from each seed.
            n: Number of layer-0 elements used for the bottom-up vote.
            vote_top_j: Number of top per-community similarities averaged into the bottom-up vote.
            beta: Mixing weight between community similarity and bottom-up vote.
            gamma: Mixing weight between the semantic and lexical similarity channels.
            pool_multiplier: Over-fetch factor applied to k/n before filtering by layer/type.
            relation_category: If given, restricts seeds, neighbours and votes to matching
                instrument roles; Phase 4's relation-category filter.

        Returns:
            The top-k selected community nodes, ranked best first.

        Raises:
            RetrieverError: If no communities exist at layer ``h0`` and no ``relation_category``
                filter was applied. When a filter is given and it matches nothing, that is a valid
                (if uninteresting) outcome rather than a graph-build problem, so an empty list of
                communities is returned instead.

        """
        layer_texts = self.driver.texts_at_layer(h0, instrument_roles=relation_category)
        if not layer_texts:
            if relation_category is not None:
                log.info(
                    "No communities matched the requested relation_category",
                    h0=h0,
                    relation_category=[role.value for role in relation_category],
                )
                return []
            raise RetrieverError(
                f"No communities found at layer {h0}; build the community graph first or choose a smaller h0."
            )
        lex_layer = min_max_normalize(Bm25Index.from_texts(layer_texts).scores(query))

        seeds = self._seed_communities(
            embedding,
            lex_layer,
            h0=h0,
            k=k,
            gamma=gamma,
            pool_multiplier=pool_multiplier,
            relation_category=relation_category,
        )
        log.info("Step 1(i): seed communities", h0=h0, k=k, seeds=[seed.name for seed in seeds])

        candidates: dict[UUID, Node] = {seed.id: seed for seed in seeds}
        neighbours: dict[UUID, Node] = {}
        for seed in seeds:
            for neighbour, _weight in self.driver.community_neighbors(seed.id, m, instrument_roles=relation_category):
                candidates.setdefault(neighbour.id, neighbour)
                neighbours.setdefault(neighbour.id, neighbour)
        log.info(
            "Step 1(ii): neighbour expansion",
            m=m,
            added=[neighbour.name for neighbour in neighbours.values()],
            candidates=[node.name for node in candidates.values()],
        )

        votes = self._bottom_up_vote(
            embedding,
            query,
            candidates,
            h0=h0,
            n=n,
            vote_top_j=vote_top_j,
            gamma=gamma,
            pool_multiplier=pool_multiplier,
            relation_category=relation_category,
        )
        log.info(
            "Step 1(iii): bottom-up vote",
            n=n,
            candidates=[node.name for node in candidates.values()],
            votes={candidates[community_id].name: round(vote, 3) for community_id, vote in votes.items()},
        )

        csim = {
            node.id: hybrid_similarity(
                cosine_similarity(embedding, node.summary_embedding), lex_layer.get(node.id, 0.0), gamma
            )
            for node in candidates.values()
        }
        scored = self._score_communities(list(candidates.values()), csim, votes, beta=beta)
        scored.sort(key=lambda item: item[1], reverse=True)
        log.info(
            "Step 1(iv): scored candidate communities",
            candidates=[{"name": node.name, "score": round(score, 3)} for node, score in scored],
        )
        selected = scored[:k]
        log.info(
            "Step 1(iv): selected communities",
            num_candidates=len(scored),
            selected=[{"name": node.name, "score": round(score, 3)} for node, score in selected],
        )
        return [node for node, _ in selected]

    def _seed_communities(  # noqa: PLR0913
        self,
        embedding: Sequence[float],
        lex_layer: dict[UUID, float],
        *,
        h0: int,
        k: int,
        gamma: float,
        pool_multiplier: int,
        relation_category: Sequence[InstrumentRole] | None,
    ) -> list[Node]:
        """
        Algorithm 2 step 1(i): rank L_h0 by community similarity over a semantic+lexical candidate pool, keep top-k.

        Args:
            embedding: The query embedding, used for the semantic channel.
            lex_layer: The min-max normalised BM25 scores of every node at layer ``h0``.
            h0: Community graph layer to select candidates from.
            k: Number of seed communities to keep.
            gamma: Mixing weight between the semantic and lexical similarity channels.
            pool_multiplier: Over-fetch factor applied to k before filtering by layer.
            relation_category: If given, restricts candidates to matching instrument roles;
                Phase 4's relation-category filter.

        Returns:
            The top-k seed community nodes, ranked best first.

        """
        pool = min(k * pool_multiplier, len(lex_layer))
        semantic = self.driver.vector_search(
            embedding, pool, layer=h0, candidate_pool=pool, instrument_roles=relation_category
        )
        candidate_nodes: dict[UUID, Node] = {node.id: node for node, _ in semantic}
        lexical_ids = [node_id for node_id, _ in _top_by_value(lex_layer, pool)]
        missing = [node_id for node_id in lexical_ids if node_id not in candidate_nodes]
        for node in self.driver.nodes_by_ids(missing):
            candidate_nodes[node.id] = node
        scored = [
            (
                node,
                hybrid_similarity(
                    cosine_similarity(embedding, node.summary_embedding), lex_layer.get(node.id, 0.0), gamma
                ),
            )
            for node in candidate_nodes.values()
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [node for node, _ in scored[:k]]

    def _bottom_up_vote(  # noqa: PLR0913
        self,
        embedding: Sequence[float],
        query: str,
        candidates: dict[UUID, Node],
        *,
        h0: int,
        n: int,
        vote_top_j: int,
        gamma: float,
        pool_multiplier: int,
        relation_category: Sequence[InstrumentRole] | None,
    ) -> dict[UUID, float]:
        """
        Algorithm 2 step 1(iii): rank L_0 by hybrid similarity, then average each community's top-j
        member similarities over min(j, |C|) -- a community with fewer than j members is not penalised
        for votes it could not have cast, while a large community that landed only one element in T_n
        still divides by j. j=1 reduces the vote to a plain max.

        Args:
            embedding: The query embedding, used for the semantic channel.
            query: The natural-language query, used for the lexical (BM25) channel.
            candidates: The candidate communities gathered so far; mutated in place to add any new
                ancestor community discovered while accumulating votes.
            h0: Community graph layer to accumulate votes at.
            n: Number of layer-0 elements used for the bottom-up vote.
            vote_top_j: Number of top per-community similarities averaged into the vote.
            gamma: Mixing weight between the semantic and lexical similarity channels.
            pool_multiplier: Over-fetch factor applied to n before filtering by layer/type.
            relation_category: If given, restricts voting elements to matching instrument roles;
                Phase 4's relation-category filter.

        Returns:
            A mapping from community id to its accumulated vote.

        """
        element_texts = self.driver.texts_at_layer(0, self.ELEMENT_TYPES, instrument_roles=relation_category)
        lex_elements = min_max_normalize(Bm25Index.from_texts(element_texts).scores(query))

        pool = min(n * pool_multiplier, len(lex_elements))
        semantic = self.driver.vector_search(
            embedding,
            pool,
            layer=0,
            node_types=self.ELEMENT_TYPES,
            candidate_pool=pool,
            instrument_roles=relation_category,
        )
        element_nodes: dict[UUID, Node] = {node.id: node for node, _ in semantic}
        lexical_ids = [node_id for node_id, _ in _top_by_value(lex_elements, pool)]
        missing = [node_id for node_id in lexical_ids if node_id not in element_nodes]
        for node in self.driver.nodes_by_ids(missing):
            element_nodes[node.id] = node

        scored = [
            (
                node,
                hybrid_similarity(
                    cosine_similarity(embedding, node.summary_embedding), lex_elements.get(node.id, 0.0), gamma
                ),
            )
            for node in element_nodes.values()
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        top_n = scored[:n]
        log.info("Step 1(iii): bottom-up vote candidates", n=n, voted_elements=[node.name for node, _ in top_n])

        similarities_by_community: dict[UUID, list[float]] = {}
        if top_n:
            ancestors = self.driver.ancestors_at_layer([node.id for node, _ in top_n], h0)
            for node, similarity in top_n:
                ancestor = ancestors.get(node.id)
                if ancestor is None:
                    log.warning("retriever_no_ancestor_at_layer", element_id=node.id, layer=h0)
                    continue
                candidates.setdefault(ancestor.id, ancestor)
                similarities_by_community.setdefault(ancestor.id, []).append(similarity)

        # Zero-padding to vote_top_j never changes the sum, so summing the top-j values and
        # dividing by vote_top_j (not by the member count) is the padded mean directly.
        sizes = self.driver.community_sizes(list(similarities_by_community.keys()))
        votes: dict[UUID, float] = {}
        for community_id, sims in similarities_by_community.items():
            size = sizes.get(community_id, 0)
            if size <= 0:
                log.warning("retriever_missing_community_size", community_id=community_id, fallback=vote_top_j)
                size = vote_top_j  # degrade to the old zero-padded behaviour rather than inflating the vote
            divisor = min(vote_top_j, size)
            votes[community_id] = sum(sorted(sims, reverse=True)[:vote_top_j]) / divisor
        return votes

    @staticmethod
    def _score_communities(
        candidates: list[Node],
        csim: dict[UUID, float],
        votes: dict[UUID, float],
        *,
        beta: float,
    ) -> list[tuple[Node, float]]:
        """
        Algorithm 2 step 1(iv): score(C) = beta*csim(q,C) + (1-beta)*cauth(q,C).

        Args:
            candidates: The candidate community nodes to score.
            csim: The community similarity of each candidate, keyed by node id.
            votes: The bottom-up vote of each candidate, keyed by node id.
            beta: Mixing weight between community similarity and bottom-up vote.

        Returns:
            ``(node, score)`` pairs for every candidate, in the same order as ``candidates``.

        """
        max_vote = max(votes.values(), default=0.0)
        return [
            (
                node,
                beta * csim.get(node.id, 0.0) + (1 - beta) * (votes.get(node.id, 0.0) / max_vote if max_vote else 0.0),
            )
            for node in candidates
        ]

    def _rank_elements(  # noqa: PLR0913
        self,
        embedding: Sequence[float],
        query: str,
        elements: list[Node],
        *,
        alpha: float,
        gamma: float,
        r: int,
        rerank_enabled: bool,
        rerank_pool: int,
        rho: float,
    ) -> list[_ElementScore]:
        """
        Algorithm 2 step 4: score, optionally rerank the shortlist (step 4(ii)), and keep the top-r elements.

        Args:
            embedding: The query embedding, used for the semantic channel.
            query: The natural-language query, used for the lexical (BM25) channel.
            elements: The candidate elements to score.
            alpha: Mixing weight between element similarity and citation authority.
            gamma: Mixing weight between the semantic and lexical similarity channels.
            r: Number of top-ranked elements to keep.
            rerank_enabled: Whether to run the step 4(ii) cascade at all.
            rerank_pool: Shortlist size p handed to the reranker.
            rho: Rerank blend weight.

        Returns:
            The top-r scored elements, ranked best first -- by score_rr over the reranked
            shortlist when the cascade ran, by score over all of E otherwise.

        """
        type_counts = self.driver.citing_type_counts([node.id for node in elements])
        citation_weights = {node.id: self._citation_weight(type_counts.get(node.id, {})) for node in elements}
        lex_elements = min_max_normalize(Bm25Index.from_nodes(elements).scores(query))
        scored = self._score_elements(embedding, elements, citation_weights, lex_elements, alpha=alpha, gamma=gamma)
        scored.sort(key=lambda item: (item.score, str(item.node.id)), reverse=True)
        log.info(
            "Step 4(i): scored candidate elements",
            candidates=[
                {"rank": rank + 1, "name": item.node.name, "score": round(item.score, 3)}
                for rank, item in enumerate(scored)
            ],
        )

        if rerank_enabled:
            node_by_id = {item.node.id: item.node for item in scored}
            first_pass = {item.node.id: item.score for item in scored}
            rerank_raw, score_rr = self._rerank(
                query, node_by_id, first_pass, first_pass, rerank_pool=rerank_pool, rho=rho
            )
            if score_rr:
                by_id = {item.node.id: item for item in scored}
                ranked_ids = sorted(score_rr.items(), key=lambda kv: (kv[1], str(kv[0])), reverse=True)[:r]
                selected = [
                    by_id[node_id]._replace(rerank_raw=rerank_raw[node_id], rerank=blended, rho=rho)
                    for node_id, blended in ranked_ids
                ]
                log.info(
                    "Step 4(ii): reranked shortlist",
                    shortlist_size=len(score_rr),
                    selected=[
                        {"name": by_id[node_id].node.name, "score": round(blended, 3), "rank": rank + 1}
                        for rank, (node_id, blended) in enumerate(ranked_ids)
                    ],
                )
                return selected

        selected = scored[:r]
        log.info(
            "Step 4: selected relevant elements",
            num_candidates=len(scored),
            selected=[{"name": item.node.name, "score": round(item.score, 3)} for item in selected],
        )
        return selected

    def _rerank_text(self, node: Node) -> str:
        """Return the text of ``node`` to hand to the reranker, per ``InferenceSettings.rerank_field``."""
        return node.text if self._settings.inference.rerank_field == "text" else node.summary

    def _rerank(  # noqa: PLR0913
        self,
        query: str,
        candidates: dict[UUID, Node],
        shortlist_key: dict[UUID, float],
        blend_target: dict[UUID, float],
        *,
        rerank_pool: int,
        rho: float,
    ) -> tuple[dict[UUID, float], dict[UUID, float]]:
        """
        Algorithm 2 steps 4(ii)/7: shortlist by ``shortlist_key``, score with the cross-encoder,
        and blend rho*rsim_P + (1-rho)*blend_target(u).

        ``shortlist_key`` and ``blend_target`` coincide at step 4 (``score(x)`` both selects the
        shortlist and is blended). They differ at step 7, where the shortlist is drawn on the
        decayed ``xsim`` but the blend keeps ``delta**depth`` outside it (Remark 8), blending
        against the undecayed ``hs_N`` instead -- callers apply decay to the result themselves.

        Args:
            query: The natural-language query.
            candidates: Every node eligible to be reranked, keyed by id.
            shortlist_key: The value used to pick the top-``rerank_pool`` ids, keyed by id.
            blend_target: The first-pass score ``s(u)`` blended with ``rsim_P``, keyed by id.
            rerank_pool: Shortlist size p.
            rho: Blend weight in [0, 1].

        Returns:
            ``(rerank_raw, score_rr)``, both keyed by the shortlisted ids only; both empty when
            ``rho <= 0``, ``rerank_pool <= 0``, or the reranker returns nothing for the shortlist
            (e.g. a :class:`~wbpoa.core.inference.rerank.NullReranker`) -- callers fall back to the
            unreranked ranking in that case.

        """
        if rho <= 0 or rerank_pool <= 0:
            return {}, {}
        shortlist_ids = [
            node_id
            for node_id, _ in sorted(shortlist_key.items(), key=lambda kv: (kv[1], str(kv[0])), reverse=True)[
                :rerank_pool
            ]
        ]
        if not shortlist_ids:
            return {}, {}
        docs = [(node_id, self._rerank_text(candidates[node_id])) for node_id in shortlist_ids]
        rerank_raw = self.reranker.scores(query, docs)
        if not rerank_raw:
            return {}, {}
        score_rr = rerank_blend(rerank_raw, {node_id: blend_target[node_id] for node_id in rerank_raw}, rho)
        return rerank_raw, score_rr

    @staticmethod
    def _score_elements(  # noqa: PLR0913
        embedding: Sequence[float],
        elements: list[Node],
        citation_weights: dict[UUID, float],
        lex_elements: dict[UUID, float],
        *,
        alpha: float,
        gamma: float,
    ) -> list[_ElementScore]:
        """
        Algorithm 2 step 4: score(x) = alpha*esim(q,x) + (1-alpha)*eauth(x,E).

        Args:
            embedding: The query embedding, used for the semantic channel.
            elements: The candidate elements to score.
            citation_weights: cw(x) for each element, keyed by node id.
            lex_elements: The min-max normalised BM25 scores of each element within E, keyed by node id.
            alpha: Mixing weight between element similarity and citation authority.
            gamma: Mixing weight between the semantic and lexical similarity channels.

        Returns:
            An ``_ElementScore`` for every element, in the same order as ``elements``.

        """
        max_cw_by_type: dict[str, float] = {}
        for node in elements:
            citation_weight = citation_weights.get(node.id, 0.0)
            max_cw_by_type[node.type] = max(max_cw_by_type.get(node.type, 0.0), citation_weight)
        # log(1+t) is increasing, so each type's max log weight is log1p of its max weight.
        max_log_by_type = {node_type: math.log1p(max_cw) for node_type, max_cw in max_cw_by_type.items()}
        scored = []
        for node in elements:
            semantic = cosine_similarity(embedding, node.summary_embedding)
            lexical = lex_elements.get(node.id, 0.0)
            similarity = hybrid_similarity(semantic, lexical, gamma)
            citation_weight = citation_weights.get(node.id, 0.0)
            max_log = max_log_by_type[node.type]
            authority = math.log1p(citation_weight) / max_log if max_log else 0.0
            score = alpha * similarity + (1 - alpha) * authority
            scored.append(_ElementScore(node, score, similarity, semantic, lexical, citation_weight, authority))
        return scored

    @classmethod
    def _citation_weight(cls, type_counts: dict[str, int]) -> float:
        """
        cw(x) = sum over citing nodes u of lambda(u), grouped by type -- Algorithm 2's definition.

        Args:
            type_counts: Number of incoming CITES arches into x, grouped by source node type.

        Returns:
            The citation weight cw(x).

        """
        return sum(count * cls.LAMBDA_BY_TYPE.get(type_name, 0.0) for type_name, count in type_counts.items())

    def _expand_nodes(  # noqa: PLR0913
        self,
        embedding: Sequence[float],
        query: str,
        relevant: list[_ElementScore],
        *,
        ell_elements: int,
        ell_docs: int,
        gamma: float,
        delta: float,
        max_hops: int,
        include_rulings: bool,
        rerank_enabled: bool,
        rerank_pool: int,
        rho: float,
        rerank_expansion: bool,
        relation_category: Sequence[InstrumentRole] | None,
        operative_expansion_enabled: bool,
        operative_decay: float,
    ) -> list[ExpansionNode]:
        """
        Algorithm 2 steps 5-7: build the citation neighbourhood N, keep its most query-similar nodes.

        N is split into two disjoint pools -- elements (Article/Recital/Annex) and documents
        (Guideline/Ruling) -- each ranked under its own lexical normalisation and cut at its own
        budget, since a paper's summary and an article's summary are different genres of text and
        are not comparable once min-max normalised together.

        Args:
            embedding: The query embedding, used for the semantic channel.
            query: The natural-language query, used for the lexical (BM25) channel.
            relevant: The relevant elements R selected in step 4.
            ell_elements: Number of element expansion nodes to keep.
            ell_docs: Number of document expansion nodes to keep.
            gamma: Mixing weight between the semantic and lexical similarity channels.
            delta: Per-hop decay applied to element expansion similarity; documents get depth 0.
            max_hops: Maximum citation hops followed by the cited element closure.
            include_rulings: Whether rulings, not just guidelines, are admitted into the expansion set G.
            rerank_enabled: Master switch for the cascade; combined with ``rerank_expansion`` below.
            rerank_pool: Shortlist size p handed to the reranker, per pool.
            rho: Rerank blend weight.
            rerank_expansion: Whether step 7 reranks its two pools at all.
            relation_category: If given, restricts operative-expansion targets to matching
                instrument roles; Phase 4's relation-category filter.
            operative_expansion_enabled: Whether relevant base-instrument elements are expanded
                toward newer amending/implementing provisions via OPERATIVE_RELATION edges.
            operative_decay: Flat decay applied to xsim for operative-expansion results, used in
                place of ``delta**depth``.

        Returns:
            The top expansion nodes, element pool first, each ranked best first within its pool.

        """
        relevant_ids = {item.node.id for item in relevant}

        # X = R union ref*(R); reached_from records which relevant elements each cited element came from.
        x_nodes: dict[UUID, Node] = {item.node.id: item.node for item in relevant}
        reached_from: dict[UUID, set[UUID]] = {}
        depths: dict[UUID, int] = {}
        for item in relevant:
            for node, depth in self.driver.cited_element_closure([item.node.id], max_hops=max_hops):
                x_nodes.setdefault(node.id, node)
                reached_from.setdefault(node.id, set()).add(item.node.id)
                depths[node.id] = min(depth, depths.get(node.id, depth))
        log.info(
            "Step 5: cited element closure",
            closure=[node.name for node_id, node in x_nodes.items() if node_id not in relevant_ids],
        )

        # One-way base-to-newer operative expansion: only relevant *base*-role elements expand
        # toward the amending/implementing provisions that operatively relate into them (Behavioral
        # decision 11); nodes reached this way share x_nodes/reached_from with the citation closure
        # above, so a node reached by both is deduplicated for free, kept once with the union of
        # both origins in reached_from.
        operative_ids: set[UUID] = set()
        operative_relations: dict[UUID, tuple[InstrumentRelationType, OperativeOperation]] = {}
        if operative_expansion_enabled:
            for item in relevant:
                if item.node.instrument_role is not InstrumentRole.BASE:
                    continue
                for node, relation_type, operation in self.driver.operative_sources([item.node.id]):
                    if relation_category is not None and node.instrument_role not in relation_category:
                        continue
                    x_nodes.setdefault(node.id, node)
                    reached_from.setdefault(node.id, set()).add(item.node.id)
                    operative_ids.add(node.id)
                    operative_relations.setdefault(node.id, (relation_type, operation))
            log.info(
                "Step 5b: operative expansion",
                expansion=[x_nodes[node_id].name for node_id in operative_ids],
            )

        documents = self.driver.citing_documents(list(x_nodes.keys()))
        if not include_rulings:
            documents = [document for document in documents if document.type != "Ruling"]
        log.info("Step 6: citing documents", documents=[document.name for document in documents])

        # N = (X \ R) union G.
        n_nodes: dict[UUID, Node] = {node_id: node for node_id, node in x_nodes.items() if node_id not in relevant_ids}
        for document in documents:
            n_nodes.setdefault(document.id, document)
        if not n_nodes:
            log.info("Step 7: no expansion candidates found")
            return []

        n_elem = {node_id: node for node_id, node in n_nodes.items() if node.type in self.ELEMENT_TYPES}
        n_doc = {node_id: node for node_id, node in n_nodes.items() if node.type not in self.ELEMENT_TYPES}
        do_rerank = rerank_enabled and rerank_expansion

        elements = self._score_expansion_pool(
            embedding,
            query,
            n_elem,
            "element",
            reached_from,
            depths,
            gamma=gamma,
            delta=delta,
            operative_ids=operative_ids,
            operative_decay=operative_decay,
            operative_relations=operative_relations,
            rerank_enabled=do_rerank,
            rerank_pool=rerank_pool,
            rho=rho,
        )
        documents_scored = self._score_expansion_pool(
            embedding,
            query,
            n_doc,
            "document",
            reached_from,
            {},
            gamma=gamma,
            delta=delta,
            operative_ids=set(),
            operative_decay=operative_decay,
            operative_relations={},
            rerank_enabled=do_rerank,
            rerank_pool=rerank_pool,
            rho=rho,
        )

        selected_elements = elements[:ell_elements]
        selected_docs = documents_scored[:ell_docs]
        log.info(
            "Step 7: selected expansion nodes (elements)",
            num_candidates=len(elements),
            selected=[node.name for node in selected_elements],
        )
        log.info(
            "Step 7: selected expansion nodes (documents)",
            num_candidates=len(documents_scored),
            selected=[node.name for node in selected_docs],
        )
        return selected_elements + selected_docs

    def _lexical_index(self, nodes: Iterable[Node], pool: Literal["element", "document"]) -> Bm25Index:
        """
        Build the BM25 index of one expansion pool.

        Args:
            nodes: The pool's candidate nodes.
            pool: Which pool they form; only the document pool honours
                ``InferenceSettings.document_lexical_field``.

        Returns:
            An index over each node's full text, or over each document's summary when
            ``document_lexical_field`` is ``summary`` and ``pool`` is ``document``.

        """
        if pool == "document" and self._settings.inference.document_lexical_field == "summary":
            return Bm25Index.from_texts((node.id, node.summary) for node in nodes)
        return Bm25Index.from_nodes(nodes)

    def _score_expansion_pool(  # noqa: PLR0913, PLR0917
        self,
        embedding: Sequence[float],
        query: str,
        nodes: dict[UUID, Node],
        pool: Literal["element", "document"],
        reached_from: dict[UUID, set[UUID]],
        depths: dict[UUID, int],
        *,
        gamma: float,
        delta: float,
        operative_ids: set[UUID],
        operative_decay: float,
        operative_relations: dict[UUID, tuple[InstrumentRelationType, OperativeOperation]],
        rerank_enabled: bool,
        rerank_pool: int,
        rho: float,
    ) -> list[ExpansionNode]:
        """
        Score, decay and rank one expansion pool: its own BM25 normalisation, its own xsim cut,
        optionally rescored by the reranker (step 7's cascade).

        Args:
            embedding: The query embedding, used for the semantic channel.
            query: The natural-language query, used for the lexical (BM25) channel.
            nodes: The candidate nodes of this pool, keyed by id.
            pool: Which pool this is, stored on each returned node.
            reached_from: Ids of the relevant elements each node was reached from, keyed by node id.
            depths: Minimum citation hops from R to each node, keyed by node id; missing means depth 0.
            gamma: Mixing weight between the semantic and lexical similarity channels.
            delta: Per-hop decay applied to xsim for citation-expansion results.
            operative_ids: Ids of nodes reached via operative expansion rather than citation
                closure; these use ``operative_decay`` instead of ``delta**depth`` (a node reached
                by both is treated as operative).
            operative_decay: Flat decay applied to xsim for operative-expansion results.
            operative_relations: The ``(relation_type, operation)`` each operative-expansion node
                was reached with, keyed by node id.
            rerank_enabled: Whether to run the cascade for this pool at all.
            rerank_pool: Shortlist size p handed to the reranker.
            rho: Rerank blend weight.

        Returns:
            This pool's nodes, ranked best first -- by xsim_rr over the reranked shortlist when
            the cascade ran, by xsim = similarity * decay otherwise.

        """
        if not nodes:
            return []
        lex_pool = min_max_normalize(self._lexical_index(nodes.values(), pool).scores(query))
        scored = []
        decay_by_id: dict[UUID, float] = {}
        similarity_by_id: dict[UUID, float] = {}
        for node in nodes.values():
            semantic = cosine_similarity(embedding, node.summary_embedding)
            lexical = lex_pool.get(node.id, 0.0)
            similarity = hybrid_similarity(semantic, lexical, gamma)
            depth = depths.get(node.id, 0)
            decay = operative_decay if node.id in operative_ids else delta**depth
            decay_by_id[node.id] = decay
            similarity_by_id[node.id] = similarity
            relation_type, operation = operative_relations.get(node.id, (None, None))
            scored.append(
                ExpansionNode(
                    id=node.id,
                    name=node.name,
                    type=node.type,
                    text=node.text,
                    summary=node.summary,
                    similarity=similarity,
                    semantic=semantic,
                    lexical=lexical,
                    pool=pool,
                    depth=depth,
                    decay=decay,
                    reached_from=sorted(reached_from.get(node.id, set())),
                    qualified_citation=node.qualified_citation,
                    relation_type=relation_type,
                    operation=operation,
                )
            )
        scored.sort(key=lambda node: (node.similarity * node.decay, str(node.id)), reverse=True)
        log.info(
            "Step 7: scored expansion candidates",
            pool=pool,
            candidates=[{"name": node.name, "xsim": round(node.similarity * node.decay, 3)} for node in scored],
        )

        if rerank_enabled:
            node_by_id = {node.id: node for node in nodes.values()}
            xsim_by_id = {
                node_id: similarity * decay_by_id[node_id] for node_id, similarity in similarity_by_id.items()
            }
            rerank_raw, score_rr = self._rerank(
                query, node_by_id, xsim_by_id, similarity_by_id, rerank_pool=rerank_pool, rho=rho
            )
            if score_rr:
                by_id = {node.id: node for node in scored}
                final_xsim = {node_id: blended * decay_by_id[node_id] for node_id, blended in score_rr.items()}
                ranked_ids = sorted(final_xsim.items(), key=lambda kv: (kv[1], str(kv[0])), reverse=True)
                reranked = [
                    by_id[node_id].model_copy(update={"rerank_raw": rerank_raw[node_id], "rerank": xsim_rr, "rho": rho})
                    for node_id, xsim_rr in ranked_ids
                ]
                log.info(
                    "Step 7: reranked expansion candidates",
                    pool=pool,
                    shortlist_size=len(score_rr),
                    candidates=[
                        {"name": by_id[node_id].name, "xsim_rr": round(xsim_rr, 3)} for node_id, xsim_rr in ranked_ids
                    ],
                )
                return reranked

        return scored


def _top_by_value(values: dict[UUID, float], count: int) -> list[tuple[UUID, float]]:
    """
    Return the ``count`` highest-valued ``(id, value)`` pairs, best first.

    Args:
        values: A mapping from id to a raw score.
        count: Number of top pairs to return.

    Returns:
        The ``count`` highest-valued ``(id, value)`` pairs, sorted descending.

    """
    return sorted(values.items(), key=lambda item: item[1], reverse=True)[:count]
