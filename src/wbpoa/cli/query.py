import argparse
from pathlib import Path

from structlog import get_logger

from wbpoa.core.corpus import CorpusRegistry
from wbpoa.core.corpus.corpus_config import InstrumentRole
from wbpoa.core.driver.neo4j_driver import Neo4jDriver
from wbpoa.core.inference import CrossEncoderReranker, RetrievalResult, Retriever, RetrieverError
from wbpoa.core.ingestion.utils import Embedder
from wbpoa.single_corpus.settings import get_single_corpus_settings

logger = get_logger()


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for querying the WBPOA graph (Algorithm 2).
    """
    parser = argparse.ArgumentParser(
        description="Retrieve the elements and their citation neighbourhood relevant to a query."
    )
    parser.add_argument("query", help="Natural-language query.")
    parser.add_argument("--corpus", required=True, help="Canonical corpus key from the corpus registry.")
    parser.add_argument("--corpora-dir", type=Path, default=Path("data/corpora"), help="Corpus manifest directory.")
    parser.add_argument("--h0", type=int, default=None, help="Community graph layer to select candidates from.")
    parser.add_argument("--k", type=int, default=None, help="Number of seed and finally selected communities.")
    parser.add_argument("--m", type=int, default=None, help="Number of neighbour communities per seed.")
    parser.add_argument("--n", type=int, default=None, help="Number of layer-0 elements used for the bottom-up vote.")
    parser.add_argument(
        "--vote-top-j",
        type=int,
        default=None,
        help="Number of top per-community similarities averaged into the bottom-up vote; 1 reduces it to a max.",
    )
    parser.add_argument("--r", type=int, default=None, help="Number of relevant elements to return.")
    parser.add_argument("--ell-elements", type=int, default=None, help="Number of element expansion nodes to return.")
    parser.add_argument("--ell-docs", type=int, default=None, help="Number of document expansion nodes to return.")
    parser.add_argument("--alpha", type=float, default=None, help="Element similarity vs. citation authority mix.")
    parser.add_argument("--beta", type=float, default=None, help="Community similarity vs. vote mix.")
    parser.add_argument("--gamma", type=float, default=None, help="Semantic vs. lexical similarity mix.")
    parser.add_argument("--delta", type=float, default=None, help="Per-hop decay applied to expansion similarity.")
    parser.add_argument(
        "--max-hops", type=int, default=None, help="Maximum citation hops in the cited element closure."
    )
    parser.add_argument(
        "--no-rulings",
        dest="include_rulings",
        action="store_false",
        default=None,
        help="Only admit citing guidelines, not rulings, into the expansion set.",
    )
    parser.add_argument(
        "--rerank",
        dest="rerank_enabled",
        action="store_true",
        default=None,
        help="Enable the cross-encoder reranking cascade (steps 4(ii) and 7).",
    )
    parser.add_argument(
        "--no-rerank",
        dest="rerank_enabled",
        action="store_false",
        default=None,
        help="Disable the reranking cascade; the algorithm reduces exactly to the pre-reranker formulation.",
    )
    parser.add_argument("--rerank-pool", type=int, default=None, help="Shortlist size p handed to the reranker.")
    parser.add_argument("--rho", type=float, default=None, help="Rerank blend: rho*rsim + (1-rho)*first-pass score.")
    parser.add_argument(
        "--relation-category",
        dest="relation_category",
        action="append",
        choices=[role.value for role in InstrumentRole],
        default=None,
        help="Restrict retrieval to one or more instrument roles (repeatable); defaults to every active instrument.",
    )
    return parser.parse_args()


def log_results(result: RetrievalResult) -> None:
    """
    Log the ranked relevant elements and the ranked expansion nodes as two separate lists.
    """
    if not result.relevant_elements:
        logger.info("No relevant elements found")
        return

    for element in result.relevant_elements:
        logger.info(
            "Relevant element",
            score=round(element.score, 3),
            alpha=element.alpha,
            gamma=element.gamma,
            similarity=round(element.similarity, 3),
            semantic=round(element.semantic, 3),
            lexical=round(element.lexical, 3),
            citation_weight=round(element.citation_weight, 3),
            authority=round(element.authority, 3),
            rerank_raw=round(element.rerank_raw, 3) if element.rerank_raw is not None else None,
            rerank=round(element.rerank, 3) if element.rerank is not None else None,
            rho=element.rho,
            type=element.type,
            name=element.name,
            instrument_key=element.qualified_citation.instrument_key if element.qualified_citation else None,
        )

    for pool_label in ("element", "document"):
        pool_nodes = [node for node in result.expansion_nodes if node.pool == pool_label]
        for node in pool_nodes:
            logger.info(
                "Expansion node",
                pool=pool_label,
                similarity=round(node.similarity, 3),
                semantic=round(node.semantic, 3),
                lexical=round(node.lexical, 3),
                depth=node.depth,
                decay=round(node.decay, 3),
                xsim=round(node.similarity * node.decay, 3),
                rerank_raw=round(node.rerank_raw, 3) if node.rerank_raw is not None else None,
                rerank=round(node.rerank, 3) if node.rerank is not None else None,
                rho=node.rho,
                type=node.type,
                name=node.name,
                reached_from=[str(element_id) for element_id in node.reached_from],
                relation_type=node.relation_type,
                operation=node.operation,
                instrument_key=node.qualified_citation.instrument_key if node.qualified_citation else None,
            )


def main() -> None:
    """Retrieve and log the elements relevant to the command-line query."""
    args = parse_args()

    embedder = Embedder()
    settings = get_single_corpus_settings()
    workspace = CorpusRegistry(args.corpora_dir).load(args.corpus)
    settings.act_name = workspace.config.name
    inference_settings = settings.inference
    relation_category = [InstrumentRole(role) for role in args.relation_category] if args.relation_category else None
    # Constructed unconditionally but lazily loaded: --rerank/--no-rerank toggles rerank_enabled
    # per call without needing to know in advance whether this run will use it.
    reranker = CrossEncoderReranker(
        inference_settings.rerank_model,
        device=inference_settings.rerank_device,
        token=settings.hf_token.get_secret_value() if settings.hf_token is not None else None,
    )
    with Neo4jDriver(settings, corpus_id=workspace.key) as driver:
        driver.ensure_schema(dimensions=embedder.dimension)
        retriever = Retriever(driver, embedder, reranker=reranker)
        try:
            result = retriever.retrieve(
                args.query,
                h0=args.h0,
                k=args.k,
                m=args.m,
                n=args.n,
                vote_top_j=args.vote_top_j,
                r=args.r,
                ell_elements=args.ell_elements,
                ell_docs=args.ell_docs,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                delta=args.delta,
                max_hops=args.max_hops,
                include_rulings=args.include_rulings,
                rerank_enabled=args.rerank_enabled,
                rerank_pool=args.rerank_pool,
                rho=args.rho,
                relation_category=relation_category,
            )
        except RetrieverError:
            logger.exception("Query failed")
            raise

    logger.info(
        "Query completed",
        query=args.query,
        num_relevant=len(result.relevant_elements),
        num_expansion=len(result.expansion_nodes),
    )
    log_results(result)


if __name__ == "__main__":
    main()
