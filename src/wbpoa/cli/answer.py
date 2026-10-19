import argparse
import json
from pathlib import Path

from structlog import get_logger

from wbpoa.core.corpus import CorpusContext, CorpusRegistry
from wbpoa.core.corpus.corpus_config import InstrumentRole
from wbpoa.core.driver.neo4j_driver import Neo4jDriver
from wbpoa.core.generation import SingleCorpusAgent, SingleCorpusAgentError
from wbpoa.core.inference import CrossEncoderReranker, Retriever
from wbpoa.core.ingestion.utils import Embedder
from wbpoa.single_corpus.settings import get_single_corpus_settings

logger = get_logger()


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for answering a query with the single-corpus agent (Algorithm 3).
    """
    parser = argparse.ArgumentParser(
        description="Answer a query with the single-corpus agent (Algorithm 3 over a search tool)."
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
        "--no-reformulation",
        dest="reformulation_enabled",
        action="store_false",
        default=None,
        help="Skip reformulation; the search tool runs Algorithm 2 on each question verbatim.",
    )
    parser.add_argument(
        "--no-redaction",
        dest="redaction_enabled",
        action="store_false",
        default=None,
        help="Skip redaction; the search tool returns its full candidate pool U.",
    )
    parser.add_argument(
        "--redaction-field",
        choices=["text", "summary"],
        default=None,
        help="Whether the redaction agent reads each node's full text or its summary.",
    )
    parser.add_argument(
        "--redaction-temperature", type=float, default=None, help="Sampling temperature for the redaction agent."
    )
    parser.add_argument(
        "--max-context-nodes",
        type=int,
        default=None,
        help="Optional hard cap on |S| per search, after redaction.",
    )
    parser.add_argument("--max-searches", type=int, default=None, help="Maximum searchs per question.")
    parser.add_argument(
        "--relation-category",
        dest="relation_category",
        action="append",
        choices=[role.value for role in InstrumentRole],
        default=None,
        help="Restrict retrieval to one or more instrument roles (repeatable); defaults to every active instrument.",
    )
    return parser.parse_args()


def main() -> None:
    """Answer the command-line query with the single-corpus agent."""
    args = parse_args()
    settings = get_single_corpus_settings()
    workspace = CorpusRegistry(args.corpora_dir).load(args.corpus)
    settings.act_name = workspace.config.name
    if args.reformulation_enabled is not None:
        settings.generation.reformulation_enabled = args.reformulation_enabled
    if args.redaction_enabled is not None:
        settings.generation.redaction_enabled = args.redaction_enabled
    if args.redaction_field is not None:
        settings.generation.redaction_field = args.redaction_field
    if args.redaction_temperature is not None:
        settings.generation.redaction_temperature = args.redaction_temperature
    if args.max_context_nodes is not None:
        settings.generation.max_context_nodes = args.max_context_nodes
    if args.max_searches is not None:
        settings.generation.max_searches = args.max_searches
    relation_category = [InstrumentRole(role) for role in args.relation_category] if args.relation_category else None

    embedder = Embedder()
    reranker = CrossEncoderReranker(
        settings.inference.rerank_model,
        device=settings.inference.rerank_device,
        token=settings.hf_token.get_secret_value() if settings.hf_token is not None else None,
    )
    with Neo4jDriver(settings, corpus_id=workspace.key) as driver:
        driver.ensure_schema(dimensions=embedder.dimension)
        retriever = Retriever(driver, embedder, reranker=reranker, settings=settings)
        agent = SingleCorpusAgent(
            retriever,
            settings=settings,
            corpus_context=CorpusContext.from_workspace(workspace),
        )
        try:
            result = agent.run(
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
        except SingleCorpusAgentError:
            logger.exception("Generation failed")
            raise

    logger.info(
        "Generation completed",
        query=args.query,
        search_questions=[call.question for call in result.searches],
        reformulated_queries=[call.reformulated_query for call in result.searches],
        num_context_nodes=len(result.grounding_context),
    )
    output = {
        "answer": result.answer,
        "search_questions": [call.question for call in result.searches],
        "provisions": [node.name for node in result.grounding_context],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))  # noqa: T201 -- the CLI's deliverable output


if __name__ == "__main__":
    main()
