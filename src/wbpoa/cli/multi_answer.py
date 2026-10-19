import argparse
import json
import sys

from structlog import get_logger

from wbpoa.core.corpus.corpus_config import InstrumentRole
from wbpoa.core.inference import CrossEncoderReranker
from wbpoa.core.ingestion.utils import Embedder
from wbpoa.multi_corpora import (
    MultiCorporaContext,
    MultiCorporaSubAgents,
    MultiCorpusAgent,
    MultiCorpusAgentError,
    add_multi_corpora_directory_arguments,
    multi_corpora_registry,
)
from wbpoa.multi_corpora.settings import get_multi_corpora_settings

logger = get_logger()


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for answering a query with the multi-corpus agent.
    """
    parser = argparse.ArgumentParser(description="Answer a query spanning two corpora via the multi-corpus agent.")
    parser.add_argument("query", help="Natural-language query.")
    parser.add_argument(
        "--multi-corpora", required=True, help="Canonical multi-corpora key from the multi-corpora registry."
    )
    add_multi_corpora_directory_arguments(parser)
    parser.add_argument(
        "--max-calls-per-sub-agent",
        type=int,
        default=None,
        help="Maximum calls the manager may make to each sub agent.",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Manager tool-calling turns before synthesis.")
    parser.add_argument(
        "--relation-category",
        dest="relation_category",
        action="append",
        choices=[role.value for role in InstrumentRole],
        default=None,
        help="Restrict both sub agents' retrieval to one or more instrument roles (repeatable).",
    )
    return parser.parse_args()


def main() -> None:
    """Answer the command-line query with the multi-corpus agent."""
    args = parse_args()
    settings = get_multi_corpora_settings()
    workspace = multi_corpora_registry(args).load(args.multi_corpora)
    settings.act_name = workspace.config.name
    if args.max_calls_per_sub_agent is not None:
        settings.multi_corpora.max_calls_per_sub_agent = args.max_calls_per_sub_agent
    if args.max_steps is not None:
        settings.multi_corpora.max_steps = args.max_steps
    if args.relation_category:
        settings.inference.relation_category = tuple(InstrumentRole(role) for role in args.relation_category)

    embedder = Embedder()
    reranker = CrossEncoderReranker(
        settings.inference.rerank_model,
        device=settings.inference.rerank_device,
        token=settings.hf_token.get_secret_value() if settings.hf_token is not None else None,
    )
    with MultiCorporaSubAgents(workspace, settings=settings, embedder=embedder, reranker=reranker) as sub_agents:
        manager = MultiCorpusAgent(sub_agents, context=MultiCorporaContext.from_workspace(workspace), settings=settings)
        try:
            result = manager.run(args.query)
        except MultiCorpusAgentError:
            logger.exception("Multi-corpora generation failed")
            raise

    logger.info(
        "Multi-corpora generation completed",
        query=args.query,
        sub_agent_calls=[(call.corpus_key, call.question, call.result is not None) for call in result.sub_agent_calls],
        num_context_nodes=len(result.grounding_context),
    )
    output = {
        "answer": result.answer,
        "provisions": [node.name for node in result.grounding_context],
        "sub_agent_calls": [
            {"corpus": call.corpus_key, "question": call.question, "error": call.error}
            for call in result.sub_agent_calls
        ],
    }
    sys.stdout.write(json.dumps(output, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
