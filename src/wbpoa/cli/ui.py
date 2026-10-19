import argparse
from pathlib import Path

from wbpoa.chat_ui.agents import AgentPool, DefaultAgentFactory
from wbpoa.chat_ui.app.chat_app import ChatApp
from wbpoa.core.corpus import CorpusRegistry
from wbpoa.core.corpus.corpus_registry import DEFAULT_CORPORA_DIR
from wbpoa.multi_corpora import MultiCorporaRegistry
from wbpoa.multi_corpora.corpus import DEFAULT_MULTI_CORPORA_DIR
from wbpoa.multi_corpora.settings import get_multi_corpora_settings


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the chat UI.
    """
    parser = argparse.ArgumentParser(
        description="Serve the WBPOA chat UI: chat with the agents, browse the citation graph, read about the project."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind.")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on.")
    parser.add_argument("--corpora-dir", type=Path, default=DEFAULT_CORPORA_DIR, help="Corpus manifest directory.")
    parser.add_argument(
        "--multi-corpora-dir", type=Path, default=DEFAULT_MULTI_CORPORA_DIR, help="Multi-corpora manifest directory."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/dataset"),
        help="Question CSVs the example questions come from.",
    )
    return parser.parse_args()


def main() -> None:
    """Serve the chat UI until interrupted."""
    args = parse_args()
    corpus_registry = CorpusRegistry(args.corpora_dir)
    pool = AgentPool(
        corpus_registry,
        MultiCorporaRegistry(args.multi_corpora_dir, corpus_registry),
        DefaultAgentFactory(get_multi_corpora_settings()),
        dataset_dir=args.dataset_dir,
    )
    ChatApp(pool, pool, on_shutdown=pool.close).run(host=args.host, port=args.port)


if __name__ in {"__main__", "__mp_main__"}:
    main()
