from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from structlog import get_logger

from wbpoa.core.corpus.corpus_registry import DEFAULT_CORPORA_DIR, CorpusRegistry, CorpusWorkspace

logger = get_logger()


def add_corpus_selection_arguments(parser: argparse.ArgumentParser, *, all_help: str) -> None:
    """
    Add the shared `--corpus KEY | --all` selector and `--corpora-dir` to a root CLI.

    Exactly one of `--corpus` and `--all` is required; there is no implicit default corpus.

    Args:
        parser: Parser to extend.
        all_help: Help text describing what `--all` does for this particular CLI.
    """
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--corpus", help="Canonical corpus key from the corpus registry.")
    selection.add_argument("--all", action="store_true", help=all_help)
    parser.add_argument("--corpora-dir", type=Path, default=DEFAULT_CORPORA_DIR, help="Corpus manifest directory.")


def select_workspaces(args: argparse.Namespace) -> list[CorpusWorkspace]:
    """
    Resolve the corpora chosen by `add_corpus_selection_arguments`.

    Args:
        args: Parsed arguments carrying `corpus`, `all`, and `corpora_dir`.

    Returns:
        Every discovered corpus in key order for `--all`, otherwise the single `--corpus` workspace.

    Raises:
        CorpusRegistryError: If the registry, the selected key, or any discovered manifest is invalid.
    """
    registry = CorpusRegistry(args.corpora_dir)
    return registry.discover() if args.all else [registry.load(args.corpus)]


def run_for_each_corpus(
    workspaces: list[CorpusWorkspace],
    action: Callable[[CorpusWorkspace], None],
    *,
    keep_going: bool,
    label: str,
) -> None:
    """
    Run `action` for every workspace in order, collecting failures.

    Args:
        workspaces: Corpora to process, in order.
        action: Per-corpus work; any exception it raises counts as that corpus failing.
        keep_going: When true (the `--all` case), a failure is logged and later corpora are still
            attempted; when false, the first failure is re-raised unchanged.
        label: Human-readable name of the work, used in log events and the failure summary.

    Raises:
        SystemExit: If `keep_going` is true and at least one corpus failed; the message lists each one.
    """
    failures: list[tuple[str, Exception]] = []
    for workspace in workspaces:
        try:
            logger.info("Corpus task started", task=label, corpus=workspace.key, name=workspace.config.name)
            action(workspace)
            logger.info("Corpus task succeeded", task=label, corpus=workspace.key)
        except Exception as error:
            if not keep_going:
                raise
            failures.append((workspace.key, error))
            logger.exception("Corpus task failed", task=label, corpus=workspace.key)
    if failures:
        summary = ", ".join(f"{key}: {type(error).__name__}" for key, error in failures)
        raise SystemExit(f"{label} failed for: {summary}")
