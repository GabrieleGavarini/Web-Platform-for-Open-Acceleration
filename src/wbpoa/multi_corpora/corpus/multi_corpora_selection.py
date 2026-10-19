from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from structlog import get_logger

from wbpoa.core.corpus.corpus_registry import DEFAULT_CORPORA_DIR, CorpusRegistry
from wbpoa.multi_corpora.corpus.multi_corpora_registry import MultiCorporaRegistry, MultiCorporaWorkspace

logger = get_logger()

DEFAULT_MULTI_CORPORA_DIR = Path("data/multi-corpora")


def add_multi_corpora_selection_arguments(parser: argparse.ArgumentParser, *, all_help: str) -> None:
    """
    Add the shared `--multi-corpora KEY | --all` selector and its registry directories to a multi-corpora CLI.

    Exactly one of `--multi-corpora` and `--all` is required; there is no implicit default pairing.

    Args:
        parser: Parser to extend.
        all_help: Help text describing what `--all` does for this particular CLI.
    """
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--multi-corpora", help="Canonical multi-corpora key from the multi-corpora registry.")
    selection.add_argument("--all", action="store_true", help=all_help)
    add_multi_corpora_directory_arguments(parser)


def add_multi_corpora_directory_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add `--multi-corpora-dir` and `--corpora-dir`, the two registries a multi-corpora key resolves through.

    Args:
        parser: Parser to extend.
    """
    parser.add_argument(
        "--multi-corpora-dir",
        type=Path,
        default=DEFAULT_MULTI_CORPORA_DIR,
        help="Multi-corpora manifest directory.",
    )
    parser.add_argument(
        "--corpora-dir", type=Path, default=DEFAULT_CORPORA_DIR, help="Corpus manifest directory for sub-corpora."
    )


def multi_corpora_registry(args: argparse.Namespace) -> MultiCorporaRegistry:
    """
    Build the multi-corpora registry named by `add_multi_corpora_directory_arguments`.

    Args:
        args: Parsed arguments carrying `multi_corpora_dir` and `corpora_dir`.

    Returns:
        A registry resolving sub-corpora against `corpora_dir`.
    """
    return MultiCorporaRegistry(args.multi_corpora_dir, CorpusRegistry(args.corpora_dir))


def select_multi_corpora_workspaces(args: argparse.Namespace) -> list[MultiCorporaWorkspace]:
    """
    Resolve the pairings chosen by `add_multi_corpora_selection_arguments`.

    Args:
        args: Parsed arguments carrying `multi_corpora`, `all`, `multi_corpora_dir`, and `corpora_dir`.

    Returns:
        Every discovered pairing in key order for `--all`, otherwise the single `--multi-corpora` workspace.

    Raises:
        MultiCorporaRegistryError: If the registry, the selected key, or any discovered manifest is invalid.
    """
    registry = multi_corpora_registry(args)
    return registry.discover() if args.all else [registry.load(args.multi_corpora)]


def run_for_each_multi_corpora(
    workspaces: list[MultiCorporaWorkspace],
    action: Callable[[MultiCorporaWorkspace], None],
    *,
    keep_going: bool,
    label: str,
) -> None:
    """
    Run `action` for every multi-corpora workspace in order, collecting failures.

    Args:
        workspaces: Pairings to process, in order.
        action: Per-pairing work; any exception it raises counts as that pairing failing.
        keep_going: When true (the `--all` case), a failure is logged and later pairings are still
            attempted; when false, the first failure is re-raised unchanged.
        label: Human-readable name of the work, used in log events and the failure summary.

    Raises:
        SystemExit: If `keep_going` is true and at least one pairing failed; the message lists each one.
    """
    failures: list[tuple[str, Exception]] = []
    for workspace in workspaces:
        try:
            logger.info(
                "Multi-corpora task started",
                task=label,
                multi_corpora=workspace.key,
                corpora=workspace.config.corpora,
            )
            action(workspace)
            logger.info("Multi-corpora task succeeded", task=label, multi_corpora=workspace.key)
        except Exception as error:
            if not keep_going:
                raise
            failures.append((workspace.key, error))
            logger.exception("Multi-corpora task failed", task=label, multi_corpora=workspace.key)
    if failures:
        summary = ", ".join(f"{key}: {type(error).__name__}" for key, error in failures)
        raise SystemExit(f"{label} failed for: {summary}")
