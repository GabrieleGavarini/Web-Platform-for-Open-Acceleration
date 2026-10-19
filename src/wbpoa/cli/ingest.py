import os

# Must be set before torch is imported transitively.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import argparse
import time
from pathlib import Path

from structlog import get_logger

from wbpoa.core.corpus import CorpusWorkspace
from wbpoa.core.corpus.corpus_config import InstrumentRelationType
from wbpoa.core.driver.neo4j_driver import Neo4jDriver
from wbpoa.core.ingestion import InstrumentIngestion, ProvisionReferenceParser, load_or_build, load_or_ingest_corpus
from wbpoa.core.ingestion.utils import Embedder
from wbpoa.core.model.element import Corpus, Guidelines, Rulings
from wbpoa.single_corpus.corpus_selection import add_corpus_selection_arguments, run_for_each_corpus, select_workspaces
from wbpoa.single_corpus.graph_builder import (
    CitationGraphBuilder,
    CoCitationGraphBuilder,
    CommunityGraphBuilder,
    InstrumentGraphBuilder,
)
from wbpoa.single_corpus.graph_builder.base_graph_builder import GraphBuilderContext
from wbpoa.single_corpus.ingestion import GuidelineIngestion, RulingIngestion
from wbpoa.single_corpus.ingestion.source_downloader import SourceDownloader
from wbpoa.single_corpus.ingestion.utils import ProgressLogger, Summarizer
from wbpoa.single_corpus.ingestion.zotero_downloader import ZoteroDownloader
from wbpoa.single_corpus.settings import get_single_corpus_settings

logger = get_logger()


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for building the co-citation graph.
    """
    parser = argparse.ArgumentParser(description="Build the WBPOA graph for one or all manifest-defined corpora.")
    add_corpus_selection_arguments(
        parser, all_help="Build every discovered corpus sequentially; a failure does not stop later corpora."
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=Path("tmp"),
        help="Folder used to cache intermediate JSON artifacts (corpus, guidelines, nodes, edges).",
    )
    parser.add_argument(
        "--refresh-sources",
        action="store_true",
        help="Refresh the legislation source and configured Zotero collection before ingestion.",
    )
    parser.add_argument(
        "--invalidate-ingestion-cache",
        action="store_true",
        help="Reingest the corpus, guidelines, and rulings from source, then rebuild all graph caches.",
    )
    parser.add_argument(
        "--invalidate-ruling-cache",
        action="store_true",
        help="Rebuild ruling cache while reusing corpus and guideline data.",
    )
    parser.add_argument(
        "--invalidate-graph-cache",
        action="store_true",
        help="Rebuild all graph caches while reusing cached corpus, guideline, and ruling data.",
    )
    parser.add_argument(
        "--invalidate-community-cache",
        action="store_true",
        help="Rebuild the community graph while reusing the source graph caches.",
    )
    return parser.parse_args()


def load_or_ingest_guidelines(  # noqa: PLR0913
    guideline_folder: Path,
    guidelines_json_path: Path,
    corpus: Corpus,
    reference_parser: ProvisionReferenceParser,
    guideline_ingestion: GuidelineIngestion,
    *,
    bypass_cache: bool = False,
) -> Guidelines:
    """
    Load guidelines from cache or ingest them and resolve family-wide provision references.
    """

    def ingest() -> Guidelines:
        logger.info("Ingesting guidelines from PDF files", folder=guideline_folder)
        guidelines = Guidelines(guidelines=[])

        guideline_paths = sorted(guideline_folder.glob("*.pdf"))
        progress = ProgressLogger(total=len(guideline_paths))
        for idx, guideline_path in enumerate(guideline_paths, start=1):
            ingest_start = time.perf_counter()
            guideline = guideline_ingestion.ingest(
                guideline_path,
                name=guideline_path.stem,
                progress=progress,
                current_document=idx,
            )
            ingest_seconds = round(time.perf_counter() - ingest_start, 3)
            if guideline is None:
                logger.info(
                    "guideline_skipped",
                    filename=guideline_path.name,
                    reason="page_limit",
                    **progress.record(idx),
                )
                continue
            reference_parsing_start = time.perf_counter()
            references = reference_parser.parse_family_references(guideline.text, corpus)
            reference_parsing_seconds = round(time.perf_counter() - reference_parsing_start, 3)
            guideline.provision_references = {reference.id for reference in references}
            guidelines.guidelines.append(guideline)
            logger.info(
                "guideline_ingested",
                filename=guideline_path.name,
                ingest_seconds=ingest_seconds,
                reference_parsing_seconds=reference_parsing_seconds,
                **progress.record(idx),
            )

        return guidelines

    return load_or_build(guidelines_json_path, Guidelines, ingest, bypass_cache=bypass_cache)


def load_or_ingest_rulings(  # noqa: PLR0913
    ruling_folder: Path,
    rulings_json_path: Path,
    corpus: Corpus,
    reference_parser: ProvisionReferenceParser,
    ruling_ingestion: RulingIngestion,
    *,
    bypass_cache: bool = False,
) -> Rulings:
    """
    Load rulings from cache or ingest them and resolve family-wide provision references.
    """

    def ingest() -> Rulings:
        logger.info("Ingesting rulings from PDF files", folder=ruling_folder)
        rulings = Rulings(rulings=[])

        if not ruling_folder.exists():
            logger.info("Ruling folder does not exist, skipping ruling ingestion", folder=ruling_folder)
            return rulings

        ruling_paths = sorted(ruling_folder.glob("*.pdf"))
        progress = ProgressLogger(total=len(ruling_paths))
        for idx, ruling_path in enumerate(ruling_paths, start=1):
            ingest_start = time.perf_counter()
            ruling = ruling_ingestion.ingest(
                ruling_path,
                name=ruling_path.stem,
                progress=progress,
                current_document=idx,
            )
            ingest_seconds = round(time.perf_counter() - ingest_start, 3)
            if ruling is None:
                logger.info(
                    "ruling_skipped",
                    filename=ruling_path.name,
                    reason="page_limit",
                    **progress.record(idx),
                )
                continue
            reference_parsing_start = time.perf_counter()
            references = reference_parser.parse_family_references(ruling.text, corpus)
            reference_parsing_seconds = round(time.perf_counter() - reference_parsing_start, 3)
            ruling.provision_references = {reference.id for reference in references}
            rulings.rulings.append(ruling)
            logger.info(
                "ruling_ingested",
                filename=ruling_path.name,
                ingest_seconds=ingest_seconds,
                reference_parsing_seconds=reference_parsing_seconds,
                **progress.record(idx),
            )

        return rulings

    return load_or_build(rulings_json_path, Rulings, ingest, bypass_cache=bypass_cache)


def build_corpus(workspace: CorpusWorkspace, args: argparse.Namespace, settings) -> None:  # noqa: PLR0912, PLR0915
    """Build one manifest-defined corpus and its corpus-local graph caches."""
    invalidate_ingestion_cache = args.invalidate_ingestion_cache
    invalidate_ruling_cache = args.invalidate_ruling_cache or invalidate_ingestion_cache
    invalidate_graph_cache = args.invalidate_graph_cache or invalidate_ingestion_cache
    invalidate_community_cache = args.invalidate_community_cache or invalidate_graph_cache or invalidate_ingestion_cache

    cache_root = (args.tmp_dir / "corpora" / workspace.key).resolve()
    graph_root = cache_root / "graph"
    corpus_json_path = cache_root / "instruments.json"
    guidelines_json_path = cache_root / "guidelines.json"
    rulings_json_path = cache_root / "rulings.json"
    nodes_path = graph_root / "co_citation_nodes.json"
    edges_path = graph_root / "co_citation_edges.json"
    citation_nodes_path = graph_root / "citation_nodes.json"
    citation_edges_path = graph_root / "citation_edges.json"
    community_nodes_path = graph_root / "community_nodes.json"
    community_edges_path = graph_root / "community_edges.json"
    instrument_nodes_path = graph_root / "instrument_nodes.json"
    instrument_edges_path = graph_root / "instrument_edges.json"

    source_downloader = SourceDownloader(
        timeout=settings.zotero.timeout_seconds,
        max_retries=settings.zotero.max_retries,
    )
    for instrument in workspace.config.instruments:
        source_downloader.download(
            instrument.url,
            workspace.instrument_source_paths[instrument.key],
            refresh=args.refresh_sources,
        )
    ZoteroDownloader(workspace.config.guidelines, settings.zotero, workspace.guidelines_dir).sync(
        refresh=args.refresh_sources
    )
    if workspace.config.rulings is not None:
        ZoteroDownloader(workspace.config.rulings, settings.zotero, workspace.rulings_dir).sync(
            refresh=args.refresh_sources
        )

    instrument_ingestion = InstrumentIngestion()
    reference_parser = ProvisionReferenceParser()
    guideline_ingestion = GuidelineIngestion(markdown_output_dir=workspace.markdown_guidelines_dir)
    ruling_ingestion = RulingIngestion(markdown_output_dir=workspace.markdown_rulings_dir)
    summarizer = Summarizer()
    embedder = Embedder()

    corpus = load_or_ingest_corpus(
        workspace,
        corpus_json_path,
        reference_parser,
        instrument_ingestion,
        bypass_cache=invalidate_ingestion_cache,
    )
    logger.info("Loaded corpus", instruments=len(corpus.instruments), provisions=len(corpus.provisions))

    guidelines = load_or_ingest_guidelines(
        workspace.guidelines_dir,
        guidelines_json_path,
        corpus,
        reference_parser,
        guideline_ingestion,
        bypass_cache=invalidate_ingestion_cache,
    )
    logger.info("Loaded guidelines", num_guidelines=len(guidelines.guidelines))

    # A corpus without a ruling collection ignores any stale ruling folder or cache left on disk.
    rulings = (
        load_or_ingest_rulings(
            workspace.rulings_dir,
            rulings_json_path,
            corpus,
            reference_parser,
            ruling_ingestion,
            bypass_cache=invalidate_ruling_cache,
        )
        if workspace.config.rulings is not None
        else Rulings(rulings=[])
    )
    logger.info("Loaded rulings", num_rulings=len(rulings.rulings))

    driver = Neo4jDriver(settings, corpus_id=workspace.key)
    context = GraphBuilderContext(driver=driver, corpus_id=workspace.key)
    graph_builder = CoCitationGraphBuilder(
        nodes_path=nodes_path,
        edges_path=edges_path,
        summarizer=summarizer,
        embedder=embedder,
        context=context,
    )

    instruments_by_key = {instrument.key: instrument for instrument in corpus.instruments}

    if graph_builder.has_node_cache() and not invalidate_graph_cache:
        logger.info("Loading nodes from cache", nodes_path=nodes_path)
        graph_builder.load_nodes()
    else:
        graph_builder.create_nodes(corpus.provisions, instruments_by_key=instruments_by_key, avoid_summarization=False)
        graph_builder.save_nodes()

    if graph_builder.has_edge_cache() and not invalidate_graph_cache:
        logger.info("Loading edges from cache", edges_path=edges_path)
        graph_builder.load_edges()
    else:
        graph_builder.create_edges(corpus.provisions)
        # Guideline and ruling edges are only replayed when edges are actually rebuilt, to avoid double-counting on reload.
        for guideline in guidelines.guidelines:
            graph_builder.add_arch(guideline)
        for ruling in rulings.rulings:
            graph_builder.add_arch(ruling)
        graph_builder.add_structure_arch(
            corpus.provisions,
            section_weight=settings.graph.section_weight,
            chapter_weight=settings.graph.chapter_weight,
        )
        # Degrees must reflect the finished citation graph, so the backbone runs last.
        if settings.graph.backbone_enabled:
            graph_builder.add_semantic_backbone(
                k=settings.graph.backbone_k,
                weight=settings.graph.backbone_weight,
                degree_threshold=settings.graph.backbone_degree_threshold,
                mutual=settings.graph.backbone_mutual,
            )
        graph_builder.save_edges()

    citation_builder = CitationGraphBuilder(
        nodes_path=citation_nodes_path,
        edges_path=citation_edges_path,
        summarizer=summarizer,
        embedder=embedder,
        context=context,
    )

    if citation_builder.has_node_cache() and not invalidate_graph_cache:
        logger.info("Loading citation nodes from cache", nodes_path=citation_nodes_path)
        citation_builder.load_nodes()
    else:
        # Reuse the co-citation graph's element nodes so summaries/embeddings aren't computed twice.
        citation_builder.adopt_element_nodes(graph_builder.nodes)
        citation_builder.add_nodes(guidelines.guidelines)
        citation_builder.add_nodes(rulings.rulings)
        citation_builder.save_nodes()

    if citation_builder.has_edge_cache() and not invalidate_graph_cache:
        logger.info("Loading citation edges from cache", edges_path=citation_edges_path)
        citation_builder.load_edges()
    else:
        citation_builder.create_edges(corpus.provisions)
        # Guideline and ruling arches are only replayed when edges are actually rebuilt, to avoid duplicates on reload.
        for guideline in guidelines.guidelines:
            citation_builder.add_arch(guideline)
        for ruling in rulings.rulings:
            citation_builder.add_arch(ruling)
        citation_builder.save_edges()

    # Idempotent: required for Algorithm 2's vector search, safe to run on every build.
    graph_builder.driver.ensure_schema(dimensions=embedder.dimension)
    graph_builder.sync_to_driver()
    # clear=False: the citation graph shares element nodes with the co-citation graph just written.
    citation_builder.sync_to_driver(clear=False)

    instrument_builder = InstrumentGraphBuilder(
        nodes_path=instrument_nodes_path,
        edges_path=instrument_edges_path,
        summarizer=summarizer,
        embedder=embedder,
        context=context,
    )
    if instrument_builder.has_node_cache() and instrument_builder.has_edge_cache() and not invalidate_graph_cache:
        logger.info("Loading instrument graph from cache", nodes_path=instrument_nodes_path)
        instrument_builder.load_nodes()
        instrument_builder.load_edges()
    else:
        provision_node_by_id = {node.id: node for node in graph_builder.nodes if node is not None}
        affinity_weights = {
            InstrumentRelationType.AMENDS: settings.graph.operative_affinity_amends_weight,
            InstrumentRelationType.IMPLEMENTS: settings.graph.operative_affinity_implements_weight,
            InstrumentRelationType.SUPPLEMENTS: settings.graph.operative_affinity_supplements_weight,
            InstrumentRelationType.REPEALS: settings.graph.operative_affinity_repeals_weight,
            InstrumentRelationType.REPLACES: settings.graph.operative_affinity_replaces_weight,
        }
        instrument_builder.create_instrument_nodes(corpus.instruments)
        instrument_builder.create_instrument_relation_edges(workspace.instrument_relations())
        instrument_builder.create_operative_edges(corpus.operative_relations, provision_node_by_id, affinity_weights)
        instrument_builder.save_nodes()
        instrument_builder.save_edges()
    # clear=False: instrument relation/operative edges reference provision nodes already written above.
    instrument_builder.sync_to_driver(clear=False)

    community_builder = CommunityGraphBuilder(
        nodes_path=community_nodes_path,
        edges_path=community_edges_path,
        summarizer=summarizer,
        embedder=embedder,
        context=context,
    )
    if community_builder.has_node_cache() and community_builder.has_edge_cache() and not invalidate_community_cache:
        community_builder.load_nodes()
        community_builder.load_edges()
    else:
        community_builder.nodes = set(graph_builder.nodes)
        community_builder.build(
            max_levels=settings.graph.community_max_levels,
            leiden_gamma=settings.graph.community_leiden_gamma,
            include_operative_affinity=settings.graph.operative_affinity_enabled,
        )
        community_builder.save_nodes()
        community_builder.save_edges()
    community_builder.sync_to_driver(clear=False)

    driver.close()


def main() -> None:
    """Build the graphs of the corpora selected on the command line."""
    args = parse_args()
    settings = get_single_corpus_settings()
    run_for_each_corpus(
        select_workspaces(args),
        lambda workspace: build_corpus(workspace, args, settings),
        keep_going=args.all,
        label="Corpus build",
    )


if __name__ == "__main__":
    main()
