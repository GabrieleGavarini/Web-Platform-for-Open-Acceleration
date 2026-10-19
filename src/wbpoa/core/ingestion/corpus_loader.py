from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel
from structlog import get_logger

from wbpoa.core.corpus import CorpusWorkspace
from wbpoa.core.ingestion.instrument_ingestion import InstrumentIngestion
from wbpoa.core.ingestion.provision_reference_parser import ProvisionReferenceParser
from wbpoa.core.model.element import Corpus
from wbpoa.core.model.operative_relation import OperativeRelation

logger = get_logger()


def load_or_build[T: BaseModel](
    path: Path,
    model: type[T],
    factory: Callable[[], T],
    *,
    bypass_cache: bool = False,
) -> T:
    """
    Load a cached pydantic model from a JSON file, or build it with factory and cache it if the file is missing.

    Args:
        path (Path): Where the JSON cache is stored.
        model (type[T]): The pydantic model class to (de)serialize.
        factory (Callable[[], T]): Builds the value from scratch on a cache miss.
        bypass_cache (bool): Whether to build and overwrite the cache even when it exists.

    Returns:
        T: The loaded or freshly built value.

    """
    if path.exists() and not bypass_cache:
        logger.info("Loading cached JSON", path=path)
        return model.model_validate_json(path.read_text(encoding="utf-8"))

    logger.info("Building from scratch", path=path, bypass_cache=bypass_cache)
    value = factory()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.model_dump_json(indent=2), encoding="utf-8")
    return value


def load_or_ingest_corpus(
    workspace: CorpusWorkspace,
    corpus_json_path: Path,
    reference_parser: ProvisionReferenceParser,
    instrument_ingestion: InstrumentIngestion,
    *,
    bypass_cache: bool = False,
) -> Corpus:
    """
    Load a legal-family corpus from cache or ingest all of its configured instruments.
    """

    def ingest() -> Corpus:
        instruments = []
        for config in workspace.config.instruments:
            source_path = workspace.instrument_source_paths[config.key]
            logger.info("Ingesting instrument from HTML file", instrument_key=config.key, path=source_path)
            instrument = instrument_ingestion.ingest(source_path, workspace.key, config)
            for provision in instrument.provisions:
                references = reference_parser.parse_for_reference(
                    text=provision.text,
                    instrument=instrument,
                    include_corpus_name=False,
                    source=provision,
                )
                provision.provision_references = {reference.id for reference in references}
            instruments.append(instrument)

        instruments_by_key = {instrument.key: instrument for instrument in instruments}
        operative_relations: list[OperativeRelation] = []
        for relation in workspace.instrument_relations():
            resolved = reference_parser.parse_operative_relations(
                source_instrument=instruments_by_key[relation.source],
                target_instrument=instruments_by_key[relation.target],
                relation_type=relation.type,
            )
            operative_relations.extend(resolved)
            log = logger.warning if not resolved else logger.info
            log(
                "operative_relations_resolved",
                corpus_id=workspace.key,
                source_instrument=relation.source,
                target_instrument=relation.target,
                relation_type=relation.type.value,
                relation_count=len(resolved),
            )

        logger.info(
            "corpus_operative_relations_summary",
            corpus_id=workspace.key,
            instrument_relation_pairs=len(workspace.instrument_relations()),
            total_relations=len(operative_relations),
        )

        return Corpus(
            id=workspace.key,
            name=workspace.config.name,
            instruments=instruments,
            operative_relations=operative_relations,
        )

    return load_or_build(corpus_json_path, Corpus, ingest, bypass_cache=bypass_cache)
