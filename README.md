# WBPOA

Graph-based retrieval and question answering over EU legal corpora: legislation, guidelines and rulings are linked into a citation graph, clustered into communities, and searched by single-corpus and multi-corpus agents.

## Setup

Requires Python 3.14, [uv](https://docs.astral.sh/uv/), Neo4j with Graph Data Science, and an OpenAI-compatible LLM endpoint.

```sh
uv sync
```

Create a `.env` with at least `NEO4J_PASSWORD`, plus `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`, and `ZOTERO_API_KEY` for ingestion. Optional: `NEO4J_URL`, `EMBED_MODEL`, `VLM_*`, `SUMMARIZER_*`, `LANGFUSE_*`, `HF_TOKEN`. Every option is in the `*settings.py` modules under `src/wbpoa/`.

## Corpora

1. Copy `data/corpora/corpus.example.toml` to `data/corpora/<key>/corpus.toml` and fill it in.
2. For cross-corpus questions, copy `data/multi-corpora/multi_corpora.example.toml` to `data/multi-corpora/<key>/multi_corpora.toml`.

## Usage

```sh
uv run wbpoa-ingest --corpus <key>          # build the graph (or --all)
uv run wbpoa-query "question" --corpus <key>       # retrieval only
uv run wbpoa-answer "question" --corpus <key>      # single-corpus answer
uv run wbpoa-multi-answer "question" --multi-corpora <key>
uv run wbpoa-ui                                     # web UI on http://127.0.0.1:8080
```

Each command takes `--help`.
