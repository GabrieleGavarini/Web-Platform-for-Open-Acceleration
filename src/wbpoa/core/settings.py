from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import structlog
import torch
from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

from wbpoa.core.corpus.corpus_config import InstrumentRole

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _default_neo4j_settings() -> Neo4jSettings:
    # Neo4jSettings.password is a required field, so the class itself is not a
    # valid zero-argument factory as far as the type checker is concerned; the
    # value is actually supplied via NEO4J_PASSWORD at runtime.
    return Neo4jSettings()  # type: ignore[call-arg]


class Neo4jSettings(BaseSettings):
    """Neo4j database configuration."""

    model_config = SettingsConfigDict(env_prefix="NEO4J_", env_file=".env", extra="ignore")

    url: str = Field(default="neo4j://localhost:7687", description="Neo4j database URL.")
    user: str = Field(default="neo4j", description="Neo4j database user.")
    password: SecretStr = Field(description="Neo4j database password.")
    database: str = Field(default="neo4j", description="Neo4j database name.")


class EmbeddingSettings(BaseSettings):
    """Embeddings configuration."""

    model_config = SettingsConfigDict(env_prefix="EMBED_", env_file=".env", extra="ignore")

    model: str = Field(default="BAAI/bge-m3", description="Embedding model name.")
    device: str = Field(
        default_factory=_default_device,
        description="Device to run embeddings on (e.g., 'cuda' or 'cpu').",
    )
    normalize: bool = Field(default=True, description="Whether to normalize embeddings.")


class LangfuseSettings(BaseSettings):
    """Langfuse configuration. Optional: leave unset to run without tracing."""

    model_config = SettingsConfigDict(env_prefix="LANGFUSE_", env_file=".env", extra="ignore")

    public_key: SecretStr | None = Field(default=None, description="Langfuse public key.")
    secret_key: SecretStr | None = Field(default=None, description="Langfuse secret key.")
    host: str = Field(
        default="http://localhost:3000",
        description="Langfuse host URL.",
    )


class LLMSettings(BaseSettings):
    """
    Conversational LLM (chat agent, OpenAI-compatible endpoint) configuration.

    Text-only model used by the citation-aware chat agents.
    Thinking-capable: it reasons before answering. Served locally by Ollama via
    its OpenAI-compatible API.
    """

    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", extra="ignore")

    base_url: str = Field(
        default="http://localhost:11434/v1", description="Base URL of the OpenAI-compatible endpoint."
    )
    model: str = Field(default="qwen3.5:122b", description="LLM model name.")
    # Ollama's OpenAI-compatible API ignores this value; kept as a placeholder,
    # not a real credential.
    api_key: str = Field(default="ollama", description="LLM API key.")
    temperature: float = Field(
        default=0.6,
        description=(
            "Sampling temperature. Qwen3 recommends 0.6 for 'thinking' mode: greedy "
            "decoding (temperature 0) causes infinite repetitions in the reasoning."
        ),
    )
    top_p: float = Field(default=0.95, description="Top-p sampling parameter.")
    top_k: int = Field(default=20, description="Top-k sampling parameter.")
    presence_penalty: float = Field(default=1.0, description="Presence penalty parameter.")
    max_tokens: int = Field(default=8192, description="Maximum number of tokens to generate.")
    context_length: int = Field(
        default=65536,
        description="Maximum total (prompt + completion) token count the served model accepts.",
    )
    timeout: float = Field(
        default=120.0,
        description=(
            "Per-request timeout for a single LLM call, in seconds, after which the client raises "
            "APITimeoutError instead of blocking indefinitely. Applies to every ChatOpenAI call built "
            "from these settings (Algorithm 3's agents, the summarizer)."
        ),
    )


class InferenceSettings(BaseSettings):
    """Algorithm 2 (inference/retrieval) parameters."""

    model_config = SettingsConfigDict(env_prefix="INFERENCE_", env_file=".env", extra="ignore")

    h0: int = Field(default=1, ge=1, description="Community graph layer to select candidate communities from.")
    k: int = Field(default=5, gt=0, description="Number of seed communities, and final selected communities.")
    m: int = Field(
        default=3, ge=0, description="Number of neighbour communities expanded from each seed; 0 disables expansion."
    )
    n: int = Field(default=10, gt=0, description="Number of layer-0 elements used for the bottom-up vote.")
    vote_top_j: int = Field(
        default=3,
        gt=0,
        description=(
            "Number of top per-community element similarities averaged into vote(q,C), zero-padded "
            "when a community has fewer; j=1 reduces the vote to a plain max, making it size-invariant."
        ),
    )
    r: int = Field(default=5, gt=0, description="Number of top-ranked elements returned as relevant documents.")
    ell_elements: int = Field(
        default=5, ge=0, description="Number of element (Article/Recital/Annex) expansion nodes to keep."
    )
    ell_docs: int = Field(default=5, ge=0, description="Number of document (Guideline/Ruling) expansion nodes to keep.")
    delta: float = Field(
        default=0.7, ge=0, le=1, description="Per-hop decay applied to xsim; 1.0 disables decay (control for sweeps)."
    )
    max_hops: int = Field(default=3, gt=0, description="Maximum citation hops followed by the cited element closure.")
    relation_category: tuple[InstrumentRole, ...] | None = Field(
        default=None,
        description="Default instrument-role filter for retrieval; None searches every active instrument.",
    )
    operative_expansion_enabled: bool = Field(
        default=True,
        description="Master switch for one-way base-to-newer operative expansion, alongside citation expansion.",
    )
    operative_decay: float = Field(
        default=0.85,
        ge=0,
        le=1,
        description="Flat decay applied once to xsim for operative-expansion results (single-hop, no depth exponent).",
    )
    alpha: float = Field(
        default=0.6, ge=0, le=1, description="Mixing weight between element similarity and citation weight."
    )
    beta: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description=(
            "Mixing weight between community similarity and bottom-up vote. Needs re-tuning after "
            "fractional counting was introduced in add_arch (steps 7/11): the resulting partition "
            "change shifts the cauth distribution, same as any other backbone/gamma sweep."
        ),
    )
    gamma: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Mixing weight between semantic and lexical similarity; 1.0 recovers purely semantic scoring.",
    )
    include_rulings: bool = Field(
        default=True, description="Whether citing rulings, not just guidelines, are admitted into the expansion set."
    )
    candidate_pool_multiplier: int = Field(
        default=20,
        gt=0,
        description=(
            "Over-fetch factor applied to k/n when a vector search is post-filtered by layer, so "
            "enough candidates survive the filter."
        ),
    )
    rerank_enabled: bool = Field(
        default=False,
        description=(
            "Master switch for the cascade stages 4(ii) and 7. Off: the algorithm reduces exactly "
            "to the pre-reranker formulation. Keep False as the A/B control until a gold query set exists."
        ),
    )
    rerank_model: str = Field(
        default="Alibaba-NLP/gte-reranker-modernbert-base",
        description=(
            "Cross-encoder used for rr(q, u). ModernBERT-class for its long window (Remark 12); "
            "BAAI/bge-reranker-v2-m3 is the multilingual alternative at a 512-token cost."
        ),
    )
    rerank_device: str = Field(
        default_factory=_default_device,
        description="Device to load the reranker model on (e.g. 'cuda' or 'cpu'), same resolution as EmbeddingSettings.",
    )
    rerank_pool: int = Field(
        default=25,
        gt=0,
        description=(
            "Shortlist size p handed to the reranker, per pool. Must satisfy p >= r at step 4 and "
            "p >= ell_elements/ell_docs at step 7, or the reranker can only reorder what the first "
            "pass already chose instead of rescuing anything."
        ),
    )
    rho: float = Field(
        default=0.7,
        ge=0,
        le=1,
        description=(
            "Rerank blend: rho*rsim + (1-rho)*first-pass score. 0.0 = reranker computed but ignored "
            "(equivalent to rerank_enabled=False except for the wasted forward passes). Tune jointly "
            "with alpha, which over-weights similarity once the reranker absorbs part of that job."
        ),
    )
    rerank_field: Literal["text", "summary"] = Field(
        default="text",
        description=(
            "Whether rr scores the node's full text (Remark 12; chunked and max-reduced if it "
            "exceeds the model's window) or its summary (never truncated, but largely redundant "
            "with the semantic channel)."
        ),
    )
    document_lexical_field: Literal["text", "summary"] = Field(
        default="text",
        description=(
            "What the BM25 (lexical) channel indexes for the document expansion pool (guidelines and "
            "rulings): the full text, which favours long documents that mention every term, or the "
            "summary, matching the semantic channel, which already embeds the summary. Elements "
            "(articles, recitals, annexes) always use their full text."
        ),
    )
    rerank_expansion: bool = Field(
        default=True,
        description=(
            "Whether step 7 reranks its two expansion pools as well as step 4. False: reranking "
            "applies to the relevant elements only, the cheaper half of the win."
        ),
    )

    @computed_field(description="ell_elements + ell_docs, kept for logging/back-compat only.")  # type: ignore[prop-decorator]
    @property
    def ell(self) -> int:
        return self.ell_elements + self.ell_docs


class GenerationSettings(BaseSettings):
    """Algorithm 3 (single-corpus agent over its search tool: reformulation, retrieval, redaction) parameters."""

    model_config = SettingsConfigDict(env_prefix="GENERATION_", env_file=".env", extra="ignore")

    reformulation_enabled: bool = Field(
        default=True,
        description=(
            "Master switch for step 1. Off: q' = q, Algorithm 2 runs on the raw query verbatim -- "
            "the A/B control against reformulation, and cheaper (one fewer LLM call)."
        ),
    )
    reformulation_enable_thinking: bool = Field(
        default=False,
        description=(
            "Disable the model's 'thinking' mode during step 1: reformulation is a short rewriting "
            "task, and reasoning is pure overhead on the request's critical path. For Qwen3 served "
            "via Ollama this is passed as ``chat_template_kwargs={'enable_thinking': False}``; when "
            "enabled, thinking is emitted inline in the response as a leading "
            "``<think>...</think>`` block, mirroring VLMSettings.enable_thinking and "
            "SummarizerSettings.enable_thinking."
        ),
    )
    redaction_enabled: bool = Field(
        default=True,
        description=(
            "Master switch for step 4. Off: S = U, the full candidate pool goes straight to "
            "generation -- cheaper (one fewer LLM call) but reintroduces the precision problem of Remark 14."
        ),
    )
    redaction_field: Literal["text", "summary"] = Field(
        default="summary",
        description=(
            "Whether the redaction agent reads (and each kept node's RedactedNode.text is populated with) "
            "each node's full text or its summary, mirroring InferenceSettings.rerank_field."
        ),
    )
    redaction_temperature: float = Field(
        default=0.2,
        description=(
            "Sampling temperature for step 4, independent of LLMSettings.temperature. Lower than "
            "LLMSettings.temperature's default since redaction is a single-correct-filter task, not "
            "open-ended generation (Remark 18)."
        ),
    )
    redaction_enable_thinking: bool = Field(
        default=False,
        description=(
            "Disable the model's 'thinking' mode during redaction: step 4 reduces to picking which "
            "candidate ids to keep, and reasoning only slows that down. For Qwen3 served via Ollama "
            "this is passed as ``chat_template_kwargs={'enable_thinking': False}``; when enabled, "
            "thinking is emitted inline in the response as a leading ``<think>...</think>`` block, "
            "mirroring VLMSettings.enable_thinking and SummarizerSettings.enable_thinking."
        ),
    )
    document_field: Literal["text", "summary", "excerpt"] = Field(
        default="excerpt",
        description=(
            "What guideline and ruling passages returned to the single-corpus agent carry: 'excerpt' "
            "(the chunks of the document most relevant to the research question, by BM25), 'summary' "
            "(what the document is about), or 'text' (the whole document, which crowds the tool "
            "result's token budget and truncates every passage to an equal share)."
        ),
    )
    document_excerpt_tokens: int = Field(
        default=300, gt=0, description="Maximum estimated tokens per excerpt chunk of a guideline or ruling."
    )
    document_excerpts: int = Field(
        default=2, gt=0, description="Maximum number of excerpt chunks shown per guideline or ruling."
    )
    max_lookups: int = Field(
        default=10,
        ge=0,
        description=(
            "Maximum number of lookup_provision calls per question: direct fetches of one specific "
            "provision, allowed only after the first search. 0 disables the tool."
        ),
    )
    max_plan_questions: int = Field(
        default=3,
        gt=0,
        description=(
            "Maximum number of broad questions the research planner writes for the first round (also "
            "capped by max_searches). Fewer, more distinct questions leave calls for follow-ups."
        ),
    )
    duplicate_question_threshold: float = Field(
        default=0.9,
        gt=0.0,
        le=1.0,
        description=(
            "Embedding cosine similarity at or above which a research question counts as a near-duplicate "
            "of one already asked in the run; near-duplicates are not researched. 1.0 only blocks exact "
            "repeats."
        ),
    )
    max_leads: int = Field(
        default=10,
        ge=0,
        description=(
            "Maximum number of leads -- provisions cited by the returned guidelines/rulings that no "
            "search has retrieved yet -- listed under each research result, most-cited first. "
            "0 disables leads."
        ),
    )
    max_context_nodes: int | None = Field(
        default=None,
        description=(
            "Optional hard cap on |S|, applied after step 4 by truncating to the redaction agent's own "
            "top-scoring order if it returns more. None trusts the redaction agent's judgement entirely."
        ),
    )
    generation_enable_thinking: bool = Field(
        default=False,
        description=(
            "Disable the model's 'thinking' mode for the single-corpus agent, which writes the research "
            "questions and the final answer. Off by default even though LLMSettings.temperature is tuned for Qwen3's "
            "thinking mode, so reasoning must be opted into explicitly. For Qwen3 served via Ollama this "
            "is passed as ``chat_template_kwargs={'enable_thinking': False}``; when enabled, thinking is "
            "emitted inline in the response as a leading ``<think>...</think>`` block, mirroring "
            "VLMSettings.enable_thinking and SummarizerSettings.enable_thinking."
        ),
    )
    planning_enabled: bool = Field(
        default=True,
        description=(
            "Whether the single-corpus agent starts with a research plan: the ResearchPlanner breaks the "
            "question down into its legal issues, one broad question each, and all are researched as the "
            "first round before the agent reviews and refines. Off: the agent writes its own first calls."
        ),
    )
    max_searches: int = Field(
        default=6,
        gt=0,
        description=(
            "Maximum number of times the single-corpus agent may call its search tool for one "
            "question. Every call is a full reformulation -> Algorithm 2 -> redaction pass, so this "
            "bounds the cost of a question; further calls get a 'limit reached' tool result."
        ),
    )
    max_steps: int = Field(
        default=8,
        gt=0,
        description=(
            "Maximum number of tool-calling turns the single-corpus agent takes before it is forced to "
            "write its final answer from the passages gathered so far."
        ),
    )


class CoreSettings(BaseSettings):
    """
    Configuration shared by every mode: storage, models, tracing, retrieval, and generation.

    Environment variable names correspond to the nested subclass prefixes
    (e.g. ``NEO4J_URL``, ``EMBED_MODEL``, ``LLM_BASE_URL``, ``ACT_NAME``). Mode-specific settings
    extend this class: `SingleCorpusSettings` adds the ingestion and graph-build groups, and
    `MultiCorporaSettings` adds the multi-corpora manager group.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    log_level: LogLevel = Field(default="INFO", description="Minimum level emitted by application logs.")

    hf_token: SecretStr | None = Field(
        default=None,
        description="Optional Hugging Face access token used by gated embedding and reranker models.",
    )
    neo4j: Neo4jSettings = Field(default_factory=_default_neo4j_settings, description="Neo4j database configuration.")
    embed: EmbeddingSettings = Field(default_factory=EmbeddingSettings, description="Embeddings configuration.")
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings, description="Langfuse configuration.")
    llm: LLMSettings = Field(
        default_factory=LLMSettings,
        description="Conversational LLM (chat agent, OpenAI-compatible endpoint) configuration.",
    )
    act_name: str = Field(default="AI Act", description="Law citations act name.")
    inference: InferenceSettings = Field(
        default_factory=InferenceSettings, description="Algorithm 2 (inference/retrieval) parameters."
    )
    generation: GenerationSettings = Field(
        default_factory=GenerationSettings, description="Algorithm 3 (retrieval-augmented generation) parameters."
    )


def get_core_settings() -> CoreSettings:
    """
    Create a CoreSettings instance reading the current environment, and configure logging from it.

    Returns:
        The shared settings.

    """
    settings = CoreSettings()
    configure_logging(settings.log_level)
    return settings


def configure_logging(level: LogLevel) -> None:
    """Configure structlog to emit records at or above ``level``.

    Args:
        level: Minimum log level to emit.
    """
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)))
