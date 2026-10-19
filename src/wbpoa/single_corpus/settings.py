from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from wbpoa.core.settings import PROJECT_ROOT, CoreSettings, configure_logging


class ZoteroSettings(BaseSettings):
    """Shared Zotero API and download configuration."""

    model_config = SettingsConfigDict(env_prefix="ZOTERO_", env_file=".env", extra="ignore")

    api_key: SecretStr | None = Field(default=None, description="Zotero API key used for configured collections.")
    max_retries: int = Field(default=3, ge=0, description="Retry count for transient Zotero API failures.")
    timeout_seconds: float = Field(default=180.0, gt=0, description="Per-request Zotero timeout in seconds.")
    rate_limit_seconds: float = Field(default=0.3, ge=0, description="Delay between Zotero file downloads.")


class GraphSettings(BaseSettings):
    """Citation graph configuration."""

    model_config = SettingsConfigDict(env_prefix="GRAPH_", env_file=".env", extra="ignore")

    community_max_levels: int = Field(
        default=3,
        description="Maximum number of Leiden community layers to build.",
    )
    community_leiden_gamma: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Gamma (resolution) parameter passed to Leiden community detection. "
            "Higher values generally produce smaller, more numerous communities. "
            "Needs re-tuning after fractional counting was introduced in add_arch (steps 7/11): "
            "co-citation weights now span orders of magnitude more than before, so a given "
            "gamma selects a different partition than it used to."
        ),
    )

    community_property: str = Field(
        default="communityId",
        description=(
            "Node property holding the community id written by "
            "LawGraphStore.build_communities, used to expand retrieved guideline "
            "chunks to other provisions in the same CITE/EXPLAIN community."
        ),
    )
    cite_weight: float = Field(
        default=1.0,
        description=("Weight stored on CITE relationships, which link law provisions to each other."),
    )
    explain_weight: float = Field(
        default=0.7,
        description=("Weight stored on EXPLAIN relationships, which link a guideline to the provisions it explains."),
    )
    backbone_enabled: bool = Field(
        default=True, description="Whether to add the kNN semantic backbone to the co-citation graph."
    )
    backbone_k: int = Field(
        default=1, gt=0, description="Number of nearest Article/Annex neighbours considered per backbone candidate."
    )
    backbone_weight: float = Field(
        default=0.25,
        gt=0,
        description=(
            "Weight stored on semantic_backbone edges. Matches Ruling.citation_weight (the level-2 "
            "per-source weight) -- a decisive edge, not a negligible floor, since weight only "
            "controls how strongly a merged orphan pulls the partition, not whether it merges "
            "(backbone_degree_threshold gates that)."
        ),
    )
    backbone_degree_threshold: int = Field(
        default=1, ge=0, description="Nodes with a co-citation degree at or below this are backbone candidates."
    )
    backbone_mutual: bool = Field(
        default=True,
        description=(
            "Keep a backbone edge only if each endpoint is in the other's top-k (the target's top-k "
            "is always computed over the whole node population, not restricted to provisions); "
            "otherwise cap in-degree at k."
        ),
    )
    section_weight: float | None = Field(
        default=2.0,
        gt=0,
        description=(
            "Weight budget w_s for edges between articles sharing a section: each pair gets "
            "w_s / sqrt(|S| - 1), |S| the section's article count. None disables section edges "
            "entirely."
        ),
    )
    chapter_weight: float | None = Field(
        default=1.0,
        gt=0,
        description=(
            "Weight budget w_c for edges between articles sharing a chapter: each pair gets "
            "w_c / sqrt(|C| - 1), |C| the chapter's article count. None disables chapter edges "
            "entirely."
        ),
    )
    operative_affinity_enabled: bool = Field(
        default=True,
        description="Whether to project OPERATIVE_AFFINITY edges alongside co-citation for Leiden.",
    )
    operative_affinity_amends_weight: float | None = Field(
        default=1.0,
        gt=0,
        description="Weight stored on OPERATIVE_AFFINITY edges derived from an amends relation. None disables them.",
    )
    operative_affinity_implements_weight: float | None = Field(
        default=1.0,
        gt=0,
        description="Weight stored on OPERATIVE_AFFINITY edges derived from an implements relation. None disables them.",
    )
    operative_affinity_supplements_weight: float | None = Field(
        default=1.0,
        gt=0,
        description="Weight stored on OPERATIVE_AFFINITY edges derived from a supplements relation. None disables them.",
    )
    operative_affinity_repeals_weight: float | None = Field(
        default=1.0,
        gt=0,
        description="Weight stored on OPERATIVE_AFFINITY edges derived from a repeals relation. None disables them.",
    )
    operative_affinity_replaces_weight: float | None = Field(
        default=1.0,
        gt=0,
        description="Weight stored on OPERATIVE_AFFINITY edges derived from a replaces relation. None disables them.",
    )


class VLMSettings(BaseSettings):
    """
    VLM (PDF -> Markdown extraction via docling) configuration.

    The converter reuses ``LLMSettings.base_url`` and ``LLMSettings.model`` so
    there is a single model endpoint to configure. These settings control
    conversion behaviour only.
    """

    model_config = SettingsConfigDict(env_prefix="VLM_", env_file=".env", extra="ignore")

    timeout: int = Field(default=300, description="Timeout for VLM conversion, in seconds.")
    scale: float = Field(default=1.0, description="Scale factor for VLM conversion.")
    concurrency: int = Field(default=4, description="Concurrency level for VLM conversion.")
    default_pdf_parser: Literal["pypdfium2", "docling", "vlm"] = Field(
        default="pypdfium2",
        description=(
            "Starting tier of the PDF text-extraction cascade: 'pypdfium2' (plain text-layer read, "
            "cheapest) -> 'docling' (native pass with OCR/table-structure disabled) -> 'vlm' (remote "
            "VLM pipeline, most expensive). Selecting a later tier skips the earlier ones entirely; "
            "the cascade still falls through to later tiers if the selected one yields no text."
        ),
    )
    enable_thinking: bool = Field(
        default=False,
        description=(
            "Disable the model's 'thinking' mode during Markdown extraction: reasoning is "
            "useless for a verbatim transcription and only slows it down. For Qwen3 served "
            "via vLLM this is passed as ``chat_template_kwargs={'enable_thinking': False}``."
        ),
    )
    prompt: str = Field(
        default=(
            "Convert this document page to clean Markdown. "
            "Preserve the structure: use # for headings, proper markdown tables "
            "with | for tabular data, bullet/numbered lists, and bold/italic. "
            "Do not summarize, do not omit any text. Output only the Markdown."
        ),
        description="Prompt sent to the VLM for PDF -> Markdown extraction.",
    )


class SummarizerSettings(BaseSettings):
    """Token budgeting for the map-reduce document summarizer (wbpoa/single_corpus/ingestion/utils/summarizer.py)."""

    model_config = SettingsConfigDict(env_prefix="SUMMARIZER_", env_file=".env", extra="ignore")

    enable_thinking: bool = Field(
        default=False,
        description=(
            "Disable the model's 'thinking' mode during summarization. For Qwen3 served "
            "via vLLM this is passed as ``chat_template_kwargs={'enable_thinking': False}``."
        ),
    )
    token_safety_margin: int = Field(
        default=2000,
        description=(
            "Fixed number of tokens reserved for the system prompt and document wrapper "
            "(constant overhead, independent of chunk size)."
        ),
    )
    token_estimate_safety_factor: float = Field(
        default=0.85,
        gt=0.0,
        le=1.0,
        description=(
            "Multiplier applied to the chunking budget to absorb the estimate (cl100k, a different "
            "tokenizer than the served model's) undercounting the actual token count. Token-dense "
            "text (URLs, footnote numbers, non-ASCII punctuation like curly quotes and soft hyphens) "
            "can tokenize to noticeably more tokens under the model's own tokenizer than under "
            "cl100k. This discrepancy scales with chunk size, so it is applied as a percentage "
            "rather than folded into the fixed token_safety_margin."
        ),
    )
    reserved_output_tokens: int = Field(
        default=2048,
        description=(
            "Tokens reserved for the summary output. A single-paragraph 120-200 word summary needs "
            "a few hundred tokens; this is a generous, fixed reservation independent of "
            "LLMSettings.max_tokens, which the summarizer doesn't forward to the model call and can "
            "be configured far larger than context_length."
        ),
    )
    min_input_tokens: int = Field(
        default=4096,
        description=(
            "Floor for the chunking budget, so a misconfigured or unusually small context_length "
            "can't drive the effective input budget to zero or negative and stall chunking."
        ),
    )
    token_encoding: str = Field(default="cl100k_base", description="tiktoken encoding used to estimate token counts.")


class IngestionSettings(BaseSettings):
    """Page-count limits applied before running PDF ingestion."""

    model_config = SettingsConfigDict(env_prefix="INGESTION_", env_file=".env", extra="ignore")

    ruling_max_pages: int | None = Field(
        default=20,
        description="Maximum page count for a ruling PDF to be ingested; longer rulings are skipped. None disables the limit.",
    )
    guideline_max_pages: int | None = Field(
        default=None,
        description=(
            "Maximum page count for a guideline PDF to be ingested; longer guidelines are skipped. "
            "None disables the limit."
        ),
    )


class SingleCorpusSettings(CoreSettings):
    """
    Settings for single-corpus work: the shared core plus ingestion and graph construction.

    Only a single corpus is ever ingested and built at a time, so the Zotero, graph, VLM,
    summarizer, and PDF-ingestion groups live here rather than in `CoreSettings`.
    """

    zotero: ZoteroSettings = Field(default_factory=ZoteroSettings, description="Zotero API configuration.")
    graph: GraphSettings = Field(default_factory=GraphSettings, description="Citation graph configuration.")
    vlm: VLMSettings = Field(
        default_factory=VLMSettings, description="VLM (PDF -> Markdown extraction via docling) configuration."
    )
    summarizer: SummarizerSettings = Field(
        default_factory=SummarizerSettings, description="Document summarizer token budgeting configuration."
    )
    ingestion: IngestionSettings = Field(
        default_factory=IngestionSettings, description="Page-count limits applied before running PDF ingestion."
    )
    pdf_dir: Path = Field(default=Path("data/pdf"), description="Directory for PDF files.")
    md_dir: Path = Field(default=Path("data/md/vlm"), description="Directory for Markdown files extracted by VLM.")

    @model_validator(mode="after")
    def _resolve_paths(self) -> Self:
        """Resolve relative paths against the project root instead of the CWD."""
        for name in ("md_dir", "pdf_dir"):
            value: Path = getattr(self, name)
            if not value.is_absolute():
                setattr(self, name, (PROJECT_ROOT / value).resolve())
        return self


def get_single_corpus_settings() -> SingleCorpusSettings:
    """
    Create a SingleCorpusSettings instance reading the current environment, and configure logging from it.

    Returns:
        The single-corpus settings.

    """
    settings = SingleCorpusSettings()
    configure_logging(settings.log_level)
    return settings
