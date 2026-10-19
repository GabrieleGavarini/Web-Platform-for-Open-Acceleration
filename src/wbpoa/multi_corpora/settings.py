from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from wbpoa.core.settings import CoreSettings, configure_logging


class MultiCorporaManagerSettings(BaseSettings):
    """Multi-corpus agent parameters: one manager delegating to two per-corpus single-corpus agents."""

    model_config = SettingsConfigDict(env_prefix="MULTI_CORPORA_", env_file=".env", extra="ignore")

    max_calls_per_sub_agent: int = Field(
        default=2,
        gt=0,
        description=(
            "Maximum number of times the manager may call each sub agent for one question. Every call is "
            "a full single-corpus agent run on that sub agent's corpus, so this bounds the cost of a question."
        ),
    )
    max_steps: int = Field(
        default=8,
        gt=0,
        description=(
            "Maximum number of tool-calling turns the manager takes before it is forced to write its "
            "final answer from the sub answers gathered so far."
        ),
    )
    manager_enable_thinking: bool = Field(
        default=False,
        description=(
            "Whether the manager's calls enable the model's 'thinking' mode. Off by default, matching "
            "GenerationSettings.generation_enable_thinking; for Qwen3 served via Ollama this is passed as "
            "``chat_template_kwargs={'enable_thinking': ...}``."
        ),
    )


class MultiCorporaSettings(CoreSettings):
    """Settings for multi-corpora work: the shared core plus the multi-corpora manager group."""

    multi_corpora: MultiCorporaManagerSettings = Field(
        default_factory=MultiCorporaManagerSettings,
        description="Multi-corpus agent (two per-corpus single-corpus agents) parameters.",
    )


def get_multi_corpora_settings() -> MultiCorporaSettings:
    """
    Create a MultiCorporaSettings instance reading the current environment, and configure logging from it.

    Returns:
        The multi-corpora settings.

    """
    settings = MultiCorporaSettings()
    configure_logging(settings.log_level)
    return settings
