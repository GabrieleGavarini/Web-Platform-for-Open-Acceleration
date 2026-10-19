from __future__ import annotations

from pydantic import BaseModel, Field

from wbpoa.core.generation.models import RedactedNode, SingleCorpusResult


class MultiCorporaSubAgentCall(BaseModel):
    """One call the multi-corpora manager made to a sub agent, successful or not."""

    corpus_key: str = Field(description="Key of the sub-corpus whose single-corpus agent was called.")
    question: str = Field(description="The sub-question the manager wrote for that corpus.")
    result: SingleCorpusResult | None = Field(
        default=None, description="The sub agent's full single-corpus result; None when the call failed."
    )
    error: str | None = Field(default=None, description="Why the call failed; None when it succeeded.")


class MultiCorporaResult(BaseModel):
    """The multi-corpora manager's answer, plus every sub-agent call it made, for logging/inspection."""

    query: str
    answer: str
    sub_agent_calls: list[MultiCorporaSubAgentCall] = Field(default_factory=list)
    grounding_context: list[RedactedNode] = Field(
        default_factory=list,
        description=(
            "Union of every successful sub-agent call's grounding context, deduplicated by node id in "
            "first-seen order: the provisions the answer may cite."
        ),
    )
