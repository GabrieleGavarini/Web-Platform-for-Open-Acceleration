from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import tiktoken
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, message_chunk_to_message
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from structlog import get_logger

from wbpoa.core.settings import LLMSettings

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

log = get_logger()

# Fixed-margin token budgeting, the same shape as Summarizer's but without a configurable
# percentage safety factor: Algorithm 3's prompts are agent instructions plus a handful of
# retrieved passages, not a single long document, so a flat reservation is precise enough.
_TIKTOKEN_ENCODING_NAME = "cl100k_base"
_RESERVED_OUTPUT_TOKENS = 2048
_TOKEN_SAFETY_MARGIN = 1000
_MIN_INPUT_TOKENS = 2048
_TRUNCATION_MARKER = " …[truncated]"

# Qwen3, when 'thinking' is enabled, emits its reasoning inline as a leading
# <think>...</think> block ahead of the real content -- there is no separate response
# field for it over the OpenAI-compatible API this served model exposes.
_THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_CONTENT_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_THINK_OPEN = "<think>"

_encoding = tiktoken.get_encoding(_TIKTOKEN_ENCODING_NAME)


@runtime_checkable
class ChatModel(Protocol):
    """Structural protocol for the object each Algorithm 3 agent calls to talk to the LLM. ChatOpenAI satisfies it."""

    def invoke(self, messages: list[BaseMessage], /, config: RunnableConfig | None = None) -> BaseMessage: ...


class StreamingRunnable(Protocol):
    """Anything whose reply can be streamed chunk by chunk; both tool-calling protocols below satisfy it."""

    def stream(self, messages: list[BaseMessage], /, config: RunnableConfig | None = None) -> Iterator[BaseMessage]: ...


class ToolCallingRunnable(Protocol):
    """The methods a tool-calling agent calls on the runnable `bind_tools` returns."""

    def invoke(self, messages: list[BaseMessage], /, config: RunnableConfig | None = None) -> AIMessage: ...

    def stream(self, messages: list[BaseMessage], /, config: RunnableConfig | None = None) -> Iterator[AIMessage]: ...


class ToolBindingChatModel(Protocol):
    """Structural protocol for a tool-calling agent's model: plain and streamed calls plus tool binding. ChatOpenAI satisfies it."""

    def invoke(self, messages: list[BaseMessage], /, config: RunnableConfig | None = None) -> BaseMessage: ...

    def stream(self, messages: list[BaseMessage], /, config: RunnableConfig | None = None) -> Iterator[BaseMessage]: ...

    def bind_tools(self, tools: Sequence[dict[str, object]], /) -> ToolCallingRunnable: ...


def build_chat_model(
    llm_settings: LLMSettings, *, temperature: float | None = None, enable_thinking: bool | None = None
) -> ChatOpenAI:
    """
    Build a ChatOpenAI client against LLMSettings, shared by every Algorithm 3 agent.

    Mirrors Summarizer._build_llm (wbpoa/single_corpus/ingestion/utils/summarizer.py), the codebase's only
    other LLM-call site, so all model construction goes through one idiom.

    Args:
        llm_settings: Endpoint/model configuration (base_url, model, api_key).
        temperature: Overrides llm_settings.temperature when given.
        enable_thinking: When not None, forces chat_template_kwargs.enable_thinking via extra_body.
            When None, no override is sent and the served model's default behaviour applies.

    Returns:
        A configured ChatOpenAI instance.

    """
    api_key = llm_settings.api_key if isinstance(llm_settings.api_key, SecretStr) else SecretStr(llm_settings.api_key)
    kwargs: dict[str, Any] = {}
    if enable_thinking is not None:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}

    log.info("llm_initialized", model=llm_settings.model, base_url=llm_settings.base_url)
    return ChatOpenAI(
        base_url=llm_settings.base_url,
        model=llm_settings.model,
        api_key=api_key,
        temperature=temperature if temperature is not None else llm_settings.temperature,
        timeout=llm_settings.timeout,
        **kwargs,
    )


def _flatten_content(content: str | list[str | dict[str, Any]]) -> str:
    """
    Flatten a ChatOpenAI message's raw content into plain text.

    Handles both a plain string content and a list of content blocks (e.g. some
    endpoints return ``[{"type": "text", "text": "..."}]``), mirroring
    Summarizer._extract_text.

    Returns:
        The unstripped, unfiltered text -- including any inline ``<think>`` block.

    """
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(parts)


def extract_text(response: BaseMessage) -> str:
    """
    Flatten a ChatOpenAI response's content into plain text, with thinking removed.

    Args:
        response: The message returned by a ChatOpenAI call.

    Returns:
        The response text with any inline ``<think>...</think>`` block (emitted when
        the served model's thinking mode is on, see ``extract_thinking``) removed, and
        stripped of leading/trailing whitespace.

    """
    text = _flatten_content(response.content)
    return _THINK_BLOCK_PATTERN.sub("", text).strip()


def extract_thinking(response: BaseMessage) -> str | None:
    """
    Extract a ChatOpenAI response's inline thinking block, if present.

    Args:
        response: The message returned by a ChatOpenAI call.

    Returns:
        The text inside the leading ``<think>...</think>`` block, stripped of
        leading/trailing whitespace, or None if the response has no such block (e.g.
        thinking mode was off, or the served model didn't emit one).

    """
    match = _THINK_CONTENT_PATTERN.search(_flatten_content(response.content))
    return match.group(1).strip() if match else None


def visible_text(raw: str) -> str:
    """
    The part of a reply still being streamed that a reader should see, thinking removed.

    The streaming counterpart of `extract_text`: a thinking block may still be open.

    Args:
        raw: The reply's text so far.

    Returns:
        `raw` without its closed ``<think>...</think>`` blocks, cut at a ``<think>`` not yet
        closed, and stripped of leading/trailing whitespace.

    """
    text = _THINK_BLOCK_PATTERN.sub("", raw)
    return text.split(_THINK_OPEN, 1)[0].strip()


def stream_message(
    model: StreamingRunnable,
    messages: list[BaseMessage],
    config: RunnableConfig | None,
    on_text: Callable[[str], None],
) -> AIMessage:
    """
    Call `model` streaming its reply, sending the visible text to `on_text` as it grows.

    Text stops being sent once the reply starts calling a tool. A reply that ends up calling a
    tool after text was sent sends ``""``, since that text was not an answer.

    Args:
        model: The chat model or tool-bound runnable to call.
        messages: The conversation so far.
        config: LangChain run config for tracing.
        on_text: Called with the visible text so far (see `visible_text`) each time it changes.

    Returns:
        The whole reply, its tool calls parsed as `invoke` would return them.

    Raises:
        TypeError: If the model streams something other than message chunks.

    """
    merged: AIMessageChunk | None = None
    shown = ""
    for chunk in model.stream(messages, config=config):
        if not isinstance(chunk, AIMessageChunk):
            message = f"Expected AIMessageChunk from a streamed model, got {type(chunk).__name__}"
            raise TypeError(message)
        merged = chunk if merged is None else merged + chunk
        if merged.tool_call_chunks:
            continue
        text = visible_text(_flatten_content(merged.content))
        if text != shown:
            shown = text
            on_text(text)
    reply = message_chunk_to_message(merged) if merged is not None else AIMessage(content="")
    if not isinstance(reply, AIMessage):
        message = f"Expected AIMessage from a streamed model, got {type(reply).__name__}"
        raise TypeError(message)
    if reply.tool_calls and shown:
        on_text("")
    return reply


def count_tokens(text: str) -> int:
    """
    Estimate the number of tokens in `text`.

    Uses a general-purpose tokenizer, not the served model's own, so this is an estimate --
    same caveat as Summarizer._count_tokens.

    Args:
        text: The text to estimate.

    Returns:
        The estimated token count.

    """
    return len(_encoding.encode(text))


def max_input_tokens(llm_settings: LLMSettings) -> int:
    """
    The token budget available for prompt input, reserving room for the model's output.

    Args:
        llm_settings: Endpoint/model configuration; only `context_length` is used.

    Returns:
        `context_length` minus a fixed reservation for output and overhead, floored at
        `_MIN_INPUT_TOKENS` so a small `context_length` can't drive the budget to zero or negative.

    """
    return max(llm_settings.context_length - _RESERVED_OUTPUT_TOKENS - _TOKEN_SAFETY_MARGIN, _MIN_INPUT_TOKENS)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate `text` to at most `max_tokens` estimated tokens.

    Args:
        text: The text to truncate.
        max_tokens: The token budget; must be non-negative.

    Returns:
        `text` unchanged if it already fits, otherwise its first `max_tokens` tokens with a
        truncation marker appended.

    """
    tokens = _encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _encoding.decode(tokens[:max_tokens]) + _TRUNCATION_MARKER


def budget_texts(texts: Sequence[str], total_budget: int) -> list[str]:
    """
    Truncate `texts` so their combined estimated token count stays within `total_budget`.

    Each text over budget is cut to an equal share of `total_budget` (an even split, not
    proportional to each text's own length) -- simple and order-independent, at the cost of
    truncating a short text that didn't need it once even one other text is long enough to
    blow the joint budget.

    Args:
        texts: The candidate texts, in the order they'll be rendered.
        total_budget: The combined token budget across all of `texts`.

    Returns:
        `texts` unchanged if their combined token count already fits `total_budget`, otherwise
        each text truncated to `total_budget // len(texts)` tokens, in the same order.

    """
    if not texts:
        return list(texts)
    total = sum(count_tokens(text) for text in texts)
    if total <= total_budget:
        return list(texts)
    per_text_budget = max(total_budget // len(texts), 0)
    return [truncate_to_tokens(text, per_text_budget) for text in texts]
