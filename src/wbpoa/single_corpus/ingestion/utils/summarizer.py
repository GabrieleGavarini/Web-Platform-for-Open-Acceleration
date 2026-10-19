from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Final, cast

import tiktoken
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from structlog import get_logger

from wbpoa.core.ingestion.utils.chunking import split_into_chunks
from wbpoa.single_corpus.settings import get_single_corpus_settings

log = get_logger()


class SummarizationError(RuntimeError):
    """Raised when the model call fails or returns unusable content."""


class Summarizer:
    """
    Summarize legal documents in a single short paragraph, backed by an OpenAI-compatible chat endpoint.

    Documents that would exceed the served model's context window are handled transparently
    through a map-reduce strategy: the text is split into token-bounded chunks, each chunk is
    summarized independently (optionally batched), and the resulting partial summaries are
    recursively combined until they fit in a single final summarization call.
    """

    SUMMARIZATION_PROMPT: Final = """\
    You are a legal analyst. You summarize legal documents — contracts, court \
    filings, judgments, statutes, regulations, policies, and legal correspondence — \
    faithfully and neutrally.

    Produce a single paragraph of 120-200 words. No headings, no bullets, no lists.

    Cover, in this order and only where the document supports it: the document type \
    and its parties, the subject matter and purpose, the two or three most \
    consequential terms (or, for a decision, the holding and disposition), and the \
    dates that govern the document's operation. Keep the concrete details that give \
    the summary its value — amounts, deadlines, notice periods, clause references — \
    rather than describing them in general terms. Skip boilerplate.

    Rules:
    1. Ground every statement in the document. Never add facts, background, or legal \
    conclusions that are not in the text, and never speculate about intent or outcome.
    2. Do not give legal advice or opinions on validity, enforceability, or risk. If \
    the document itself states such a position, report it and attribute it to the source.
    3. Preserve precision exactly as written: party names, defined terms (with their \
    original capitalization), dates, notice periods and deadlines, monetary amounts \
    and currencies, percentages, clause and section numbers, jurisdictions, and \
    governing law. Never round, convert, or normalize these.
    4. If the document is silent, ambiguous, or internally inconsistent on a point \
    material to the summary, say so in a clause rather than resolving it.
    5. If the text is an excerpt or breaks off mid-sentence, note that and summarize \
    only what is present.
    6. Keep an objective register. No advocacy, no persuasive framing, no filler.
    7. Write in the language of the document unless instructed otherwise.
    8. Return the paragraph only — no preamble, no sign-off, no meta-commentary.\
    """

    AGGREGATE_SUMMARIZATION_PROMPT: Final = """\
    You are a legal analyst. You are given several already-summarized legal items, each \
    introduced by its own title on its own line. They were grouped by a clustering algorithm, \
    so they may or may not share a subject.

    Write a single paragraph mapping what this group covers, such that a reader looking for any \
    one of these items can recognize that it is here. Prose only: no headings, no bullets, no lists.

    Cover the group, do not distil it. Give every item at least one clause of its own, naming what \
    it governs in the words that item itself uses. Order the paragraph by how many items share a \
    subject — subjects common to several items first, subjects belonging to a single item after — \
    but never drop the single-item subjects. Aim for roughly 25 words per item, and between 100 \
    and 400 words in total.

    Rules:
    1. Ground every statement in the supplied items. Never add facts, background, or legal \
    conclusions that are not in the text, and never speculate.
    2. Reuse each item's own terminology verbatim: defined terms, terms of art, and the words it \
    uses for the things it governs. Do not paraphrase them into general language, and do not \
    invent an umbrella term the items do not use.
    3. Preserve precision exactly as written: defined terms, designators, dates, figures, \
    jurisdictions. Never round, convert, or normalize these.
    4. Refer to each item by the designator in its title, so a reader can tell which part of the \
    paragraph belongs to which item.
    5. An instrument other than the one the items belong to is mentioned only where an item \
    governs or amends it, and never as the subject of a sentence: name the obligation first and \
    the external instrument last.
    6. Describe subject matter, not the group. Never state what the items collectively address, \
    and never comment on whether they are related, cohesive, or divergent.
    7. Keep an objective register. No advocacy, no persuasive framing, no filler.
    8. Write in the language of the items unless instructed otherwise.
    9. Return the paragraph only — no preamble, no sign-off, no meta-commentary.\
    """

    TITLE_PROMPT: Final = """\
    You read a short summary of a legal document or a group of related legal documents and \
    produce a concise descriptive title for it.

    Rules:
    1. 3 to 8 words, Title Case, no ending punctuation, no quotes.
    2. Capture the common subject matter, not generic words like "summary" or "document".
    3. Ground the title only in the supplied text; never invent facts.
    4. Return the title only — no preamble, no sign-off, no meta-commentary.\
    """

    def __init__(self) -> None:
        settings = get_single_corpus_settings()

        self.llm_model = settings.llm.model
        self.llm_base_url = settings.llm.base_url
        self.llm_api_key = settings.llm.api_key
        self.llm_temperature = settings.llm.temperature
        self.llm_timeout = settings.llm.timeout
        self.summarizer_enable_thinking = settings.summarizer.enable_thinking

        self.llm = self._build_llm()

        self._encoding = tiktoken.get_encoding(settings.summarizer.token_encoding)
        # The percentage-based factor is applied before the fixed margin: it scales with chunk
        # size to absorb the cl100k estimate undercounting the served model's own tokenization of
        # token-dense text, while the margin covers the (roughly constant) prompt/wrapper overhead.
        self.max_input_tokens = max(
            int(
                (settings.llm.context_length - settings.summarizer.reserved_output_tokens)
                * settings.summarizer.token_estimate_safety_factor
            )
            - settings.summarizer.token_safety_margin,
            settings.summarizer.min_input_tokens,
        )

    def _build_llm(self) -> ChatOpenAI:
        """
        Create the LLM client for an OpenAI-compatible endpoint.

        Returns:
            the :class:`ChatOpenAI` instance.

        """
        api_key = self.llm_api_key if isinstance(self.llm_api_key, SecretStr) else SecretStr(self.llm_api_key)

        log.info("llm_initialized", model=self.llm_model, base_url=self.llm_base_url)
        return ChatOpenAI(
            base_url=self.llm_base_url,
            model=self.llm_model,
            api_key=api_key,
            temperature=self.llm_temperature,
            timeout=self.llm_timeout,
            extra_body={"chat_template_kwargs": {"enable_thinking": self.summarizer_enable_thinking}},
        )

    @staticmethod
    def _build_messages(
        text: str,
        system_prompt: str,
        document_type: str | None,
        focus: str | None,
    ) -> list[BaseMessage]:
        """
        Build the system/user message pair for one summarization request.

        Returns:
            the messages to send to the model.

        """
        system_parts = [system_prompt]
        if document_type:
            system_parts.append(f"The document is described as: {document_type}.")
        if focus:
            system_parts.append(
                f"Give particular weight to the following, without omitting other material terms: {focus}."
            )

        user_content = f"<document>\n{text.strip()}\n</document>"

        return [
            SystemMessage(content="\n\n".join(system_parts)),
            HumanMessage(content=user_content),
        ]

    @staticmethod
    def _extract_text(message: BaseMessage) -> str:
        """
        Flatten a message's content into plain text.

        Returns:
            the message text.

        """
        content = message.content
        if isinstance(content, str):
            return content.strip()

        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(parts).strip()

    def create_summary(
        self,
        text: str,
        document_type: str | None = None,
        focus: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """
        Summarize a legal document in a single paragraph.

        Documents whose estimated token count would exceed the model's context window are
        summarized in a map-reduce pass instead: split into chunks, each chunk summarized, then
        the chunk summaries combined into the final single-paragraph summary.

        Args:
            text: the document text to summarize.
            document_type: optional hint, e.g. ``"share purchase agreement"``.
            focus: optional aspects to prioritize, e.g. ``"termination rights"``.
            system_prompt: optional prompt strategy override, e.g. :attr:`AGGREGATE_SUMMARIZATION_PROMPT`
                to synthesize several already-titled summaries instead of one raw document. Defaults
                to :attr:`SUMMARIZATION_PROMPT`.

        Returns:
            the summary as a string.

        Raises:
            ValueError: if ``text`` is empty or whitespace only.
            SummarizationError: if the model call fails or returns no content.

        """
        if not text or not text.strip():
            msg = "Cannot summarize empty text."
            raise ValueError(msg)
        prompt = system_prompt or self.SUMMARIZATION_PROMPT

        if self._count_tokens(text) > self.max_input_tokens:
            return self._summarize_long_text(text, prompt, document_type, focus)

        return self._summarize_once(text, prompt, document_type, focus)

    def _count_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in ``text``.

        Uses a general-purpose tokenizer, not the served model's own, so this is an estimate:
        ``max_input_tokens`` is computed with a safety margin to absorb the difference.

        Returns:
            the estimated token count.

        """
        return len(self._encoding.encode(text))

    def _split_into_chunks(self, text: str, max_tokens: int) -> list[str]:
        """
        Split ``text`` into chunks of at most ``max_tokens`` estimated tokens each.

        Splits on paragraph boundaries where possible; a single paragraph longer than
        ``max_tokens`` is hard-split on raw token boundaries as a last resort.

        Returns:
            the chunks, in the original order.

        """
        return split_into_chunks(
            text,
            max_tokens,
            count_tokens=self._count_tokens,
            encode=self._encoding.encode,
            decode=self._encoding.decode,
        )

    def _summarize_long_text(
        self,
        text: str,
        system_prompt: str,
        document_type: str | None,
        focus: str | None,
    ) -> str:
        """
        Map-reduce a document too long for a single call: summarize each chunk, then summarize
        the combined chunk summaries. Recurses if the combined summaries are themselves still
        too long to combine in one call.

        Returns:
            the final single-paragraph summary.

        Raises:
            SummarizationError: if the text cannot be split into more than one chunk despite
                exceeding the model's context window.

        """
        chunks = self._split_into_chunks(text, self.max_input_tokens)
        if len(chunks) <= 1:
            log.error("summarization_chunking_failed", estimated_tokens=self._count_tokens(text))
            msg = "Document exceeds the model's context window and could not be split into smaller chunks."
            raise SummarizationError(msg)
        log.info("summarization_chunked", chunks=len(chunks), estimated_tokens=self._count_tokens(text))

        partial_summaries = self.create_summaries(
            chunks, document_type=document_type, focus=focus, system_prompt=system_prompt
        )
        combined = "\n\n".join(partial_summaries)

        if self._count_tokens(combined) > self.max_input_tokens:
            return self._summarize_long_text(combined, system_prompt, document_type, focus)

        return self._summarize_once(combined, system_prompt, document_type, focus)

    def _summarize_once(
        self,
        text: str,
        system_prompt: str,
        document_type: str | None,
        focus: str | None,
    ) -> str:
        """
        Summarize a document in a single model call, without any length check.

        Returns:
            the summary as a string.

        Raises:
            SummarizationError: if the model call fails or returns no content.

        """
        messages = self._build_messages(text, system_prompt, document_type, focus)

        log.info("summarization_started", chars=len(text))
        start = time.perf_counter()
        try:
            response = self.llm.invoke(messages)
        except Exception as e:
            log.exception("llm_invoke_failed", duration_seconds=round(time.perf_counter() - start, 3))
            msg = "Summarization request failed."
            raise SummarizationError(msg) from e
        duration = time.perf_counter() - start

        summary = self._extract_text(response)
        if not summary:
            log.error("summarization_empty_response", duration_seconds=round(duration, 3))
            msg = "The model returned an empty summary."
            raise SummarizationError(msg)

        log.info("summarization_completed", chars=len(summary), duration_seconds=round(duration, 3))
        return summary

    def _summarize_batch(
        self,
        texts: Sequence[str],
        system_prompt: str,
        document_type: str | None,
        focus: str | None,
    ) -> list[str]:
        """
        Summarize several documents in one batched model call, without any length check.

        Mirrors :meth:`_summarize_once`'s error handling so single and batched calls fail
        the same way.

        Returns:
            the summaries, in the same order as ``texts``.

        Raises:
            SummarizationError: if the batched model call fails or returns unusable content.

        """
        messages: list[LanguageModelInput] = [
            self._build_messages(text, system_prompt, document_type, focus) for text in texts
        ]

        log.info("batch_summarization_started", count=len(messages))
        start = time.perf_counter()
        try:
            responses = self.llm.batch(messages)
        except Exception as e:
            log.exception(
                "llm_batch_invoke_failed", count=len(messages), duration_seconds=round(time.perf_counter() - start, 3)
            )
            msg = "Batched summarization request failed."
            raise SummarizationError(msg) from e
        duration = time.perf_counter() - start

        summaries = [self._extract_text(response) for response in responses]
        if any(not summary for summary in summaries):
            log.error("batch_summarization_empty_response", duration_seconds=round(duration, 3))
            msg = "The model returned an empty summary for at least one document in the batch."
            raise SummarizationError(msg)

        log.info("batch_summarization_completed", count=len(messages), duration_seconds=round(duration, 3))
        return summaries

    def create_summaries(
        self,
        texts: Sequence[str],
        document_type: str | None = None,
        focus: str | None = None,
        system_prompt: str | None = None,
    ) -> list[str]:
        """
        Summarize several documents, batching the calls where possible.

        Applies the same length check as :meth:`create_summary` to each text: texts short
        enough to fit the model's context window are summarized together in a single batched
        call, while texts that would exceed it are individually map-reduced through
        :meth:`create_summary`'s chunking strategy (:meth:`_summarize_long_text`).

        Args:
            texts: the document texts to summarize.
            document_type: optional hint, e.g. ``"share purchase agreement"``.
            focus: optional aspects to prioritize, e.g. ``"termination rights"``.
            system_prompt: optional prompt strategy override, see :meth:`create_summary`.

        Returns:
            the summaries, in the same order as ``texts``.

        Raises:
            SummarizationError: if a model call fails or returns unusable content.

        """
        prompt = system_prompt or self.SUMMARIZATION_PROMPT

        summaries: list[str | None] = [None] * len(texts)
        short_indices: list[int] = []
        for i, text in enumerate(texts):
            if self._count_tokens(text) > self.max_input_tokens:
                summaries[i] = self._summarize_long_text(text, prompt, document_type, focus)
            else:
                short_indices.append(i)

        if short_indices:
            short_summaries = self._summarize_batch([texts[i] for i in short_indices], prompt, document_type, focus)
            for i, summary in zip(short_indices, short_summaries, strict=True):
                summaries[i] = summary

        return cast("list[str]", summaries)

    def create_name(self, text: str) -> str:
        """
        Generate a short descriptive title from an already-condensed summary.

        Unlike :meth:`create_summary`, this never chunks the input: it is only ever called on
        text short enough to fit in one call (a summary, not a raw document).

        Args:
            text: the (summary) text to title.

        Returns:
            the generated title.

        Raises:
            ValueError: if ``text`` is empty or whitespace only.
            SummarizationError: if the model call fails or returns no content.

        """
        if not text or not text.strip():
            msg = "Cannot title empty text."
            raise ValueError(msg)
        return self._summarize_once(text, self.TITLE_PROMPT, None, None)

    def create_names(self, texts: Sequence[str]) -> list[str]:
        """
        Generate short descriptive titles for several already-condensed summaries in one batch.

        Returns:
            the titles, in the same order as ``texts``.

        Raises:
            SummarizationError: if the batched model call fails.

        """
        batches: list[LanguageModelInput] = [
            self._build_messages(text, self.TITLE_PROMPT, None, None) for text in texts
        ]

        log.info("Batch title generation started", count=len(batches))
        start = time.perf_counter()
        try:
            responses = self.llm.batch(batches)
        except Exception as e:
            log.exception(
                "Batch title generation failed",
                count=len(batches),
                duration_seconds=round(time.perf_counter() - start, 3),
            )
            msg = "Batched title generation request failed."
            raise SummarizationError(msg) from e

        log.info(
            "Batch title generation completed",
            count=len(batches),
            duration_seconds=round(time.perf_counter() - start, 3),
        )
        return [self._extract_text(response) for response in responses]
