from __future__ import annotations

from typing import TYPE_CHECKING, Final

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from structlog import get_logger

from wbpoa.core.generation.chat_model import (
    ToolBindingChatModel,
    ToolCallingRunnable,
    build_chat_model,
    extract_text,
    stream_message,
)
from wbpoa.core.generation.models import RedactedNode
from wbpoa.core.generation.research_observer import (
    ResearchEvent,
    ResearchEventKind,
    ResearchEventPhase,
    ResearchObserver,
)
from wbpoa.core.generation.tracing import LangfuseTracer
from wbpoa.multi_corpora.generation.models import MultiCorporaResult, MultiCorporaSubAgentCall
from wbpoa.multi_corpora.settings import MultiCorporaSettings, get_multi_corpora_settings

if TYPE_CHECKING:
    from langchain_core.messages import ToolCall
    from langchain_core.runnables import RunnableConfig

    from wbpoa.core.generation.single_corpus_agent import SingleCorpusAgent
    from wbpoa.multi_corpora.corpus.multi_corpora_context import MultiCorporaContext

log = get_logger()


class MultiCorpusAgentError(RuntimeError):
    """Raised when the manager's model call fails, returns an empty answer, or every sub-agent call failed."""


class MultiCorpusAgent:
    """
    Answer a question spanning two corpora by delegating to two unchanged per-corpus single-corpus agents.

    The manager is a tool-calling loop: each sub agent -- a full `SingleCorpusAgent` bound to
    one sub-corpus -- is exposed as one tool taking a single ``question``. The manager writes a
    focused sub-question per corpus, chooses the order, may follow up on an answer, and finally
    synthesizes one answer citing only what the sub agents cited. It must consult both sub
    agents: a final answer given before both were called is sent back with a reminder, and when
    the step budget runs out, any sub agent still uncalled is run on the original question
    before a forced synthesis. A failed sub-agent call is logged, reported back to the manager
    as a tool result, and counts as having called that sub agent; only when every call failed
    does the manager raise.

    Attributes:
        corpus_context: Identity of the pairing, exposing both sub-corpora's merged instruments.
        llm: The chat model the manager reasons, calls tools, and answers with.

    """

    MANAGER_PROMPT: Final = """You are the manager of a legal question-answering system that spans two \
bodies of EU law. You cannot read the law yourself: each body of law is covered by a specialist sub agent, \
which you consult through its tool.

{corpora}

The user message gives the question in <question> tags.

Consulting the sub agents:
- Consult both sub agents before you answer: every question gets at least one call to each tool.
- Decide which corpus to consult first. When what one corpus says shapes what to ask the other, consult
them in that order and use the first answer to sharpen the second question.
- Write each sub agent a focused, self-contained question about its own corpus only. Name the concepts,
actors, and obligations it should look for, and never ask it about the other corpus.
- Each sub agent answers only from its own corpus and cites its sources. A sub agent may say its corpus
does not cover the question; that is a valid answer, not a failure.
- You may call a sub agent again with a sharper follow-up question when its answer leaves a gap the
question needs filled, but each sub agent can be called at most {max_calls} times.
- If a tool reports that a sub agent failed, answer from the other sub agent and say in one sentence
which part of the question could not be checked.

Grounding:
- Answer only from the sub agents' answers. Never answer from memory, and never add legal content they do
not contain, however well you know it.
- If neither answer covers the question, say plainly that the corpora do not cover it, and stop there.
Do not guess or fall back on your own knowledge.
- If the answers cover only part of the question, answer that part and say in one sentence which part
the corpora do not cover.
- If the two corpora pull in different directions, state what each requires and attribute each to its
own instrument, rather than picking one.

Citations:
- Cite only the bracketed citations that appear in the sub agents' answers, copied exactly as written,
e.g. [Article 6 AI Act] or [Article 35(1) GDPR]. Never invent a citation, never change the instrument
name inside a bracket, and never cite a source no sub agent cited.
- Every legal claim carries at least one citation, placed at the end of the sentence or clause it
supports. When several sources back the same claim, put their brackets side by side. A statement that
the corpora do not cover something needs no citation.

Style:
- The first sentence answers the question. For a yes/no question, open with the verdict ("Yes.", "No.",
"Not necessarily.") and then give the reason.
- Answer every part of a multi-part question, and show how the two bodies of law fit together when both
apply.
- For rules with conditions or fallbacks, set them out in the order they apply.
- Paraphrase the law in plain, precise language, as a lawyer would answer a client; state the law as fact
and let the citation show where it comes from. Do not quote long passages.
- Be concise: usually two to six sentences in a single paragraph, never more than two short paragraphs.
Plain prose only: no bullets, headings, tables, bold, or closing summary.
- Never describe the sub agents, the tools, or how you gathered the answer: no "according to the sub
agent", "the GDPR agent says", "based on the answers". Refer to the law, not to your sources.
- Do not restate the question or add a preamble. Never ask a clarifying question.
- Answer in English, whatever language the question is in.
"""

    REMINDER: Final = (
        "You have not consulted {missing} yet. Every question must be checked against both corpora: "
        "call {missing} now before you write the final answer."
    )

    SYNTHESIS_INSTRUCTION: Final = (
        "The sub agents' answers are given in <sub_answers> tags, one <sub_answer> per call. "
        "No more tool calls are possible: write the final answer now, following every rule above."
    )

    def __init__(
        self,
        sub_agents: tuple[SingleCorpusAgent, SingleCorpusAgent],
        *,
        context: MultiCorporaContext,
        settings: MultiCorporaSettings | None = None,
        llm: ToolBindingChatModel | None = None,
    ) -> None:
        """
        Args:
            sub_agents: The two per-corpus single-corpus agents, used unmodified; each one's
                `corpus_context.key` names its sub-corpus.
            context: Identity of the pairing; its `corpus_keys` must be exactly the sub agents' keys.
            settings: Overrides for MultiCorporaSettings; defaults to the process-wide settings.
                `settings.multi_corpora` supplies the call cap, step budget, and thinking switch.
            llm: Chat model override; defaults to one built from `settings.llm`.

        Raises:
            ValueError: If the two sub agents share a corpus key, or their keys differ from
                `context.corpus_keys`.

        """
        keys = [agent.corpus_context.key for agent in sub_agents]
        if len(set(keys)) != len(keys):
            message = f"Multi-corpora sub agents must cover distinct corpora, got {keys}"
            raise ValueError(message)
        if set(keys) != set(context.corpus_keys):
            message = f"Sub agent corpora {keys} do not match the pairing's corpora {list(context.corpus_keys)}"
            raise ValueError(message)

        self._settings = settings if settings is not None else get_multi_corpora_settings()
        self.corpus_context = context
        self._sub_agents = {self._tool_name(agent.corpus_context.key): agent for agent in sub_agents}
        self._tools = [self._tool_spec(name, agent) for name, agent in self._sub_agents.items()]
        self._prompt = self.MANAGER_PROMPT.replace("{corpora}", self._corpora_block()).replace(
            "{max_calls}", str(self._settings.multi_corpora.max_calls_per_sub_agent)
        )
        self.llm: ToolBindingChatModel = (
            llm
            if llm is not None
            else build_chat_model(
                self._settings.llm, enable_thinking=self._settings.multi_corpora.manager_enable_thinking
            )
        )

    @staticmethod
    def _tool_name(corpus_key: str) -> str:
        """Return the tool name exposing the sub agent of `corpus_key`, e.g. ``ask_ai_act``."""
        return f"ask_{corpus_key.replace('-', '_')}"

    @staticmethod
    def _instruments(agent: SingleCorpusAgent) -> str:
        """Return the sub agent's instrument names, comma-separated, for prompts and tool descriptions."""
        return ", ".join(agent.corpus_context.instrument_names.values())

    def _tool_spec(self, name: str, agent: SingleCorpusAgent) -> dict[str, object]:
        """
        Build the OpenAI function-tool definition for one sub agent.

        Args:
            name: The tool name, from `_tool_name`.
            agent: The sub agent the tool calls.

        Returns:
            A ``{"type": "function", ...}`` definition with a single required ``question`` argument.

        """
        corpus = agent.corpus_context
        instruments = self._instruments(agent)
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": (
                    f"Ask the {corpus.name} sub agent one question. It answers only from the {corpus.name} "
                    f"corpus ({instruments}), with a short cited answer, or says the corpus does not cover it."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": (f"A focused, self-contained question about the {corpus.name} corpus only."),
                        }
                    },
                    "required": ["question"],
                },
            },
        }

    def _corpora_block(self) -> str:
        """Render the prompt's list of sub agents: one line per tool, naming its corpus and instruments."""
        return "The two sub agents are:\n" + "\n".join(
            f"- {name}: the {agent.corpus_context.name} corpus ({self._instruments(agent)})."
            for name, agent in self._sub_agents.items()
        )

    def run_metadata(self) -> dict[str, object]:
        """
        Describe the manager's configuration for a run's metadata.

        Returns:
            ``sub_corpora``, the two sub-corpus keys, and ``multi_corpora``, the manager settings.

        """
        return {
            "sub_corpora": list(self.corpus_context.corpus_keys),
            "multi_corpora": self._settings.multi_corpora.model_dump(),
        }

    def run(self, query: str, /) -> MultiCorporaResult:
        """
        Answer `query` from both sub-corpora.

        Args:
            query: The raw user question.

        Returns:
            The manager's answer, every sub-agent call it made, and the union of those calls'
            grounding contexts.

        Raises:
            MultiCorpusAgentError: If a manager model call fails, the final answer is empty, or
                every sub-agent call failed.

        """
        tracer = LangfuseTracer(self._settings.langfuse)
        if tracer.client is None:
            return self._run(query, tracer.config)

        with tracer.client.start_as_current_observation(
            name="Multi-Corpus Agent", as_type="chain", input=query
        ) as root:
            result = self._run(query, tracer.config)
            root.update(output=result.answer)
        return result

    def _run(self, query: str, config: RunnableConfig | None) -> MultiCorporaResult:
        """
        Run the tool-calling loop, assuming any Langfuse trace has already been opened by `run`.

        Args:
            query: The raw user question.
            config: LangChain run config threaded into every manager model call.

        Returns:
            The manager's result.

        """
        messages: list[BaseMessage] = [
            SystemMessage(self._prompt),
            HumanMessage(f"<question>\n{query}\n</question>"),
        ]
        calls: list[MultiCorporaSubAgentCall] = []
        tool_llm = self.llm.bind_tools(self._tools)

        for step in range(self._settings.multi_corpora.max_steps):
            # Until both sub agents are consulted, a text reply is sent back with a reminder, so it is not streamed.
            response = self._invoke(tool_llm, messages, config, stream=not self._uncalled(calls))
            messages.append(response)
            if response.tool_calls:
                messages.extend(self._answer_tool_call(tool_call, query, calls) for tool_call in response.tool_calls)
                continue

            missing = self._uncalled(calls)
            if missing:
                log.info("multi_corpus_agent_reminded", step=step, missing=missing)
                messages.append(HumanMessage(self.REMINDER.format(missing=" and ".join(missing))))
                continue

            return self._result(query, extract_text(response), calls)

        log.warning("multi_corpus_agent_step_budget_exhausted", max_steps=self._settings.multi_corpora.max_steps)
        calls.extend(self._call_sub_agent(name, query) for name in self._uncalled(calls))
        return self._result(query, self._synthesize(query, calls, config), calls)

    @staticmethod
    def _invoke(
        model: ToolCallingRunnable | ToolBindingChatModel,
        messages: list[BaseMessage],
        config: RunnableConfig | None,
        *,
        stream: bool,
    ) -> AIMessage:
        """
        Call the manager model, normalizing failures to `MultiCorpusAgentError`.

        When `stream` is set and the `ResearchObserver` has an answer listener, the reply is
        streamed to it as it is written (see `stream_message`), so the final answer shows while
        it is being generated.

        Args:
            model: The tool-bound runnable or the plain chat model.
            messages: The conversation so far.
            config: LangChain run config for tracing.
            stream: Whether the reply's text could be the final answer, and so may be streamed.

        Returns:
            The model's reply; a non-AI reply is wrapped as an `AIMessage` with no tool calls.

        Raises:
            MultiCorpusAgentError: If the model call raises.

        """
        try:
            if stream and ResearchObserver.streams_answer():
                return stream_message(model, messages, config, ResearchObserver.report_answer)
            response = model.invoke(messages, config=config)
        except Exception as error:
            message = f"Multi-corpus agent model call failed: {error}"
            raise MultiCorpusAgentError(message) from error
        return response if isinstance(response, AIMessage) else AIMessage(content=response.content)

    def _answer_tool_call(self, tool_call: ToolCall, query: str, calls: list[MultiCorporaSubAgentCall]) -> ToolMessage:
        """
        Execute one tool call from the manager, within the per-sub-agent call cap.

        Args:
            tool_call: The tool call to answer.
            query: The original question, used when the call carries no usable ``question``.
            calls: Every sub-agent call made so far; a made call is appended.

        Returns:
            The tool result: the sub agent's answer, or why no answer was produced.

        """
        name = tool_call["name"]
        call_id = tool_call.get("id") or f"call_{len(calls)}"
        agent = self._sub_agents.get(name)
        if agent is None:
            log.warning("multi_corpus_agent_unknown_tool", tool=name)
            available = ", ".join(self._sub_agents)
            return ToolMessage(f"There is no tool named {name!r}. Available tools: {available}.", tool_call_id=call_id)

        cap = self._settings.multi_corpora.max_calls_per_sub_agent
        if sum(call.corpus_key == agent.corpus_context.key for call in calls) >= cap:
            log.info("multi_corpus_agent_call_cap_reached", tool=name, cap=cap)
            return ToolMessage(
                f"{name} has already been called {cap} times, its limit. Answer from what you have.",
                tool_call_id=call_id,
            )

        question = tool_call["args"].get("question")
        if not isinstance(question, str) or not question.strip():
            log.warning("multi_corpus_agent_missing_sub_question", tool=name)
            question = query

        call = self._call_sub_agent(name, question)
        calls.append(call)
        if call.result is None:
            return ToolMessage(
                f"The {agent.corpus_context.name} sub agent failed and gave no answer ({call.error}).",
                tool_call_id=call_id,
            )
        return ToolMessage(call.result.answer, tool_call_id=call_id)

    def _call_sub_agent(self, name: str, question: str) -> MultiCorporaSubAgentCall:
        """
        Run one sub agent end to end, recording rather than raising its failure.

        The call is reported to the `ResearchObserver` as it starts and again, under the same id,
        once it has answered or failed. The sub agent's own answer is never streamed: only the
        manager's answer is.

        Args:
            name: Tool name of the sub agent to run.
            question: The sub-question to answer.

        Returns:
            The call record, carrying the sub agent's result or its error.

        """
        agent = self._sub_agents[name]
        corpus_key = agent.corpus_context.key
        log.info("multi_corpora_sub_agent_called", corpus=corpus_key, question=question)
        started = ResearchEvent(
            kind=ResearchEventKind.DELEGATION, phase=ResearchEventPhase.STARTED, corpus_key=corpus_key, text=question
        )
        ResearchObserver.report(started)
        finished = started.model_copy(update={"phase": ResearchEventPhase.FINISHED})
        try:
            with ResearchObserver.quiet_answers():
                result = agent.run(question)
        except Exception as error:
            log.exception("multi_corpora_sub_agent_failed", corpus=corpus_key)
            failure = f"{type(error).__name__}: {error}"
            ResearchObserver.report(finished.model_copy(update={"error": failure}))
            return MultiCorporaSubAgentCall(corpus_key=corpus_key, question=question, error=failure)
        ResearchObserver.report(finished)
        return MultiCorporaSubAgentCall(corpus_key=corpus_key, question=question, result=result)

    def _uncalled(self, calls: list[MultiCorporaSubAgentCall]) -> list[str]:
        """Return the tool names of sub agents never called, failed calls counting as called."""
        called = {call.corpus_key for call in calls}
        return [name for name, agent in self._sub_agents.items() if agent.corpus_context.key not in called]

    def _synthesize(self, query: str, calls: list[MultiCorporaSubAgentCall], config: RunnableConfig | None) -> str:
        """
        Force a final answer, without tools, from every sub answer gathered so far.

        Starts a fresh conversation rather than continuing the tool-calling one, so the model sees
        the sub answers as plain context and cannot attempt another tool call.

        Args:
            query: The original question.
            calls: Every sub-agent call made.
            config: LangChain run config for tracing.

        Returns:
            The final answer text.

        """
        blocks = [
            f'<sub_answer corpus="{call.corpus_key}" question="{call.question}">\n'
            f"{call.result.answer if call.result is not None else f'The sub agent failed: {call.error}'}\n"
            "</sub_answer>"
            for call in calls
        ]
        messages: list[BaseMessage] = [
            SystemMessage(self._prompt),
            HumanMessage(
                f"<question>\n{query}\n</question>\n\n<sub_answers>\n"
                + "\n".join(blocks)
                + f"\n</sub_answers>\n\n{self.SYNTHESIS_INSTRUCTION}"
            ),
        ]
        return extract_text(self._invoke(self.llm, messages, config, stream=True))

    @staticmethod
    def _result(query: str, answer: str, calls: list[MultiCorporaSubAgentCall]) -> MultiCorporaResult:
        """
        Assemble the final result, rejecting an empty answer or a run where every call failed.

        Args:
            query: The original question.
            answer: The manager's final answer text.
            calls: Every sub-agent call made.

        Returns:
            The result, whose grounding context is the union of every successful call's grounding
            context, deduplicated by node id in first-seen order.

        Raises:
            MultiCorpusAgentError: If every sub-agent call failed, or `answer` is blank.

        """
        if all(call.result is None for call in calls):
            message = f"Every multi-corpora sub-agent call failed ({len(calls)} calls)"
            raise MultiCorpusAgentError(message)
        if not answer.strip():
            message = "Multi-corpus agent returned an empty answer"
            raise MultiCorpusAgentError(message)

        grounding_context: dict[str, RedactedNode] = {}
        for call in calls:
            if call.result is None:
                continue
            for node in call.result.grounding_context:
                grounding_context.setdefault(str(node.id), node)
        log.info(
            "multi_corpus_agent_answered",
            calls=[(call.corpus_key, call.result is not None) for call in calls],
            num_context_nodes=len(grounding_context),
        )
        return MultiCorporaResult(
            query=query, answer=answer, sub_agent_calls=calls, grounding_context=list(grounding_context.values())
        )
