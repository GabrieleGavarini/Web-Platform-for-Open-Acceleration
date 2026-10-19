from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, TypedDict, Unpack
from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from structlog import get_logger

from wbpoa.core.corpus import CorpusContext
from wbpoa.core.generation.chat_model import (
    ToolBindingChatModel,
    ToolCallingRunnable,
    build_chat_model,
    count_tokens,
    extract_text,
    extract_thinking,
    max_input_tokens,
    stream_message,
)
from wbpoa.core.generation.context_formatter import ContextFormatter
from wbpoa.core.generation.models import RedactedNode, SearchResult, SingleCorpusResult
from wbpoa.core.generation.provision_lookup import ProvisionLookup, ProvisionLookupError
from wbpoa.core.generation.redaction_agent import RedactionAgent
from wbpoa.core.generation.reformulation_agent import ReformulationAgent
from wbpoa.core.generation.research_observer import (
    ResearchEvent,
    ResearchEventKind,
    ResearchEventPhase,
    ResearchObserver,
)
from wbpoa.core.generation.research_planner import ResearchPlanError, ResearchPlanner
from wbpoa.core.generation.search_tool import RetrievalOptions, SearchTool
from wbpoa.core.generation.tracing import LangfuseTracer
from wbpoa.core.inference.similarity import cosine_similarity
from wbpoa.core.settings import CoreSettings, get_core_settings

if TYPE_CHECKING:
    from langchain_core.messages import ToolCall
    from langchain_core.runnables import RunnableConfig

    from wbpoa.core.inference import Retriever

log = get_logger()

TOOL_NAME: Final = "search_corpus"
LOOKUP_TOOL_NAME: Final = "lookup_provision"

# A bracketed citation of a guideline or ruling, as rendered by ContextFormatter.header for
# nodes without a qualified citation (or a shortened form of it), with the whitespace before it.
_DOCUMENT_CITATION: Final = re.compile(r"[ \t]*\[(?:Guideline|Ruling)\b[^\]]*\]")


class SingleCorpusAgentError(RuntimeError):
    """Raised when the agent's model call fails, returns an empty answer, or every search failed."""


class SingleCorpusAgentOptions(TypedDict, total=False):
    """
    Optional keyword overrides for `SingleCorpusAgent.__init__`.

    Attributes:
        settings: Overrides for CoreSettings; defaults to the process-wide settings.
        corpus_context: Corpus identity used by prompts, citation headers, and tracing metadata.
        llm: Chat model override; defaults to one built from `settings.llm`.
        planner: Research planner override; defaults to a new ResearchPlanner.

    """

    settings: CoreSettings
    corpus_context: CorpusContext
    llm: ToolBindingChatModel
    planner: ResearchPlanner


@dataclass
class _Session:
    """The mutable state of one `SingleCorpusAgent.run`: its inputs and every search so far."""

    query: str
    tracer: LangfuseTracer
    retrieval_options: RetrievalOptions
    searches: list[SearchResult] = field(default_factory=list)
    seen_ids: set[UUID] = field(default_factory=set)
    looked_up: list[RedactedNode] = field(default_factory=list)
    lookups: int = 0
    asked: list[tuple[str, list[float]]] = field(default_factory=list)
    attempts: int = 0


class SingleCorpusAgent:
    """
    Answer a question over one corpus with a tool-calling loop over a search tool and a provision lookup.

    The agent is exposed a single tool, `search_corpus`, taking one ``question``; each call
    runs the `SearchTool` (reformulation -> Algorithm 2 -> redaction, each honouring its
    GenerationSettings switch) and returns the kept passages rendered as grounding context, each
    headed by the citation to copy. Unless planning is disabled, the `ResearchPlanner` first
    breaks the question down into its legal issues, one broad question each, and the agent runs
    them all as its first research round (recorded as the agent's own tool calls). A review
    instruction then asks it to check each issue against the passages and refine the unsettled
    ones with narrower follow-up questions, before answering citing only the passages it was shown. It must consult the research
    agent at least once: a final answer given before any call is sent back with a reminder, and
    when the step budget runs out without a call, the original question is researched before a
    forced synthesis. A failed search is logged and reported back as a tool result; only
    when every call failed does the agent raise.

    Attributes:
        corpus_context: Identity of the corpus answered from.
        search_tool: The search tool each search_corpus call runs.
        planner: Breaks the question down into the broad questions of the first research round.
        llm: The chat model the agent reasons, calls tools, and answers with.

    """

    AGENT_PROMPT: Final = """You are a legal assistant for the {corpus}, its official guidelines, and rulings. \
You cannot read the law yourself: you search the corpus with the tool {tool}. Once it has returned \
passages, you can also read one specific provision of one specific instrument directly with {lookup_tool}.

The user message gives the question in <question> tags.

Researching the question:
- Search before you answer: every question gets at least one call to {tool}.
- Plan first. Break the question down into the separate legal issues a complete answer must settle: the
definitions and scope it turns on, who the actors are (e.g. provider, deployer, authority), what obligations,
rights, or prohibitions apply, the conditions, exceptions, and deadlines, and who decides, supervises, or
enforces. Keep only the issues the answer needs.
- Broad pass. Ask one broad question per issue, all in the same turn -- usually two or three questions, rarely
more. Make them genuinely different: each must target a different part of the law. If the same provisions
would likely answer two issues (e.g. what support exists and who grants it), ask one question covering both
instead of two near-identical ones. Describe each topic in the corpus's own terms rather than guessing article
numbers, so the search surfaces every provision that may bear on it.
- Each question is self-contained: name the concepts, actors, obligations, and legal terms it should look for,
keep the question's own terms, and resolve every pronoun or implicit reference so the question reads on its own.
- Each call returns zero or more passages, each introduced by a header naming its source; "(no context
retrieved)" means the search tool found nothing relevant. A passage already returned by an earlier call is
shown by its header only.
- Guideline, ruling, and paper passages are excerpts of the document chosen for your question (or, failing
that, its summary). They are research aids, never sources: read them to learn which provisions govern each
issue and how those provisions fit together, then make sure those provisions are among your passages.
- A result may end with "Leads": provisions that the returned guidelines and rulings cite but that no call
has returned yet, most-cited first. Leads are the main way to find provisions the first searches missed.
- Review. Check each issue against the passages. An issue is settled when a passage states the rule outright;
it is not settled when nothing was found, when the passages only mention it in passing, or when a passage
refers to a provision, definition, exception, or procedure you have not seen yet.
- Refine, building on what you learned. Each follow-up states what the passages already establish, with the
article it comes from, and asks only for the missing piece -- e.g. "Article <n> of <instrument> requires
<rule>; which authority decides on <the open point>?". Use the passages' own terms and name the specific
article, annex, or defined term they point to. If a broad question found nothing, ask about the same issue
with different legal terms.
- Follow leads with {lookup_tool}. It does not search: given an instrument (exactly as named in headers and
leads) and a provision ("Article 62", "Recital 28", "Annex III"), it returns that provision's full text. Use it,
only after {tool} has returned passages, for every lead that bears on the question -- above all those cited by
several guidelines or rulings -- and for any provision a passage refers to or a guideline or ruling names when
discussing an issue. Several lookups can go in one turn. Keep {tool} for open issues, not for fetching a
provision you can already name. Ignore leads that do not bear on the question.
- Never repeat or paraphrase a question already asked: a near-duplicate is not researched and wastes a turn.
Stop when every issue is settled and no relevant lead or named provision is left unretrieved, or when the
calls run out. Missing a provision that governs the question is worse than one extra call.
- You can call the search tool at most {max_calls} times in total, so spend the calls on the issues the
answer needs.
- A passage's header names the specific instrument it comes from (its legal act), since the corpus can contain
more than one related instrument; a header may also note, outside the brackets, that its instrument plays a
non-base role (amending, implementing, supplementing, repealing, or replacing) -- that note is metadata for
you, never part of a citation.

Grounding:
- Answer only from the passages the search tool returned. Never answer from memory, and never add legal
content that is not in the passages, however well you know it.
- Cover every provision that governs the answer: the rule itself, the provisions that define its terms or set
its conditions and exceptions, and those that say who decides or enforces. Leave out only passages that are
off-topic or merely repeat another. Say nothing about the passages you leave out.
- Base the answer on provisions (articles, recitals, annexes). A point you learned only from a guideline,
ruling, or paper goes into the answer only if a provision among the passages supports it, cited by that
provision; otherwise leave it out.
- If the search tool found nothing, or not enough to answer, say plainly that the corpus does not cover the
question, and stop there. Do not guess or fall back on your own knowledge.
- If the passages answer only part of the question, answer that part and say in one sentence which part the
corpus does not cover.
- If two passages conflict, give both and attribute each to its source rather than picking one.

Citations:
- Cite only provisions -- articles, recitals, and annexes -- present in the passages. Never cite a guideline,
ruling, or paper, even when it is among the passages. Never invent a citation, never cite a provision that is
not in the passages, and never cite a lead: a lead becomes citable only once {tool} or {lookup_tool} returns it as a
passage.
- Use exactly the form shown in each passage's own bracketed header, e.g. [Article 6 AI Act],
[Article 6(2) AI Act], [Recital 51 Digital Omnibus], [Annex III AI Act].
Always copy the instrument name from that specific passage's header -- never substitute a different
instrument's name, and never drop it in favor of one fixed name for every citation.
- Every legal claim carries at least one citation, placed at the end of the sentence or clause it supports.
When several sources back the same claim, put their brackets side by side, e.g. [Article 3(1) AI Act]
[Recital 12 AI Act]. A statement that the corpus does not cover something needs no citation.

Multi-instrument grounding:
- When an amending passage and the base passage it modifies are both in the passages, treat the amending
passage as the current law: lead with its citation and cite the base passage alongside it.
- A passage noted as implementing is an independent source: cite it next to the base, never as if it replaces,
supersedes, or consolidates into the base instrument's text.
- Never claim or imply that a single consolidated version of two passages exists. When passages from different
instruments both matter, state what each says and cite each by its own instrument.

Style:
- The first sentence answers the question. For a yes/no question, open with the verdict ("Yes.", "No.",
"Not necessarily.") and then give the reason.
- Answer every part of a multi-part question.
- For a "why" question, explain the purpose or rationale behind the rule in plain terms, rather than listing
provisions.
- For rules with conditions or fallbacks, set them out in the order they apply: the main route, then the
alternative, then the last resort.
- Paraphrase the law in plain, precise language, as a lawyer would answer a client; state the law as fact and
let the citation show where it comes from. Briefly define a key term when the answer turns on it. Do not quote
long passages.
- Be concise: usually two to six sentences in a single paragraph, never more than two short paragraphs. Plain
prose only: no bullets, headings, tables, bold, or closing summary.
- Never describe the material you were given, the search tool, or how you used them: no "the provided
text states", "the search tool found", "based on the passages", "the passage does not mention". Refer to
the law, not to the passage.
- Do not restate the question or add a preamble. Never ask a clarifying question.
- Answer in English, whatever language the question is in.
"""

    REVIEW_INSTRUCTION: Final = (
        "Review the passages before answering. For each issue the question raises, check whether a passage "
        "states the rule outright. Fetch with {lookup_tool} every relevant lead, and every provision a passage "
        "refers to or a guideline or ruling names for an issue, that is not yet among your passages. If an issue "
        "is still not settled, call {tool} again with a follow-up that states what the passages already "
        "establish (with the article) and asks only for the missing piece. Do not repeat or paraphrase an "
        "earlier question. Otherwise, write the final answer, citing provisions only."
    )

    DUPLICATE_NOTICE: Final = (
        "Not researched: this question is nearly the same as one already asked ({asked!r}), so it would return "
        "the same passages. Ask about what those passages leave open, stating what they already establish, "
        "or write the final answer."
    )

    REMINDER: Final = (
        "You have not searched the corpus yet. Every answer must rest on passages from the corpus: "
        "call {tool} now before you write the final answer."
    )

    SYNTHESIS_INSTRUCTION: Final = (
        "The passages the search tool returned are given in <context> tags. "
        "No more tool calls are possible: write the final answer now, following every rule above."
    )

    def __init__(
        self,
        retriever: Retriever,
        reformulation_agent: ReformulationAgent | None = None,
        redaction_agent: RedactionAgent | None = None,
        **options: Unpack[SingleCorpusAgentOptions],
    ) -> None:
        """
        Args:
            retriever: Algorithm 2's retriever, used unmodified by every search.
            reformulation_agent: The search tool's step 1 agent; defaults to a new ReformulationAgent.
            redaction_agent: The search tool's step 4 agent; defaults to a new RedactionAgent.
            options: Optional `settings`, `corpus_context`, `llm`, and `planner` keyword overrides
                (see SingleCorpusAgentOptions).

        """
        settings = options.get("settings")
        self._settings = settings if settings is not None else get_core_settings()
        self.corpus_context = options.get("corpus_context") or CorpusContext(key="legacy", name="AI Act")
        generation = self._settings.generation
        self.search_tool = SearchTool(
            retriever,
            reformulation_agent
            if reformulation_agent is not None
            else ReformulationAgent(self._settings, self.corpus_context),
            redaction_agent if redaction_agent is not None else RedactionAgent(self._settings, self.corpus_context),
            generation,
        )
        self.planner = options.get("planner") or ResearchPlanner(self._settings, self.corpus_context)
        self._formatter = ContextFormatter(self.corpus_context)
        self._embedder = retriever.embedder
        self.provision_lookup = ProvisionLookup(retriever.driver, self.corpus_context)
        self._tools = [self._tool_spec()]
        self._tool_names = [TOOL_NAME]
        if self._settings.generation.max_lookups > 0:
            self._tools.append(self._lookup_tool_spec())
            self._tool_names.append(LOOKUP_TOOL_NAME)
        self._prompt = (
            self.AGENT_PROMPT.replace("{corpus}", self.corpus_context.name)
            .replace("{tool}", TOOL_NAME)
            .replace("{lookup_tool}", LOOKUP_TOOL_NAME)
            .replace("{max_calls}", str(generation.max_searches))
        )
        self._input_budget = max(max_input_tokens(self._settings.llm) - count_tokens(self._prompt), 0)
        llm = options.get("llm")
        self.llm: ToolBindingChatModel = (
            llm
            if llm is not None
            else build_chat_model(self._settings.llm, enable_thinking=generation.generation_enable_thinking)
        )

    def _instruments(self) -> str:
        """Return the corpus's instrument names, comma-separated, or the corpus name when none are known."""
        return ", ".join(self.corpus_context.instrument_names.values()) or self.corpus_context.name

    def _tool_spec(self) -> dict[str, object]:
        """
        Build the OpenAI function-tool definition of the search tool.

        Returns:
            A ``{"type": "function", ...}`` definition with a single required ``question`` argument.

        """
        name = self.corpus_context.name
        return {
            "type": "function",
            "function": {
                "name": TOOL_NAME,
                "description": (
                    f"Search the {name} corpus ({self._instruments()}, guidelines and rulings) with one question: "
                    "one search, returning the passages that bear on it, each headed by the "
                    "bracketed citation to copy, or '(no context retrieved)' when nothing relevant was found."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": (
                                f"A focused, self-contained question about one piece of information in the {name} "
                                "corpus, naming the concepts, actors, and obligations to look for."
                            ),
                        }
                    },
                    "required": ["question"],
                },
            },
        }

    def _lookup_tool_spec(self) -> dict[str, object]:
        """
        Build the OpenAI function-tool definition of the provision lookup.

        Returns:
            A ``{"type": "function", ...}`` definition with required ``instrument`` (one of the
            corpus's instrument names) and ``provision`` arguments.

        """
        return {
            "type": "function",
            "function": {
                "name": LOOKUP_TOOL_NAME,
                "description": (
                    "Fetch the full text of one specific provision of one specific instrument, by its number. "
                    "No search: it returns exactly that provision, or says it does not exist. Only usable after "
                    f"{TOOL_NAME} has run, for provisions research has surfaced: leads, or provisions a passage "
                    "refers to or a guideline or ruling names."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "instrument": {
                            "type": "string",
                            "enum": self.provision_lookup.instrument_names(),
                            "description": "The instrument, exactly as named in passage headers and leads.",
                        },
                        "provision": {
                            "type": "string",
                            "description": "The provision, e.g. 'Article 62', 'Recital 28', or 'Annex III'.",
                        },
                    },
                    "required": ["instrument", "provision"],
                },
            },
        }

    def run(self, query: str, /, **retrieval_options: Unpack[RetrievalOptions]) -> SingleCorpusResult:
        """
        Answer `query` from the corpus.

        Args:
            query: The raw user question.
            retrieval_options: Algorithm 2 overrides applied to every search (see
                RetrievalOptions); omitted keys fall back to InferenceSettings.

        Returns:
            The agent's answer, every search it made, and the union of those calls' passages.

        Raises:
            SingleCorpusAgentError: If a model call fails, the final answer is empty, or every
                search failed.

        """
        tracer = LangfuseTracer(self._settings.langfuse)
        session = _Session(query=query, tracer=tracer, retrieval_options=retrieval_options)
        if tracer.client is None:
            return self._run(session)

        with tracer.client.start_as_current_observation(
            name="Single Corpus Agent", as_type="agent", input=query
        ) as root:
            result = self._run(session)
            root.update(output=result.answer)
        return result

    def _run(self, session: _Session) -> SingleCorpusResult:
        """
        Run the tool-calling loop, assuming any Langfuse trace has already been opened by `run`.

        Args:
            session: The run's inputs and research record.

        Returns:
            The agent's result.

        """
        config = session.tracer.config
        messages: list[BaseMessage] = [
            SystemMessage(self._prompt),
            HumanMessage(f"<question>\n{session.query.strip()}\n</question>"),
        ]
        tool_llm = self.llm.bind_tools(self._tools)
        if self._settings.generation.planning_enabled:
            messages.extend(self._broad_round(session))

        for step in range(self._settings.generation.max_steps):
            # Before any search, a text reply is sent back with a reminder, so it is not streamed.
            response = self._invoke(tool_llm, messages, config, stream=session.attempts > 0)
            messages.append(response)
            if response.tool_calls:
                first_round = session.attempts == 0
                messages.extend(self._answer_tool_call(tool_call, session) for tool_call in response.tool_calls)
                if first_round:
                    messages.extend(self._review(session))
                continue

            if session.attempts == 0:
                log.info("single_corpus_agent_reminded", step=step)
                messages.append(HumanMessage(self.REMINDER.format(tool=TOOL_NAME)))
                continue

            return self._result(session, self._answer_text(response))

        log.warning("single_corpus_agent_step_budget_exhausted", max_steps=self._settings.generation.max_steps)
        if session.attempts == 0:
            self._search(session.query, session)
        return self._result(session, self._synthesize(session))

    def _broad_round(self, session: _Session) -> list[BaseMessage]:
        """
        Plan the question and research every planned question, as the agent's first tool calls.

        The round is recorded in the conversation as one assistant turn calling the tool once per
        planned question, followed by the tool results and the review instruction, so the agent
        continues from it exactly as if it had made those calls itself. A planning failure falls
        back to researching the original question alone.

        Args:
            session: The run's inputs and research record; the calls made are recorded on it.

        Returns:
            The messages to append: the tool-calling turn, its tool results, and the review instruction.

        """
        generation = self._settings.generation
        cap = min(generation.max_plan_questions, generation.max_searches)
        try:
            planned = self.planner.plan(session.query, cap, config=session.tracer.config)
        except ResearchPlanError:
            log.warning("single_corpus_agent_plan_failed", exc_info=True)
            planned = [session.query]
        questions = [question for question in planned if self._duplicate_of(question, session) is None]
        turn = AIMessage(
            content="",
            tool_calls=[
                {"name": TOOL_NAME, "args": {"question": question}, "id": f"plan_{index}", "type": "tool_call"}
                for index, question in enumerate(questions)
            ],
        )
        # Planned questions were already checked for duplicates above, and never exceed the call cap.
        results = [
            ToolMessage(self._search(question, session), tool_call_id=f"plan_{index}")
            for index, question in enumerate(questions)
        ]
        return [turn, *results, *self._review(session)]

    def _review(self, session: _Session) -> list[BaseMessage]:
        """Return the review instruction after the first research round, or nothing when no call is left."""
        if session.attempts >= self._settings.generation.max_searches:
            return []
        return [HumanMessage(self.REVIEW_INSTRUCTION.format(tool=TOOL_NAME, lookup_tool=LOOKUP_TOOL_NAME))]

    @staticmethod
    def _invoke(
        model: ToolCallingRunnable | ToolBindingChatModel,
        messages: list[BaseMessage],
        config: RunnableConfig | None,
        *,
        stream: bool,
    ) -> AIMessage:
        """
        Call the agent's model, normalizing failures to `SingleCorpusAgentError`.

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
            SingleCorpusAgentError: If the model call raises.

        """
        try:
            if stream and ResearchObserver.streams_answer():
                return stream_message(model, messages, config, ResearchObserver.report_answer)
            response = model.invoke(messages, config=config)
        except Exception as error:
            message = f"Single-corpus agent model call failed: {error}"
            raise SingleCorpusAgentError(message) from error
        return response if isinstance(response, AIMessage) else AIMessage(content=response.content)

    @staticmethod
    def _answer_text(response: BaseMessage) -> str:
        """Return the answer text of `response`, logging any inline thinking block."""
        thinking = extract_thinking(response)
        if thinking is not None:
            log.info("single_corpus_agent_thinking", thinking=thinking)
        return extract_text(response)

    def _answer_tool_call(self, tool_call: ToolCall, session: _Session) -> ToolMessage:
        """
        Execute one tool call from the agent, within the search cap.

        Args:
            tool_call: The tool call to answer.
            session: The run's inputs and research record; a call made is recorded on it.

        Returns:
            The tool result: the search tool's passages rendered as grounding context, or why
            none were returned.

        """
        name = tool_call["name"]
        call_id = tool_call.get("id") or f"call_{session.attempts}"
        if name == LOOKUP_TOOL_NAME and self._settings.generation.max_lookups > 0:
            return ToolMessage(self._lookup(tool_call["args"], session), tool_call_id=call_id)
        if name != TOOL_NAME:
            log.warning("single_corpus_agent_unknown_tool", tool=name)
            available = ", ".join(self._tool_names)
            return ToolMessage(f"There is no tool named {name!r}. Available tools: {available}.", tool_call_id=call_id)

        cap = self._settings.generation.max_searches
        if session.attempts >= cap:
            log.info("single_corpus_agent_call_cap_reached", cap=cap)
            return ToolMessage(
                f"{name} has already been called {cap} times, its limit. Answer from what you have.",
                tool_call_id=call_id,
            )

        question = tool_call["args"].get("question")
        if not isinstance(question, str) or not question.strip():
            log.warning("single_corpus_agent_missing_question")
            question = session.query
        question = question.strip()

        asked = self._duplicate_of(question, session)
        if asked is not None:
            return ToolMessage(self.DUPLICATE_NOTICE.format(asked=asked), tool_call_id=call_id)
        return ToolMessage(self._search(question, session), tool_call_id=call_id)

    def _lookup(self, args: dict[str, object], session: _Session) -> str:
        """
        Fetch one provision for the `lookup_provision` tool, within its cap and only after research.

        Args:
            args: The tool call's arguments: ``instrument`` and ``provision``.
            session: The run's record; a found provision is added to its lookups and seen passages.

        Returns:
            The provision rendered as a passage, or why it was not fetched.

        """
        refusal = self._lookup_refusal(session)
        if refusal is not None:
            return refusal
        instrument, provision = args.get("instrument"), args.get("provision")
        if not isinstance(instrument, str) or not isinstance(provision, str):
            return f"{LOOKUP_TOOL_NAME} needs both an 'instrument' and a 'provision'."

        session.lookups += 1
        try:
            node = self._traced_lookup(instrument, provision, session.tracer)
        except ProvisionLookupError as error:
            return str(error)
        except Exception as error:
            log.exception("single_corpus_agent_lookup_failed", instrument=instrument, provision=provision)
            return f"The lookup failed ({type(error).__name__}: {error})."
        if node is None:
            return f"{instrument} has no {provision} in this corpus."

        ResearchObserver.report(
            ResearchEvent(
                kind=ResearchEventKind.LOOKUP,
                corpus_key=self.corpus_context.key,
                text=f"{provision} {instrument}",
                passages=[node],
            )
        )
        already_shown = set(session.seen_ids)
        session.seen_ids.add(node.id)
        if all(looked.id != node.id for looked in session.looked_up):
            session.looked_up.append(node)
        budget = (self._input_budget - count_tokens(session.query)) // (self._settings.generation.max_searches + 1)
        return self._formatter.format([node], budget, already_shown)

    def _traced_lookup(self, instrument: str, provision: str, tracer: LangfuseTracer) -> RedactedNode | None:
        """Run the lookup, wrapped in a `Provision Lookup` Langfuse span when tracing is enabled."""
        if tracer.client is None:
            return self.provision_lookup.lookup(instrument, provision)
        with tracer.client.start_as_current_observation(
            name="Provision Lookup", as_type="tool", input={"instrument": instrument, "provision": provision}
        ) as span:
            node = self.provision_lookup.lookup(instrument, provision)
            span.update(output=node.name if node is not None else None)
        return node

    def _lookup_refusal(self, session: _Session) -> str | None:
        """Return why a lookup may not run yet (no research so far, or its cap reached), or None."""
        if not session.searches:
            return (
                f"{LOOKUP_TOOL_NAME} is only usable after {TOOL_NAME} has returned passages: call {TOOL_NAME} "
                "first, then look up the provisions it surfaces."
            )
        cap = self._settings.generation.max_lookups
        if session.lookups >= cap:
            return f"{LOOKUP_TOOL_NAME} has already been called {cap} times, its limit. Answer from what you have."
        return None

    def _duplicate_of(self, question: str, session: _Session) -> str | None:
        """
        Return the earlier question `question` nearly duplicates, or record it as asked.

        Compares the question's embedding against every question already asked in the run;
        at or above `GenerationSettings.duplicate_question_threshold` cosine similarity it is
        a near-duplicate. An embedding failure never blocks a question.

        Args:
            question: The question about to be researched.
            session: The run's record of asked questions; a non-duplicate is appended to it.

        Returns:
            The earlier question it duplicates, or None when it is new.

        """
        try:
            embedding = self._embedder.create_embedding(question)
        except Exception:
            log.warning("single_corpus_agent_question_embedding_failed", exc_info=True)
            return None
        threshold = self._settings.generation.duplicate_question_threshold
        for asked, asked_embedding in session.asked:
            if cosine_similarity(embedding, asked_embedding) >= threshold:
                log.info("single_corpus_agent_duplicate_question", question=question, duplicate_of=asked)
                return asked
        session.asked.append((question, embedding))
        return None

    def _search(self, question: str, session: _Session) -> str:
        """
        Run the search tool once, recording rather than raising its failure.

        The search is reported to the `ResearchObserver` as it starts and again, under the same id,
        once it has finished or failed.

        Args:
            question: The question to research.
            session: The run's inputs and research record; the attempt, and a successful call,
                are recorded on it.

        Returns:
            The search tool's passages rendered as grounding context, followed by the leads
            footer (cited provisions no call has returned yet), or why the call failed.

        """
        session.attempts += 1
        log.info("single_corpus_agent_search_called", question=question, attempt=session.attempts)
        started = ResearchEvent(
            kind=ResearchEventKind.SEARCH,
            phase=ResearchEventPhase.STARTED,
            corpus_key=self.corpus_context.key,
            text=question,
        )
        ResearchObserver.report(started)
        finished = started.model_copy(update={"phase": ResearchEventPhase.FINISHED})
        try:
            result = self.search_tool.search(question, session.tracer, session.retrieval_options)
        except Exception as error:
            log.exception("single_corpus_agent_search_failed", question=question)
            failure = f"{type(error).__name__}: {error}"
            ResearchObserver.report(finished.model_copy(update={"error": failure}))
            return f"The search tool failed and returned no passages ({failure})."
        session.searches.append(result)
        ResearchObserver.report(finished.model_copy(update={"passages": result.passages}))
        already_shown = set(session.seen_ids)
        session.seen_ids.update(node.id for node in result.passages)
        budget = (self._input_budget - count_tokens(session.query)) // (self._settings.generation.max_searches + 1)
        passages = self._formatter.format(result.passages, budget, already_shown)
        leads = [provision for provision in result.cited_provisions if provision.id not in session.seen_ids]
        footer = self._formatter.leads(leads[: self._settings.generation.max_leads])
        return f"{passages}\n\n{footer}" if footer else passages

    def _synthesize(self, session: _Session) -> str:
        """
        Force a final answer, without tools, from every passage gathered so far.

        Starts a fresh conversation rather than continuing the tool-calling one, so the model sees
        the passages as plain context and cannot attempt another tool call.

        Args:
            session: The run's inputs and research record.

        Returns:
            The final answer text.

        """
        budget = self._input_budget - count_tokens(session.query) - count_tokens(self.SYNTHESIS_INSTRUCTION)
        context = self._formatter.format(self._grounding_context(session), budget)
        messages: list[BaseMessage] = [
            SystemMessage(self._prompt),
            HumanMessage(
                f"<question>\n{session.query.strip()}\n</question>\n\n<context>\n{context}\n</context>\n\n"
                f"{self.SYNTHESIS_INSTRUCTION}"
            ),
        ]
        return self._answer_text(self._invoke(self.llm, messages, session.tracer.config, stream=True))

    @staticmethod
    def _grounding_context(session: _Session) -> list[RedactedNode]:
        """Return every search's passages, then every looked-up provision, deduplicated by id."""
        passages: dict[UUID, RedactedNode] = {}
        for node in [*(node for call in session.searches for node in call.passages), *session.looked_up]:
            passages.setdefault(node.id, node)
        return list(passages.values())

    @staticmethod
    def _strip_document_citations(answer: str) -> str:
        """
        Remove any guideline or ruling citation from `answer`: only provisions may be cited.

        Guidelines, rulings, and papers are research aids the prompt forbids citing; this
        enforces that when the model cites one anyway, e.g. ``[Ruling Malik - 2025 - ...]``.

        Args:
            answer: The agent's final answer text.

        Returns:
            `answer` without ``[Guideline ...]``/``[Ruling ...]`` brackets (and the space before
            them), stripped of surrounding whitespace.

        """
        stripped, count = _DOCUMENT_CITATION.subn("", answer)
        if count:
            log.info("single_corpus_agent_document_citations_removed", count=count)
        return stripped.strip()

    def _result(self, session: _Session, answer: str) -> SingleCorpusResult:
        """
        Assemble the final result, rejecting an empty answer or a run where every search failed.

        Args:
            session: The run's inputs and research record.
            answer: The agent's final answer text.

        Returns:
            The result, whose grounding context is the union of every search's passages.

        Raises:
            SingleCorpusAgentError: If every search failed, or `answer` is blank once any
                guideline/ruling citation is removed.

        """
        if not session.searches:
            message = f"Every search failed ({session.attempts} calls)"
            raise SingleCorpusAgentError(message)
        answer = self._strip_document_citations(answer)
        if not answer.strip():
            message = "Single-corpus agent returned an empty answer"
            raise SingleCorpusAgentError(message)

        grounding_context = self._grounding_context(session)
        log.info(
            "single_corpus_agent_answered",
            questions=[call.question for call in session.searches],
            num_context_nodes=len(grounding_context),
        )
        return SingleCorpusResult(
            query=session.query,
            answer=answer,
            searches=session.searches,
            looked_up=session.looked_up,
            grounding_context=grounding_context,
        )
