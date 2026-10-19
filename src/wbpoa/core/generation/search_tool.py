from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final, TypedDict
from uuid import UUID

from structlog import get_logger

from wbpoa.core.corpus.corpus_config import InstrumentRole
from wbpoa.core.generation.document_excerpter import DocumentExcerpter
from wbpoa.core.generation.models import CandidateNode, CitedProvision, RedactedNode, SearchResult
from wbpoa.core.inference import RetrievalResult, Retriever

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

    from wbpoa.core.generation.redaction_agent import RedactionAgent
    from wbpoa.core.generation.reformulation_agent import ReformulationAgent
    from wbpoa.core.generation.tracing import LangfuseTracer
    from wbpoa.core.settings import GenerationSettings

log = get_logger()

ELEMENT_TYPES: Final = frozenset(Retriever.ELEMENT_TYPES)


class RetrievalOptions(TypedDict, total=False):
    """
    Algorithm 2 overrides accepted by `SingleCorpusAgent.run`, forwarded to `Retriever.retrieve`.

    Any key left out falls back to InferenceSettings, exactly as a `None` argument to
    `Retriever.retrieve` does.

    Attributes:
        h0: Community graph layer to select candidate communities from.
        k: Number of seed communities, and number of finally selected communities.
        m: Number of neighbour communities expanded from each seed.
        n: Number of layer-0 elements used for the bottom-up vote.
        vote_top_j: Number of top per-community similarities averaged into the bottom-up vote.
        r: Number of top-ranked elements returned as relevant documents.
        ell_elements: Number of element expansion nodes returned from the citation neighbourhood.
        ell_docs: Number of document expansion nodes returned from the citation neighbourhood.
        alpha: Mixing weight between element similarity and citation authority.
        beta: Mixing weight between community similarity and bottom-up vote.
        gamma: Mixing weight between the semantic and lexical similarity channels.
        delta: Per-hop decay applied to expansion similarity; 1.0 disables decay.
        max_hops: Maximum citation hops followed by the cited element closure.
        include_rulings: Whether rulings, not just guidelines, are admitted into the expansion set.
        rerank_enabled: Master switch for Algorithm 2's cascade stages 4(ii) and 7.
        rerank_pool: Shortlist size p handed to the reranker, per pool.
        rho: Rerank blend weight, rho*rsim + (1-rho)*first-pass score.
        rerank_expansion: Whether step 7 reranks its two pools as well as step 4.
        relation_category: If given, restricts retrieval to matching instrument roles; None
            searches every active instrument.
        operative_expansion_enabled: Whether relevant base-instrument elements are expanded
            toward newer amending/implementing provisions via OPERATIVE_RELATION edges.
        operative_decay: Flat decay applied to xsim for operative-expansion results.

    """

    h0: int | None
    k: int | None
    m: int | None
    n: int | None
    vote_top_j: int | None
    r: int | None
    ell_elements: int | None
    ell_docs: int | None
    alpha: float | None
    beta: float | None
    gamma: float | None
    delta: float | None
    max_hops: int | None
    include_rulings: bool | None
    rerank_enabled: bool | None
    rerank_pool: int | None
    rho: float | None
    rerank_expansion: bool | None
    relation_category: Sequence[InstrumentRole] | None
    operative_expansion_enabled: bool | None
    operative_decay: float | None


class SearchTool:
    """
    The single-corpus agent's search tool: reformulation -> Algorithm 2 retrieval -> redaction.

    One search is Algorithm 3's steps 1-4 for one question, with exactly one search: the
    question is reformulated into one query q' (unless reformulation is disabled, when q' is the
    question), Algorithm 2 runs once on q', its results are pooled into U, and U is filtered into
    S by the redaction agent (unless redaction is disabled, when S = U). Every step gets its own Langfuse span when tracing is enabled.

    Attributes:
        retriever: Algorithm 2's retriever, used unmodified.
        reformulation_agent: Step 1's agent.
        redaction_agent: Step 4's agent.

    """

    def __init__(
        self,
        retriever: Retriever,
        reformulation_agent: ReformulationAgent,
        redaction_agent: RedactionAgent,
        settings: GenerationSettings,
    ) -> None:
        """
        Args:
            retriever: Algorithm 2's retriever, used unmodified.
            reformulation_agent: Step 1's agent.
            redaction_agent: Step 4's agent.
            settings: The reformulation/redaction switches and the `max_context_nodes` cap.

        """
        self.retriever = retriever
        self.reformulation_agent = reformulation_agent
        self.redaction_agent = redaction_agent
        self._settings = settings
        self._excerpter = DocumentExcerpter(settings.document_excerpt_tokens, settings.document_excerpts)

    def search(self, question: str, tracer: LangfuseTracer, retrieval_options: RetrievalOptions) -> SearchResult:
        """
        Search for `question`, wrapped in a `Search Tool` span when tracing is enabled.

        Args:
            question: The question the single-corpus agent asked.
            tracer: The Langfuse tracer opened by the single-corpus agent's `run`.
            retrieval_options: Algorithm 2 overrides, forwarded verbatim to `Retriever.retrieve`.

        Returns:
            The call's reformulated queries, retrievals, candidate pool, and passages.

        Raises:
            ReformulationError: Propagated from the reformulation agent.
            RetrieverError: Propagated from Retriever.retrieve.
            RedactionError: Propagated from the redaction agent.

        """
        if tracer.client is None:
            return self._search(question, tracer, retrieval_options)

        with tracer.client.start_as_current_observation(name="Search Tool", as_type="tool", input=question) as span:
            result = self._search(question, tracer, retrieval_options)
            span.update(output=self._passages_trace_output([(node.id, {}) for node in result.passages]))
        return result

    def _search(self, question: str, tracer: LangfuseTracer, retrieval_options: RetrievalOptions) -> SearchResult:
        """
        Run steps 1-4, assuming any enclosing Langfuse span has already been opened.

        Args:
            question: The question the single-corpus agent asked.
            tracer: The Langfuse tracer opened by the single-corpus agent's `run`.
            retrieval_options: Algorithm 2 overrides, forwarded verbatim to `Retriever.retrieve`.

        Returns:
            The call's reformulated query, retrieval, candidate pool, and passages.

        """
        reformulated_query = self._reformulate(question, tracer)
        log.info("step_1_reformulation", question=question, reformulated_query=reformulated_query)

        retrieval = self._retrieve(reformulated_query, tracer, retrieval_options)

        candidate_pool = self._build_candidate_pool(retrieval)
        log.info("steps_2_3_candidate_pool", num_candidates=len(candidate_pool))

        passages = self._redact(question, candidate_pool, tracer)
        log.info("step_4_redaction", num_selected=len(passages))

        cited_provisions = self._cited_provisions(passages)
        log.info("search_leads", leads=[provision.name for provision in cited_provisions])

        return SearchResult(
            question=question,
            reformulated_query=reformulated_query,
            retrieval=retrieval,
            candidate_pool=candidate_pool,
            passages=passages,
            cited_provisions=cited_provisions,
        )

    def _cited_provisions(self, passages: list[RedactedNode]) -> list[CitedProvision]:
        """
        Collect the provisions the guideline/ruling passages cite that are not passages themselves.

        Args:
            passages: S, the passages returned by this call.

        Returns:
            One entry per distinct cited provision, carrying the names of the documents citing
            it, ordered by number of citing documents (most first), then by name.

        """
        documents = {node.id: node for node in passages if node.type not in ELEMENT_TYPES}
        if not documents:
            return []
        passage_ids = {node.id for node in passages}
        provisions: dict[UUID, CitedProvision] = {}
        for document_id, node in self.retriever.driver.cited_provisions(list(documents)):
            if node.id in passage_ids:
                continue
            provision = provisions.setdefault(
                node.id,
                CitedProvision(id=node.id, name=node.name, type=node.type, qualified_citation=node.qualified_citation),
            )
            document_name = documents[document_id].name
            if document_name not in provision.cited_by:
                provision.cited_by.append(document_name)
        return sorted(provisions.values(), key=lambda provision: (-len(provision.cited_by), provision.name))

    def _reformulate(self, question: str, tracer: LangfuseTracer) -> str:
        """
        Step 1, wrapped in its own Langfuse span when tracing is enabled.

        Always produces the `Reformulation Agent` span, whether or not the reformulation agent
        itself ran, so the trace has a consistent step for what query fed Algorithm 2.

        Args:
            question: The question being researched.
            tracer: The Langfuse tracer opened by the single-corpus agent's `run`.

        Returns:
            q', the one query handed to Algorithm 2: the reformulated question, or the question
            itself when reformulation is disabled.

        """
        reformulation_enabled = self._settings.reformulation_enabled
        if tracer.client is None:
            if reformulation_enabled:
                return self.reformulation_agent.reformulate(question, config=tracer.config)
            return question

        with tracer.client.start_as_current_observation(name="Reformulation Agent", input=question) as span:
            reformulated_query = (
                self.reformulation_agent.reformulate(question, config=tracer.config)
                if reformulation_enabled
                else question
            )
            span.update(output=reformulated_query)
        return reformulated_query

    def _select_context(
        self, question: str, candidate_pool: list[CandidateNode], config: RunnableConfig | None
    ) -> list[RedactedNode]:
        """
        Step 4: select S subseteq U, via the redaction agent or, when disabled, the full pool.

        Args:
            question: The question being researched, which redaction judges passages against.
            candidate_pool: U, the candidate pool built from Algorithm 2's output.
            config: Optional LangChain run config (e.g. carrying a Langfuse callback handler).

        Returns:
            S, truncated to `GenerationSettings.max_context_nodes` when set.

        """
        if self._settings.redaction_enabled:
            context = self.redaction_agent.redact(question, candidate_pool, config=config)
        else:
            context = [
                RedactedNode(id=c.id, name=c.name, type=c.type, text=c.text, qualified_citation=c.qualified_citation)
                for c in candidate_pool
            ]
        if self._settings.max_context_nodes is not None:
            context = context[: self._settings.max_context_nodes]
        by_id = {candidate.id: candidate for candidate in candidate_pool}
        return [self._document_text(node, by_id.get(node.id), question) for node in context]

    def _document_text(self, node: RedactedNode, candidate: CandidateNode | None, question: str) -> RedactedNode:
        """
        Return `node` with a guideline/ruling's text set per `GenerationSettings.document_field`.

        Args:
            node: A kept passage; provisions are returned unchanged.
            candidate: The pool entry `node` came from, carrying the document's text and summary.
            question: The question being researched, which an excerpt is chosen against.

        Returns:
            `node`, its text replaced by the document's excerpt, summary, or full text. An excerpt
            falls back to the summary when no part of the document shares a term with `question`.

        """
        if node.type in ELEMENT_TYPES or candidate is None:
            return node
        match self._settings.document_field:
            case "summary":
                text = candidate.summary
            case "text":
                text = candidate.text
            case "excerpt":
                text = self._excerpter.excerpt(candidate.text, question) or candidate.summary
        return node.model_copy(update={"text": text})

    def _redact(self, question: str, candidate_pool: list[CandidateNode], tracer: LangfuseTracer) -> list[RedactedNode]:
        """
        Step 4, wrapped in its own Langfuse span when tracing is enabled.

        Always produces the `Redaction Agent` span, whether or not the redaction agent itself
        ran, so the trace has a consistent step for what ended up in S.

        Args:
            question: The question being researched.
            candidate_pool: U, the candidate pool built from Algorithm 2's output.
            tracer: The Langfuse tracer opened by the single-corpus agent's `run`.

        Returns:
            S, the passages returned to the single-corpus agent.

        """
        if tracer.client is None:
            return self._select_context(question, candidate_pool, tracer.config)

        with tracer.client.start_as_current_observation(name="Redaction Agent", as_type="tool", input=question) as span:
            context = self._select_context(question, candidate_pool, tracer.config)
            span.update(output=self._passages_trace_output([(node.id, {}) for node in context]))
        return context

    def _retrieve(
        self, reformulated_query: str, tracer: LangfuseTracer, retrieval_options: RetrievalOptions
    ) -> RetrievalResult:
        """
        Steps 2-3: run Algorithm 2, wrapped in its own Langfuse span when tracing is enabled.

        Retriever is never itself instrumented (Remark 17): this wraps the call from the outside.

        Args:
            reformulated_query: q', Algorithm 2's input.
            tracer: The Langfuse tracer opened by the single-corpus agent's `run`.
            retrieval_options: Algorithm 2 overrides, forwarded verbatim to `Retriever.retrieve`.

        Returns:
            Algorithm 2's result: the relevant elements R and the two expansion pools.

        """
        if tracer.client is None:
            return self.retriever.retrieve(reformulated_query, **retrieval_options)

        with tracer.client.start_as_current_observation(
            name="Retriever", as_type="retriever", input=reformulated_query
        ) as span:
            retrieval = self.retriever.retrieve(reformulated_query, **retrieval_options)
            span.update(output=self._retrieval_trace_output(retrieval))
        return retrieval

    def _retrieval_trace_output(self, retrieval: RetrievalResult) -> dict[str, list[dict[str, Any]]]:
        """
        Build the `Retriever` Langfuse span's output: relevant elements and expansion nodes.

        Args:
            retrieval: Algorithm 2's result.

        Returns:
            Two lists keyed "relevant_elements" and "expansion_nodes", each entry the full node
            content (see `_passages_trace_output`); expansion node entries also carry `pool`.

        """
        return {
            "relevant_elements": self._passages_trace_output(
                [(element.id, {}) for element in retrieval.relevant_elements]
            ),
            "expansion_nodes": self._passages_trace_output(
                [(node.id, {"pool": node.pool}) for node in retrieval.expansion_nodes]
            ),
        }

    def _passages_trace_output(self, entries: Sequence[tuple[UUID, dict[str, Any]]]) -> list[dict[str, Any]]:
        """
        Fetch each id's full Node by id and return one labeled dict per passage, input order preserved.

        Looks the ids up against `self.retriever.driver` -- the canonical stored node -- rather
        than serialising whatever lightweight wrapper (CandidateNode/RedactedNode) the caller
        holds, so agents never need to carry more than id/name/type/text and Langfuse still gets
        the full passage.

        Args:
            entries: `(id, extra_fields)` pairs, in the order they should appear in the trace;
                `extra_fields` is merged into that passage's dict (e.g. `{"pool": "element"}`).

        Returns:
            One dict per resolvable id -- id/name/type/text/summary/layer plus `extra_fields` --
            excluding `summary_embedding`. Ids `nodes_by_ids` doesn't resolve are skipped.

        """
        if not entries:
            return []
        nodes_by_id = {
            node.id: node for node in self.retriever.driver.nodes_by_ids([node_id for node_id, _ in entries])
        }
        output: list[dict[str, Any]] = []
        for node_id, extra in entries:
            node = nodes_by_id.get(node_id)
            if node is None:
                log.warning("passage_trace_lookup_missing", id=str(node_id))
                continue
            output.append(
                {
                    "id": str(node.id),
                    "name": node.name,
                    "type": node.type,
                    "text": node.text,
                    "summary": node.summary,
                    "layer": node.layer,
                    **extra,
                }
            )
        return output

    @staticmethod
    def _build_candidate_pool(retrieval: RetrievalResult) -> list[CandidateNode]:
        """
        Build U, the union of R and both expansion pools of the one retrieval, deduplicated by id.

        Args:
            retrieval: Algorithm 2's result for the reformulated query q'.

        Returns:
            One CandidateNode per distinct node of R, N_elem and N_doc, relevant elements first.

        """
        candidates: dict[UUID, CandidateNode] = {}
        for element in retrieval.relevant_elements:
            candidates.setdefault(
                element.id,
                CandidateNode(
                    id=element.id,
                    name=element.name,
                    type=element.type,
                    text=element.text,
                    summary=element.summary,
                    origin="relevant",
                    qualified_citation=element.qualified_citation,
                ),
            )
        for node in retrieval.expansion_nodes:
            origin = "element_expansion" if node.pool == "element" else "document_expansion"
            candidates.setdefault(
                node.id,
                CandidateNode(
                    id=node.id,
                    name=node.name,
                    type=node.type,
                    text=node.text,
                    summary=node.summary,
                    origin=origin,
                    qualified_citation=node.qualified_citation,
                ),
            )
        return list(candidates.values())
