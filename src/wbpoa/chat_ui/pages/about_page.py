from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING, Final

from nicegui import ui

from wbpoa.chat_ui.agents.chat_target import ChatTargetKind
from wbpoa.chat_ui.app import routes
from wbpoa.chat_ui.app.routes import Routes
from wbpoa.chat_ui.components.page_frame import PageFrame

if TYPE_CHECKING:
    from wbpoa.chat_ui.agents.types import TargetCatalog

SECTIONS: Final = (
    ("why", "Why a graph"),
    ("how", "How an answer is made"),
    ("corpora", "Corpora"),
    ("reading", "Reading an answer"),
    ("stack", "Models and infrastructure"),
)

_WHY: Final = """
<p>Legal text is a network, not a pile of passages. Provisions refer to other provisions, and
guidelines from authorities, court rulings, and academic papers cite provisions too. Which
provisions are cited together is strong evidence of which provisions belong together. WBPOA
builds that network for each body of law and retrieves through it.</p>
<p>A law is also a family of instruments: a base act, and the acts that amend, implement, or
supplement it. WBPOA keeps every act as its own citable instrument and never makes up a
consolidated text that no authority published, so every claim traces back to a real act.</p>
"""

_HOW: Final = """
<ol>
<li><strong>Ingestion.</strong> Each instrument is split into Articles, Recitals, and Annexes.
References between them, and from guidelines and rulings to them, become the edges of a citation
graph. A co-citation graph links provisions that are cited together, and Leiden clustering groups
them into a hierarchy of topical communities, each with a summary.</li>
<li><strong>Retrieval.</strong> A question is matched against the community summaries first.
Provisions inside the chosen communities are ranked by similarity and citation authority, then
expanded along citation chains, together with the guidelines and rulings that cite them.</li>
<li><strong>Answering.</strong> A research planner breaks the question down into its legal issues.
The agent searches the corpus for each, follows leads (provisions that the retrieved guidelines and
rulings cite), fetches specific provisions, and writes an answer that cites provisions only.</li>
<li><strong>Two laws at once.</strong> For a pairing such as GDPR and NIS2, a manager agent asks one
single-corpus agent per law and combines their answers, citing only what they cited.</li>
</ol>
"""

_READING: Final = f"""
<p>Each bracketed citation the agent writes becomes a footnote marker. The footnotes under the
answer list only the provisions the answer actually cites, and only when they are among the
passages the agent was shown; a citation it was not shown stays in the text as written, in
brackets. Every footnote opens the provision in the <a href="{routes.GRAPH}">citation graph</a>,
where you can follow what cites it and what it cites.</p>
<p>Guidelines and rulings are research aids: the agent reads them to find the governing
provisions but never cites them. The research trail under each answer lists the searches,
lookups, or sub-agent questions behind it.</p>
"""

_STACK: Final = """
<table>
<thead><tr><th>Role</th><th>Model or system</th></tr></thead>
<tbody>
<tr><td>Language model for summaries and agents</td><td>Qwen3.5-35B served by vLLM</td></tr>
<tr><td>Embeddings</td><td>Qwen3-Embedding-0.6B, 1024 dimensions</td></tr>
<tr><td>Graph store and clustering</td><td>Neo4j with Graph Data Science (Leiden) and vector indexes</td></tr>
<tr><td>Lexical index</td><td>Okapi BM25</td></tr>
<tr><td>Tracing</td><td>Langfuse</td></tr>
</tbody>
</table>
<p>Legislative texts come from the EU Publications Office by their CELEX identifiers.</p>
"""


class AboutPage:
    """
    What WBPOA is, how it answers, and how to read this interface, for someone new to the project.

    Attributes:
        catalog: Lists the corpora and pairings the corpora section names.
    """

    def __init__(self, catalog: TargetCatalog) -> None:
        """
        Args:
            catalog: Lists the corpora and pairings.
        """
        self.catalog = catalog

    def render(self) -> None:
        """Build the page."""
        ui.page_title("About · WBPOA")
        with PageFrame.render(routes.ABOUT, ["About WBPOA"]), ui.element("div").classes("wbpoa-reading"):
            ui.html(self.document(), sanitize=False).classes("wbpoa-doc wbpoa-reading-main")
            with (
                ui.element("div").classes("wbpoa-reading-side"),
                ui.element("nav").classes("wbpoa-toc").props('aria-label="On this page"'),
            ):
                ui.html('<h2 class="wbpoa-section-title">On this page</h2>', sanitize=False)
                items = "".join(f'<li><a href="#{anchor}">{title}</a></li>' for anchor, title in SECTIONS)
                ui.html(f"<ol>{items}</ol>", sanitize=False)

    def document(self) -> str:
        """
        Write the page's text.

        Returns:
            The page body as HTML.
        """
        bodies = {
            "why": _WHY,
            "how": _HOW,
            "corpora": self._corpora(),
            "reading": _READING,
            "stack": _STACK,
        }
        sections = "".join(f'<h2 id="{anchor}">{title}</h2>{bodies[anchor]}' for anchor, title in SECTIONS)
        return (
            "<h1>About WBPOA</h1>"
            '<p class="lede">WBPOA, the Web Based Platform for Open Acceleration, answers questions about EU legislation and cites the Articles, Recitals, and '
            "Annexes each answer rests on. It is a research system of the Scuola Superiore Sant&#39;Anna.</p>"
            + sections
        )

    def _corpora(self) -> str:
        """The corpora and pairings, read from their manifests."""
        targets = self.catalog.targets()
        rows = "".join(
            f'<tr><td><a href="{escape(Routes.corpus_url(target.key), quote=True)}">{escape(target.name)}</a></td>'
            f"<td>{escape('; '.join(target.instrument_names))}</td></tr>"
            for target in targets
            if target.kind == ChatTargetKind.SINGLE
        )
        pairings = ", ".join(escape(target.name) for target in targets if target.kind == ChatTargetKind.MULTI)
        table = f"<table><thead><tr><th>Corpus</th><th>Instruments</th></tr></thead><tbody>{rows}</tbody></table>"
        pairing_note = f"<p>Pairings answered by the manager agent: {pairings}.</p>" if pairings else ""
        return (
            "<p>Each corpus is a legal family, kept in its own partition of the graph database so that "
            "rebuilding one law never disturbs another.</p>" + table + pairing_note
        )
