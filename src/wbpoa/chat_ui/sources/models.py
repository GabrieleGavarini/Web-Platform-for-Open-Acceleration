from uuid import UUID

from pydantic import BaseModel, Field

from wbpoa.core.generation.research_observer import ResearchEventKind


class UsedSource(BaseModel):
    """A provision the answer actually cites, numbered as its footnote."""

    number: int = Field(description="Footnote number, in order of first citation in the answer.")
    corpus_key: str = Field(description="Corpus partition holding the provision, for its graph link.")
    node_id: UUID
    provision: str = Field(description="The provision's path, e.g. 'Article 6'.")
    instrument_name: str
    celex_id: str
    excerpt: str = Field(description="The opening of the provision's text, for the footnote.")
    url: str = Field(description="The provision's page in the citation graph.")
    anchor: str = Field(description="Id of its footnote, unique on the page, which the answer's citations link to.")


class FoundPassage(BaseModel):
    """A passage a search returned or a lookup fetched, linked to its graph page."""

    name: str
    type: str = Field(description="Article, Recital, Annex, Guideline, or Ruling.")
    instrument_name: str | None = Field(default=None, description="The provision's instrument; None for documents.")
    url: str


class ResearchStep(BaseModel):
    """One step of the research behind an answer, shown live while it runs and in the reply's trail."""

    kind: ResearchEventKind
    corpus_key: str
    text: str = Field(description="The question searched or delegated, or the provision looked up.")
    found: list[FoundPassage] = Field(default_factory=list, description="What the step returned.")
    error: str | None = Field(default=None, description="Why a search or delegation failed; None when it succeeded.")
    running: bool = Field(default=False, description="True while the step has started but not finished yet.")
    agent: str | None = Field(
        default=None,
        description="Name of the sub agent's corpus the step belongs to, in a pairing; None for a single corpus.",
    )
    agent_slot: int = Field(default=0, description="Position of that corpus in the pairing, which picks its colour.")


class ChatReply(BaseModel):
    """One question and what the chat shows for it: the footnoted answer, its sources, and its research."""

    question: str
    target_name: str
    answer_html: str = Field(
        default="", description="The answer as escaped HTML, citations turned into footnote links."
    )
    sources: list[UsedSource] = Field(default_factory=list, description="Only the provisions the answer cites.")
    research: list[ResearchStep] = Field(default_factory=list)
    error: str | None = Field(default=None, description="Why no answer could be produced; None on success.")
    seconds: float = Field(default=0.0, description="How long the agent took.")
