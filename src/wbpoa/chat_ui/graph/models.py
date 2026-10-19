from uuid import UUID

from pydantic import BaseModel, Field


class NodeSummary(BaseModel):
    """One node of the citation graph as listed by the browser: enough to label and link it."""

    id: UUID
    name: str
    type: str = Field(description="Article, Recital, Annex, Guideline, or Ruling.")
    instrument_name: str | None = Field(
        default=None, description="Name of the provision's instrument; None for guidelines and rulings."
    )

    @property
    def is_provision(self) -> bool:
        """Whether the node is an Article, Recital, or Annex of an instrument."""
        return self.instrument_name is not None


class NeighbourGroup(BaseModel):
    """The neighbours of one type on one side of a node: what it cites, or what cites it."""

    type: str
    nodes: list[NodeSummary] = Field(default_factory=list)


class CoCitation(BaseModel):
    """A provision cited together with a node, and how strongly: the co-citation arch's weight."""

    node: NodeSummary
    weight: float


class CommunityView(BaseModel):
    """A topical community a provision belongs to: its title, summary, and the provisions it groups."""

    id: UUID
    name: str
    layer: int = Field(description="Hierarchy level, 1 being the finest.")
    summary: str
    members: list[NodeSummary] = Field(default_factory=list, description="Its provisions, naturally ordered.")


class GraphNodeView(BaseModel):
    """Everything the node page shows: the node, its text, and its citation and co-citation neighbourhoods."""

    corpus_key: str
    corpus_name: str
    node: NodeSummary
    title: str | None = Field(default=None, description="An article's heading, e.g. 'Lawfulness of processing'.")
    celex_id: str | None = None
    text: str
    summary: str
    cites: list[NeighbourGroup] = Field(default_factory=list, description="Nodes this node cites, grouped by type.")
    cited_by: list[NeighbourGroup] = Field(default_factory=list, description="Nodes citing this node, grouped by type.")
    co_cited: list[CoCitation] = Field(
        default_factory=list, description="A provision's co-cited provisions, heaviest first; empty for documents."
    )
    communities: list[CommunityView] = Field(
        default_factory=list, description="The communities holding a provision, finest first; empty for documents."
    )

    @property
    def cites_count(self) -> int:
        """Number of nodes this node cites."""
        return sum(len(group.nodes) for group in self.cites)

    @property
    def cited_by_count(self) -> int:
        """Number of nodes citing this node."""
        return sum(len(group.nodes) for group in self.cited_by)
