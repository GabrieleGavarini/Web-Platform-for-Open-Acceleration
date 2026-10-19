from pydantic import Field

from wbpoa.core.model.graph.edge import Edge
from wbpoa.core.model.graph.node import Node


class EdgeUndirected(Edge):
    nodes: tuple[Node, Node] = Field(
        ...,
        description="The pair of nodes connected by the undirected edge.",
    )

    def __hash__(self) -> int:
        return hash(self.id)
