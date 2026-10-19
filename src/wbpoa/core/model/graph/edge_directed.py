from pydantic import Field

from wbpoa.core.model.graph.edge import Edge
from wbpoa.core.model.graph.node import Node


class EdgeDirected(Edge):
    source: Node = Field(
        ...,
        description="The source node of the directed edge.",
    )
    target: Node = Field(
        ...,
        description="The target node of the directed edge.",
    )

    def __hash__(self) -> int:
        return hash(self.id)
