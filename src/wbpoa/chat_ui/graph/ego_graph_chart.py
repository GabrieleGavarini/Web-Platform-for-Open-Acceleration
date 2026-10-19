from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from wbpoa.chat_ui.app.routes import Routes

if TYPE_CHECKING:
    from wbpoa.chat_ui.graph.models import GraphNodeView, NodeSummary

MAX_NEIGHBOURS: Final = 16
"""Co-cited provisions drawn at most, the heaviest ties kept: eight a side, each with room for its label."""

SPREAD_MAX: Final = 72.0
"""Half the arc, in degrees, a side's neighbours fan over at most."""

SPREAD_STEP: Final = 9.0
"""Degrees of arc each further neighbour adds, so a few stay near the horizontal."""

CROWDED: Final = 8
"""Neighbours on a side beyond which every other one moves to the inner arc, so labels do not touch."""

INNER_RADIUS: Final = 0.72
"""The inner arc's radius, as a share of the outer one."""

WIDTH_RATIO: Final = 1.0
"""How much wider than tall the drawing is: the pane's box is barely wider than tall, and the labels take the sides."""

RADIUS: Final = 100.0
"""The outer arc's radius in drawing units; ECharts misplaces near-horizontal edges at unit scale."""

MAX_WIDTH: Final = 4.0
"""Line width, in pixels, of the heaviest tie drawn; the others are thinner in proportion to their weight."""

MIN_WIDTH: Final = 0.5
"""The thinnest a line is drawn, so a light tie stays visible."""

EDGE_OPACITY: Final = 0.8
"""Opacity of every line: the width alone tells the weight."""


@dataclass(frozen=True, slots=True)
class EgoGraphPalette:
    """
    Colours of the ego graph, taken from the page theme so the chart follows it.

    Attributes:
        focus: The node the page is about.
        provision: Articles, Recitals, and Annexes.
        edge: Co-citation lines.
        label: Node labels.
    """

    focus: str
    provision: str
    edge: str
    label: str


@dataclass(frozen=True, slots=True)
class EgoGraph:
    """
    An ECharts option drawing a node's co-citation neighbourhood, and where each drawn node links to.

    Attributes:
        options: The ECharts option.
        links: The URL of each drawn node, indexed like the series data; None for the focus node.
    """

    options: dict[str, object]
    links: list[str | None] = field(default_factory=list)

    def link_at(self, data_type: str | None, data_index: int) -> str | None:
        """
        The page a click on the drawing opens.

        Args:
            data_type: What ECharts says was clicked: ``node`` or ``edge``.
            data_index: Its index in the series' nodes or links.

        Returns:
            The clicked neighbour's URL; None for the focus, a line, or an index out of range.
        """
        if data_type != "node" or not 0 <= data_index < len(self.links):
            return None
        return self.links[data_index]


class EgoGraphChart:
    """
    Draw a node and the provisions cited together with it, each tie as thick as it is heavy.

    The node sits in the middle and its co-cited provisions fan out on an arc to either side, taken
    in turn by falling weight: the heaviest straight to its right, the next straight to its left,
    the lighter ones further up and down each arc; when a side is crowded, every other one moves to
    an inner arc. A line joins the node to each of them, its width in proportion to the co-citation
    weight. The positions are fixed, so the drawing fills its box at any size and is the same on
    every visit.
    """

    def __init__(self, palette: EgoGraphPalette, max_neighbours: int = MAX_NEIGHBOURS) -> None:
        """
        Args:
            palette: Colours from the page theme.
            max_neighbours: Co-cited provisions drawn, at most.
        """
        self.palette = palette
        self.max_neighbours = max_neighbours

    def build(self, view: GraphNodeView) -> EgoGraph:
        """
        Build the chart of one node's co-citation neighbourhood.

        Args:
            view: The node and its co-cited provisions, heaviest first.

        Returns:
            The ECharts option and the link of every drawn node; only the focus is drawn when no
            provision is cited together with it.
        """
        focus_id = str(view.node.id)
        ties = [tie for tie in view.co_cited if str(tie.node.id) != focus_id][: self.max_neighbours]
        heaviest = max((tie.weight for tie in ties), default=0.0)

        data: list[dict[str, object]] = [self._point(view.node, (0.0, 0.0), focus=True)]
        links: list[str | None] = [None]
        edges: list[dict[str, object]] = []
        for index, tie in enumerate(ties):
            data.append(self._point(tie.node, self._place(index, len(ties)), focus=False, weight=tie.weight))
            links.append(Routes.node_url(view.corpus_key, tie.node.id))
            share = tie.weight / heaviest if heaviest > 0 else 0.0
            edges.append(
                {
                    "source": focus_id,
                    "target": str(tie.node.id),
                    "value": tie.weight,
                    "lineStyle": {"width": round(max(MIN_WIDTH, MAX_WIDTH * share), 2)},
                    "tooltip": {"formatter": f"{view.node.name} and {tie.node.name} · weight {tie.weight:.1f}"},
                }
            )

        options: dict[str, object] = {
            "animationDurationUpdate": 200,
            "tooltip": {"trigger": "item", "confine": True},
            "series": [
                {
                    "type": "graph",
                    "layout": "none",
                    "roam": True,
                    "zoom": 0.8,
                    "labelLayout": {"hideOverlap": True},
                    "edgeSymbol": ["none", "none"],
                    "lineStyle": {"color": self.palette.edge, "opacity": EDGE_OPACITY, "curveness": 0},
                    "emphasis": {"focus": "adjacency", "label": {"show": True}},
                    "label": {"color": self.palette.label, "fontSize": 11},
                    "data": data,
                    "links": edges,
                }
            ],
        }
        return EgoGraph(options=options, links=links)

    @staticmethod
    def _place(rank: int, count: int) -> tuple[float, float]:
        """
        Where the neighbour of weight rank `rank`, out of `count`, sits.

        Ranks alternate right and left; on its side, a neighbour takes the free slot of the arc
        nearest the horizontal, so the heavier ties sit level with the focus. When the side is
        crowded, every other slot, counted from the middle one, is on the inner arc.

        Args:
            rank: The neighbour's place in falling weight, 0 the heaviest.
            count: Neighbours drawn on both sides.

        Returns:
            The ``(x, y)`` position, the focus at the origin.
        """
        side = 1 if rank % 2 == 0 else -1
        on_side = (count + 1) // 2 if side == 1 else count // 2
        middle = (on_side - 1) / 2
        slots = sorted(range(on_side), key=lambda slot: (abs(slot - middle), slot))
        slot = slots[rank // 2]
        spread = min(SPREAD_MAX, SPREAD_STEP * (on_side - 1))
        angle = math.radians(-spread + 2 * spread * slot / (on_side - 1)) if on_side > 1 else 0.0
        inner = on_side > CROWDED and (slot - slots[0]) % 2 == 1
        radius = RADIUS * (INNER_RADIUS if inner else 1.0)
        return side * radius * WIDTH_RATIO * math.cos(angle), radius * math.sin(angle)

    def _point(
        self, node: NodeSummary, at: tuple[float, float], *, focus: bool, weight: float = 0.0
    ) -> dict[str, object]:
        """
        One drawn node at `at`, labelled away from the focus: the focus large and bold, its neighbours small.

        Every node carries a value, its tie's weight (none for the focus): NiceGUI drops a click on
        a node without one.
        """
        colour, size = (self.palette.focus, 22) if focus else (self.palette.provision, 11)
        tooltip = f"{node.name}, {node.instrument_name}" if node.instrument_name else node.name
        return {
            "id": str(node.id),
            "name": node.name,
            "value": weight,
            "x": at[0],
            "y": at[1],
            "symbolSize": size,
            "itemStyle": {"color": colour},
            "label": {
                "show": True,
                "position": "bottom" if focus else "right" if at[0] > 0 else "left",
                "fontWeight": "bold" if focus else "normal",
            },
            "tooltip": {"formatter": tooltip},
        }
