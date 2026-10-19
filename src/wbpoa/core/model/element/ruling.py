from typing import ClassVar

from wbpoa.core.model.element.element import Element


class Ruling(Element):
    """
    Represents a ruling element.
    Has a citation weight of 0.25.
    """

    citation_weight: ClassVar[float] = 0.25
