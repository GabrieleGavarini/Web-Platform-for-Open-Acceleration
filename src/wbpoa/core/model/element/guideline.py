from typing import ClassVar

from pydantic import BaseModel

from wbpoa.core.model.element.element import Element


class Guideline(Element):
    """
    Represents a guideline element.
    Has a citation weight of 0.5.
    """

    citation_weight: ClassVar[float] = 0.5


class Guidelines(BaseModel):
    """
    Represents a collection of guidelines.
    """

    guidelines: list[Guideline]
