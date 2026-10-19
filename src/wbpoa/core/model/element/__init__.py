from pydantic import BaseModel

from wbpoa.core.model.element.corpus import Annex, Article, Corpus, Instrument, Provision, Recital
from wbpoa.core.model.element.corpus.types import ValidProvision
from wbpoa.core.model.element.element import Element
from wbpoa.core.model.element.guideline import Guideline, Guidelines
from wbpoa.core.model.element.ruling import Ruling


class Rulings(BaseModel):
    """
    Represents a collection of rulings.
    """

    rulings: list[Ruling]


__all__ = [
    "Element",
    "Provision",
    "Annex",
    "Article",
    "Recital",
    "Guideline",
    "Guidelines",
    "Ruling",
    "Rulings",
    "Corpus",
    "Instrument",
    "ValidProvision",
]
