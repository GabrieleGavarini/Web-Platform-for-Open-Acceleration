from typing import Annotated

from pydantic import Field

from wbpoa.core.model.element.corpus.annex import Annex
from wbpoa.core.model.element.corpus.article import Article
from wbpoa.core.model.element.corpus.recital import Recital

# Explicit type alias so type checkers treat this consistently across modules, instead of
# occasionally inferring it as an ordinary variable when imported (which broke covariant
# `Sequence[Element]` assignability checks at call sites like `_create_nodes_batch`).
type ValidProvision = Annotated[Article | Annex | Recital, Field(discriminator="kind")]
