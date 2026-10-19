from typing import Literal

from pydantic import Field, computed_field

from wbpoa.core.model.element.corpus.provision import Provision


class Article(Provision):
    kind: Literal["Article"] = "Article"
    """Discriminator used to recover the concrete type when deserializing a ValidProvision."""

    chapter: str | None = Field(
        default=None,
        description=(
            "Label of the chapter containing this article (e.g. 'chapter III'), or None if the "
            "article is not organised under any chapter."
        ),
    )
    section: str | None = Field(
        default=None,
        description=(
            "Label of the section containing this article (e.g. 'section 1'), or None if the "
            "article is not organised under any section. Section labels repeat across different "
            "chapters, so section membership must always be read together with chapter, never "
            "from section alone."
        ),
    )

    @computed_field
    @property
    def to_citation_names(self) -> set[str]:
        """
        Returns a set of string representations of the article's name suitable for citation.
        """
        return {
            self.name,
            self.name.replace("Article", "Art.", 1),
        }
