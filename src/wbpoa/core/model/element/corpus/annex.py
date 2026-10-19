from typing import Literal

from pydantic import computed_field

from wbpoa.core.model.element.corpus.provision import Provision


class Annex(Provision):
    kind: Literal["Annex"] = "Annex"
    """Discriminator used to recover the concrete type when deserializing a ValidProvision."""

    @computed_field
    @property
    def to_citation_names(self) -> set[str]:
        """
        Returns a set of string representations of the annex's name suitable for citation.
        """
        return {
            self.name,
        }
