from typing import Literal

from pydantic import computed_field

from wbpoa.core.model.element.corpus.provision import Provision


class Recital(Provision):
    kind: Literal["Recital"] = "Recital"
    """Discriminator used to recover the concrete type when deserializing a ValidProvision."""

    @computed_field
    @property
    def to_citation_names(self) -> set[str]:
        """
        Returns a set of string representations of the recital's name suitable for citation.
        """
        return {
            self.name,
            self.name.replace("Recital", "Rec.", 1),
        }
