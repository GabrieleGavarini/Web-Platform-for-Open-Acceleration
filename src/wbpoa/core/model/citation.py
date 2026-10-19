from dataclasses import dataclass
from typing import Final

PROVISION_TYPES: Final = ("Article", "Recital", "Annex")
OTHER_PROVISION_TYPE: Final = "Other"


@dataclass(frozen=True)
class Citation:
    """One cited top-level provision.

    Attributes:
        instrument_key: Instrument the provision belongs to.
        provision_path: Top-level provision path, such as ``Article 5``.
    """

    instrument_key: str
    provision_path: str

    @property
    def provision_type(self) -> str:
        """Kind of provision: ``Article``, ``Recital``, ``Annex`` or ``Other``.

        Returns:
            The first word of `provision_path` when it is a known type,
            otherwise `OTHER_PROVISION_TYPE`.
        """
        head = self.provision_path.split(" ", 1)[0]
        return head if head in PROVISION_TYPES else OTHER_PROVISION_TYPE

    def __str__(self) -> str:
        """Render as ``instrument_key:provision_path``."""
        return f"{self.instrument_key}:{self.provision_path}"
