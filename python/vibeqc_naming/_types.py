"""Shared types for the vibeqc_naming package.

These types are used by multiple modules in the package and are kept
in a single location to avoid circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class NamingSource(str, Enum):
    """Provenance of a naming result."""

    TRIVIAL_IUPAC = "trivial_iupac"
    ORGANIC_SYSTEMATIC = "organic_systematic"
    INORGANIC_COMPOSITIONAL = "inorganic_compositional"
    COORDINATION = "coordination_additive"
    EXTERNAL_RDKIT = "external_rdkit"
    EXTERNAL_CACTUS = "external_cactus"
    EXTERNAL_PUBCHEM = "external_pubchem"


class Confidence(str, Enum):
    """Confidence level for a naming result."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True, repr=False)
class NamedResult:
    """Naming result with provenance and confidence.

    Attributes
    ----------
    name :
        The IUPAC name (or compositional name).
    source :
        Where the name came from.
    confidence :
        Confidence in the accuracy of this name.
    formula :
        Hill-system formula for reference.
    is_trivial :
        Whether this is a retained IUPAC trivial name.
    """

    name: str
    source: NamingSource
    confidence: Confidence
    formula: str = ""
    is_trivial: bool = False

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return (
            f"NamedResult(name={self.name!r}, source={self.source.value!r}, "
            f"confidence={self.confidence.value!r})"
        )

    def _replace(self, **kwargs) -> NamedResult:
        """Create a copy with updated fields (like namedtuple._replace)."""
        d = {
            f.name: kwargs.get(f.name, getattr(self, f.name))
            for f in self.__dataclass_fields__.values()
        }
        return NamedResult(**d)


@dataclass
class OrganicResult:
    """Result of an organic nomenclature attempt."""

    name: str
    confidence: float  # 0.0 – 1.0
    principal_chain_length: int = 0
    has_rings: bool = False
    ring_count: int = 0
    substituents: list[str] = field(default_factory=list)
    functional_groups: list[str] = field(default_factory=list)
