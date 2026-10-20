"""Canonical basis-set data model for vibe-qc.

This module defines the typed, structured representation for Gaussian-type
orbital (GTO) basis sets, effective core potentials (ECPs), auxiliary
basis sets, and their metadata.  It is the single source of truth for
basis-set data inside vibe-qc -- all importers produce these types and
all exporters consume them.

Design principles
-----------------
- **No ambiguity.**  Every field has a well-defined semantic.
  ``harmonic_type`` is explicit; ``angular_momentum`` is a list of ints;
  contraction coefficients follow the G94 convention (unnormalized
  primitives, contraction coefficient includes normalization).
- **Serializable.**  All types are ``@dataclass`` with ``asdict()``-style
  serialisation to plain dicts.  Deserialisation from dicts is
  constructor-only (no custom ``__init__``).
- **Validation-ready.**  Every type is simple enough that a JSON Schema
  can validate it without custom logic.  ``Optional`` fields default to
  ``None``; required fields have no default.
- **BSE / QCSchema aligned.**  The model is designed to accept BSE JSON
  as its primary import format and QCSchema-style structured JSON as an
  alternative.  The element-based (rather than center-based) organisation
  matches BSE's layout.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

__all__ = [
    "HarmonicType",
    "BasisRole",
    "Primitive",
    "Shell",
    "ECPPotential",
    "ElementBasis",
    "ECPEntry",
    "Reference",
    "Provenance",
    "ProvenanceKind",
    "Upstream",
    "CitationRoute",
    "BasisSetData",
    "LossReport",
    "SCHEMA_VERSION",
]

#: The QVF-Basis schema version this model writes. 1.1.0 widened
#: ``provenance`` (kind, upstream identity, content hash, per-range citation
#: routes); every 1.0.0 record still validates and loads.
SCHEMA_VERSION = "1.1.0"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class HarmonicType(str, Enum):
    """Spherical vs. Cartesian harmonic convention.

    Spherical: (2L+1) functions (the physical convention).
    Cartesian: (L+1)(L+2)/2 functions (the "6d" / "10f" convention).
    """

    SPHERICAL = "spherical"
    CARTESIAN = "cartesian"


class BasisRole(str, Enum):
    """High-level role a basis set plays in a calculation."""

    ORBITAL = "orbital"
    AUXILIARY = "auxiliary"
    FITTING = "fitting"
    ECP = "ecp"


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


@dataclass
class Primitive:
    """A single primitive Gaussian function.

    Coefficients follow the G94 / Gaussian convention: the primitive is
    *unnormalized* and the contraction coefficient includes normalisation.
    """

    exponent: float
    coefficient: float

    def to_dict(self) -> Dict[str, Any]:
        return {"exponent": self.exponent, "coefficient": self.coefficient}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Primitive":
        return cls(exponent=float(d["exponent"]), coefficient=float(d["coefficient"]))


@dataclass
class Shell:
    """One contracted shell of Gaussian primitives.

    Attributes
    ----------
    angular_momentum:
        List of angular-momentum quantum numbers this shell carries.
        ``[0]`` = S, ``[1]`` = P, ``[2]`` = D, ``[0, 1]`` = SP (shared
        exponents with separate s- and p-contraction coefficients).
    harmonic_type:
        Whether the shell produces spherical or Cartesian functions.
    primitives:
        Primitive Gaussians, ordered by exponent descending (standard
        convention for G94 compatibility).  The length of this list must
        equal the length of each coefficient row in the source format.
        For an SP shell the coefficients are interleaved: the first
        ``N`` coefficients are the s contraction and the next ``N`` are
        the p contraction, where ``N == len(primitives)``.
    """

    angular_momentum: List[int]  # length 1 for pure, 2 for SP
    harmonic_type: HarmonicType
    primitives: List[Primitive]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "angular_momentum": [int(am) for am in self.angular_momentum],
            "harmonic_type": self.harmonic_type.value,
            "primitives": [p.to_dict() for p in self.primitives],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Shell":
        return cls(
            angular_momentum=[int(am) for am in d["angular_momentum"]],
            harmonic_type=HarmonicType(d["harmonic_type"]),
            primitives=[Primitive.from_dict(p) for p in d["primitives"]],
        )


@dataclass
class ECPPotential:
    """One angular-momentum channel of an effective core potential.

    The potential is expressed as a sum of Gaussians:

        U_L(r) = r^{-2} S_i coefficient_i * r^{n_i} * exp(-exponent_i * r^2)
    """

    angular_momentum: int  # L quantum number for this channel
    terms: List[ECPTerm] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "angular_momentum": self.angular_momentum,
            "terms": [t.to_dict() for t in self.terms],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ECPPotential":
        return cls(
            angular_momentum=int(d["angular_momentum"]),
            terms=[ECPTerm.from_dict(t) for t in d.get("terms", [])],
        )


@dataclass
class ECPTerm:
    """A single Gaussian term in an ECP potential channel."""

    coefficient: float
    exponent: float
    # Gaussian input includes the Jacobian: libecpint subtracts two from n.
    # https://libecpint.readthedocs.io/en/latest/usage.html#important-note-about-ecp-definitions
    power: int  # Gaussian n=0..4 in coefficient * r^(n-2) * exp(-exponent * r^2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coefficient": self.coefficient,
            "exponent": self.exponent,
            "power": self.power,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ECPTerm":
        return cls(
            coefficient=float(d["coefficient"]),
            exponent=float(d["exponent"]),
            power=int(d["power"]),
        )


@dataclass
class ElementBasis:
    """Basis functions for a single chemical element.

    ``ecp_id``, when set, names the ECP entry paired with this basis.
    """

    element: str  # canonical element symbol, e.g. "H", "Fe"
    shells: List[Shell] = field(default_factory=list)
    ecp_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "element": self.element,
            "shells": [s.to_dict() for s in self.shells],
        }
        if self.ecp_id is not None:
            d["ecp_id"] = self.ecp_id
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ElementBasis":
        return cls(
            element=str(d["element"]),
            shells=[Shell.from_dict(s) for s in d.get("shells", [])],
            ecp_id=d.get("ecp_id"),
        )


@dataclass
class ECPEntry:
    """Effective core potential definition for a single element."""

    element: str
    n_core_electrons: int
    angular_momentum_max: int  # highest L with an explicit potential
    potentials: List[ECPPotential] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "element": self.element,
            "n_core_electrons": self.n_core_electrons,
            "angular_momentum_max": self.angular_momentum_max,
            "potentials": [p.to_dict() for p in self.potentials],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ECPEntry":
        return cls(
            element=str(d["element"]),
            n_core_electrons=int(d["n_core_electrons"]),
            angular_momentum_max=int(d.get("angular_momentum_max", 0)),
            potentials=[ECPPotential.from_dict(p) for p in d.get("potentials", [])],
        )


@dataclass
class Reference:
    """A bibliographic reference for the basis set or its components."""

    key: str  # short citation key, e.g. "dunning1989"
    description: str = ""
    doi: Optional[str] = None
    bibtex: Optional[str] = None
    url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"key": self.key, "description": self.description}
        if self.doi:
            d["doi"] = self.doi
        if self.bibtex:
            d["bibtex"] = self.bibtex
        if self.url:
            d["url"] = self.url
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Reference":
        return cls(
            key=str(d["key"]),
            description=str(d.get("description", "")),
            doi=d.get("doi"),
            bibtex=d.get("bibtex"),
            url=d.get("url"),
        )


class ProvenanceKind(str, Enum):
    """What class of record this is (#295 decision 2: one store, one
    namespace, provenance says what a record is; nothing downstream
    branches on it).

    * ``VENDOR``: bundled by the vibe-qc maintainers from an upstream
      distribution, such as libint's basis bundle or a publishing group's
      archive, possibly completed with reviewed element blocks from the
      Basis Set Exchange.
    * ``BSE``: a whole record rendered from the Basis Set Exchange
      catalogue, bundled or fetched on demand.
    * ``HAND_WRITTEN``: supplied by a user through the import tool.
    * ``VIBE_BASIS``: produced by the vibe-basis optimiser.
    """

    VENDOR = "vendor"
    BSE = "bse"
    HAND_WRITTEN = "hand-written"
    VIBE_BASIS = "vibe-basis"


@dataclass
class Upstream:
    """One upstream source a record draws on.

    ``name`` is the distribution or archive (``"libint"``,
    ``"basis-set-exchange"``, ``"Bredow group basis-set archive"``),
    ``version`` its version or catalogue release, ``url`` where it was
    obtained, ``identifier`` the record's name in that source, and
    ``detail`` which part of this record came from it (``"H-Kr"``).
    """

    name: str
    version: Optional[str] = None
    url: Optional[str] = None
    identifier: Optional[str] = None
    detail: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"name": self.name}
        for key in ("version", "url", "identifier", "detail"):
            value = getattr(self, key)
            if value:
                d[key] = value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Upstream":
        return cls(
            name=str(d["name"]),
            version=d.get("version"),
            url=d.get("url"),
            identifier=d.get("identifier"),
            detail=d.get("detail"),
        )


@dataclass
class CitationRoute:
    """Which publications cover which elements of the record.

    ``keys`` are citation keys in ``vibeqc.output.citations`` (its
    ``database.toml`` entries). ``elements`` lists the symbols the route
    covers; ``None`` means every element of the record. pob-TZVP-rev2 needs
    four routes because its 78 records come from four papers (#288).
    """

    keys: List[str]
    elements: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"keys": list(self.keys)}
        if self.elements is not None:
            d["elements"] = list(self.elements)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CitationRoute":
        elements = d.get("elements")
        return cls(
            keys=[str(k) for k in d.get("keys", [])],
            elements=[str(e) for e in elements] if elements is not None else None,
        )


@dataclass
class Provenance:
    """How this basis set was obtained and any upstream source identity.

    ``origin`` is the 1.0.0 field (which importer produced the record);
    ``kind`` and the rest arrived with schema 1.1.0 (#407).
    ``content_sha256`` is :meth:`BasisSetData.content_sha256`, the digest
    of the numerical content alone, so renaming or re-describing a record
    does not change its identity and a cache can be addressed by it.
    """

    origin: Literal["bse", "manual", "crystal", "file", "generated"] = "bse"
    retrieval_date: Optional[str] = None  # ISO 8601 date
    source_doi: Optional[str] = None
    source_version: Optional[str] = None
    source_checksum: Optional[str] = None  # sha256 of the source file
    kind: Optional[ProvenanceKind] = None
    upstream: List[Upstream] = field(default_factory=list)
    content_sha256: Optional[str] = None
    citations: List[CitationRoute] = field(default_factory=list)
    run_id: Optional[str] = None  # vibe-basis run identity
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"origin": self.origin}
        if self.kind is not None:
            d["kind"] = ProvenanceKind(self.kind).value
        if self.upstream:
            d["upstream"] = [u.to_dict() for u in self.upstream]
        if self.retrieval_date:
            d["retrieval_date"] = self.retrieval_date
        if self.source_doi:
            d["source_doi"] = self.source_doi
        if self.source_version:
            d["source_version"] = self.source_version
        if self.source_checksum:
            d["source_checksum"] = self.source_checksum
        if self.content_sha256:
            d["content_sha256"] = self.content_sha256
        if self.citations:
            d["citations"] = [c.to_dict() for c in self.citations]
        if self.run_id:
            d["run_id"] = self.run_id
        if self.notes:
            d["notes"] = self.notes
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Provenance":
        kind = d.get("kind")
        return cls(
            origin=d.get("origin", "bse"),
            retrieval_date=d.get("retrieval_date"),
            source_doi=d.get("source_doi"),
            source_version=d.get("source_version"),
            source_checksum=d.get("source_checksum"),
            kind=ProvenanceKind(kind) if kind is not None else None,
            upstream=[Upstream.from_dict(u) for u in d.get("upstream", [])],
            content_sha256=d.get("content_sha256"),
            citations=[CitationRoute.from_dict(c) for c in d.get("citations", [])],
            run_id=d.get("run_id"),
            notes=d.get("notes"),
        )


# ---------------------------------------------------------------------------
# Top-level container
# ---------------------------------------------------------------------------


@dataclass
class BasisSetData:
    """Top-level canonical basis-set object.

    This is the root object serialised in a ``.qvf.json`` file and the
    payload carried inside a QVF-Basis ``.qvf`` archive's ``basis`` section.

    Attributes
    ----------
    schema_version:
        Semantic version of the QVF-Basis schema this file conforms to.
    name:
        Short identifier, e.g. ``"cc-pVDZ"``, ``"def2-SVP"``.
    description:
        Human-readable one-liner.
    role:
        Intended use: orbital, auxiliary, fitting, or ECP.
    basis_family:
        Optional family grouping, e.g. ``"dunning"``, ``"ahlrichs"``.
    elements:
        Per-element basis definitions, keyed by element symbol.
    ecps:
        Per-element ECP definitions, keyed by element symbol.
    references:
        Bibliographic references for the basis set.
    provenance:
        Origin and retrieval metadata.
    """

    schema_version: str
    name: str
    description: str = ""
    role: BasisRole = BasisRole.ORBITAL
    basis_family: Optional[str] = None
    elements: Dict[str, ElementBasis] = field(default_factory=dict)
    ecps: Optional[Dict[str, ECPEntry]] = None
    references: List[Reference] = field(default_factory=list)
    provenance: Optional[Provenance] = None

    def elems_sorted(self) -> List[str]:
        """Return element symbols sorted by atomic number."""
        from ._element_data import ATOMIC_NUMBER

        return sorted(self.elements.keys(), key=lambda s: ATOMIC_NUMBER.get(s, 999))

    def content_dict(self) -> Dict[str, Any]:
        """The numerical content alone: elements and ECPs, nothing else.

        Keys are sorted on serialisation, so two records with the same
        shells and potentials hash alike whatever their name, description,
        references or provenance say.
        """
        d: Dict[str, Any] = {
            "elements": {k: v.to_dict() for k, v in self.elements.items()},
        }
        if self.ecps:
            d["ecps"] = {k: v.to_dict() for k, v in self.ecps.items()}
        return d

    def content_sha256(self) -> str:
        """SHA-256 of :meth:`content_dict` in canonical JSON.

        This is the identity the content-addressed store (#408) and the
        provenance block (#407) carry. Floats serialise with ``repr``, so the
        digest changes exactly when a double changes.
        """
        payload = json.dumps(
            self.content_dict(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        )
        return hashlib.sha256(payload.encode("ascii")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "name": self.name,
            "description": self.description,
            "role": self.role.value,
            "elements": {k: v.to_dict() for k, v in self.elements.items()},
        }
        if self.basis_family:
            d["basis_family"] = self.basis_family
        if self.ecps:
            d["ecps"] = {k: v.to_dict() for k, v in self.ecps.items()}
        if self.references:
            d["references"] = [r.to_dict() for r in self.references]
        if self.provenance:
            d["provenance"] = self.provenance.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BasisSetData":
        return cls(
            schema_version=str(d["schema_version"]),
            name=str(d["name"]),
            description=str(d.get("description", "")),
            role=BasisRole(d.get("role", "orbital")),
            basis_family=d.get("basis_family"),
            elements={k: ElementBasis.from_dict(v) for k, v in d["elements"].items()},
            ecps={k: ECPEntry.from_dict(v) for k, v in d["ecps"].items()}
            if d.get("ecps")
            else None,
            references=[Reference.from_dict(r) for r in d.get("references", [])],
            provenance=Provenance.from_dict(d["provenance"])
            if d.get("provenance")
            else None,
        )


# ---------------------------------------------------------------------------
# Loss report
# ---------------------------------------------------------------------------


@dataclass
class LossReport:
    """Records which information was lost during a format conversion."""

    fields_dropped: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    is_lossless: bool = True

    def add_loss(self, field: str, reason: str = "") -> None:
        self.fields_dropped.append(field)
        self.is_lossless = False
        if reason:
            self.warnings.append(f"{field}: {reason}")

    def merge(self, other: "LossReport") -> None:
        self.fields_dropped.extend(other.fields_dropped)
        self.warnings.extend(other.warnings)
        self.is_lossless = self.is_lossless and other.is_lossless

    def summary(self) -> str:
        if self.is_lossless:
            return "Conversion was lossless."
        lines = ["Conversion lost the following information:"]
        for w in self.warnings:
            lines.append(f"  - {w}")
        return "\n".join(lines)
