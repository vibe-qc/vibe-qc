"""Canonical reference set for the v1 round-trip smoke test.

These are the structures the fetcher must round-trip end-to-end:
    fetch -> emit SPEC -> emit input -> run vibe-qc -> match
    examples/regression/expected/<id>__sto-3g__rks-lda.json within
    tolerance.

The v1 verification harness (``vqfetch canonical <slug>``) walks this
table. **IDs here were resolved against the live API** (the spec doc
explicitly forbids hard-coding training-data IDs); the table below was
populated by querying each provider with
``chemical_formula_reduced=<formula>`` and picking the first hit
whose spglib-derived space group matches the canonical Bravais
lattice (Fm-3m for rocksalt, Fd-3m for diamond).

Populated by direct OPTIMADE id-lookup against
https://optimade.materialsproject.org on 2026-05-05 (UTC). Re-run the
``--quick`` smoke harness against the live API to verify a given
entry is still resolvable.

Note on formula syntax: ``chemical_formula_reduced`` is alphabetised
per OPTIMADE spec -- "NaCl" round-trips as ``"ClNa"``, "LiH" as
``"HLi"``. We store both the human-readable formula and the OPTIMADE
form for clarity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CanonicalEntry:
    slug: str
    """filesystem-safe id used for emission."""
    formula: str
    """Human-readable formula (``"MgO"``, ``"NaCl"``, ...)."""
    optimade_formula: str
    """OPTIMADE alphabetised form ("ClNa" for NaCl, etc.)."""
    family: str
    """rocksalt / diamond / molecule."""
    primary_provider: str
    """Short OPTIMADE provider key (``"mp"``, ``"cod"``, ...)."""
    primary_id: str
    """Provider-specific id (``"mp-1265"``, ``"9006411"``)."""
    primary_url: str
    """Permalink for citation."""
    expected_space_group: str
    """spglib international symbol the fetched cell should standardise to."""
    expected_nsites_conventional: int
    """Atom count after standardising to the conventional cell."""
    expected_a_ang: float
    """Conventional-cell lattice parameter (Å), for sanity-checking."""
    notes: str = ""
    secondary_provider: Optional[str] = None
    secondary_id: Optional[str] = None


# Resolved 2026-05-05 against optimade.materialsproject.org.
# Order matches docs/tutorial/external_data_fetcher.md Sec. 10.
CANONICAL_SET: tuple[CanonicalEntry, ...] = (
    CanonicalEntry(
        slug="mgo_rocksalt",
        formula="MgO",
        optimade_formula="MgO",
        family="rocksalt",
        primary_provider="mp",
        primary_id="mp-1265",
        primary_url="https://next-gen.materialsproject.org/materials/mp-1265",
        expected_space_group="Fm-3m (225)",
        expected_nsites_conventional=8,
        expected_a_ang=4.194,
        notes="Already in regression suite -- direct A/B vs hand-curated SPEC.",
    ),
    CanonicalEntry(
        slug="nacl_rocksalt",
        formula="NaCl",
        optimade_formula="ClNa",
        family="rocksalt",
        primary_provider="mp",
        primary_id="mp-22862",
        primary_url="https://next-gen.materialsproject.org/materials/mp-22862",
        expected_space_group="Fm-3m (225)",
        expected_nsites_conventional=8,
        expected_a_ang=5.588,
        notes="Known-good baseline.",
    ),
    CanonicalEntry(
        slug="lih_rocksalt",
        formula="LiH",
        optimade_formula="HLi",
        family="rocksalt",
        primary_provider="mp",
        primary_id="mp-23703",
        primary_url="https://next-gen.materialsproject.org/materials/mp-23703",
        expected_space_group="Fm-3m (225)",
        expected_nsites_conventional=8,
        expected_a_ang=4.017,
        notes="Lightest ionic -- fastest SCF.",
    ),
    CanonicalEntry(
        slug="si_diamond",
        formula="Si",
        optimade_formula="Si",
        family="diamond",
        primary_provider="mp",
        primary_id="mp-149",
        primary_url="https://next-gen.materialsproject.org/materials/mp-149",
        expected_space_group="Fd-3m (227)",
        expected_nsites_conventional=8,
        expected_a_ang=5.444,
        notes="Covalent, wide-gap, canonical.",
    ),
    CanonicalEntry(
        slug="c_diamond",
        formula="C",
        optimade_formula="C",
        family="diamond",
        primary_provider="mp",
        primary_id="mp-66",
        primary_url="https://next-gen.materialsproject.org/materials/mp-66",
        expected_space_group="Fd-3m (227)",
        expected_nsites_conventional=8,
        expected_a_ang=3.561,
        notes="All-light-atom covalent.",
    ),
)


def find(slug: str) -> CanonicalEntry:
    for e in CANONICAL_SET:
        if e.slug == slug:
            return e
    raise KeyError(
        f"unknown canonical slug {slug!r}; "
        f"known: {[c.slug for c in CANONICAL_SET]}"
    )
