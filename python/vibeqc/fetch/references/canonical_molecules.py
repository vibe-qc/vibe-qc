"""Canonical molecule reference set for the v1 round-trip smoke test.

These are the molecules the fetcher must successfully round-trip
through ``vqfetch reference --source cccbdb`` end-to-end. Same
pattern as phase-1's ``canonical_set.py`` -- a frozen dataclass
table of CAS numbers resolved against the live API.

Resolved 2026-05-09 against ``cccbdb.nist.gov/exp2x.asp?casno=...``.
Each entry was verified to return a real "Experimental data for X"
master page (HTTP 200, ~50 KB, >= 10 data tables) -- not a 500-error
or empty form page.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CanonicalMolecule:
    slug: str
    """Filesystem-safe id used for emission (matches ``MoleculeSpec.id``
    on the existing regression suite when applicable)."""
    formula: str
    """Hill-system molecular formula, e.g. ``"H2O"``, ``"CH4"``."""
    name: str
    """IUPAC / common name, e.g. ``"Water"``, ``"Methane"``."""
    cas: str
    """CAS registry number, hyphenated. Stored as a string to preserve
    leading zeros (``"00057-13-6"`` for urea is not 57136)."""
    cccbdb_url: str
    """Direct link to the molecule's CCCBDB ``exp2x`` page."""
    notes: str = ""


# Resolved 2026-05-09. Order matches docs/tutorial/external_data_fetcher.md Sec. 10.
CANONICAL_MOLECULES: tuple[CanonicalMolecule, ...] = (
    CanonicalMolecule(
        slug="h2o",
        formula="H2O",
        name="Water",
        cas="7732-18-5",
        cccbdb_url="https://cccbdb.nist.gov/exp2x.asp?casno=7732185",
        notes="Workhorse; exists as MoleculeSpec in the regression suite.",
    ),
    CanonicalMolecule(
        slug="ch4",
        formula="CH4",
        name="Methane",
        cas="74-82-8",
        cccbdb_url="https://cccbdb.nist.gov/exp2x.asp?casno=74828",
        notes="Closed-shell organic; exists as MoleculeSpec.",
    ),
    CanonicalMolecule(
        slug="nh3",
        formula="NH3",
        name="Ammonia",
        cas="7664-41-7",
        cccbdb_url="https://cccbdb.nist.gov/exp2x.asp?casno=7664417",
        notes="Tests umbrella vibrational mode parsing.",
    ),
    CanonicalMolecule(
        slug="hf",
        formula="HF",
        name="Hydrogen fluoride",
        cas="7664-39-3",
        cccbdb_url="https://cccbdb.nist.gov/exp2x.asp?casno=7664393",
        notes="Diatomic; simplest atomization-energy ground truth.",
    ),
    CanonicalMolecule(
        slug="o2",
        formula="O2",
        name="Oxygen",
        cas="7782-44-7",
        cccbdb_url="https://cccbdb.nist.gov/exp2x.asp?casno=7782447",
        notes="Open-shell triplet; tests multiplicity=3 flow.",
    ),
    CanonicalMolecule(
        slug="o3",
        formula="O3",
        name="Ozone",
        cas="10028-15-6",
        cccbdb_url="https://cccbdb.nist.gov/exp2x.asp?casno=10028156",
        notes="Singlet biradical; tests strong-correlation reference.",
    ),
    CanonicalMolecule(
        slug="co2",
        formula="CO2",
        name="Carbon dioxide",
        cas="124-38-9",
        cccbdb_url="https://cccbdb.nist.gov/exp2x.asp?casno=124389",
        notes="Linear triatomic; benchmark anchor.",
    ),
    CanonicalMolecule(
        slug="h2co",
        formula="CH2O",
        name="Formaldehyde",
        cas="50-00-0",
        cccbdb_url="https://cccbdb.nist.gov/exp2x.asp?casno=50000",
        notes="Carbonyl; exists as MoleculeSpec.",
    ),
)


def find(slug: str) -> CanonicalMolecule:
    for m in CANONICAL_MOLECULES:
        if m.slug == slug:
            return m
    raise KeyError(
        f"unknown canonical molecule slug {slug!r}; "
        f"known: {[c.slug for c in CANONICAL_MOLECULES]}"
    )


def cas_to_url(cas: str) -> str:
    """Strip the CAS hyphens and form the ``exp2x.asp`` URL."""
    return f"https://cccbdb.nist.gov/exp2x.asp?casno={cas.replace('-', '')}"


def composition_from_formula(formula: str) -> dict[str, int]:
    """Crude Hill-formula parser -> ``{symbol: count}``.

    Handles two-letter symbols (``"Cl"``, ``"Br"``) and digits. Doesn't
    handle parentheses (e.g. ``"Ca(OH)2"``) -- extend if/when a
    canonical molecule needs it. None of the v1 set use parens.
    """
    import re

    out: dict[str, int] = {}
    for sym, count in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if not sym:
            continue
        out[sym] = out.get(sym, 0) + (int(count) if count else 1)
    return out
