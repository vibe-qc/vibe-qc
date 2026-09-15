"""Citation-database coverage for the SCF-accelerator family.

The sibling of [`test_basis_citation_coverage.py`](test_basis_citation_coverage.py),
for `SCFAccelerator` instead of bundled bases. Three contracts, all per
[CLAUDE.md § 8](../CLAUDE.md) / [AGENTS.md § 8](../AGENTS.md):

1. **Every accelerator is routed.** Each `vibeqc.SCFAccelerator` member
   has a `[routes.methods]` entry keyed by its lower-cased enum name
   (the key `runner._detect_scf_accelerator` emits). An unrouted
   accelerator silently credits nobody.

2. **Every route resolves.** Each key a route names exists as an
   `[entries.<key>]` block. A dangling route is broken attribution the
   moment the route fires.

3. **Originating references are pinned by DOI.** `_ORIGINATING_REFERENCE`
   pins, per accelerator, the entry key + DOI of the publication the
   algorithm actually comes from. This is the contract that review alone
   failed to hold: between the KDIIS landing and 2026-07-10, `kdiis`
   routed `garza_scuseria_kdiis_2012` (Garza & Scuseria, *Comparison of
   self-consistent field convergence acceleration techniques*, J. Chem.
   Phys. 137, 054110 (2012)). That paper compares ADIIS / LIST /
   EDIIS+DIIS and its text never mentions Kollmar or KDIIS, so anyone
   running `scf_accelerator="kdiis"` got a references block that never
   named the originating paper. Pinning the DOI here makes a repeat
   mechanically impossible.

DOIs below are verified against Crossref, not transcribed from a
docstring. Plain `diis` is deliberately absent from the pinned map: its
originating references *are* the Pulay pair that every route carries.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import vibeqc as _vq


DATABASE_TOML = (
    Path(_vq.__file__).resolve().parent
    / "output" / "citations" / "database.toml"
)

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover — repo requires 3.11+ per AGENTS.md
    import tomli as tomllib


# Pulay's B-matrix extrapolation underlies every member of the family,
# including KDIIS (which changes only the error vector). Every route
# therefore carries both Pulay papers.
_PULAY = ("pulay_diis_1980", "pulay_diis_1982")

# accelerator route key -> (originating entry key, DOI of that entry)
#
# Add a row here in the SAME merge that adds an SCFAccelerator member.
_ORIGINATING_REFERENCE: dict[str, tuple[str, str]] = {
    "kdiis": (
        # C. Kollmar, Int. J. Quantum Chem. 62, 617-637 (1997).
        "kollmar_kdiis_1997",
        "10.1002/(SICI)1097-461X(1997)62:6<617::AID-QUA5>3.0.CO;2-Z",
    ),
    "ediis": (
        # Kudin, Scuseria & Cancès, J. Chem. Phys. 116, 8255 (2002).
        "kudin_ediis_2002",
        "10.1063/1.1470195",
    ),
    "ediis_diis": (
        # Garza & Scuseria, J. Chem. Phys. 137, 054110 (2012) — the
        # paper that recommends the EDIIS+DIIS hybrid.
        "garza_scuseria_2012",
        "10.1063/1.4740249",
    ),
    "adiis": (
        # Hu & Yang, J. Chem. Phys. 132, 054109 (2010).
        "hu_yang_adiis_2010",
        "10.1063/1.3304922",
    ),
    "adiis_diis": ("hu_yang_adiis_2010", "10.1063/1.3304922"),
    "r_cdiis": (
        # Chupin, Dupuy, Legendre & Séré, ESAIM: M2AN 55, 2785 (2021).
        "chupin_adaptive_diis_2021",
        "10.1051/m2an/2021069",
    ),
    "ad_cdiis": ("chupin_adaptive_diis_2021", "10.1051/m2an/2021069"),
}

# Entries that must NOT appear on a given route, with the reason. Guards
# against a mis-attribution silently coming back.
_FORBIDDEN_ON_ROUTE: dict[str, dict[str, str]] = {
    "kdiis": {
        "garza_scuseria_2012": (
            "Garza & Scuseria 2012 compares ADIIS / LIST / EDIIS+DIIS and "
            "never discusses Kollmar's KDIIS; it is not a KDIIS reference"
        ),
    },
}


def _database() -> dict:
    with DATABASE_TOML.open("rb") as fh:
        return tomllib.load(fh)


def _method_routes() -> dict[str, list[str]]:
    return _database()["routes"]["methods"]


def _accelerator_route_keys() -> list[str]:
    """The route key `runner._detect_scf_accelerator` emits per member."""
    return [name.lower() for name in _vq.SCFAccelerator.__members__]


@pytest.mark.parametrize("key", sorted(_accelerator_route_keys()))
def test_every_accelerator_is_routed(key: str) -> None:
    routes = _method_routes()
    assert key in routes, (
        f"SCFAccelerator.{key.upper()} has no [routes.methods] entry in "
        f"{DATABASE_TOML.name}. A job selecting it would credit nobody. "
        f"See CLAUDE.md § 8 step 2."
    )
    for pulay in _PULAY:
        assert pulay in routes[key], (
            f"route {key!r} omits {pulay!r}; every accelerator in the "
            f"family reuses Pulay's B-matrix extrapolation"
        )


@pytest.mark.parametrize("key", sorted(_accelerator_route_keys()))
def test_accelerator_route_resolves_to_entries(key: str) -> None:
    db = _database()
    entries = db["entries"]
    for entry_key in db["routes"]["methods"][key]:
        assert entry_key in entries, (
            f"route {key!r} names {entry_key!r}, which has no "
            f"[entries.{entry_key}] block: a dangling route means the "
            f"citation silently resolves to nothing when it fires."
        )


@pytest.mark.parametrize(
    "key,entry_key,doi",
    sorted((k, e, d) for k, (e, d) in _ORIGINATING_REFERENCE.items()),
)
def test_originating_reference_is_routed_with_pinned_doi(
    key: str, entry_key: str, doi: str
) -> None:
    db = _database()
    assert entry_key in db["routes"]["methods"][key], (
        f"route {key!r} no longer cites its originating publication "
        f"{entry_key!r}. Users running scf_accelerator={key!r} would get "
        f"a references block that never names the paper the method comes "
        f"from."
    )
    entry = db["entries"][entry_key]
    assert entry.get("doi") == doi, (
        f"[entries.{entry_key}] DOI is {entry.get('doi')!r}, expected "
        f"{doi!r} (verified against Crossref). Do not edit this pin "
        f"without re-verifying against the literature."
    )
    # print defaults to true; an originating reference must reach the
    # user-facing references block, not just the .system manifest.
    assert entry.get("print", True) is True, (
        f"[entries.{entry_key}] is print=false, so it never reaches the "
        f".out references block. Originating publications are exactly "
        f"what a paper's Methods section must cite (CLAUDE.md § 8 step 3)."
    )


@pytest.mark.parametrize(
    "key,forbidden,reason",
    sorted(
        (k, f, r)
        for k, forb in _FORBIDDEN_ON_ROUTE.items()
        for f, r in forb.items()
    ),
)
def test_mis_attribution_stays_off_route(
    key: str, forbidden: str, reason: str
) -> None:
    routed = _method_routes()[key]
    assert forbidden not in routed, f"{forbidden!r} back on {key!r}: {reason}"
