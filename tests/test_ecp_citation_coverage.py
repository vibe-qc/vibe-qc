"""Citation-database coverage for effective core potentials.

Sibling of [`test_scf_accelerator_citation_coverage.py`](test_scf_accelerator_citation_coverage.py),
for the ECP surface, per [CLAUDE.md § 8](../CLAUDE.md) / [AGENTS.md § 8](../AGENTS.md).
CLAUDE.md § 8 names ECPs explicitly in the mandatory citation surface: "ECPs
used: per the libecpint per-element references."

The ECP citation surface has two halves, and this file pins both:

1. **The integral library.** `uses_ecp=True` fires
   `routes.libraries.libecpint`, so any job with an ECP credits libecpint.

2. **The potentials themselves.** An ECP is a *fitted potential*, so the
   paper that fitted it is what a Methods section must cite. vibe-qc ships
   two ECP families: the Hay-Wadt LANL potentials (reached from the `lanl*`
   basis keys) and Peterson's relativistic small-core PPs (reached from the
   `*-pp-rifit` auxiliary keys).

`_UNROUTED_ECP_ENTRIES` records ECP entries that exist but no route reaches,
so the debt is tracked mechanically rather than rediscovered. Per CLAUDE.md
§ 8 step 2, "a citation without a route is silently dead weight".

DOIs below are verified against Crossref, not transcribed from a docstring.
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


# The ECP integral library, fired by `uses_ecp=True`.
_LIBECPINT = ("shaw_gilbert_libecpint", "10.1063/1.4986887")

# entry key -> Crossref-verified DOI, for every ECP-family paper that a
# route can reach today.
_ECP_REFERENCE: dict[str, str] = {
    # Hay & Wadt, J. Chem. Phys. 82, 299 (1985) -- the LANL potentials.
    "hay_wadt_lanl_1985": "10.1063/1.448799",
    # Peterson, J. Chem. Phys. 119, 11099 (2003) -- relativistic small-core
    # PPs underlying the cc-pV*Z-PP family and the def2-ECP for Te-Xe / Po-Rn.
    "peterson_pp_2003": "10.1063/1.1622924",
    # The def2-ECP set the def2, def2-m* and pob-TZVP-rev2 sidecars carry
    # (per element: Andrae 1990 Y-Cd / Hf-Hg, Metz 2000 In-Sb / Tl-Bi,
    # Leininger 1996 Rb / Cs, Kaupp 1991 Sr / Ba, Dolg 1989 lanthanides).
    "andrae_ecp_1990": "10.1007/BF01114537",
    "metz_stoll_dolg_ecp_2000": "10.1063/1.1305880",
    "leininger_nicklass_ecp_1996": "10.1016/0009-2614(96)00382-X",
    "kaupp_schleyer_stoll_preuss_ecp_1991": "10.1063/1.459993",
    "dolg_stoll_savin_preuss_lanthanide_ecp_1989": "10.1007/BF00528565",
    # Weigend & Baldes, J. Chem. Phys. 133, 174102 (2010) -- the dhf-*
    # sets and their Dirac-Fock ECPs.
    "weigend_baldes_dhf_2010": "10.1063/1.3495681",
}

# route key -> the ECP entry it must cite.
_ROUTED_FROM: dict[str, str] = {
    "lanl2dz": "hay_wadt_lanl_1985",
    "lanl2dzdp": "hay_wadt_lanl_1985",
    "lanl2tz": "hay_wadt_lanl_1985",
    "lanl08": "hay_wadt_lanl_1985",
    "lanl08(d)": "hay_wadt_lanl_1985",
    "lanl08(f)": "hay_wadt_lanl_1985",
    "cc-pvdz-pp-rifit": "peterson_pp_2003",
    "cc-pvtz-pp-rifit": "peterson_pp_2003",
    "aug-cc-pvtz-pp-rifit": "peterson_pp_2003",
    # def2 beyond Kr, the 3c bases and pob-TZVP-rev2 carry the def2-ECP /
    # Stuttgart potentials as sidecars (2026-09); dhf-* carry Dirac-Fock ones.
    "def2-svp": "andrae_ecp_1990",
    "def2-tzvp": "metz_stoll_dolg_ecp_2000",
    "def2-qzvpp": "leininger_nicklass_ecp_1996",
    "def2-tzvppd": "kaupp_schleyer_stoll_preuss_ecp_1991",
    "def2-msvp": "dolg_stoll_savin_preuss_lanthanide_ecp_1989",
    "def2-mtzvp": "andrae_ecp_1990",
    "def2-mtzvpp": "andrae_ecp_1990",
    "pob-tzvp-rev2": "andrae_ecp_1990",
    "dhf-tzvp": "weigend_baldes_dhf_2010",
    "vdzp": "mueller_wb97x3c_2023",
}

# ECP entries that exist but that no [routes.*] table reaches. Each is dead
# weight until something routes it: it can never appear in a user's
# references block. Shrink this set; do not grow it.
#
# Empty since 2026-09: every bundled ECP-bearing basis routes the papers
# that fitted the potentials it carries (the def2-ECP set from its def2,
# def2-m* and pob-TZVP-rev2 sidecars, the Dirac-Fock set from dhf-*,
# vDZP's own). The route is still keyed on the basis, so a light molecule
# in such a basis cites the potentials too, as LANL2DZ always has.
#
# This set is the ECP-specific view. The repo-wide dead-entry guard lives in
# tests/test_citation_no_dead_entries.py::_KNOWN_DEAD_ENTRIES, which also
# accounts for entries reached dynamically via assemble(extra_entries=...) --
# a route-table-only scan reports those as dead when they are not. Keep the two
# in sync: an entry wired here must leave both allow-lists.
_UNROUTED_ECP_ENTRIES = frozenset()


def _database() -> dict:
    with DATABASE_TOML.open("rb") as fh:
        return tomllib.load(fh)


def _all_routed_keys(db: dict) -> set[str]:
    routed: set[str] = set()
    for table in db["routes"].values():
        for value in table.values():
            routed.update(value if isinstance(value, list) else [value])
    return routed


def test_libecpint_is_routed_with_pinned_doi() -> None:
    db = _database()
    key, doi = _LIBECPINT
    assert key in db["routes"]["libraries"]["libecpint"], (
        "routes.libraries.libecpint no longer cites "
        f"{key!r}. Every job with uses_ecp=True must credit the ECP "
        "integral library (CLAUDE.md § 8)."
    )
    entry = db["entries"][key]
    assert entry.get("doi") == doi, (
        f"[entries.{key}] DOI is {entry.get('doi')!r}, expected {doi!r} "
        f"(verified against Crossref)."
    )
    assert entry.get("print", True) is True, (
        f"[entries.{key}] is print=false. libecpint is a scientific "
        f"dependency, not link-time-only infrastructure, so it belongs in "
        f"the user-facing references block (CLAUDE.md § 8 step 3)."
    )


@pytest.mark.parametrize("key,doi", sorted(_ECP_REFERENCE.items()))
def test_ecp_reference_doi_is_pinned(key: str, doi: str) -> None:
    entry = _database()["entries"][key]
    assert entry.get("doi") == doi, (
        f"[entries.{key}] DOI is {entry.get('doi')!r}, expected {doi!r} "
        f"(verified against Crossref). Do not edit this pin without "
        f"re-verifying against the literature."
    )
    assert entry.get("print", True) is True, (
        f"[entries.{key}] is print=false, so the ECP a job actually used "
        f"never reaches the references block."
    )


@pytest.mark.parametrize("route_key,entry_key", sorted(_ROUTED_FROM.items()))
def test_ecp_basis_route_cites_its_potential(
    route_key: str, entry_key: str
) -> None:
    routes = _database()["routes"]["basis_sets"]
    assert route_key in routes, (
        f"routes.basis_sets.{route_key!r} disappeared; an ECP basis with no "
        f"route credits nobody for the potential it uses."
    )
    assert entry_key in routes[route_key], (
        f"routes.basis_sets.{route_key!r} no longer cites {entry_key!r}, "
        f"the paper that fitted the potential it carries."
    )


def test_unrouted_ecp_entries_are_exactly_the_known_set() -> None:
    """Guard both directions: no new dead ECP citation, and the known dead
    ones are noticed the moment somebody routes them."""
    db = _database()
    routed = _all_routed_keys(db)
    ecp_entries = set(_ECP_REFERENCE) | set(_UNROUTED_ECP_ENTRIES)
    actually_unrouted = {k for k in ecp_entries if k not in routed}
    assert actually_unrouted == set(_UNROUTED_ECP_ENTRIES), (
        "the set of unrouted ECP entries changed.\n"
        f"  newly dead (routed nowhere): "
        f"{sorted(actually_unrouted - _UNROUTED_ECP_ENTRIES)}\n"
        f"  now routed, drop from _UNROUTED_ECP_ENTRIES: "
        f"{sorted(_UNROUTED_ECP_ENTRIES - actually_unrouted)}\n"
        "A citation without a route is silently dead weight "
        "(CLAUDE.md § 8 step 2)."
    )
