"""Citation-database coverage for functionals and composite methods.

Sibling of [`test_scf_accelerator_citation_coverage.py`](test_scf_accelerator_citation_coverage.py),
applying the same three contracts to `[routes.functionals]` and to the
composite ("3c" / double-hybrid) method routes, per
[CLAUDE.md § 8](../CLAUDE.md) / [AGENTS.md § 8](../AGENTS.md):

1. **Every route resolves.** Each key a route names exists as an
   `[entries.<key>]` block.

2. **Component references are pinned.** A hybrid or composite functional
   is a *recipe*: it must cite the papers for the parts it actually
   evaluates. `_REQUIRED_COMPONENTS` pins, per functional, the entry keys
   that must be routed. This is what review alone did not hold: `b2plyp`
   shipped without ever citing **B88**, the semilocal exchange it uses
   ("the combination of B88 and LYP yields the best results" — Grimme,
   *J. Chem. Phys.* 124, 034108 (2006)), while `blyp` cited it correctly.

3. **Originating references are pinned by DOI.** `_ORIGINATING_REFERENCE`
   pins, per composite, the entry key + Crossref-verified DOI of the paper
   that defines the recipe. `hse-3c` shipped with no defining-paper
   citation at all, unlike every sibling 3c method.

`_FORBIDDEN_ON_ROUTE` guards the two mis-attributions the 2026-07-10 audit
removed, so neither can silently return.

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


# composite / double-hybrid route key -> (originating entry key, DOI)
#
# Add a row here in the SAME merge that adds a composite method.
_ORIGINATING_REFERENCE: dict[str, tuple[str, str]] = {
    "b2plyp": ("grimme_b2plyp_2006", "10.1063/1.2148954"),
    "hf-3c": ("sure_grimme_hf3c_2013", "10.1002/jcc.23317"),
    "pbeh-3c": ("grimme_pbeh3c_2015", "10.1063/1.4927476"),
    "b97-3c": ("brandenburg_b973c_2018", "10.1063/1.5012601"),
    "r2scan-3c": ("grimme_r2scan3c_2021", "10.1063/5.0040021"),
    "wb97x-3c": ("mueller_wb97x3c_2023", "10.1063/5.0133026"),
    # Brandenburg, Caldeweyher & Grimme, PCCP 18, 15519-15523 (2016).
    # Absent from this route until the 2026-07-10 audit.
    "hse-3c": ("brandenburg_hse3c_2016", "10.1039/c6cp01697a"),
}

# route key -> entry keys that must be routed, because the method
# evaluates that component. Not exhaustive: a route may carry more.
_REQUIRED_COMPONENTS: dict[str, tuple[str, ...]] = {
    # B2PLYP = HF/B88 exchange + LYP correlation + PT2. The B88 pin is the
    # regression guard for the defect this file was written for.
    "b2plyp": ("becke_b88_1988", "lee_yang_parr_1988", "moller_plesset_1934"),
    "blyp": ("becke_b88_1988", "lee_yang_parr_1988"),
    # r2SCAN-3c pairs D4 (not D3) with the refitted gCP.
    "r2scan-3c": ("caldeweyher_d4_2019", "kruse_grimme_gcp_2012"),
    # The D3(BJ)-based 3c methods carry both D3 papers plus gCP.
    "hf-3c": ("kruse_grimme_gcp_2012", "grimme_d3_2010", "grimme_d3bj_2011"),
    "pbeh-3c": ("kruse_grimme_gcp_2012", "grimme_d3_2010", "grimme_d3bj_2011"),
    "hse-3c": ("kruse_grimme_gcp_2012", "grimme_d3_2010", "grimme_d3bj_2011"),
}

# Entries that must NOT appear on a given route, with the reason.
_FORBIDDEN_ON_ROUTE: dict[str, dict[str, str]] = {
    "r2scan-3c": {
        "grimme_d3_2010": (
            "r2SCAN-3c uses D4 (composites.py sets dispersion='d4'); the "
            "r2SCAN-3c paper's every 'D3' mention is a comparison "
            "functional, never its own dispersion correction"
        ),
    },
}


def _database() -> dict:
    with DATABASE_TOML.open("rb") as fh:
        return tomllib.load(fh)


def _functional_routes() -> dict[str, list[str]]:
    return _database()["routes"]["functionals"]


def _method_routes() -> dict[str, list[str]]:
    return _database()["routes"]["methods"]


def _route_for(key: str) -> list[str]:
    """Composites live in routes.methods; plain functionals in
    routes.functionals. Several appear in both."""
    db = _database()
    return list(
        db["routes"]["methods"].get(key)
        or db["routes"]["functionals"].get(key)
        or []
    )


@pytest.mark.parametrize("key", sorted(_functional_routes()))
def test_functional_route_resolves_to_entries(key: str) -> None:
    db = _database()
    entries = db["entries"]
    for entry_key in db["routes"]["functionals"][key]:
        assert entry_key in entries, (
            f"routes.functionals.{key!r} names {entry_key!r}, which has no "
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
    routed = _route_for(key)
    assert routed, f"no route named {key!r} in routes.methods or .functionals"
    assert entry_key in routed, (
        f"route {key!r} does not cite its defining publication "
        f"{entry_key!r}. Users running {key!r} would get a references block "
        f"that never names the paper the method comes from."
    )
    entry = _database()["entries"][entry_key]
    assert entry.get("doi") == doi, (
        f"[entries.{entry_key}] DOI is {entry.get('doi')!r}, expected "
        f"{doi!r} (verified against Crossref). Do not edit this pin "
        f"without re-verifying against the literature."
    )
    assert entry.get("print", True) is True, (
        f"[entries.{entry_key}] is print=false, so it never reaches the "
        f".out references block. Defining publications are exactly what a "
        f"paper's Methods section must cite (CLAUDE.md § 8 step 3)."
    )


@pytest.mark.parametrize(
    "key,component",
    sorted((k, c) for k, comps in _REQUIRED_COMPONENTS.items() for c in comps),
)
def test_component_papers_are_routed(key: str, component: str) -> None:
    routed = _route_for(key)
    assert routed, f"no route named {key!r}"
    assert component in routed, (
        f"route {key!r} omits {component!r}, a component it evaluates. A "
        f"functional is a recipe: every part it computes must be citable "
        f"from the run that computes it."
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
    assert forbidden not in _route_for(key), (
        f"{forbidden!r} back on {key!r}: {reason}"
    )
