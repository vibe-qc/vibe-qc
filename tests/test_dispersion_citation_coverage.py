"""Citation-database coverage for the dispersion-correction family.

Sibling of [`test_scf_accelerator_citation_coverage.py`](test_scf_accelerator_citation_coverage.py),
for D2 / D3 / D3(BJ) / D4 and their per-functional reparameterisations, per
[CLAUDE.md § 8](../CLAUDE.md) / [AGENTS.md § 8](../AGENTS.md).

Three contracts:

1. **Every route resolves.** Each key `[routes.methods]` and
   `[routes.dispersion_params]` name exists as an `[entries.<key>]` block.

2. **Originating references are pinned by DOI.** A dispersion model is a
   parameterisation, so the paper that fits the parameters is exactly what a
   Methods section must cite. `_ORIGINATING_REFERENCE` pins each.

3. **Reparameterisation routes are pinned.** `routes.dispersion_params` maps
   `d4:<functional>` to the paper that fitted *those* D4 parameters. Routing
   the generic D4 paper there would silently mis-credit the refit.

The `_FORBIDDEN_ON_ROUTE` guard records that r2SCAN-3c uses D4, not D3: the
`r2scan-3c` route carried `grimme_d3_2010` until the 2026-07-10 audit, even
though `composites.py` sets `dispersion="d4"` and the r2SCAN-3c paper's every
"D3" mention is a comparison functional.

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


# routes.methods key -> (originating entry key, Crossref-verified DOI)
_ORIGINATING_REFERENCE: dict[str, tuple[str, str]] = {
    "d3": ("grimme_d3_2010", "10.1063/1.3382344"),
    "d3bj": ("grimme_d3bj_2011", "10.1002/jcc.21759"),
    "d4": ("caldeweyher_d4_2019", "10.1063/1.5090222"),
}

# D3(BJ) is D3 plus Becke-Johnson damping: both papers must fire.
_REQUIRED_COMPONENTS: dict[str, tuple[str, ...]] = {
    "d3bj": ("grimme_d3_2010", "grimme_d3bj_2011"),
    # The D4 model spans three papers; all three ship and are distinct.
    "d4": (
        "caldeweyher_d4_2017",
        "caldeweyher_d4_2019",
        "caldeweyher_d4_2020",
    ),
}

# routes.dispersion_params key -> (entry key, DOI of the paper that fitted
# *these* parameters). One row per reparameterisation vibe-qc ships.
_REPARAM_REFERENCE: dict[str, tuple[str, str]] = {
    "d4:rscan": ("ehlert_r2scan_d4_2021", "10.1063/5.0041008"),
    "d4:r2scan": ("ehlert_r2scan_d4_2021", "10.1063/5.0041008"),
    "d4:r2scanh": ("bursch_r2scan_hybrids_2022", "10.1063/5.0086040"),
    "d4:r2scan0": ("bursch_r2scan_hybrids_2022", "10.1063/5.0086040"),
    "d4:r2scan50": ("bursch_r2scan_hybrids_2022", "10.1063/5.0086040"),
    "d4:wb97x": ("najibi_goerigk_d4_2020", "10.1002/jcc.26411"),
    "d4:lcwpbe": ("friede_tuned_rsh_d4_2023", "10.1021/acs.jctc.3c00717"),
    "d4:revdsdblyp": ("santra_martin_revdsd_2019", "10.1021/acs.jpca.9b03157"),
    "d4:revdsdpbep86": (
        "santra_martin_revdsd_2019", "10.1021/acs.jpca.9b03157",
    ),
    "d4:revdsdpbe": ("santra_martin_revdsd_2019", "10.1021/acs.jpca.9b03157"),
    "d4:revdodpbep86": (
        "santra_martin_revdsd_2019", "10.1021/acs.jpca.9b03157",
    ),
    "d4:r2scan3c": ("grimme_r2scan3c_2021", "10.1063/5.0040021"),
    "d4:gfn2xtb": ("bannwarth_gfn2_2019", "10.1021/acs.jctc.8b01176"),
    "d4:gfn1xtb": ("grimme_gfn1_2017", "10.1021/acs.jctc.7b00118"),
}

_FORBIDDEN_ON_ROUTE: dict[str, dict[str, str]] = {
    "r2scan-3c": {
        "grimme_d3_2010": (
            "r2SCAN-3c uses D4, not D3 (composites.py sets dispersion='d4')"
        ),
    },
    "wb97x-3c": {
        "grimme_d3_2010": "wB97X-3c uses D4",
        "grimme_d3bj_2011": "wB97X-3c uses D4",
    },
}


def _database() -> dict:
    with DATABASE_TOML.open("rb") as fh:
        return tomllib.load(fh)


def _route_for(key: str) -> list[str]:
    db = _database()
    return list(db["routes"]["methods"].get(key, ()))


def test_every_dispersion_params_route_is_pinned() -> None:
    """A reparameterisation added without a pin here is unattributed."""
    routed = set(_database()["routes"]["dispersion_params"])
    assert routed == set(_REPARAM_REFERENCE), (
        "routes.dispersion_params and _REPARAM_REFERENCE disagree.\n"
        f"  unpinned in this test: {sorted(routed - set(_REPARAM_REFERENCE))}\n"
        f"  pinned but unrouted  : {sorted(set(_REPARAM_REFERENCE) - routed)}\n"
        "Add a row here in the SAME merge that adds the reparameterisation "
        "(CLAUDE.md § 8 step 4)."
    )


@pytest.mark.parametrize("key", sorted(_database()["routes"]["dispersion_params"]))
def test_dispersion_params_route_resolves_to_entries(key: str) -> None:
    db = _database()
    for entry_key in db["routes"]["dispersion_params"][key]:
        assert entry_key in db["entries"], (
            f"routes.dispersion_params.{key!r} names {entry_key!r}, which "
            f"has no [entries.{entry_key}] block."
        )


@pytest.mark.parametrize(
    "key,entry_key,doi",
    sorted((k, e, d) for k, (e, d) in _ORIGINATING_REFERENCE.items()),
)
def test_originating_reference_is_routed_with_pinned_doi(
    key: str, entry_key: str, doi: str
) -> None:
    routed = _route_for(key)
    assert entry_key in routed, (
        f"route {key!r} no longer cites its originating publication "
        f"{entry_key!r}."
    )
    entry = _database()["entries"][entry_key]
    assert entry.get("doi") == doi, (
        f"[entries.{entry_key}] DOI is {entry.get('doi')!r}, expected "
        f"{doi!r} (verified against Crossref)."
    )
    assert entry.get("print", True) is True, (
        f"[entries.{entry_key}] is print=false and never reaches the .out "
        f"references block."
    )


@pytest.mark.parametrize(
    "key,entry_key,doi",
    sorted((k, e, d) for k, (e, d) in _REPARAM_REFERENCE.items()),
)
def test_reparameterisation_cites_the_paper_that_fitted_it(
    key: str, entry_key: str, doi: str
) -> None:
    db = _database()
    routed = db["routes"]["dispersion_params"][key]
    assert entry_key in routed, (
        f"routes.dispersion_params.{key!r} does not cite {entry_key!r}, the "
        f"paper that fitted those parameters. Citing the generic D4 paper "
        f"instead silently mis-credits the refit."
    )
    entry = db["entries"][entry_key]
    assert entry.get("doi") == doi, (
        f"[entries.{entry_key}] DOI is {entry.get('doi')!r}, expected "
        f"{doi!r} (verified against Crossref)."
    )


@pytest.mark.parametrize(
    "key,component",
    sorted((k, c) for k, comps in _REQUIRED_COMPONENTS.items() for c in comps),
)
def test_component_papers_are_routed(key: str, component: str) -> None:
    assert component in _route_for(key), (
        f"route {key!r} omits {component!r}, a paper the model is built from."
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
