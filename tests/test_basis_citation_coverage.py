"""Citation-database coverage for bundled basis families.

Two contracts, both per [AGENTS.md § 8](../AGENTS.md):

1. **No dangling routes.** Every key in `[routes.basis_sets]` in
   ``python/vibeqc/output/citations/database.toml`` must resolve to
   a basis that actually ships in
   ``python/vibeqc/basis_library/basis/``. A dangling route is a
   citation pointing at nothing — silently broken attribution if the
   route ever fires.

2. **Pinned coverage doesn't regress.** A snapshot of the bundled
   bases that currently have citation coverage (either a route or an
   inline ``! Originating publication: …`` header) is pinned in
   ``CURRENTLY_COVERED``. The test fails if any pinned basis loses
   its coverage — e.g. the citation route is dropped, or a custom
   ``.g94`` is regenerated from a source that strips the provenance
   header.

We do **not** assert full coverage of all shipped bases: most of
libint's standard set arrives without inline citations and the route
table covers only the primary user targets.

3. **Routes are derivable from the records** (#407). Every bundled
   QVF-Basis record carries ``provenance.citations``, per element range,
   and the route table's keys for a shipped basis are exactly the union
   of its record's citation keys. The route table is therefore a view
   over the records, not a second hand-kept source.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

import vibeqc as _vq


BUNDLED_BASIS_DIR = Path(_vq.__file__).resolve().parent / "basis_library"
BASIS_DIR = BUNDLED_BASIS_DIR / "basis"
DATABASE_TOML = (
    Path(_vq.__file__).resolve().parent
    / "output" / "citations" / "database.toml"
)

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover — repo requires 3.11+ per AGENTS.md
    import tomli as tomllib

# Pople-syntax aliases: a route name on the left is considered to
# cover the shipped basis on the right. Gaussian convention treats
# ``6-31G(d)`` and ``6-31G*`` as the same basis; the route exists so
# user input that uses the parens form picks up the same citation.
# Update both halves together when adding a new alias route.
_ALIASES = {
    "6-31g(d)": "6-31g*",
    "6-311+g(3df,2p)": "6-311+g3df2p",
    "631g*": "6-31g*",
    "sto3g": "sto-3g",
    "def2-svp-jkfit": "def2-svp-jk",
}

# Regex for "this file carries its own provenance header." Mirrors
# the conventions used by scripts/basisset_dev/fetch_from_bse.py
# (``! Originating publication: …``, ``! Fetched from: …``) and by
# hand-curated custom .g94 sources.
_PROVENANCE_RE = re.compile(
    r"^\s*!\s*("
    r"originating publication|citation|reference|fetched from|"
    r"cite|published in|published as|doi"
    r")\b",
    re.IGNORECASE,
)

# Pinned coverage snapshot — every basis listed here has either a
# citation route OR an inline provenance header today. The test
# enforces that this stays true. When you intentionally remove a
# basis (or its provenance), remove the corresponding entry here in
# the **same** commit.
CURRENTLY_COVERED: frozenset[str] = frozenset({
    # routed in [routes.basis_sets]
    "6-31g", "6-31g*",
    "6-31g**", "6-31gs", "6-31gss", "6-31g**-rifit",
    "6-311+g3df2p",
    "6-311g**", "6-311gss", "6-311g**-rifit",
    "3-21g",
    "cc-pvdz", "cc-pvtz", "cc-pvqz",
    "cc-pv5z", "cc-pv6z", "cc-pv7z", "cc-pvtz-mini",
    "cc-pvdz-ri", "cc-pvdz-rifit",
    "cc-pvtz-ri", "cc-pvtz-rifit",
    "cc-pvqz-ri", "cc-pvqz-rifit",
    "cc-pv5z-ri", "cc-pv5z-rifit",
    "cc-pv6z-ri", "cc-pv6z-rifit",
    "cc-pvdz-jkfit", "cc-pvtz-jkfit", "cc-pvqz-jkfit", "cc-pv5z-jkfit",
    "cc-pvdz-f12", "cc-pvtz-f12", "cc-pvqz-f12",
    "cc-pvdz-pp-rifit", "cc-pvtz-pp-rifit", "cc-pvqz-pp-rifit", "cc-pv5z-pp-rifit",
    "cc-pwcvdz-rifit", "cc-pwcvtz-rifit", "cc-pwcvqz-rifit", "cc-pwcv5z-rifit",
    "cc-pwcvdz-pp-rifit", "cc-pwcvtz-pp-rifit",
    "cc-pwcvqz-pp-rifit", "cc-pwcv5z-pp-rifit",
    "augmentation-cc-pvdz", "augmentation-cc-pvtz", "augmentation-cc-pvqz",
    "augmentation-cc-pv5z", "augmentation-cc-pv6z", "augmentation-cc-pv7z",
    "augmentation-cc-pvdz-ri", "augmentation-cc-pvtz-ri",
    "augmentation-cc-pvqz-ri", "augmentation-cc-pv5z-ri",
    "augmentation-cc-pv6z-ri",
    "augmentation-cc-pvdz-jkfit", "augmentation-cc-pvtz-jkfit",
    "augmentation-cc-pvqz-jkfit", "augmentation-cc-pv5z-jkfit",
    "aug-cc-pvdz-cabs", "aug-cc-pvtz-cabs",
    "aug-cc-pvqz-cabs", "aug-cc-pv5z-cabs",
    "aug-cc-pvdz-rifit", "aug-cc-pvtz-rifit",
    "aug-cc-pvqz-rifit", "aug-cc-pv5z-rifit", "aug-cc-pv6z-rifit",
    "aug-cc-pvdz-pp-rifit", "aug-cc-pvtz-pp-rifit",
    "aug-cc-pvqz-pp-rifit", "aug-cc-pv5z-pp-rifit",
    "aug-cc-pwcvdz-rifit", "aug-cc-pwcvtz-rifit",
    "aug-cc-pwcvqz-rifit", "aug-cc-pwcv5z-rifit",
    "aug-cc-pwcvdz-pp-rifit", "aug-cc-pwcvtz-pp-rifit",
    "aug-cc-pwcvqz-pp-rifit", "aug-cc-pwcv5z-pp-rifit",
    # The PP orbital sets those fits belong to (shipped 2026-09-08). Each
    # routes the union of its 39 elements' publications -- orbital blocks and
    # Stuttgart-Koeln ECP blocks are separate papers, and both differ per
    # element, so losing the route loses most of the attribution.
    "cc-pvdz-pp", "aug-cc-pvdz-pp",
    "cc-pwcvdz-pp", "aug-cc-pwcvdz-pp",
    "cc-pvtz-pp", "aug-cc-pvtz-pp",
    "cc-pwcvtz-pp", "aug-cc-pwcvtz-pp",
    "cc-pvqz-pp", "aug-cc-pvqz-pp",
    "cc-pwcvqz-pp", "aug-cc-pwcvqz-pp",
    "cc-pv5z-pp", "aug-cc-pv5z-pp",
    "cc-pwcv5z-pp", "aug-cc-pwcv5z-pp",
    "def2-svp", "def2-tzvp", "def2-qzvp",
    "def2-tzvpp", "def2-qzvpp",
    "def2-sv", "def2-sv(p)",
    "def2-svpd", "def2-tzvpd", "def2-tzvppd",
    "def2-qzvpd", "def2-qzvppd",
    "def2-svp-jk", "def2-tzvp-jk",
    "def2-sv(p)-jk", "def2-sv(p)-jkfit",
    "def2-tzvpp-jk", "def2-qzvp-jk", "def2-qzvpp-jk",
    "def2-universal-jkfit", "def2-universal-jfit",
    "def2-svp-j", "def2-sv(p)-j",
    "def2-tzvp-j", "def2-tzvpp-j",
    "def2-qzvp-j", "def2-qzvpp-j",
    "def2-svp-rifit",
    "def2-sv(p)-rifit",
    "def2-tzvp-rifit", "def2-tzvpd-rifit",
    "def2-tzvpp-rifit", "def2-tzvppd-rifit",
    "def2-qzvp-rifit", "def2-qzvpp-rifit", "def2-qzvppd-rifit",
    "def2-svpd-rifit",
    "def2-svp-c", "def2-sv(p)-c", "def2-svpd-c",
    "def2-tzvp-c", "def2-tzvpd-c",
    "def2-tzvpp-c", "def2-tzvppd-c",
    "def2-qzvp-c", "def2-qzvpp-c", "def2-qzvppd-c",
    "mini",
    "ano-rcc", "ano-rcc-mb",
    "sap_grasp_large", "sap_helfem_large",
    "pob-tzvp", "pob-dzvp-rev2", "pob-tzvp-rev2",
    "sto-3g", "sto-6g",
    # roadmap basis-library expansion (v0.8.0 BS sweep) — routed here in the
    # 2026-06 citation-database expansion.
    "pc-0", "pc-1", "pc-2", "pc-3", "pc-4",
    "aug-pc-0", "aug-pc-1", "aug-pc-2", "aug-pc-3", "aug-pc-4",
    "pcseg-0", "pcseg-1", "pcseg-2", "pcseg-3", "pcseg-4",
    "aug-pcseg-0", "aug-pcseg-1", "aug-pcseg-2", "aug-pcseg-3", "aug-pcseg-4",
    "pcseg-s-0", "pcseg-s-1", "pcseg-s-2",
    "pcs-0", "pcs-1", "pcs-2", "pcs-3", "aug-pcs-1", "aug-pcs-2",
    "pcj-0", "pcj-1", "pcj-2", "pcj-3",
    "x2c-tzvpall", "x2c-tzvpall-2c", "x2c-tzvpall-s",
    "sapporo-dkh3-dzp", "sapporo-dkh3-tzp", "sapporo-dkh3-qzp",
    "ano-r", "ano-r0", "ano-r1", "ano-r2", "ano-r3",
    "lanl2dz", "lanl2dzdp", "lanl2tz", "lanl08", "lanl08(d)", "lanl08(f)",
    "cc-pv(d+d)z", "cc-pv(t+d)z", "cc-pv(q+d)z", "cc-pv(5+d)z",
    "aug-cc-pv(d+d)z", "aug-cc-pv(t+d)z", "aug-cc-pv(q+d)z", "aug-cc-pv(5+d)z",
    # provenance via inline ``.g94`` header
    "cc-pvdz-f12-cabs", "cc-pvqz-f12-cabs", "cc-pvtz-f12-cabs",
    "def2-msvp", "def2-mtzvp", "def2-mtzvpp",
    "minix",
    "vdzp",
})


def _shipped_stems() -> set[str]:
    return {p.stem.lower() for p in BASIS_DIR.glob("*.g94")}


def _routes() -> dict[str, list[str]]:
    db = tomllib.loads(DATABASE_TOML.read_text())
    return db.get("routes", {}).get("basis_sets", {})


def _has_provenance_header(stem: str) -> bool:
    g94 = BASIS_DIR / f"{stem}.g94"
    if not g94.is_file():
        return False
    # Provenance headers live at the top of the file; reading the
    # first 50 lines is plenty and avoids loading the full basis.
    head = g94.read_text(errors="replace").splitlines()[:50]
    return any(_PROVENANCE_RE.match(L) for L in head)


def _has_route(stem: str, routes: dict[str, list[str]]) -> bool:
    routes_lc = {r.lower() for r in routes}
    if stem in routes_lc:
        return True
    # Check aliases that resolve to this stem.
    for alias, target in _ALIASES.items():
        if target == stem and alias in routes_lc:
            return True
    return False


def test_no_dangling_basis_routes():
    """Every `[routes.basis_sets]` key resolves to a shipped basis,
    either directly or via the Pople-syntax alias table."""
    shipped = _shipped_stems()
    dangling: list[str] = []
    for route in _routes():
        r_lc = route.lower()
        # Underscore-prefixed keys are synthetic non-basis routes, not basis
        # names (the same convention as functionals' _libxc_always /
        # _grid_always). `_bse_bundled_provenance` fires the BSE library
        # citation alongside any bundled basis; it has no .g94 of its own.
        if r_lc.startswith("_"):
            continue
        if r_lc in shipped:
            continue
        if r_lc in _ALIASES and _ALIASES[r_lc] in shipped:
            continue
        dangling.append(route)
    assert not dangling, (
        f"citation routes point at bases not shipped in "
        f"basis_library/basis/: {dangling}. Either ship the basis or "
        "remove the route from database.toml (and update _ALIASES "
        "if this is a Pople-syntax alias)."
    )


@pytest.mark.parametrize("stem", sorted(CURRENTLY_COVERED))
def test_pinned_basis_still_has_citation(stem: str):
    """A basis pinned in CURRENTLY_COVERED must retain its citation
    coverage — either a route in the database or an inline provenance
    header in its `.g94`."""
    g94 = BASIS_DIR / f"{stem}.g94"
    assert g94.is_file(), (
        f"basis {stem!r} was pinned in CURRENTLY_COVERED but its "
        f".g94 no longer ships — re-run setup_basis_library.sh or "
        f"remove the entry from the pinned set in the same commit."
    )
    routes = _routes()
    if _has_route(stem, routes) or _has_provenance_header(stem):
        return
    pytest.fail(
        f"basis {stem!r} lost its citation coverage: neither a route "
        f"in [routes.basis_sets] (database.toml) nor an inline "
        f"provenance header in basis_library/basis/{stem}.g94. "
        "Restore the route, restore the header, or remove from "
        "CURRENTLY_COVERED with justification."
    )


# Bases that must carry a concrete inline citation in the `.g94`
# header — not just a route. The route table is the primary
# attribution channel, but for hand-curated custom bases the inline
# header is the failsafe that survives the libint / BSE pipeline,
# mirroring, and any downstream fetcher per CLAUDE.md § 1 + § 8.
#
# Commit 22aface3 (May 2026 pob basis-set audit) regenerated these
# three files with `! Cite: see python/vibeqc/basis_library/README.md`
# — a README pointer that passes the lenient _PROVENANCE_RE check
# but doesn't actually carry the citation. This pin requires a real
# DOI (10.NNNN/…) in the inline header so that regression class
# can't recur silently.
_INLINE_DOI_REQUIRED: frozenset[str] = frozenset({
    "pob-tzvp",
    "pob-tzvp-rev2",
    "pob-dzvp-rev2",
})

_DOI_RE = re.compile(r"10\.\d{4,}/\S+")


@pytest.mark.parametrize("stem", sorted(_INLINE_DOI_REQUIRED))
def test_inline_doi_in_header(stem: str):
    """Pinned custom bases must carry a concrete DOI in the inline
    header, not a pointer to README.md or another file. See module
    docstring + comment on `_INLINE_DOI_REQUIRED` for the regression
    motivating this pin (commit 22aface3, May 2026).
    """
    g94 = BASIS_DIR / f"{stem}.g94"
    assert g94.is_file(), (
        f"{stem!r} is pinned in _INLINE_DOI_REQUIRED but its .g94 "
        "no longer ships — re-run setup_basis_library.sh or remove "
        "the entry from the pinned set in the same commit."
    )
    head = g94.read_text(errors="replace").splitlines()[:50]
    if any(_DOI_RE.search(L) for L in head):
        return
    pytest.fail(
        f"basis {stem!r} has no concrete DOI in its inline header "
        f"(basis_library/basis/{stem}.g94, first 50 lines). A "
        "pointer like `! Cite: see README.md` is not a substitute — "
        "the per-publication reference must travel with the basis "
        "data through the libint / BSE pipeline (CLAUDE.md § 1 + "
        "§ 8). Restore the publication's DOI to the header."
    )


# pob-TZVP-rev2's records come from five papers (#288): the 2013 parent,
# H-Br (2019), Rb-I (2022), Cs-Ba + Hf-Po (2021) and La-Lu (2023). Every
# human-read surface that names the basis's provenance must name all five;
# the route table is pinned in tests/test_citations.py.
_POB_TZVP_REV2_DOIS = (
    "10.1002/jcc.23153",
    "10.1002/jcc.26013",
    "10.1002/jcc.26839",
    "10.1002/jcc.26521",
    "10.1002/jcc.27097",
)
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _pob_tzvp_rev2_surfaces() -> dict[str, str]:
    from vibeqc import basis_crystal

    surfaces = {
        "basis_crystal._BREDOW_CITATIONS['pob-tzvp-rev2']": (
            basis_crystal._BREDOW_CITATIONS["pob-tzvp-rev2"]
        ),
    }
    # The .g94 headers were missed in #288 and kept the 2019 paper alone.
    for sub in ("basis", "custom"):
        for name in ("pob-tzvp-rev2.ecp", "pob-tzvp-rev2.g94"):
            path = BUNDLED_BASIS_DIR / sub / name
            if path.is_file():
                head = [
                    line for line in path.read_text().splitlines()[:60]
                    if line.startswith("!")
                ]
                surfaces[f"{sub}/{name} header"] = "\n".join(head)
    license_md = _REPO_ROOT / "docs" / "license.md"
    if license_md.is_file():  # absent in an installed wheel
        rows = [
            line for line in license_md.read_text().splitlines()
            if line.startswith("| `pob-tzvp-rev2.g94`")
        ]
        assert len(rows) == 1, rows
        surfaces["docs/license.md pob-tzvp-rev2.g94 row"] = rows[0]
    return surfaces


def test_pob_tzvp_rev2_provenance_surfaces_name_all_five_papers() -> None:
    """#288: the fetch-time citation, the ECP sidecar header and the
    license table each credited a subset (the 2019 paper alone, or the
    2021 paper alone), so each named papers that do not contain most of
    the basis's 78 records."""
    surfaces = _pob_tzvp_rev2_surfaces()
    assert "basis/pob-tzvp-rev2.ecp header" in surfaces
    assert "basis/pob-tzvp-rev2.g94 header" in surfaces
    missing = {
        name: [d for d in _POB_TZVP_REV2_DOIS if d not in text]
        for name, text in surfaces.items()
    }
    missing = {k: v for k, v in missing.items() if v}
    assert not missing, missing


# ---------------------------------------------------------------------------
# #407: the records carry the citation routes; database.toml is derivable.
# ---------------------------------------------------------------------------


def _record_citation_keys(stem: str) -> set[str] | None:
    import json

    path = BUNDLED_BASIS_DIR / "qvf" / f"{stem}.qvf.json"
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    routes = (record.get("provenance") or {}).get("citations") or []
    return {key for route in routes for key in route["keys"]}


def test_every_citation_route_key_exists_in_the_database():
    """A record may only route to a publication the database can print."""
    db = tomllib.loads(DATABASE_TOML.read_text())
    entries = set(db.get("entries", {}))
    missing: dict[str, set[str]] = {}
    for path in sorted((BUNDLED_BASIS_DIR / "qvf").glob("*.qvf.json")):
        stem = path.name[: -len(".qvf.json")]
        keys = _record_citation_keys(stem) or set()
        if keys - entries:
            missing[stem] = keys - entries
    assert not missing, missing


def test_route_table_is_the_union_of_the_record_citation_keys():
    """For every shipped basis with a route, ``[routes.basis_sets]`` lists
    exactly the keys its record's citation routes carry, so the table can
    be regenerated from the records and never drifts from them."""
    routes = _routes()
    drift: dict[str, tuple[set[str], set[str]]] = {}
    for stem in sorted(_shipped_stems()):
        if stem not in routes:
            continue
        expected = set(routes[stem])
        recorded = _record_citation_keys(stem)
        assert recorded is not None, f"{stem}: no record"
        if recorded != expected:
            drift[stem] = (expected - recorded, recorded - expected)
    assert not drift, (
        "route table and record citations disagree (route-only, record-only): "
        f"{drift}"
    )


def test_pob_tzvp_rev2_record_routes_each_range_to_its_paper():
    """#288 at the record level: the four valence papers cover disjoint
    element ranges, and the Stuttgart ECP papers cover only ECP elements."""
    import json

    record = json.loads(
        (BUNDLED_BASIS_DIR / "qvf" / "pob-tzvp-rev2.qvf.json").read_text(encoding="utf-8")
    )
    by_key: dict[str, list[str] | None] = {}
    for route in record["provenance"]["citations"]:
        for key in route["keys"]:
            by_key[key] = route.get("elements")
    assert by_key["peintinger_pob_tzvp_2013"] is None
    assert set(by_key["vilela_oliveira_pob_rev2_2019"]) == set(record["elements"]) - set(record["ecps"])
    assert "Ag" in by_key["laun_pob_rev2_fifth_period_2022"]
    assert "W" in by_key["laun_pob_rev2_sixth_period_2021"]
    assert "Gd" in by_key["seidler_pob_rev2_lanthanides_2023"]
    assert set(by_key["andrae_ecp_1990"]) <= set(record["ecps"])
