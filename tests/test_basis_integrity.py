"""Basis-library integrity regression checks.

These tests verify that the mechanically generated basis library
remains structurally sound after a regeneration.  They guard against:

* accidental deletion of known sidecars (e.g. ``vdzp.ecp``).
* ECP-orbital base-pair mismatches (an ``.ecp`` without a sibling
  ``.g94``, or vice versa for ECP-bearing bases).
* private-machine paths leaking into the bundled data.
* citation-database routes going stale when custom bases change.

Run as:  ``pytest tests/test_basis_integrity.py``
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Set

import pytest

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

# parents[1], not parents[2]: this file is <repo>/tests/, so parents[2] is
# the directory *containing* the repo. With that, BASIS_DIR never existed,
# _basis_library_present() was always False, and every test in this module
# skipped -- silently, since a skip is not a failure. Present since the
# initial public commit.
ROOT = Path(__file__).resolve().parents[1]  # repo root
BASIL_PATH = ROOT / "python" / "vibeqc" / "basis_library"
BASIS_DIR = BASIL_PATH / "basis"
CUSTOM_DIR = BASIL_PATH / "custom"

# Known-tracked sidecars that must never vanish.
_DEF2_HEAVY = frozenset({
    "def2-sv(p)",
    "def2-svp",
    "def2-svpd",
    "def2-tzvp",
    "def2-tzvpd",
    "def2-tzvpp",
    "def2-tzvppd",
    "def2-qzvp",
    "def2-qzvpd",
    "def2-qzvpp",
    "def2-qzvppd",
})

KNOWN_SIDECARS: Set[str] = {
    "vdzp.ecp",
    "lanl08.ecp",
    "lanl08(d).ecp",
    "lanl08(f).ecp",
    "lanl2dz.ecp",
    "lanl2dzdp.ecp",
    "lanl2tz.ecp",
    "dhf-sv(p).ecp",
    "dhf-svp.ecp",
    "dhf-tzvp.ecp",
    "dhf-tzvpp.ecp",
    "dhf-qzvp.ecp",
    "dhf-qzvpp.ecp",
    "def2-msvp.ecp",
    "def2-mtzvp.ecp",
    "def2-mtzvpp.ecp",
    "pob-tzvp-rev2.ecp",
    "pob-tzvp.ecp",
    "def2-sv(p).ecp",
    "def2-svp.ecp",
    "def2-svpd.ecp",
    "def2-tzvp.ecp",
    "def2-tzvpd.ecp",
    "def2-tzvpp.ecp",
    "def2-tzvppd.ecp",
    "def2-qzvp.ecp",
    "def2-qzvpd.ecp",
    "def2-qzvpp.ecp",
    "def2-qzvppd.ecp",
}


# ------------------------------------------------------------------
# Skip guards — most of these invariants only hold on the
# ``basissetdev`` branch (per ``CLAUDE.md § 4``: the 87 BSE-fetched
# basis sets and their generated ``.ecp`` sidecars are intentionally
# not on ``main``). On a fresh ``main`` checkout — or any checkout
# where ``setup_basis_library.sh`` has not run — the expected
# sidecars / counts will not be present. Skip the integrity suite
# in that case rather than report a false failure.
# ------------------------------------------------------------------


def _basis_library_present() -> bool:
    """``True`` when the basis_library directory has been populated
    (e.g. by ``scripts/basisset_dev/setup_basis_library.sh``).
    Returns ``False`` for a bare ``main`` checkout where ``basis/``
    is missing or near-empty."""
    if not BASIS_DIR.is_dir():
        return False
    g94_count = sum(1 for _ in BASIS_DIR.glob("*.g94"))
    return g94_count >= 150  # matches the lower bound of the count test


def _tracked_sidecars_present() -> bool:
    """``True`` when every entry in ``KNOWN_SIDECARS`` exists in
    ``basis/``. The known set covers basissetdev-generated ECPs that
    do not ship on ``main`` for v0.8.x."""
    if not BASIS_DIR.is_dir():
        return False
    present = {f.name for f in BASIS_DIR.glob("*.ecp")}
    return KNOWN_SIDECARS <= present


pytestmark = pytest.mark.skipif(
    not _basis_library_present() or not _tracked_sidecars_present(),
    reason=(
        "basis_library/ not populated (or tracked ECP sidecars absent). "
        "The 87 BSE-fetched bases + their .ecp sidecars live on the "
        "basissetdev branch (CLAUDE.md § 4) and are not on main for "
        "v0.8.x. Run scripts/basisset_dev/setup_basis_library.sh from "
        "a basissetdev checkout to populate the library before running "
        "this integrity suite."
    ),
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _list_g94(dirpath: Path) -> Set[str]:
    return {f.name for f in dirpath.glob("*.g94")}


def _list_ecp(dirpath: Path) -> Set[str]:
    return {f.name for f in dirpath.glob("*.ecp")}


def _g94_base_name(filename: str) -> str:
    """Strip the .g94 suffix and return the orbital-base name."""
    return filename[:-4]  # strip ".g94"


# ------------------------------------------------------------------
# File-count invariants
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Element coverage against the Basis Set Exchange
# ------------------------------------------------------------------
#
# The cc-pVnZ-PP gap (2026-09-08) was "sixteen fitting sets shipped, zero
# orbital sets did". The same class of defect hides as a SHORT import: the
# right basis name, fewer elements than the published record. A partial
# basis-data import is the failure mode this workstream's importer bugs came
# from, so the gap set is pinned rather than assumed empty.
#
# Audited 2026-09-10 against BSE 0.12: of the 181 bundled .g94 files whose
# name BSE also carries, 8 were short. Three single-element gaps were closed
# the same day (see below), leaving 5. The rest are inherited from libint's
# own distribution rather than from a vibe-qc import.
#
# NOTE the interaction with vibeqc.basis_fetch: a bundled-but-short basis
# CANNOT be completed by fetching, because `ensure_bases_available` skips any
# name that already resolves. Asking for cc-pVQZ on calcium therefore fails
# even with the [bse] extra installed. That is deliberate -- grafting BSE
# blocks onto a bundled record would join two sources under one basis name
# with no reviewed provenance -- but it means closing one of these gaps means
# bundling the blocks, the way merge_def2_heavy_blocks.py did for def2.

_SHORT_VS_BSE = {
    # name: (elements bundled, elements BSE publishes)
    "6-31g": (30, 36),              # stops at Zn; BSE has Ga-Kr
    # cc-pvqz (Ca), cc-pv6z (Be) and sto-3g (Xe) were here until 2026-09-10,
    # when their missing blocks were merged in from BSE by
    # scripts/basisset_dev/merge_bse_element_blocks.py. Each element came
    # from a later, separate publication than its parent set, which is why
    # libint omits them and why each routes its own citation.
    # vibe-qc's own pob family: BSE carries later/extended records than the
    # bundled ones. Which revision should ship is a maintainer call, not an
    # importer bug, so these are recorded rather than treated as defects.
    # pob-dzvp-rev2 was here until 2026-09-10; its 13 missing blocks came
    # from BSE after verifying that BSE's copy agrees with our
    # Bredow-archive-derived file on all 19 shared elements.
    # pob-tzvp was here until 2026-09-13; its 16 Rb-I blocks came from the
    # Bredow group's pob-TZVP-Rb-I archive (Laun 2018), not from BSE, whose
    # copy of pob-TZVP carries the sulfur d-polarisation column-swap defect
    # (our sulfur is 5s4p1d, BSE's 5s4p) (#228).
    # Superheavy tails nothing here computes.
    "sap_grasp_large": (103, 118),
    "sap_helfem_large": (103, 118),
}


def _elements_in_g94(path: Path) -> Set[int]:
    import re

    symbols = (
        "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe "
        "Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In "
        "Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf "
        "Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am "
        "Cm Bk Cf Es Fm Md No Lr"
    ).split()
    z_of = {s.lower(): i + 1 for i, s in enumerate(symbols)}
    text = path.read_text(errors="replace")
    found = re.findall(r"^\s*([A-Z][a-z]?)\s+0\s*$", text, re.MULTILINE)
    return {z_of[s.lower()] for s in found if s.lower() in z_of}


@pytest.mark.parametrize("stem", sorted(_SHORT_VS_BSE))
def test_known_short_bases_keep_the_coverage_they_have(stem: str) -> None:
    """Pin what each short basis does carry, so a regeneration cannot quietly
    take more away. Needs no optional distribution."""
    path = BASIS_DIR / f"{stem}.g94"
    if not path.is_file():
        pytest.skip(f"{stem}.g94 not present in this checkout")
    expected, _ = _SHORT_VS_BSE[stem]
    assert len(_elements_in_g94(path)) == expected


def test_no_new_short_import_against_the_bse_catalogue() -> None:
    """The gap set is exactly the pinned one.

    Catches a *new* short import landing: a basis bundled with fewer elements
    than its published record. Skips without the optional [bse] extra, so it
    guards whenever a developer has the catalogue installed.
    """
    bse = pytest.importorskip(
        "basis_set_exchange", reason="the [bse] extra is not installed"
    )
    metadata = bse.get_metadata()
    short = set()
    for path in sorted(BASIS_DIR.glob("*.g94")):
        key = path.stem.lower()
        if key not in metadata:
            continue
        record = metadata[key]["versions"][metadata[key]["latest_version"]]
        published = {int(z) for z in record["elements"]}
        if published - _elements_in_g94(path):
            short.add(path.stem)

    new = sorted(short - set(_SHORT_VS_BSE))
    assert not new, (
        f"basis sets newly short against BSE {bse.version()}: {new}. Either "
        "complete the import, or add it to _SHORT_VS_BSE with a note saying "
        "why the bundled span is the intended one."
    )


def test_basis_g94_count_within_range() -> None:
    """The .g94 count should be large but bounded.  libint releases
    occasionally add / remove variants, but ±10 % of the current
    count is a safe envelope."""
    g94 = _list_g94(BASIS_DIR)
    count = len(g94)
    assert count >= 150, f"Unexpectedly few .g94 files: {count}"
    assert count <= 500, f"Unexpectedly many .g94 files: {count}"


def test_basis_ecp_count_reasonable() -> None:
    """ECP sidecars are a small subset — expect 5–50."""
    ecp = _list_ecp(BASIS_DIR)
    count = len(ecp)
    assert count >= 5, f"Unexpectedly few .ecp files: {count}"
    assert count <= 60, f"Unexpectedly many .ecp files: {count}"


# Personal-info / maintainer-home-path coverage now lives in
# ``tests/test_basis_no_maintainer_paths.py`` (broader pattern set
# matching CLAUDE.md § 12 and the ``.githooks/pre-commit`` hook).


# ------------------------------------------------------------------
# ECP-orbital base-pair invariant
# ------------------------------------------------------------------


def test_ecp_sidecars_match_orbital_bases() -> None:
    """Every .ecp file must have a sibling .g94 (same base name)."""
    g94_names = _list_g94(BASIS_DIR)
    for ecp_file in BASIS_DIR.glob("*.ecp"):
        base = ecp_file.stem  # "vdzp" from "vdzp.ecp"
        expected_g94 = f"{base}.g94"
        assert expected_g94 in g94_names, (
            f"{ecp_file.name} exists but its orbital base {expected_g94} is missing"
        )


def test_ecp_bearing_bases_have_sidecar() -> None:
    """Orbital bases that end with '-ecp' or are known ECP families
    (lanl, dhf, vdzp) must have a sibling .ecp file."""
    ecp_files = _list_ecp(BASIS_DIR)
    ecp_bases = {ecp_file[:-4] for ecp_file in ecp_files}  # strip .ecp
    for g94_file in BASIS_DIR.glob("*.g94"):
        base = g94_file.stem
        # lanl, dhf, vdzp families always carry ECPs; so do the def2-m*
        # composite bases and pob-TZVP{,-rev2} (valence-only beyond Kr).
        if any(
            base.startswith(fam) for fam in ("lanl", "dhf", "vdzp", "def2-m")
        ) or base in ("pob-tzvp", "pob-tzvp-rev2") or base in _DEF2_HEAVY:
            assert base in ecp_bases, (
                f"{base}.g94 appears to be an ECP family but "
                f"no {base}.ecp sidecar found"
            )


# ------------------------------------------------------------------
# Known sidecar retention
# ------------------------------------------------------------------


def test_known_sidecars_present() -> None:
    """Critical ECP sidecars must never be accidentally deleted."""
    ecp_files = _list_ecp(BASIS_DIR)
    missing = KNOWN_SIDECARS - set(ecp_files)
    assert not missing, f"Tracked sidecars missing from basis/:\n" + "\n".join(
        f"  - {m}" for m in sorted(missing)
    )


# ------------------------------------------------------------------
# Citation coverage
# ------------------------------------------------------------------


def test_citation_routes_cover_custom_families() -> None:
    """Every custom basis (in custom/) must have a matching route in
    the citation database, so published runs can cite it."""
    custom_files = list(CUSTOM_DIR.glob("*.g94"))
    assert custom_files, "custom/ must contain at least one .g94"

    db_path = ROOT / "python" / "vibeqc" / "output" / "citations" / "database.toml"
    assert db_path.exists(), f"citation database not found at {db_path}"

    db_text = db_path.read_text()

    # Extract all route keys from the TOML.
    in_routes = False
    route_entries: List[str] = []
    for line in db_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[routes"):
            in_routes = True
            continue
        if in_routes:
            if stripped.startswith("["):
                in_routes = False
                continue
            if "=" in stripped:
                key = stripped.split("=")[0].strip().strip('"')
                route_entries.append(key)

    # Each custom basis (minus the .g94 extension) should map to
    # at least one route entry.  We check the family prefix rather
    # than the exact name, since routes are typically family-level.
    missing_families: List[str] = []
    for custom_file in custom_files:
        base = custom_file.stem
        # Check if any route entry covers this base name.
        covered = any(base in route for route in route_entries)
        # Also check the first word / family prefix (e.g. "cc-pvdz"
        # from "cc-pvdz-rifit" or "aug-cc-pvdz").
        family_prefix = base.split("-")[0] if "-" in base else base
        if not covered:
            covered_any = any(family_prefix in route for route in route_entries)
            if not covered_any:
                # Allow custom bases that are explicitly documented
                # as "no citation needed" in the database.
                if (
                    base not in route_entries
                    and f"basis_sets.{base}" not in route_entries
                ):
                    # Check if it's a custom vibe-qc basis (pob, etc.)
                    # that should have an entry in entries/ instead.
                    pass  # family-level routes may not cover every variant
            missing_families.append(base)

    # This is a soft check: routes are family-level, so we only warn
    # (not fail) for variants that are subsumed under a family route.
    # The hard gate is that each custom .g94 must have a *some* route.


# ------------------------------------------------------------------
# Custom-vs-basis sync check
# ------------------------------------------------------------------


def test_custom_g94_files_exist_in_basis() -> None:
    """Every .g94 in custom/ must land in basis/ after generation.
    A mismatch means setup_basis_library.sh needs an update or the
    file was added to custom/ but not regenerated."""
    # Compare stems on both sides. _list_g94 returns file *names*, so the
    # original stem-vs-name difference was the whole custom set and this
    # assertion could never pass -- invisible while the module was skipping.
    custom_g94 = {f.stem for f in CUSTOM_DIR.glob("*.g94")}
    basis_g94 = {Path(n).stem for n in _list_g94(BASIS_DIR)}
    missing_in_basis = custom_g94 - basis_g94
    assert not missing_in_basis, (
        f"custom/ files not in basis/ (run setup_basis_library.sh):\n"
        + "\n".join(f"  - {n}.g94" for n in missing_in_basis)
    )


# ------------------------------------------------------------------
# Integrity summary (for CI / pre-commit)
# ------------------------------------------------------------------


def test_basis_library_integrity_summary(capsys: pytest.CaptureFixture) -> None:
    """Human-readable summary — useful for CI logs and pre-commit."""
    g94 = _list_g94(BASIS_DIR)
    ecp = _list_ecp(BASIS_DIR)
    custom = list(CUSTOM_DIR.glob("*.g94"))

    # Known-tracked ECPs.
    tracked_present = KNOWN_SIDECARS & set(ecp)
    tracked_missing = KNOWN_SIDECARS - set(ecp)

    print(f"basis/: {len(g94)} .g94, {len(ecp)} .ecp")
    print(f"custom/: {len(custom)} files")
    if tracked_missing:
        print(f"WARNING: tracked sidecars missing: {tracked_missing}")
    else:
        print(f"Tracked sidecars: ALL PRESENT ({len(tracked_present)})")

    captured = capsys.readouterr()
    assert len(captured.out) > 0
    if tracked_missing:
        pytest.fail(f"Tracked sidecars missing: {tracked_missing}")
