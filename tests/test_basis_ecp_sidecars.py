"""ECP `.g94` + `.ecp` sidecar contract for the bundled basis library.

ECP-bearing basis sets (vDZP, LANL2DZ family, dhf-*, …) arrive from
BSE as a single `.g94` that mixes orbital blocks with `<Sym>-ECP`
blocks. libint2 cannot parse the ECP blocks, so
``scripts/basisset_dev/split_ecp_g94.py`` (driven by
``scripts/setup_basis_library.sh``) splits each affected file into
an orbital-only ``<name>.g94`` plus a ``<name>.ecp`` sidecar that
vibe-qc reads via libecpint.

These tests pin the post-split contract on the runtime-shipped
``python/vibeqc/basis_library/basis/`` tree so an accidental deletion
of a sidecar — easy to miss inside a large basis-data diff — fails
loudly instead of silently breaking ECP SCF for the affected basis.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import vibeqc as _vq


BUNDLED_BASIS_DIR = Path(_vq.__file__).resolve().parent / "basis_library"
BASIS_DIR = BUNDLED_BASIS_DIR / "basis"
CUSTOM_DIR = BUNDLED_BASIS_DIR / "custom"

# Same regex as scripts/basisset_dev/split_ecp_g94.py — matches the
# BSE convention "<Symbol>-ECP <ncore> <lmax>", accepting both
# mixed-case (Na) and the Pople-era all-caps form (NA) used by
# LANL2DZ and friends.
_ECP_HEADER_RE = re.compile(r"^\s*([A-Z][A-Za-z]?)-ECP\s+(\d+)\s+(\d+)\s*$")


def _has_ecp_block(path: Path) -> bool:
    text = path.read_text(errors="replace")
    return any(_ECP_HEADER_RE.match(line) for line in text.splitlines())


# Pinned inventory: every basis set the build pipeline is expected to
# emit an `.ecp` sidecar for. Update intentionally when adding a new
# ECP-bearing basis to ``custom/`` (or when libint ships a new one).
EXPECTED_ECP_BASES = frozenset({
    "dhf-qzvp", "dhf-qzvpp",
    "dhf-sv(p)", "dhf-svp",
    "dhf-tzvp", "dhf-tzvpp",
    "lanl08", "lanl08(d)", "lanl08(f)",
    "lanl2dz", "lanl2dzdp", "lanl2tz",
    "vdzp",
    # The 3c composite bases carry the def2-ECP beyond Kr; pob-TZVP-rev2
    # carries the Stuttgart ECPs of its CRYSTAL Z+200 records (2026-09).
    "def2-msvp", "def2-mtzvp", "def2-mtzvpp",
    "pob-tzvp-rev2",
    # The def2 family beyond Kr (BSE Rb-Rn blocks appended to libint's
    # H-Kr files) ships the def2-ECP as a sidecar (2026-09).
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
    # The Peterson/Figgen PP orbital sets (2026-09-08). BSE ships the ECP
    # blocks inside each orbital file; the promotion splits them out. All 16
    # cover the same 39 elements (Cu-Kr, Y-Xe, Hf-Rn), one ECP block each.
    "aug-cc-pv5z-pp",
    "aug-cc-pvdz-pp",
    "aug-cc-pvqz-pp",
    "aug-cc-pvtz-pp",
    "aug-cc-pwcv5z-pp",
    "aug-cc-pwcvdz-pp",
    "aug-cc-pwcvqz-pp",
    "aug-cc-pwcvtz-pp",
    "cc-pv5z-pp",
    "cc-pvdz-pp",
    "cc-pvqz-pp",
    "cc-pvtz-pp",
    "cc-pwcv5z-pp",
    "cc-pwcvdz-pp",
    "cc-pwcvqz-pp",
    "cc-pwcvtz-pp",
})


def test_every_expected_ecp_basis_has_sidecar():
    """Pinned-list check: each known ECP-bearing basis ships a non-empty
    ``.ecp`` next to its orbital ``.g94``."""
    missing: list[str] = []
    empty: list[str] = []
    no_orbital: list[str] = []
    for stem in sorted(EXPECTED_ECP_BASES):
        ecp = BASIS_DIR / f"{stem}.ecp"
        g94 = BASIS_DIR / f"{stem}.g94"
        if not g94.is_file():
            no_orbital.append(stem)
            continue
        if not ecp.is_file():
            missing.append(stem)
            continue
        if ecp.stat().st_size == 0 or not _has_ecp_block(ecp):
            empty.append(stem)
    problems = []
    if no_orbital:
        problems.append(f"orbital .g94 missing: {no_orbital}")
    if missing:
        problems.append(f".ecp sidecar missing: {missing}")
    if empty:
        problems.append(f".ecp sidecar present but contains no ECP block: {empty}")
    assert not problems, (
        "Bundled basis library is missing expected ECP sidecars — "
        "re-run scripts/setup_basis_library.sh. "
        + " | ".join(problems)
    )


def test_no_orphan_ecp_sidecars():
    """Every `.ecp` in the bundle has a matching `.g94` of the same stem.
    An orphan sidecar means either the `.g94` was deleted or the sidecar
    was generated against a basis name that no longer ships."""
    orphans = [
        p.name for p in sorted(BASIS_DIR.glob("*.ecp"))
        if not p.with_suffix(".g94").is_file()
    ]
    assert not orphans, f"Orphan .ecp sidecars (no matching .g94): {orphans}"


def test_no_unsplit_ecp_blocks_in_orbital_g94():
    """After ``setup_basis_library.sh`` runs, no shipped `.g94` should
    still contain `<Sym>-ECP` header lines — libint2 would reject the
    file on load. A failure here means the build-time split did not
    run on a file it should have."""
    unsplit = [p.name for p in sorted(BASIS_DIR.glob("*.g94")) if _has_ecp_block(p)]
    assert not unsplit, (
        f".g94 files still contain unsplit ECP blocks: {unsplit} — "
        "re-run scripts/setup_basis_library.sh"
    )


def test_custom_ecp_sources_all_have_sidecar():
    """Source-of-truth cross-check: every ``custom/*.g94`` that carries
    ECP blocks must have a corresponding sidecar in ``basis/``. Catches
    the case where a new ECP-bearing basis lands in ``custom/`` but
    ``setup_basis_library.sh`` was never re-run."""
    if not CUSTOM_DIR.is_dir():
        pytest.skip(f"{CUSTOM_DIR} not present in this checkout")
    missing: list[str] = []
    for src in sorted(CUSTOM_DIR.glob("*.g94")):
        if not _has_ecp_block(src):
            continue
        if not (BASIS_DIR / f"{src.stem}.ecp").is_file():
            missing.append(src.name)
    assert not missing, (
        "custom/*.g94 sources contain ECP blocks but no sidecar shipped in "
        f"basis/: {missing} — re-run scripts/setup_basis_library.sh"
    )


def test_custom_ecp_sidecar_sources_land_in_basis():
    """Pre-split ``custom/*.ecp`` sources must be copied into ``basis/``.

    This catches already-split custom bases such as vDZP: the orbital
    ``custom/vdzp.g94`` has no ECP block left, so the splitter cannot
    recreate ``basis/vdzp.ecp`` unless the setup script copies the
    source sidecar first.
    """
    if not CUSTOM_DIR.is_dir():
        pytest.skip(f"{CUSTOM_DIR} not present in this checkout")
    missing: list[str] = []
    changed: list[str] = []
    for src in sorted(CUSTOM_DIR.glob("*.ecp")):
        dst = BASIS_DIR / src.name
        if not dst.is_file():
            missing.append(src.name)
            continue
        if dst.read_bytes() != src.read_bytes():
            changed.append(src.name)
    problems = []
    if missing:
        problems.append(f"missing from basis/: {missing}")
    if changed:
        problems.append(f"basis/ copy differs from custom/: {changed}")
    assert not problems, (
        "custom/*.ecp sidecars did not land cleanly: "
        + " | ".join(problems)
    )
