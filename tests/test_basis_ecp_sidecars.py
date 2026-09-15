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
    # The 3c composite bases carry the def2-ECP beyond Kr; pob-TZVP-rev2 and
    # pob-TZVP carry the Stuttgart ECPs of their CRYSTAL Z+200 records
    # (2026-09; pob-TZVP's Rb-I from the Laun 2018 archive, #228).
    "def2-msvp", "def2-mtzvp", "def2-mtzvpp",
    "pob-tzvp-rev2", "pob-tzvp",
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


# ---------------------------------------------------------------------------
# #207: the CRYSTAL -> libecpint power and local-channel conventions.
#
# CRYSTAL's NKL is the literal power of r (CRYSTAL23 eqs. 3.18-3.19); libecpint
# reads a Gaussian-format power and stores n - 2. The pob bridge passed NKL
# through unconverted, so every Stuttgart r^0 term acted as a singular r^-2
# one, and it emitted no local channel, so libecpint promoted the f projector
# to the local potential on the periodic route.
# ---------------------------------------------------------------------------

# basis -> (ECP records bundled, cc-pVDZ-PP overlap floor, Ag V_ECP trace and
# AgCl RHF energy from PySCF 2.14). The PySCF inputs were built straight from
# the CRYSTAL source records with the +2 power conversion written by hand, so
# they share no code with vibe-qc's bridge. pob-TZVP's Rb-I records carry the
# same potentials as pob-TZVP-rev2's; only the orbital shells differ (#228).
_POB_ECP_CASES = {
    "pob-tzvp-rev2": (46, 25, 312.6082409892, -605.5273792995),
    "pob-tzvp": (16, 14, 314.3796712067, -605.5487242661),
}


def _parse_ecp_terms(path: Path) -> dict[str, list[tuple[int, int, float, float]]]:
    """``{symbol: [(am, power, exponent, coefficient), ...]}`` from a sidecar.

    Deliberately a local parser: these tests pin the shipped *text*, so they
    must not inherit an interpretation from the production reader.
    """
    lines = path.read_text(errors="replace").splitlines()
    out: dict[str, list[tuple[int, int, float, float]]] = {}
    i = 0
    while i < len(lines):
        header = _ECP_HEADER_RE.match(lines[i])
        if not header:
            i += 1
            continue
        symbol, lmax = header.group(1).capitalize(), int(header.group(2))
        i += 1
        terms: list[tuple[int, int, float, float]] = []
        for channel in range(lmax + 1):
            i += 1                                   # "<letter> potential"
            count = int(lines[i].split()[0])
            i += 1
            for _ in range(count):
                power, exponent, coefficient = lines[i].split()[:3]
                am = lmax if channel == 0 else channel - 1
                terms.append((am, int(power), float(exponent), float(coefficient)))
                i += 1
        out[symbol] = terms
    return out


@pytest.mark.parametrize("name", sorted(_POB_ECP_CASES))
def test_pob_ecp_powers_agree_with_the_bse_derived_set(name):
    """pob-TZVP{,-rev2} and cc-pVDZ-PP ship the same Stuttgart-Cologne ECPs
    for the elements they share. cc-pVDZ-PP comes from the Basis Set Exchange
    in Gaussian convention, so the powers must agree element for element."""
    pob = _parse_ecp_terms(BASIS_DIR / f"{name}.ecp")
    bse = _parse_ecp_terms(BASIS_DIR / "cc-pvdz-pp.ecp")
    shared = sorted(set(pob) & set(bse))
    assert len(shared) >= _POB_ECP_CASES[name][1], (
        f"expected the heavy overlap, found {shared}"
    )
    mismatched = []
    for symbol in shared:
        by_params = {
            (am, round(exp, 6), round(coef, 6)): power
            for am, power, exp, coef in bse[symbol]
        }
        for am, power, exp, coef in pob[symbol]:
            key = (am, round(exp, 6), round(coef, 6))
            if key in by_params and by_params[key] != power:
                mismatched.append(f"{symbol}: am={am} exp={exp} pob n={power} bse n={by_params[key]}")
    assert not mismatched, (
        f"{name} radial powers disagree with the BSE-derived copy of the "
        "same ECPs (#207):\n  " + "\n  ".join(mismatched[:12])
    )


@pytest.mark.parametrize("name", sorted(_POB_ECP_CASES))
def test_every_pob_record_carries_a_local_channel(name):
    """libecpint reads its local channel off the highest angular momentum
    present. Without a primitive at ``local_ell`` the highest projector is
    silently promoted to the local potential (#207)."""
    from vibeqc.basis_crystal import crystal_ecp_to_libecpint_arrays
    from vibeqc.periodic_runner import _bundled_pob_source_atoms

    missing = []
    checked = 0
    for atom in _bundled_pob_source_atoms(name):
        if not (atom.has_ecp and atom.ecp is not None):
            continue
        arrays = crystal_ecp_to_libecpint_arrays(atom.ecp)
        checked += 1
        if arrays.local_ell not in arrays.ams:
            missing.append(int(atom.Z))
    assert checked == _POB_ECP_CASES[name][0]
    assert not missing, f"no local channel emitted for Z={missing}"


@pytest.mark.parametrize("name", sorted(_POB_ECP_CASES))
def test_pob_silver_ecp_matrix_matches_the_independent_reference(name):
    """The Ag V_ECP over an AgCl pair, against PySCF 2.14 ``ECPscalar`` fed
    the same basis and the same ECP parameters in Gaussian convention.

    Reference traces in ``_POB_ECP_CASES``; the largest element is
    1.659358e+02 Ha for both bases (PySCF 2.14, spherical AOs). Before #207
    the pob-TZVP-rev2 trace was +18101.65 on the molecular route and
    +16254.03 on the periodic one.
    """
    import numpy as np

    from vibeqc._vibeqc_core import compute_ecp_matrix_from_primitives
    from vibeqc.ecp_metadata import inline_ecp_data_for

    half = 5.55 * 1.8897261246257702 / 2      # the #52 AgCl cell, two centres
    molecule = _vq.Molecule(
        [_vq.Atom(47, [0.0, 0.0, 0.0]), _vq.Atom(17, [half, half, half])]
    )
    basis = _vq.BasisSet(molecule, name)
    blocks, centers, _eff_z, ncore = inline_ecp_data_for(molecule, name)
    assert ncore == 28 and len(blocks) == 1
    matrix = np.asarray(
        compute_ecp_matrix_from_primitives(basis, list(centers[0]), [blocks[0]]),
        dtype=float,
    )
    trace = _POB_ECP_CASES[name][2]
    assert float(np.trace(matrix)) == pytest.approx(trace, abs=1e-6)
    assert float(np.abs(matrix).max()) == pytest.approx(165.9358, abs=1e-3)


@pytest.mark.parametrize("name", sorted(_POB_ECP_CASES))
def test_pob_silver_chloride_rhf_energy_matches_the_independent_reference(name, tmp_path):
    """End-to-end: RHF on AgCl at r_e, against PySCF 2.14 RHF with the same
    basis and ECP (energies in ``_POB_ECP_CASES``). The shipped
    pob-TZVP-rev2 ECP gave -564.5993603093 Ha before #207, 40.93 Ha too
    high."""
    molecule = _vq.Molecule(
        [_vq.Atom(47, [0.0, 0.0, 0.0]), _vq.Atom(17, [0.0, 0.0, 4.3101])]
    )
    result = _vq.run_job(
        molecule, basis=name, method="RHF", output=str(tmp_path / "agcl_pob_ecp"),
        write_molden_file=False, write_population_file=False,
        write_xyz_file=False, citations=False,
    )
    energy = _POB_ECP_CASES[name][3]
    assert float(result.energy) == pytest.approx(energy, abs=1e-6)
