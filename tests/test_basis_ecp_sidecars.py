"""ECP `.g94` + `.ecp` sidecar contract for the bundled basis library.

An ECP-bearing basis set (vDZP, the LANL2DZ family, dhf-*, def2 beyond
Kr, the pob sets, ...) is one QVF-Basis record carrying both its shells
and its potentials. libint2 cannot parse ECP blocks, so the record is
rendered as an orbital-only ``<name>.g94`` plus a ``<name>.ecp`` sidecar
that vibe-qc reads via libecpint (``vibeqc.basis_toolkit.exporter_g94``).

These tests pin that contract on the runtime-shipped
``python/vibeqc/basis_library/basis/`` tree so an accidental deletion
of a sidecar, easy to miss inside a large basis-data diff, fails
loudly instead of silently breaking ECP SCF for the affected basis.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import vibeqc as _vq


BUNDLED_BASIS_DIR = Path(_vq.__file__).resolve().parent / "basis_library"
BASIS_DIR = BUNDLED_BASIS_DIR / "basis"

# Matches the BSE convention "<Symbol>-ECP <lmax> <ncore>", accepting both
# mixed-case (Na) and the Pople-era all-caps form (NA).
_ECP_HEADER_RE = re.compile(r"^\s*([A-Z][A-Za-z]?)-ECP\s+(\d+)\s+(\d+)\s*$")


def _has_ecp_block(path: Path) -> bool:
    text = path.read_text(errors="replace")
    return any(_ECP_HEADER_RE.match(line) for line in text.splitlines())


# Pinned inventory: every record the renderer is expected to emit an
# `.ecp` sidecar for. Update intentionally when a new ECP-bearing record
# joins ``basis_library/qvf/``.
EXPECTED_ECP_BASES = frozenset({
    "dhf-qzvp", "dhf-qzvpp",
    "dhf-sv(p)", "dhf-svp",
    "dhf-tzvp", "dhf-tzvpp",
    "lanl08", "lanl08(d)", "lanl08(f)",
    "lanl2dz", "lanl2dzdp", "lanl2tz",
    "vdzp",
    # The 3c composite bases carry the def2-ECP beyond Kr; the pob sets carry
    # the Stuttgart ECPs of their CRYSTAL Z+200 records (2026-09; pob-TZVP's
    # Rb-I from the Laun 2018 archive, #228; pob-DZVP-rev2's Rb-I from the
    # same paper's pob-DZVP archive, 2026-10, #292).
    "def2-msvp", "def2-mtzvp", "def2-mtzvpp",
    "pob-tzvp-rev2", "pob-tzvp", "pob-dzvp-rev2",
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
        "Bundled basis library is missing expected ECP sidecars; "
        "re-render with scripts/render_basis_library.py. "
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
    still contain `<Sym>-ECP` header lines: libint2 would reject the
    file on load. The renderer writes potentials to the sidecar only."""
    unsplit = [p.name for p in sorted(BASIS_DIR.glob("*.g94")) if _has_ecp_block(p)]
    assert not unsplit, (
        f".g94 files still contain ECP blocks: {unsplit}; "
        "re-render with scripts/render_basis_library.py"
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

# basis -> (ECP records bundled, cc-pVDZ-PP overlap floor, Ag V_ECP trace,
# AgCl RHF energy and the largest |V_ECP| element, from PySCF 2.14). The
# PySCF inputs were built straight from the CRYSTAL source records with the
# +2 power conversion written by hand, so they share no code with vibe-qc's
# bridge. All three pob sets carry the same potentials block for block; only
# the orbital shells differ (#228, #292), which is why the traces and the
# maxima differ while the potentials agree.
_POB_ECP_CASES = {
    "pob-tzvp-rev2": (46, 25, 312.60824098823844, -605.5273792995, 165.9358),
    "pob-tzvp": (16, 14, 314.37967120671487, -605.5487242661, 165.9358),
    "pob-dzvp-rev2": (16, 14, 111.9109001638, -605.3124001041, 79.39038),
}

# #515: the full-precision Ag V_ECP reference of the two pob-TZVP sets, from
# examples/regression/pob_ag_ecp_reference.py (PySCF 2.14 fed the CRYSTAL
# source records directly). vibe-qc agrees with it to 7e-12 on the trace and
# 3e-12 on the sorted generalized eigenvalues; the pre-#291 rounded
# pob-tzvp-rev2.g94 misses it by 1.0e-9 and 3.3e-9, so 1e-10 catches that.
_POB_ECP_REFERENCE = json.loads(
    (Path(__file__).parent / "data" / "pob_ag_ecp_reference.json").read_text()
)["bases"]
_POB_ECP_TOLERANCE = 1e-10


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

    Reference traces and largest elements in ``_POB_ECP_CASES`` (PySCF 2.14,
    spherical AOs). Before #207 the pob-TZVP-rev2 trace was +18101.65 on the
    molecular route and +16254.03 on the periodic one.
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
    trace, largest = _POB_ECP_CASES[name][2], _POB_ECP_CASES[name][4]
    assert float(np.abs(matrix).max()) == pytest.approx(largest, abs=1e-3)
    reference = _POB_ECP_REFERENCE.get(name)
    if reference is None:
        assert float(np.trace(matrix)) == pytest.approx(trace, abs=1e-6)
        return
    # Both pob-TZVP sets at the same tolerance, so a gap between them shows
    # (#228, #291): the trace, and the whole matrix through its generalized
    # eigenvalues against the AO overlap, which do not depend on AO order.
    import scipy.linalg

    from vibeqc._vibeqc_core import compute_overlap

    assert basis.nbasis == reference["n_ao"]
    assert float(np.trace(matrix)) == pytest.approx(reference["trace"], abs=_POB_ECP_TOLERANCE)
    assert reference["trace"] == trace
    eigenvalues = np.sort(scipy.linalg.eigh(
        matrix, np.asarray(compute_overlap(basis)), eigvals_only=True))
    deviation = np.abs(eigenvalues - np.asarray(reference["generalized_eigenvalues"]))
    assert float(deviation.max()) < _POB_ECP_TOLERANCE


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
