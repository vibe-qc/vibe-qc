"""CRYSTAL-format basis-set parser and Bredow-archive integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import vibeqc as vq
from vibeqc.basis_crystal import (
    CrystalAtomBasis,
    CrystalShell,
    emit_g94,
    parse_crystal_atom_basis,
)

# --------------------------------------------------------------------------
# Parser basics
# --------------------------------------------------------------------------

POB_TZVP_H = """\
1 4
0 0 3 1.0 1.0
  34.061341000      0.00602519780
  5.1235746000      0.04502109400
  1.1646626000      0.20189726000
0 0 1 0.0 1.0
  0.4157455100      1.00000000000
0 0 1 0.0 1.0
  0.1795111000      1.00000000000
0 2 1 0.0 1.0
  0.8000000000      1.00000000000
"""

POB_TZVP_C = """\
6 8
0 0 6 2.0 1.0
  13575.349682      0.00022245814352
  2035.2333680      0.00172327382520
  463.22562359      0.00892557153140
  131.20019598      0.03572798450200
  42.853015891      0.11076259931000
  15.584185766      0.24295627626000
0 0 2 2.0 1.0
  6.2067138508      0.41440263448000
  2.5764896527      0.23744968655000
0 0 1 0.0 1.0
  0.4941102000      1.00000000000000
0 0 1 0.0 1.0
  0.1644071000      1.00000000000000
0 2 4 2.0 1.0
  34.697232244      0.00533336578050
  7.9582622826      0.03586410909200
  2.3780826883      0.14215873329000
  0.8143320818      0.34270471845000
0 2 1 0.0 1.0
  0.5662417100      1.00000000000000
0 2 1 0.0 1.0
  0.2673545000      1.00000000000000
0 3 1 0.0 1.0
  0.8791584200      1.00000000000000
"""


def test_parse_hydrogen_pob_tzvp_shape():
    atom = parse_crystal_atom_basis(POB_TZVP_H, "H")
    assert atom.Z == 1
    assert atom.element_symbol == "H"
    assert atom.has_ecp is False
    # 3 S shells + 1 P shell = 4 total.
    assert len(atom.shells) == 4
    types = [s.shell_type for s in atom.shells]
    assert types == ["S", "S", "S", "P"]
    # Contracted first shell: 3 primitives; remaining are 1.
    primitives = [s.n_primitives for s in atom.shells]
    assert primitives == [3, 1, 1, 1]


def test_parse_carbon_pob_tzvp_values_match_source():
    atom = parse_crystal_atom_basis(POB_TZVP_C, "C")
    assert atom.Z == 6
    assert len(atom.shells) == 8
    types = [s.shell_type for s in atom.shells]
    assert types == ["S", "S", "S", "S", "P", "P", "P", "D"]
    # Spot-check the largest exponent of the first S shell (tight core).
    assert atom.shells[0].exponents[0] == pytest.approx(13575.349682)
    assert atom.shells[0].coefficients[0] == pytest.approx(0.00022245814352)
    # D-polarization exponent.
    assert atom.shells[-1].exponents[0] == pytest.approx(0.8791584200)


def test_parse_rejects_non_user_defined_shell():
    bad = "1 1\n1 0 1 0.0 1.0\n  1.0  1.0\n"  # ITYPE=1 (built-in), not 0.
    with pytest.raises(ValueError, match="ITYPE != 0"):
        parse_crystal_atom_basis(bad, "<test>")


def test_parse_ecp_block_parses_ecp_header():
    # ECP parsing is now supported (basissetdev libecpint integration).
    # A minimal ECP header with insufficient primitive data raises ValueError
    # because the header declares more terms than the data provides.
    ecp = "253 1\nINPUT\n25. 0 3 4 4 4 0\n  1.0 1.0 0\n"
    with pytest.raises(ValueError, match="malformed CRYSTAL ECP record"):
        parse_crystal_atom_basis(ecp, "<test>")


# --------------------------------------------------------------------------
# Regression: the Bredow ``16_S`` SCAL typo ``0 3 1 0.0 1 0``
# --------------------------------------------------------------------------
#
# The d-polarization shell header in the upstream Bredow pob-TZVP and
# pob-TZVP-rev2 sulfur files mis-keys SCAL ``1.0`` as two tokens ``1 0`` (the
# decimal point typed as a space). A line-blind parser reads only five tokens
# for the header, leaving the stray ``0`` to be consumed as the d-shell
# exponent and the real exponent (0.5207… / 0.4107…) as the coefficient — a
# physically invalid exp=0 function, silently produced with no error.
# ``parse_crystal_atom_basis`` tolerates the typo exactly the way
# ``scripts/basisset_dev/pob_basis_verify.py`` does, so the in-package parser
# and the canonical verify/regenerate tool agree (CLAUDE.md § 1).

# A two-shell fragment: an s-shell, then the typo'd d-shell, so the tolerance
# is exercised mid-stream (not just as the very first shell of the file).
SULFUR_D_TYPO = """\
16 2
0 0 1 0.0 1.0
  0.1155009100      1.00000000000000
0 3 1 0.0 1 0
  0.5207010100      1.00000000000000
"""


def test_parse_tolerates_sulfur_d_shell_scal_typo():
    atom = parse_crystal_atom_basis(SULFUR_D_TYPO, "16_S")
    d = atom.shells[-1]
    assert d.shell_type == "D"
    # The stray ``0`` must NOT leak into the data: the real exponent is
    # recovered and the coefficient is the genuine 1.0 (not the exponent
    # mis-read as the coefficient, which was the latent bug).
    assert d.exponents == [pytest.approx(0.5207010100)]
    assert d.coefficients == [pytest.approx(1.0)]
    assert d.scale_factor == pytest.approx(1.0)


def test_parse_rev2_sulfur_d_shell_scal_typo():
    # pob-TZVP-rev2/16_S carries the same typo with a different exponent.
    rev2 = "16 1\n0 3 1 0.0 1 0\n  0.4107010100  1.0\n"
    d = parse_crystal_atom_basis(rev2, "16_S-rev2").shells[-1]
    assert d.shell_type == "D"
    assert d.exponents == [pytest.approx(0.4107010100)]
    assert d.coefficients == [pytest.approx(1.0)]


def test_parse_integer_scal_on_own_line_is_not_treated_as_typo():
    # pob-DZVP-rev2/16_S writes the d-shell header as ``0 3 1 0 1`` — a
    # legitimate five-token header with integer CHE and SCAL, the next record
    # on its own line. The typo tolerance keys on the extra token sharing the
    # header's *physical line*, so this must parse unchanged: exp=0.15.
    dzvp = "16 1\n0 3 1 0 1\n0.1500000000 1.0000000000000\n"
    d = parse_crystal_atom_basis(dzvp, "16_S-dzvp").shells[-1]
    assert d.exponents == [pytest.approx(0.15)]
    assert d.coefficients == [pytest.approx(1.0)]
    assert d.scale_factor == pytest.approx(1.0)


def test_parse_rejects_non_positive_exponent():
    # A zero/negative Gaussian exponent is physically invalid; the parser must
    # raise rather than build a bogus diffuse function.
    bad = "16 1\n0 3 1 0.0 1.0\n  0.0  1.0\n"
    with pytest.raises(ValueError, match="non-positive Gaussian exponent"):
        parse_crystal_atom_basis(bad, "<zero-exp>")


def test_parse_rejects_stray_shell_header_token():
    # An extra header token that is NOT the recoverable ``<int> <int>`` SCAL
    # typo is a genuine malformation and must be rejected loudly rather than
    # silently shifting the token stream.
    bad = "16 1\n0 3 1 0.0 1.0 x\n  0.52  1.0\n"
    with pytest.raises(ValueError, match="unexpected extra token"):
        parse_crystal_atom_basis(bad, "<bad-extra>")


_SOURCES_DIR = (
    Path(__file__).resolve().parent.parent
    / "python" / "vibeqc" / "basis_library" / "sources"
)


@pytest.mark.parametrize(
    "rel_path, d_exponent",
    [
        ("pob-TZVP/16_S", 0.5207010100),
        ("pob-TZVP-rev2/16_S", 0.4107010100),
    ],
)
def test_shipped_sulfur_source_has_valid_d_shell(rel_path, d_exponent):
    # Guards the data fix itself: the vendored sulfur source files must yield a
    # well-formed d-polarization shell (exp>0, coef=1.0) — whether the in-tree
    # file uses the corrected ``1.0`` header or, after any future re-fetch from
    # the still-typo'd upstream archive, the tolerated ``1 0`` form.
    path = _SOURCES_DIR / rel_path
    if not path.exists():
        pytest.skip(f"source file not present: {path}")
    atom = parse_crystal_atom_basis(path.read_text(), str(path))
    d = atom.shells[-1]
    assert d.shell_type == "D"
    assert d.exponents == [pytest.approx(d_exponent)]
    assert d.coefficients == [pytest.approx(1.0)]


# --------------------------------------------------------------------------
# NWChem emitter
# --------------------------------------------------------------------------


def test_emit_g94_round_trip_through_libint():
    """Parse → emit → write to a temp basis library → load via BasisSet."""
    atom_H = parse_crystal_atom_basis(POB_TZVP_H, "H")
    atom_C = parse_crystal_atom_basis(POB_TZVP_C, "C")

    # Write a temporary basis file and point libint at it via the env var.
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        basis_lib = Path(td) / "basis"
        basis_lib.mkdir()
        (basis_lib / "custom-pob.g94").write_text(emit_g94([atom_H, atom_C]))
        old = os.environ.get("LIBINT_DATA_PATH")
        os.environ["LIBINT_DATA_PATH"] = td
        try:
            mol = vq.Molecule(
                [
                    vq.Atom(1, [0, 0, 0]),
                    vq.Atom(1, [0, 0, 1.4]),
                    vq.Atom(6, [5.0, 0, 0]),
                ]
            )
            # libint lower-cases basis names when matching filenames.
            basis = vq.BasisSet(mol, "custom-pob")
            # Count basis functions per atom:
            #   H  → 3 S + 1 P = 3 + 3 = 6
            #   C  → 4 S + 3 P + 1 D = 4 + 9 + 5 = 18 (spherical d)
            # Two H + one C → 6·2 + 18 = 30.
            assert basis.nbasis == 30
        finally:
            if old is not None:
                os.environ["LIBINT_DATA_PATH"] = old
            else:
                del os.environ["LIBINT_DATA_PATH"]


# --------------------------------------------------------------------------
# Full Bredow-basis integration (uses the fetched .g94 files in the repo)
# --------------------------------------------------------------------------

_BASIS_LIBRARY = Path(__file__).parent.parent / "basis_library" / "basis"

pob_bases_available = pytest.mark.skipif(
    not any(
        (_BASIS_LIBRARY / f"{name}.g94").exists()
        for name in ("pob-tzvp", "pob-tzvp-rev2", "pob-dzvp-rev2")
    ),
    reason=(
        "Bredow .g94 files not present — run "
        "`python -m vibeqc.basis_crystal fetch` once to populate "
        "basis_library/basis/."
    ),
)


@pob_bases_available
@pytest.mark.parametrize("basis_name", ["pob-tzvp", "pob-tzvp-rev2"])
def test_pob_tzvp_h2o_rhf_runs(basis_name):
    H2O = [
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 1.43, -0.98]),
        vq.Atom(1, [0.0, -1.43, -0.98]),
    ]
    mol = vq.Molecule(H2O, 0, 1)
    basis = vq.BasisSet(mol, basis_name)
    # pob-TZVP on H2O: triple-zeta + polarization → well above minimal basis.
    assert basis.nbasis >= 25
    res = vq.run_rhf(mol, basis)
    assert res.converged
    # Published HF/pob-TZVP H2O energy ≈ −76.05 Ha (this is roughly TZ
    # quality). Anchor the test with a loose band.
    assert -76.3 < res.energy < -75.9


@pob_bases_available
def test_pob_tzvp_periodic_ks_matches_molecular_rks():
    """In the molecular limit pob-TZVP gives the same energy periodic
    as molecular, confirming nothing about the basis is tripping the
    new periodic machinery."""
    H2O = [
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 1.43, -0.98]),
        vq.Atom(1, [0.0, -1.43, -0.98]),
    ]
    sysp = vq.PeriodicSystem(3, np.eye(3) * 50.0, H2O)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "pob-tzvp")

    opts = vq.PeriodicKSOptions()
    opts.functional = "PBE"
    opts.lattice_opts.cutoff_bohr = 15.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.conv_tol_energy = 1e-10
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    res_p = vq.run_rks_periodic(sysp, basis, km, opts)

    mopts = vq.RKSOptions()
    mopts.functional = "PBE"
    mopts.conv_tol_energy = 1e-10
    res_m = vq.run_rks(vq.Molecule(H2O, 0, 1), basis, mopts)

    assert abs(res_p.energy - res_m.energy) < 1e-9
