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
# The SCAL rule: CRYSTAL multiplies exponents by SCAL**2
# --------------------------------------------------------------------------
#
# The CRYSTAL23 manual (section 2.2.1, p. 25) calls SCAL a "scale factor"
# without saying how it enters. CRYSTAL23 1.0.1 settled it: RHF on H2 (H at
# z = +-0.37 Angstrom, MOLECULE, TOLINTEG 12 12 12 12 30, TOLDEE 11) with the
# shells below. Each energy is CRYSTAL's "SCF ENDED" value.
#
#   s shell (zeta=1 STO-3G exponents a, SCAL 1.24) + p (0.8, SCAL 1.3)
#       (a, SCAL s)          -1.1204936187730
#       (a*s**2, SCAL 1)     -1.1204936187730   identical; CRYSTAL prints a*s**2
#       (a*s**2, SCAL 0)     -1.1204936187730   SCAL 0 is "as written" here
#       (a*s, SCAL 1)        -1.1196883207051   linear scaling is 0.8 mEh off
#   STO-3G s (SCAL 1) + SP (0.5, SCAL 1.1, coefficients 1 1)
#       (0.5, SCAL 1.1)      -1.1198633063153
#       (0.605, SCAL 1)      -1.1198633063153
#       (0.55, SCAL 1)       -1.1193178600547
#
# The s exponents with SCAL 1.24 also reproduce CRYSTAL's built-in Pople
# STO-3G (ITYB=1, SCAL 0.) to 1e-10 Eh: 1.24 is H's standard Pople zeta.

_STO3G_H_EXP = (3.42525091, 0.62391373, 0.16885540)
_STO3G_H_COEF = (0.15432897, 0.53532814, 0.44463454)
_ZETA = 1.24
_ZETA1_EXP = tuple(e / _ZETA**2 for e in _STO3G_H_EXP)

SCAL_H = (
    "1 2\n"
    "0 0 3 1.0 1.24\n"
    + "".join(f"{e!r} {c!r}\n" for e, c in zip(_ZETA1_EXP, _STO3G_H_COEF))
    + "0 2 1 0.0 1.3\n"
    "0.8 1.0\n"
)
SCAL_SP_H = (
    "1 2\n"
    "0 0 3 1.0 1.0\n"
    + "".join(f"{e!r} {c!r}\n" for e, c in zip(_STO3G_H_EXP, _STO3G_H_COEF))
    + "0 1 1 0.0 1.1\n"
    "0.5 1.0 1.0\n"
)
E_CRYSTAL23 = {"scal": -1.1204936187730, "sp": -1.1198633063153}
_BOHR_ANGSTROM = 0.529177210903


def _vibe_basis_path():
    import sys

    src = str(Path(__file__).resolve().parents[1] / "vibe-basis" / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def test_crystal_exponent_factor_is_scal_squared():
    from vibeqc.basis_crystal import crystal_exponent_factor

    assert crystal_exponent_factor(1.0) == 1.0
    assert crystal_exponent_factor(0.0) == 1.0
    assert crystal_exponent_factor(1.3) == pytest.approx(1.69)
    assert crystal_exponent_factor(1.24) == pytest.approx(1.5376)
    assert crystal_exponent_factor(0.5) == pytest.approx(0.25)


def test_vibe_basis_carries_the_same_scal_rule():
    """vibe-basis cannot import vibe-qc, so it has a copy; the two must agree."""
    _vibe_basis_path()
    from vibe_basis.io.export import crystal_exponent_factor as vb_factor
    from vibeqc.basis_crystal import crystal_exponent_factor

    for scal in (0.0, 1.0, 1.3, 1.24, 0.5, 2.0, 0.93):
        assert vb_factor(scal) == crystal_exponent_factor(scal)


def test_parse_applies_scal_squared_once_and_returns_unit_scale():
    atom = parse_crystal_atom_basis(SCAL_H, "H")
    s, p = atom.shells
    assert s.exponents == pytest.approx(list(_STO3G_H_EXP), rel=1e-12)
    assert p.exponents == pytest.approx([0.8 * 1.3**2], rel=1e-12)
    # Applied on the way in, so nothing downstream applies it again.
    assert s.scale_factor == 1.0 and p.scale_factor == 1.0
    # SCAL 0.0 on a general basis is "as written", not a zero factor.
    as_written = parse_crystal_atom_basis("1 1\n0 2 1 0.0 0.0\n0.8 1.0\n", "H")
    assert as_written.shells[0].exponents == [0.8]
    assert as_written.shells[0].scale_factor == 1.0


def test_emitters_apply_the_scal_of_a_hand_built_shell():
    """A shell built with raw exponents and a SCAL comes out scaled by SCAL**2."""
    from vibeqc.basis_crystal import emit_crystal

    atom = CrystalAtomBasis(Z=1, has_ecp=False, shells=[
        CrystalShell("P", 0.0, 1.3, exponents=[0.8], coefficients=[1.0]),
    ])
    from vibeqc.basis_toolkit import from_g94

    g94 = emit_g94([atom])
    (prim,) = from_g94(g94).elements["H"].shells[0].primitives
    assert prim.exponent == pytest.approx(0.8 * 1.69, rel=1e-14)
    deck = emit_crystal([atom])
    assert "0 2   1  0.0 0.0" in deck
    assert parse_crystal_atom_basis(deck, "H").shells[0].exponents == [
        pytest.approx(0.8 * 1.69, rel=1e-10)
    ]


def test_scal_round_trip_crystal_text_and_g94():
    """SCAL != 1 text -> parse -> emit_crystal -> parse keeps the exponents,
    and every route to .g94 lands on the same numbers."""
    from vibeqc.basis_crystal import emit_crystal
    from vibeqc.basis_toolkit import from_crystal_atom, from_g94, to_g94

    for text in (SCAL_H, SCAL_SP_H):
        first = parse_crystal_atom_basis(text, "H")
        again = parse_crystal_atom_basis(emit_crystal([first]), "H")
        for a, b in zip(first.shells, again.shells):
            assert b.exponents == pytest.approx(a.exponents, rel=1e-10)
            assert b.coefficients == pytest.approx(a.coefficients, rel=1e-10)
            assert b.coefficients_p == pytest.approx(a.coefficients_p, rel=1e-10)

        def exponents(g94: str) -> list[float]:
            data = from_g94(g94)
            return [p.exponent for sh in data.elements["H"].shells for p in sh.primitives]

        via_convert = exponents(emit_g94([first]))
        via_import = exponents(to_g94(from_crystal_atom(first, name="scal")))
        assert via_import == pytest.approx(via_convert, rel=1e-9)


def _h2_rhf_energy(g94: str, name: str, tmp_path, monkeypatch) -> float:
    basis_dir = tmp_path / name / "basis"
    basis_dir.mkdir(parents=True)
    (basis_dir / f"{name}.g94").write_text(g94)
    monkeypatch.setenv("LIBINT_DATA_PATH", str(basis_dir.parent))
    z = 0.37 / _BOHR_ANGSTROM
    mol = vq.Molecule([vq.Atom(1, [0, 0, z]), vq.Atom(1, [0, 0, -z])], 0, 1)
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-12
    res = vq.run_rhf(mol, vq.BasisSet(mol, name), opts)
    assert res.converged
    return res.energy


def _vibe_basis_g94(text: str) -> str:
    """Raw exponents and SCAL straight from CRYSTAL text into vibe-basis."""
    _vibe_basis_path()
    from vibe_basis.io.export import basis_to_g94
    from vibe_basis.topology import BasisTopology, Shell

    rows = [line.split() for line in text.splitlines()[1:] if line.strip()]
    shells, i = [], 0
    while i < len(rows):
        _, lat, ng, che, scal = rows[i]
        prims = rows[i + 1:i + 1 + int(ng)]
        shells.append(Shell(
            {"0": "S", "1": "SP", "2": "P"}[lat],
            tuple(float(p[0]) for p in prims),
            tuple(float(p[1]) for p in prims),
            tuple(float(p[2]) for p in prims) if lat == "1" else (),
            occupancy=float(che), scale_factor=float(scal),
        ))
        i += 1 + int(ng)
    return basis_to_g94(BasisTopology({"H": tuple(shells)}), name="scal")


@pytest.mark.parametrize("case", ["scal", "sp"])
def test_scal_energies_match_crystal23(case, tmp_path, monkeypatch):
    """The converter, the import tool's bridge and vibe-basis's exporter all
    reproduce CRYSTAL23's energy for a SCAL != 1 basis. Before the rule was
    shared, the import bridge was 34.5 mEh off (SCAL**3) and vibe-basis
    0.8 mEh off (linear) on the first case."""
    from vibeqc.basis_toolkit import from_crystal_atom, to_g94

    text = {"scal": SCAL_H, "sp": SCAL_SP_H}[case]
    atom = parse_crystal_atom_basis(text, "H")
    routes = {
        "convert": emit_g94([atom]),
        "import": to_g94(from_crystal_atom(atom, name=f"scal-import-{case}")),
        "vibebasis": _vibe_basis_g94(text),
    }
    for route, g94 in routes.items():
        energy = _h2_rhf_energy(g94, f"scal-{route}-{case}", tmp_path, monkeypatch)
        assert energy == pytest.approx(E_CRYSTAL23[case], abs=1e-8), route


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
        "`python -m vibeqc.basis_crystal fetch --dest basis_library/basis` "
        "once to populate basis_library/basis/."
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


# --------------------------------------------------------------------------
# Bredow fetcher: which archives each basis pulls (#289)
# --------------------------------------------------------------------------


def _one_record_tarball() -> bytes:
    """A tar.gz carrying one parseable per-element record (H)."""
    import io
    import tarfile

    body = POB_TZVP_H.encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo("archive/01_H")
        info.size = len(body)
        tf.addfile(info, io.BytesIO(body))
    return buf.getvalue()


def test_pob_tzvp_fetch_never_downloads_a_rev2_archive(monkeypatch, tmp_path):
    """A pob-TZVP fetch pulled the rev2 La-Lu archive and offered its
    lanthanide records under the pob-TZVP name (#289). pob-TZVP has no
    lanthanides; its publications stop at I."""
    from vibeqc import basis_crystal

    urls: list[str] = []
    payload = _one_record_tarball()

    def fake_download(url, *args, **kwargs):
        urls.append(url)
        return payload

    monkeypatch.setattr(basis_crystal, "_download", fake_download)
    basis_crystal.fetch_bredow_basis_sets(
        tmp_path, names=["pob-tzvp"], verbose=False,
    )
    assert urls, "the fetcher downloaded nothing"
    assert urls[0] == basis_crystal._BREDOW_ARCHIVES["pob-tzvp"][0]
    rev2 = [u for u in urls if "rev2" in u.lower()]
    assert not rev2, rev2


def test_no_rev2_ecp_archive_pairs_with_a_non_rev2_basis():
    """The pairing table must not hand one pob generation's heavy records
    to another under the wrong name (#289)."""
    from vibeqc.basis_crystal import _BREDOW_AE_TO_ECP_KEYS

    wrong = [
        (ae, ek)
        for ae, eks in _BREDOW_AE_TO_ECP_KEYS.items()
        for ek in eks
        if "rev2" in ek and "rev2" not in ae
    ]
    assert not wrong, wrong


# --------------------------------------------------------------------------
# Bredow fetcher: no implicit destination (#393)
# --------------------------------------------------------------------------


def _package_dir_in(monkeypatch, tmp_path):
    """Point the fetcher's idea of its own package directory into tmp_path.

    Until #393 an omitted ``dest_dir`` meant ``<package>/basis_library/custom``,
    and the ECP tests and the runtime ECP fallback rewrote the tracked
    ``pob-tzvp-rev2.g94`` of whatever tree they ran in. Redirecting the module
    path keeps a failing run of these tests out of the real package tree.
    """
    from vibeqc import basis_crystal

    package = tmp_path / "package" / "vibeqc"
    package.mkdir(parents=True)
    monkeypatch.setattr(basis_crystal, "__file__", str(package / "basis_crystal.py"))
    monkeypatch.setattr(
        basis_crystal, "_download", lambda url, *args, **kwargs: _one_record_tarball()
    )
    return basis_crystal


def _files_under(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def test_fetch_without_dest_dir_parses_and_writes_nothing(monkeypatch, tmp_path):
    basis_crystal = _package_dir_in(monkeypatch, tmp_path)

    paths, atoms = basis_crystal.fetch_bredow_basis_sets(
        names=["pob-tzvp"], verbose=False, return_atoms=True,
    )

    assert paths == {}
    assert [atom.element_symbol for atom in atoms["pob-tzvp"]] == ["H"]
    assert _files_under(tmp_path) == []


def test_fetch_without_dest_dir_or_records_is_refused(monkeypatch, tmp_path):
    basis_crystal = _package_dir_in(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="dest_dir"):
        basis_crystal.fetch_bredow_basis_sets(names=["pob-tzvp"], verbose=False)
    assert _files_under(tmp_path) == []


def test_fetch_writes_into_an_explicit_dest_dir(monkeypatch, tmp_path):
    basis_crystal = _package_dir_in(monkeypatch, tmp_path)
    dest = tmp_path / "out"

    paths = basis_crystal.fetch_bredow_basis_sets(dest, names=["pob-tzvp"], verbose=False)

    assert paths == {"pob-tzvp": dest / "pob-tzvp.g94"}
    assert _files_under(tmp_path) == ["out/pob-tzvp.g94"]


def test_fetch_cli_needs_dest(monkeypatch, tmp_path, capsys):
    basis_crystal = _package_dir_in(monkeypatch, tmp_path)

    assert basis_crystal._main(["fetch", "pob-tzvp"]) == 2
    assert "--dest" in capsys.readouterr().out
    assert _files_under(tmp_path) == []

    dest = tmp_path / "out"
    assert basis_crystal._main(["fetch", "--dest", str(dest), "pob-tzvp"]) == 0
    assert _files_under(tmp_path) == ["out/pob-tzvp.g94"]


def test_plone_listing_follows_pages_and_element_labels(monkeypatch):
    from vibeqc import basis_crystal as bc
    root = "https://www.chemie.uni-bonn.de/bredow/de/software/pob-dzvp-rev2"
    page2 = root + "?b_start:int=20"
    si = root + "/copy9_of_04_be"
    cr = root + "/24_cr"
    pages = {
        page2: (f'<a href="{cr}">24_Cr</a><a href="{page2}">2</a>'
                f'<a href="{root}">1</a><a href="{si}">14_Si</a>').encode(),
        si: b"<p>14 1<br />0 0 1 2 1<br />1.0 1.0</p>",
        cr: b"<p>24 1<br />0 0 1 2 1<br />2.0 1.0</p>",
    }
    requested = []
    def download(url):
        requested.append(url)
        return pages[url]
    monkeypatch.setattr(bc, "_download", download)
    first = (f'<a href="{si}"><strong>14_Si</strong></a>'
             f'<a href="{page2}">Next 20 items</a>'
             '<a href="https://example.org/24_cr">24_Cr</a>'
             f'<a href="{root}-other/24_cr">24_Cr</a>').encode()
    records = list(bc._iter_plone_listing_members(first, root))
    assert [name for name, _ in records] == ["14_Si", "24_Cr"]
    assert requested.count(si) == 1
    assert requested.count(page2) == 1


def test_plone_listing_refuses_mislabeled_element(monkeypatch):
    from vibeqc import basis_crystal as bc
    root = "https://www.chemie.uni-bonn.de/bredow/de/software/pob-dzvp-rev2"
    monkeypatch.setattr(bc, "_download", lambda url: b"<p>4 1<br />0 0 1 2 1<br />1 1</p>")
    with pytest.raises(ValueError, match="contains element Z=4"):
        list(bc._iter_plone_listing_members(
            f'<a href="{root}/copy9_of_04_be">14_Si</a>'.encode(), root))
