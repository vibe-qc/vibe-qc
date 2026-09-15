"""Round-trip and integration tests for the vibe-qc basis toolkit.

Tests cover BSE import, QVF serialization (text + packaged),
G94 / ORCA / NWChem export, validation, loss reports, and
auto-detection load/save.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from vibeqc.basis_toolkit import (
    BasisRole,
    BasisSetData,
    from_bse_file,
    from_qvf_bytes,
    from_qvf_json,
    load_any,
    load_qvf,
    loss_report_g94,
    save_any,
    save_qvf,
    save_qvf_json,
    to_g94,
    to_nwchem,
    to_orca,
    to_qvf_bytes,
    to_qvf_json,
    validate_basis_dict,
)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def _load_sample(name: str) -> BasisSetData:
    return from_bse_file(SAMPLES / name)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sto3g() -> BasisSetData:
    return _load_sample("sto-3g_h2o.bse.json")


@pytest.fixture(scope="module")
def aux_ri() -> BasisSetData:
    return _load_sample("def2-svp-ri.bse.json")


@pytest.fixture(scope="module")
def cu_ecp() -> BasisSetData:
    return _load_sample("def2-svp_cu_ecp.bse.json")


# ===================================================================
# Test 1 -- Import STO-3G from BSE
# ===================================================================


def test_import_sto3g_from_bse(sto3g: BasisSetData) -> None:
    """Import sto-3g_h2o.bse.json and verify structure."""
    assert sto3g.name == "STO-3G"
    assert sto3g.role == BasisRole.ORBITAL

    assert "H" in sto3g.elements
    assert "O" in sto3g.elements

    h = sto3g.elements["H"]
    assert len(h.shells) == 1
    assert h.shells[0].angular_momentum == [0]

    o = sto3g.elements["O"]
    assert len(o.shells) == 2
    # First shell is S
    assert o.shells[0].angular_momentum == [0]
    # Second shell is SP
    assert o.shells[1].angular_momentum == [0, 1]


# ===================================================================
# Test 2 -- Round-trip through QVF JSON + to_dict/from_dict
# ===================================================================


def test_roundtrip_qvf_json(sto3g: BasisSetData) -> None:
    """bse -> to_qvf_json -> from_qvf_json -> compare."""
    json_text = to_qvf_json(sto3g)
    roundtripped = from_qvf_json(json_text)

    assert roundtripped.name == sto3g.name
    assert roundtripped.elements.keys() == sto3g.elements.keys()

    # Also verify to_dict / from_dict round-trip
    d = sto3g.to_dict()
    rt2 = BasisSetData.from_dict(d)
    assert rt2.name == sto3g.name
    assert rt2.elements.keys() == sto3g.elements.keys()


# ===================================================================
# Test 3 -- Round-trip through QVF packaged (bytes)
# ===================================================================


def test_roundtrip_qvf_packaged(sto3g: BasisSetData) -> None:
    """bse -> to_qvf_bytes -> from_qvf_bytes -> compare."""
    qvf_bytes = to_qvf_bytes(sto3g)
    roundtripped = from_qvf_bytes(qvf_bytes)

    assert roundtripped.name == sto3g.name
    assert roundtripped.elements.keys() == sto3g.elements.keys()
    # Per-element shell counts
    for sym, elem in sto3g.elements.items():
        assert sym in roundtripped.elements
        assert len(roundtripped.elements[sym].shells) == len(elem.shells)


# ===================================================================
# Test 4 -- G94 export
# ===================================================================


def test_export_g94(sto3g: BasisSetData) -> None:
    """bse -> to_g94 -> verify structure markers."""
    g94 = to_g94(sto3g)
    assert "****" in g94
    assert "H     0" in g94
    assert "O     0" in g94
    assert "SP" in g94


# ===================================================================
# Test 5 -- ORCA export
# ===================================================================


def test_export_orca(sto3g: BasisSetData) -> None:
    """bse -> to_orca -> verify block markers."""
    orca = to_orca(sto3g)
    assert "%basis" in orca
    assert "NewGTO H" in orca
    assert "NewGTO O" in orca
    assert orca.strip().endswith("end")


# ===================================================================
# Test 6 -- NWChem export
# ===================================================================


def test_export_nwchem(sto3g: BasisSetData) -> None:
    """bse -> to_nwchem -> verify block markers."""
    nw = to_nwchem(sto3g)
    assert 'basis "STO-3G"' in nw
    assert "H    S" in nw
    assert "O    S" in nw
    assert "O    P" in nw
    assert "end" in nw


# ===================================================================
# Test 7 -- G94 loss report
# ===================================================================


def test_loss_report_g94_not_lossless(sto3g: BasisSetData) -> None:
    """loss_report_g94 returns a non-lossless report with 7 dropped fields."""
    report = loss_report_g94(sto3g)
    assert report.is_lossless is False
    assert len(report.fields_dropped) == 7


# ===================================================================
# Test 8 -- Validate a valid dict
# ===================================================================


def test_validate_valid(sto3g: BasisSetData) -> None:
    """validate_basis_dict on a valid to_dict() result returns <= 1 error."""
    d = sto3g.to_dict()
    errors = validate_basis_dict(d)
    # When jsonschema is available: 0 errors.
    # When it's not: 1 warning about the lightweight fallback.
    assert len(errors) <= 1


# ===================================================================
# Test 9 -- Validate an invalid dict
# ===================================================================


def test_validate_invalid() -> None:
    """validate_basis_dict on invalid dict returns non-empty errors."""
    # Missing required 'name', bad role value
    bad = {
        "schema_version": "1.0.0",
        "role": "not_a_real_role",
        "elements": {},
    }
    errors = validate_basis_dict(bad)
    assert len(errors) > 0


# ===================================================================
# Test 10 -- Import auxiliary basis
# ===================================================================


def test_import_aux_basis(aux_ri: BasisSetData) -> None:
    """Import def2-svp-ri.bse.json; verify role + elements."""
    assert aux_ri.role == BasisRole.AUXILIARY
    assert "C" in aux_ri.elements
    assert "H" in aux_ri.elements


# ===================================================================
# Test 11 -- Import ECP basis
# ===================================================================


def test_import_ecp_basis(cu_ecp: BasisSetData) -> None:
    """Import def2-svp_cu_ecp.bse.json; verify Cu + provenance."""
    assert cu_ecp.name == "def2-SVP"
    assert "Cu" in cu_ecp.elements
    # Provenance is always set by the BSE importer
    assert cu_ecp.provenance is not None
    assert cu_ecp.provenance.origin == "bse"


# ===================================================================
# Test 12 -- Save and load .qvf.json (tempfile)
# ===================================================================


def test_save_and_load_qvf_json(sto3g: BasisSetData) -> None:
    """save_qvf_json -> load via from_qvf_json -> round-trip."""
    with tempfile.NamedTemporaryFile(suffix=".qvf.json", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        save_qvf_json(sto3g, tmp_path)
        loaded = from_qvf_json(tmp_path)
        assert loaded.name == sto3g.name
        assert loaded.elements.keys() == sto3g.elements.keys()
    finally:
        tmp_path.unlink(missing_ok=True)


# ===================================================================
# Test 13 -- Save and load .qvf (tempfile)
# ===================================================================


def test_save_and_load_qvf(sto3g: BasisSetData) -> None:
    """save_qvf -> load_qvf -> round-trip."""
    with tempfile.NamedTemporaryFile(suffix=".qvf", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        save_qvf(sto3g, tmp_path)
        loaded = load_qvf(tmp_path)
        assert loaded.name == sto3g.name
        assert loaded.elements.keys() == sto3g.elements.keys()
    finally:
        tmp_path.unlink(missing_ok=True)


# ===================================================================
# Test 14 -- Auto-detect load/save
# ===================================================================


def test_auto_detect_load(sto3g: BasisSetData) -> None:
    """save_any(.qvf.json) -> load_any; save_any(.qvf) -> load_any."""

    # --- .qvf.json path ---
    with tempfile.NamedTemporaryFile(suffix=".qvf.json", delete=False) as f:
        tmp_json = Path(f.name)
    try:
        save_any(sto3g, tmp_json)
        loaded_json = load_any(tmp_json)
        assert loaded_json.name == sto3g.name
    finally:
        tmp_json.unlink(missing_ok=True)

    # --- .qvf path ---
    with tempfile.NamedTemporaryFile(suffix=".qvf", delete=False) as f:
        tmp_qvf = Path(f.name)
    try:
        save_any(sto3g, tmp_qvf)
        loaded_qvf = load_any(tmp_qvf)
        assert loaded_qvf.name == sto3g.name
    finally:
        tmp_qvf.unlink(missing_ok=True)


# ===================================================================
# Test 15 -- G94 round-trip (export then import)
# ===================================================================


def test_g94_roundtrip(sto3g: BasisSetData) -> None:
    """Export to G94, import back, verify same elements and shells."""
    from vibeqc.basis_toolkit import from_g94, to_g94

    g94_text = to_g94(sto3g)
    imported = from_g94(g94_text)

    assert imported.name == sto3g.name
    assert set(imported.elements.keys()) == set(sto3g.elements.keys())

    # Per-element shell counts must match.
    for sym in sto3g.elements:
        orig = sto3g.elements[sym]
        imp = imported.elements[sym]
        assert len(imp.shells) == len(orig.shells)
        for os_, ns in zip(orig.shells, imp.shells):
            assert ns.angular_momentum == os_.angular_momentum
            assert len(ns.primitives) == len(os_.primitives)
            # Exponents match to 6 decimal places (G94 format precision).
            assert abs(ns.primitives[0].exponent - os_.primitives[0].exponent) < 1e-6


def test_g94_sp_shell_roundtrip_preserves_coefficients() -> None:
    """SP (general-L) shells round-trip through G94 without scrambling the
    separate s- and p-contraction coefficients.

    Regression: ``from_g94`` once stored SP primitives interleaved
    ``(s0, p0, s1, p1, ...)`` while ``to_g94`` (and orca/nwchem) read them
    blocked ``(s0, s1, ..., p0, p1, ...)``. The mismatch duplicated one
    exponent, dropped another, and scrambled the s/p coefficients on every
    Pople basis (STO-3G, 6-31G, ...) that uses SP shells. The old
    ``test_g94_roundtrip`` only checked ``primitives[0].exponent`` and so
    missed it; this test pins every primitive.
    """
    from vibeqc.basis_toolkit import from_g94

    g94 = (
        "****\n"
        "H     0\n"
        "SP   2   1.00\n"
        "      5.0000000   0.1000000   0.7000000\n"
        "      1.0000000   0.2000000   0.8000000\n"
        "****\n"
    )
    sh = from_g94(g94).elements["H"].shells[0]
    assert sh.angular_momentum == [0, 1]
    n = len(sh.primitives) // 2
    assert n == 2
    exps = [5.0, 1.0]
    s_co = [0.1, 0.2]
    p_co = [0.7, 0.8]
    for k in range(n):
        # s-block then p-block, exponents repeated.
        assert sh.primitives[k].exponent == pytest.approx(exps[k], abs=1e-7)
        assert sh.primitives[k].coefficient == pytest.approx(s_co[k], abs=1e-7)
        assert sh.primitives[k + n].exponent == pytest.approx(exps[k], abs=1e-7)
        assert sh.primitives[k + n].coefficient == pytest.approx(p_co[k], abs=1e-7)

    # Full export -> import round-trip preserves every (exponent, coefficient).
    rt = from_g94(to_g94(from_g94(g94))).elements["H"].shells[0]
    assert [(p.exponent, p.coefficient) for p in rt.primitives] == [
        pytest.approx((p.exponent, p.coefficient)) for p in sh.primitives
    ]


def test_g94_roundtrip_file(sto3g: BasisSetData) -> None:
    """Export to G94 file, import back via file path."""
    from vibeqc.basis_toolkit import from_g94_file, to_g94_file

    with tempfile.NamedTemporaryFile(suffix=".g94", delete=False) as f:
        tmp_path = Path(f.name)
    try:
        to_g94_file(sto3g, tmp_path)
        imported = from_g94_file(tmp_path)
        assert imported.name == sto3g.name
        assert set(imported.elements.keys()) == set(sto3g.elements.keys())
    finally:
        tmp_path.unlink(missing_ok=True)


def test_g94_import_parses_role(sto3g: BasisSetData) -> None:
    """G94 import extracts role and family from ! Role: comment."""
    from vibeqc.basis_toolkit import from_g94, to_g94

    g94_text = to_g94(sto3g)
    imported = from_g94(g94_text)
    # Our exporter writes "! Role: orbital pople"
    assert imported.role.value == "orbital"
    assert imported.basis_family == "pople"


# ===================================================================
# Test 16 -- Direct C++ BasisSet bridge
# ===================================================================


def test_bridge_to_libint(sto3g: BasisSetData) -> None:
    """Convert BasisSetData -> C++ BasisSet via bridge, verify shells."""
    from vibeqc import Atom, Molecule
    from vibeqc.basis_toolkit import to_libint_basis

    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.98]),
            Atom(1, [0.0, -1.43, -0.98]),
        ],
        charge=0,
        multiplicity=1,
    )

    cpp_basis = to_libint_basis(sto3g, mol)
    # STO-3G on water: O (3 shells: S, S, P) + 2x H (1 shell each: S) = 5 shells
    assert cpp_basis.nshells == 5
    assert cpp_basis.nbasis == 7

    shells = cpp_basis.shells()
    l_values = [s.l for s in shells]
    assert l_values == [0, 0, 1, 0, 0]  # O: S,S,P; H: S; H: S


def test_bridge_roundtrip(sto3g: BasisSetData) -> None:
    """BasisSetData -> C++ BasisSet -> BasisSetData round-trip.

    SP shells are expanded into separate S and P libint shells, so the
    round-tripped BasisSetData has more shells than the original.
    The total basis-function count and per-element presence are preserved.
    """
    from vibeqc import Atom, Molecule
    from vibeqc.basis_toolkit import from_libint_basis, to_libint_basis

    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.98]),
            Atom(1, [0.0, -1.43, -0.98]),
        ],
        charge=0,
        multiplicity=1,
    )

    cpp_basis = to_libint_basis(sto3g, mol)
    roundtripped = from_libint_basis(cpp_basis, molecule=mol)

    assert roundtripped.name == sto3g.name
    assert set(roundtripped.elements.keys()) == set(sto3g.elements.keys())

    # Round-trip through the bridge a second time: data2 -> C++ -> basis
    # functions must match the original C++ basis.
    cpp2 = to_libint_basis(roundtripped, mol)
    assert cpp2.nbasis == cpp_basis.nbasis
    assert cpp2.nshells == cpp_basis.nshells


def test_bridge_aux_basis(aux_ri: BasisSetData) -> None:
    """Bridge works for auxiliary basis sets."""
    from vibeqc import Atom, Molecule
    from vibeqc.basis_toolkit import to_libint_basis

    # CH4-like for C + 4H
    mol = Molecule(
        [
            Atom(6, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 0.0, 1.09]),
            Atom(1, [0.0, 1.03, -0.36]),
            Atom(1, [0.89, -0.51, -0.36]),
            Atom(1, [-0.89, -0.51, -0.36]),
        ],
        charge=0,
        multiplicity=1,
    )

    cpp_basis = to_libint_basis(aux_ri, mol)
    assert cpp_basis.nshells > 0
    assert cpp_basis.nbasis > 0


def test_bridge_g94_parity(sto3g: BasisSetData) -> None:
    """C++ BasisSet via bridge matches C++ BasisSet loaded from G94 name."""
    from vibeqc import Atom, Molecule
    from vibeqc._vibeqc_core import BasisSet as CppBasisSet
    from vibeqc.basis_toolkit import to_libint_basis

    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.98]),
            Atom(1, [0.0, -1.43, -0.98]),
        ],
        charge=0,
        multiplicity=1,
    )

    bridge_basis = to_libint_basis(sto3g, mol)
    name_basis = CppBasisSet(mol, "sto-3g")

    # Same shell count and basis function count.
    assert bridge_basis.nshells == name_basis.nshells
    assert bridge_basis.nbasis == name_basis.nbasis

    # Same angular momentum sequence.
    bridge_l = [s.l for s in bridge_basis.shells()]
    name_l = [s.l for s in name_basis.shells()]
    assert bridge_l == name_l


# ===================================================================
# Test 17 -- Cross-format round-trips: BSE -> G94 -> (import) -> ORCA -> (import)
# ===================================================================


def test_cross_format_roundtrip(sto3g: BasisSetData) -> None:
    """BSE -> G94 text -> import G94 -> ORCA text -> import ORCA.

    Verifies that the data survives a chain of different format exports
    and imports, preserving element keys and shell structure.
    """
    from vibeqc.basis_toolkit import (
        from_g94,
        from_orca,
        to_g94,
        to_orca,
    )

    # Step 1: BSE -> G94 -> import
    g94_text = to_g94(sto3g)
    from_g94_data = from_g94(g94_text)
    assert from_g94_data.name == sto3g.name
    assert set(from_g94_data.elements.keys()) == set(sto3g.elements.keys())

    # Step 2: Imported G94 -> ORCA -> import
    orca_text = to_orca(from_g94_data)
    from_orca_data = from_orca(orca_text)
    assert set(from_orca_data.elements.keys()) == set(sto3g.elements.keys())

    # Per-element shell counts: ORCA splits SP shells, so O may differ.
    for sym in sto3g.elements:
        if sym == "O":
            # SP shell becomes S+P in ORCA round-trip
            assert len(from_orca_data.elements[sym].shells) >= len(
                sto3g.elements[sym].shells
            )
        else:
            assert len(from_orca_data.elements[sym].shells) == len(
                sto3g.elements[sym].shells
            )


def test_orca_roundtrip(sto3g: BasisSetData) -> None:
    """Export ORCA, import back.  SP shells are split into S+P."""
    from vibeqc.basis_toolkit import from_orca, to_orca

    orca_text = to_orca(sto3g)
    imported = from_orca(orca_text)
    assert set(imported.elements.keys()) == set(sto3g.elements.keys())
    # All angular momenta from the original must be present.
    for sym in sto3g.elements:
        orig_am = [tuple(s.angular_momentum) for s in sto3g.elements[sym].shells]
        imp_am = [tuple(s.angular_momentum) for s in imported.elements[sym].shells]
        for am in orig_am:
            # SP [0,1] becomes two separate S+P: [0] and [1]
            if am == (0, 1):
                assert (0,) in imp_am and (1,) in imp_am, (
                    f"Missing S or P for SP shell on {sym}"
                )
            else:
                assert am in imp_am, f"Missing shell {am} for {sym}"


def test_nwchem_roundtrip(sto3g: BasisSetData) -> None:
    """Export NWChem, import back.  SP shells split into S+P."""
    from vibeqc.basis_toolkit import from_nwchem, to_nwchem

    nw_text = to_nwchem(sto3g)
    imported = from_nwchem(nw_text)
    assert set(imported.elements.keys()) == set(sto3g.elements.keys())
    # All angular momenta from the original must be present.
    for sym in sto3g.elements:
        orig_am = [tuple(s.angular_momentum) for s in sto3g.elements[sym].shells]
        imp_am = [tuple(s.angular_momentum) for s in imported.elements[sym].shells]
        for am in orig_am:
            # SP [0,1] becomes two separate S+P: [0] and [1]
            if am == (0, 1):
                assert (0,) in imp_am and (1,) in imp_am, (
                    f"Missing S or P for SP shell on {sym}"
                )
            else:
                assert am in imp_am, f"Missing shell {am} for {sym}"


def test_nwchem_parses_name() -> None:
    """NWChem import extracts basis name from header."""
    from vibeqc.basis_toolkit import from_nwchem

    text = 'basis "my-test-basis" spherical\n  H    S\n    3.425   0.1543\nend'
    imported = from_nwchem(text)
    assert imported.name == "my-test-basis"


def test_orca_import_from_text() -> None:
    """ORCA import from hand-written text."""
    from vibeqc.basis_toolkit import from_orca

    text = """%basis
  NewGTO H
    S  1
    0  3.425  0.1543
  end
end"""
    imported = from_orca(text)
    assert "H" in imported.elements
    assert len(imported.elements["H"].shells) == 1


# ===================================================================
# Test 18 -- Library writer
# ===================================================================


def test_write_to_library_g94(sto3g: BasisSetData) -> None:
    """Write G94 to library, read back, verify."""
    import re

    from vibeqc.basis_toolkit import (
        read_from_library,
        write_to_library,
    )

    path = write_to_library(sto3g, fmt="g94", subdir="custom")
    safe_name = re.sub(r"[^a-z0-9-]", "-", sto3g.name.lower()).strip("-")
    try:
        assert path.exists()
        assert path.suffix == ".g94"
        # Read back -- name is lowercased by library writer.
        reloaded = read_from_library(safe_name, fmt="g94", subdir="custom")
        assert reloaded.name.lower() == sto3g.name.lower()
        assert set(reloaded.elements.keys()) == set(sto3g.elements.keys())
    finally:
        path.unlink(missing_ok=True)


def test_write_to_library_qvf(sto3g: BasisSetData) -> None:
    """Write QVF JSON to library, read back."""
    import re

    from vibeqc.basis_toolkit import (
        read_from_library,
        write_to_library,
    )

    path = write_to_library(sto3g, fmt="qvf_json", subdir="custom")
    safe_name = re.sub(r"[^a-z0-9-]", "-", sto3g.name.lower()).strip("-")
    try:
        assert path.name.endswith(".qvf.json")
        reloaded = read_from_library(safe_name, fmt="qvf_json", subdir="custom")
        assert reloaded.name.lower() == sto3g.name.lower()
    finally:
        path.unlink(missing_ok=True)
