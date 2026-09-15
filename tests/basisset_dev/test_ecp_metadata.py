"""Tests for :mod:`vibeqc.ecp_metadata` — ECP sidecar routing.

Exercises both the pure-Python helpers (parsing, library resolution)
and the end-to-end SCF round-trips with XML-library and inline-primitive
ECP data.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

PKG_PARENT = Path(__file__).resolve().parents[2] / "python"
if str(PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(PKG_PARENT))

# Drop a leftover namespace shim from sister tests; keep an already-loaded real
# package intact so every collected module shares one set of Python classes.
_loaded_vibeqc = sys.modules.get("vibeqc")
if _loaded_vibeqc is None or getattr(_loaded_vibeqc, "__file__", None) is None:
    for _mod_name in list(sys.modules):
        if _mod_name == "vibeqc" or _mod_name.startswith("vibeqc."):
            # Keep vibeqc._vibeqc_core* registered: the single-phase-init C
            # extension never re-runs PyInit on re-import, so deleting those
            # entries would strip the pybind11 def_submodule registrations
            # (...semiempirical.{nddo,xtb,indo}) for the rest of the pytest
            # process and break every later dotted import of them.
            if _mod_name == "vibeqc._vibeqc_core" or _mod_name.startswith(
                "vibeqc._vibeqc_core."
            ):
                continue
            del sys.modules[_mod_name]
try:
    import vibeqc as vq  # type: ignore[import-not-found]
except Exception as exc:  # noqa: BLE001
    pytest.skip(
        f"vibeqc not importable: {exc!r}; build with `pip install -e .`",
        allow_module_level=True,
    )


# ---------- Sidecar parsing -------------------------------------------------


def test_real_package_identity_survives_sister_module_collection():
    """Later ECP test collection must not reload the real package."""
    assert sys.modules["vibeqc"] is vq


def test_parse_sidecar_lanl2dz_has_expected_count(tmp_path):
    """LANL2DZ ships an ECP for everything from Na onwards. The
    bundled sidecar should parse cleanly and yield ≥30 elements.
    """
    sp = vq.sidecar_path_for("lanl2dz")
    assert sp is not None, "lanl2dz.ecp sidecar should ship in the bundle"
    headers = vq.parse_sidecar_path(sp)
    assert len(headers) >= 30
    # Sodium is one of the first entries.
    na = next(h for h in headers if h.symbol == "Na")
    assert na.Z == 11
    assert na.ncore == 10
    assert na.lmax == 2


def test_parse_sidecar_synthetic(tmp_path):
    """Round-trip a tiny synthetic sidecar end-to-end, with both
    all-caps (Pople-era) and title-case (modern) symbols."""
    payload = textwrap.dedent("""
        ! synthetic sidecar
        NA-ECP    2    10
        Rb-ECP    4    28
        ! comment line
        Cs-ECP    4    46
    """).strip() + "\n"
    p = tmp_path / "fake.ecp"
    p.write_text(payload)
    headers = vq.parse_sidecar_path(p)
    assert [h.symbol for h in headers] == ["Na", "Rb", "Cs"]
    assert [h.ncore for h in headers] == [10, 28, 46]


def test_parse_sidecar_rejects_unknown_element(tmp_path):
    p = tmp_path / "fake.ecp"
    p.write_text("Xx-ECP    2    10\n")
    with pytest.raises(ValueError, match="unknown element"):
        vq.parse_sidecar_path(p)


# ---------- Library-name resolution ----------------------------------------


def test_library_for_lanl_family_returns_lanl2dz():
    """Every basis whose name starts with the LANL prefix routes to
    the single bundled ``lanl2dz.xml``, regardless of ncore."""
    for name in ("lanl2dz", "lanl2dzdp", "lanl2tz", "lanl08",
                 "lanl08(d)", "lanl08(f)"):
        for ncore in (10, 18, 28, 46, 60, 78):
            assert vq.library_for(name, ncore) == "lanl2dz", \
                f"{name} ncore={ncore} should map to lanl2dz.xml"


def test_library_for_dhf_routes_via_ncore():
    """dhf-tzvp (and friends) aren't a single libecpint XML —
    each row picks the matching Stuttgart-Köln MDF library
    (ecp10mdf, ecp28mdf, ecp46mdf, ...)."""
    assert vq.library_for("dhf-tzvp", 28) == "ecp28mdf"
    assert vq.library_for("dhf-tzvp", 46) == "ecp46mdf"
    assert vq.library_for("dhf-tzvp", 60) == "ecp60mdf"
    assert vq.library_for("dhf-tzvp", 78) == "ecp78mdf"


def test_library_for_vdzp_nonstandard_returns_none():
    """vDZP uses non-standard ncore-2 / ncore-3 customs. No bundled
    XML covers them; return None so SCF wrappers attach inline ECPs."""
    assert vq.library_for("vdzp", 2) is None
    assert vq.library_for("vdzp", 3) is None


# ---------- auto_ecp_centers — happy paths ---------------------------------


def test_auto_ecp_si_lanl2dz_single_centre():
    m = vq.Molecule([vq.Atom(14, [0.0, 0.0, 0.0])], multiplicity=3)
    centres, lib = vq.auto_ecp_centers(m, "lanl2dz")
    assert lib == "lanl2dz"
    assert len(centres) == 1
    assert centres[0].Z == 14
    assert list(centres[0].xyz) == [0.0, 0.0, 0.0]


def test_auto_ecp_pt_lanl2dz_single_centre():
    m = vq.Molecule([vq.Atom(78, [0.0, 0.0, 0.0])], multiplicity=3)
    centres, lib = vq.auto_ecp_centers(m, "lanl2dz")
    assert lib == "lanl2dz"
    assert len(centres) == 1
    assert centres[0].Z == 78


def test_auto_ecp_passes_through_all_electron_basis():
    """A basis with no .ecp sidecar (e.g. sto-3g) yields empty
    centres + empty library."""
    m = vq.Molecule([vq.Atom(1, [0, 0, 0])], multiplicity=2)
    centres, lib = vq.auto_ecp_centers(m, "sto-3g")
    assert centres == []
    assert lib == ""


def test_auto_ecp_skips_non_ecp_atoms():
    """A heavy + light mix (e.g. Pt + 2 H) should emit a centre only
    for the heavy atom (2 H atoms are all-electron in LANL2DZ).

    Use Pt + 2 H so the total electron count is even (80) and
    Molecule accepts the construction; the test is about
    ECP-centre routing, not SCF correctness.
    """
    m = vq.Molecule(
        [vq.Atom(78, [0, 0, 0]),
         vq.Atom(1,  [0, 0, 3.0]),
         vq.Atom(1,  [0, 0, -3.0])],
        multiplicity=3,
    )
    centres, lib = vq.auto_ecp_centers(m, "lanl2dz")
    assert lib == "lanl2dz"
    assert len(centres) == 1
    assert centres[0].Z == 78


# ---------- auto_ecp_centers — error paths ---------------------------------


def test_auto_ecp_mixed_library_raises_for_xml_helper():
    """Rb (ncore=28 → ecp28mdf) + Cs (ncore=46 → ecp46mdf) under
    dhf-tzvp would need two libecpint libraries; the SCF drivers
    accept only one. Auto-population must error clearly."""
    m = vq.Molecule(
        [vq.Atom(37, [0, 0, 0]), vq.Atom(55, [3, 0, 0])],
        multiplicity=1,
    )
    with pytest.raises(ValueError, match="mixed-row ECP"):
        vq.auto_ecp_centers(m, "dhf-tzvp")


def test_auto_ecp_vdzp_nonstandard_raises_for_legacy_xml_helper():
    """The XML-only auto-populator still fails closed for vDZP customs.

    Molecular SCF wrappers consume the same sidecar through inline primitive
    options; this legacy helper intentionally returns only XML-library ECPs.
    """
    m = vq.Molecule([vq.Atom(5, [0, 0, 0])], multiplicity=2)
    with pytest.raises(NotImplementedError, match="inline-primitive"):
        vq.auto_ecp_centers(m, "vdzp")


def test_parse_inline_ecp_sidecar_vdzp_custom_core():
    """vDZP custom ECP blocks are consumable as inline primitive arrays."""
    sp = vq.sidecar_path_for("vdzp")
    assert sp is not None
    records = vq.parse_inline_ecp_sidecar(sp)
    oxygen = records[8]
    assert oxygen.header.symbol == "O"
    assert oxygen.header.ncore == 2
    assert oxygen.header.lmax == 3
    assert oxygen.n_primitives == 4
    assert oxygen.ams[0] == 3  # local f channel
    assert oxygen.ams[1:] == (0, 1, 2)


def test_inline_ecp_primitives_match_lanl2dz_xml_matrix():
    """The sidecar parser's convention must reproduce the XML ECP matrix."""
    mol = vq.Molecule([vq.Atom(11, [0.0, 0.0, 0.0])], multiplicity=2)
    basis = vq.BasisSet(mol, "lanl2dz")

    blocks, centers, effective_z, total_ncore = vq.inline_ecp_data_for(
        mol, "lanl2dz"
    )
    assert len(blocks) == 1
    assert centers == [[0.0, 0.0, 0.0]]
    assert effective_z == [1.0]
    assert total_ncore == 10
    flat_centers = [x for xyz in centers for x in xyz]
    v_inline = vq.compute_ecp_matrix_from_primitives(
        basis, flat_centers, blocks
    )

    xml_centers, xml_lib = vq.auto_ecp_centers(mol, "lanl2dz")
    v_xml = vq.compute_ecp_matrix(basis, xml_centers, xml_lib)
    np.testing.assert_allclose(v_inline, v_xml, atol=2e-10, rtol=0.0)


def test_rks_vdzp_inline_ecp_keeps_physical_water_energy(tmp_path):
    """vDZP/O must attach its custom core ECP before SCF starts.

    Without the inline ECP, the valence-only O contraction returns the raw
    no-ECP branch near -36 Ha. The exact PBE value is not the target here; the
    regression pins the pseudopotential electron-count branch.
    """
    mol = vq.Molecule([
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [1.43, 0.0, 1.10]),
        vq.Atom(1, [-1.43, 0.0, 1.10]),
    ], 0, 1)
    # Configure inline ECP explicitly (BUG 99 guard requires it for ECP-paired
    # bases; the auto-attach convenience is gated behind explicit ECP intent).
    blocks, centres, eff_charges, ncore = vq.inline_ecp_data_for(mol, "vdzp")
    rks_opts = vq.RKSOptions()
    rks_opts.ecp_primitive_blocks = blocks
    rks_opts.ecp_primitive_centers = centres
    rks_opts.ecp_effective_charges = eff_charges
    rks_opts.ecp_total_ncore = int(ncore)
    res = vq.run_job(
        mol,
        basis="vdzp",
        method="rks",
        functional="pbe",
        rks_options=rks_opts,
        output=str(tmp_path / "pbe_vdzp_h2o"),
        progress=False,
    )
    assert res.converged
    assert res.ecp_total_ncore == int(ncore)
    assert vq.rhf_result_from_rks(res).ecp_total_ncore == int(ncore)
    assert -18.5 < res.energy < -16.0



# ---------- is_ecp_paired_basis — pattern detection --------------------------


@pytest.mark.parametrize("name, expected", [
    ("dhf-tzvp", True),
    ("dhf-tzvpp", True),
    ("dhf-svp", True),
    ("dhf-qzvp", True),
    ("cc-pvdz-pp", True),
    ("cc-pvtz-pp", True),
    ("cc-pvqz-pp", True),
    ("aug-cc-pvtz-pp", True),
    ("cc-pwcvdz-pp", True),
    ("lanl2dz", True),
    ("lanl2dzdp", True),
    ("lanl2tz", True),
    ("lanl08", True),
    ("lanl08(d)", True),
    ("vdzp", True),
    ("x2c-tzvpall", True),
    ("x2c-tzvpall-s", True),
    ("ecp-something", True),
    ("something-ecp", True),
    ("sto-3g", False),
    ("6-31g*", False),
    ("cc-pvdz", False),
    ("cc-pvtz", False),
    ("def2-tzvp", False),
    ("def2-svp", False),
    ("ano-rcc-vdzp", False),
])
def test_is_ecp_paired_basis(name, expected):
    assert vq.is_ecp_paired_basis(name) == expected


def test_is_ecp_paired_basis_case_insensitive():
    """Pattern matching is case-insensitive."""
    assert vq.is_ecp_paired_basis("DHF-TZVPP") is True
    assert vq.is_ecp_paired_basis("Cc-PvDz-Pp") is True
    assert vq.is_ecp_paired_basis("LANL2DZ") is True


# ---------- validate_ecp_required — BUG 99 guard ----------------------------


def test_validate_ecp_required_passes_when_ecp_centers_set():
    """Options with populated ecp_centers pass validation."""
    mol = vq.Molecule([vq.Atom(53, [0, 0, 0])], multiplicity=2)
    opts = vq.RHFOptions()
    opts.ecp_centers, opts.ecp_library = vq.auto_ecp_centers(mol, "dhf-tzvp")
    # Must not raise.
    vq.validate_ecp_required(opts, mol, "dhf-tzvp")


def test_validate_ecp_required_passes_all_electron_basis():
    """All-electron basis (sto-3g) passes — not ECP-paired."""
    mol = vq.Molecule([vq.Atom(53, [0, 0, 0])], multiplicity=2)
    opts = vq.RHFOptions()
    vq.validate_ecp_required(opts, mol, "sto-3g")


def test_validate_ecp_required_passes_def2_basis_on_light_atoms():
    """def2-TZVP is all-electron through Kr -- must pass on light atoms."""
    mol = vq.Molecule([vq.Atom(35, [0, 0, 0])], multiplicity=2)
    opts = vq.RHFOptions()
    vq.validate_ecp_required(opts, mol, "def2-tzvp")


def test_validate_ecp_required_refuses_def2_beyond_kr_when_the_ecp_is_bypassed():
    """def2-TZVP beyond Kr is valence-only and ships the def2-ECP sidecar;
    options that bypassed the automatic attachment are refused, naming the
    atom, instead of running iodine's valence block all-electron."""
    mol = vq.Molecule([vq.Atom(53, [0, 0, 0])], multiplicity=2)
    opts = vq.RHFOptions()
    with pytest.raises(ValueError, match=r"Atoms that need an ECP: \['I'\]"):
        vq.validate_ecp_required(opts, mol, "def2-tzvp")


def test_validate_ecp_required_refuses_a_family_that_ships_no_ecp_data():
    """The diffuse def2-*D sets are valence-only past Kr but carry no
    lanthanide block and no ECP for one, so the requirement has no data and
    the guard says so rather than running cerium all-electron.

    cc-pVTZ-PP was this test's example until its family shipped its orbital
    files and sidecars (2026-09-08); it now takes the "no ECP centers were
    configured" branch instead, pinned by
    ``test_validate_ecp_required_raises_for_pp_basis_without_ecp`` below.
    """
    mol = vq.Molecule([vq.Atom(58, [0, 0, 0])], multiplicity=1)
    with pytest.raises(ValueError, match="ships no ECP data"):
        vq.validate_ecp_required(vq.RHFOptions(), mol, "def2-tzvpd")


def test_validate_ecp_required_raises_for_dhf_without_ecp():
    """BUG 99: dhf-tzvp + iodine, no ECP centres -> ValueError."""
    mol = vq.Molecule([vq.Atom(53, [0, 0, 0])], multiplicity=2)
    opts = vq.RHFOptions()
    with pytest.raises(ValueError, match="no ECP centers were configured"):
        vq.validate_ecp_required(opts, mol, "dhf-tzvp")


def test_validate_ecp_required_raises_for_pp_basis_without_ecp():
    """cc-pVTZ-PP + iodine, no ECP centres -> ValueError."""
    mol = vq.Molecule([vq.Atom(53, [0, 0, 0])], multiplicity=2)
    opts = vq.RHFOptions()
    with pytest.raises(ValueError, match="no ECP centers were configured"):
        vq.validate_ecp_required(opts, mol, "cc-pvtz-pp")


def test_validate_ecp_required_raises_for_lanl_without_ecp():
    """lanl2dz + Pt, no ECP centres -> ValueError."""
    mol = vq.Molecule([vq.Atom(78, [0, 0, 0])], multiplicity=3)
    opts = vq.UHFOptions()
    with pytest.raises(ValueError, match="no ECP centers were configured"):
        vq.validate_ecp_required(opts, mol, "lanl2dz")


def test_validate_ecp_required_passes_with_primitive_blocks():
    """Options with ecp_primitive_blocks (inline ECP, e.g. vDZP) pass."""
    mol = vq.Molecule([vq.Atom(5, [0, 0, 0])], multiplicity=2)
    opts = vq.RHFOptions()
    blocks, centres, eff_charges, ncore = vq.inline_ecp_data_for(mol, "vdzp")
    opts.ecp_primitive_blocks = blocks
    opts.ecp_primitive_centers = centres
    opts.ecp_effective_charges = eff_charges
    opts.ecp_total_ncore = ncore
    # Must not raise -- primitive blocks satisfy the guard.
    vq.validate_ecp_required(opts, mol, "vdzp")


def test_validate_ecp_required_passes_when_options_none():
    """None options pass silently (post-SCF paths may not have SCF opts)."""
    mol = vq.Molecule([vq.Atom(53, [0, 0, 0])], multiplicity=2)
    vq.validate_ecp_required(None, mol, "dhf-tzvp")


def test_validate_ecp_required_error_names_heavy_atoms():
    """The error message identifies heavy atoms (Z > 36)."""
    mol = vq.Molecule([vq.Atom(53, [0, 0, 0])], multiplicity=2)
    opts = vq.RHFOptions()
    with pytest.raises(ValueError, match=r"Atoms that need an ECP: \['I'\]"):
        vq.validate_ecp_required(opts, mol, "dhf-tzvp")


def test_validate_ecp_required_passes_light_molecule_in_ecp_family_basis():
    """A light molecule in an ECP-family basis is an all-electron run.

    dhf-TZVP and LANL2DZ carry no ECP for carbon; refusing them there (the
    pre-2026-09 name heuristic did) blocked correct calculations. The
    decision is per element through the sidecar now.
    """
    mol = vq.Molecule(
        [vq.Atom(6, [0, 0, 0])],  # C atom, 6 e-
        multiplicity=3,           # triplet
    )
    opts = vq.RHFOptions()
    vq.validate_ecp_required(opts, mol, "dhf-tzvp")
    vq.validate_ecp_required(opts, mol, "lanl2dz")


def test_validate_ecp_required_passes_all_electron_relativistic_basis():
    """x2c-TZVPall is all-electron on every element; gold must pass."""
    mol = vq.Molecule([vq.Atom(79, [0, 0, 0])], multiplicity=2)
    opts = vq.RHFOptions()
    vq.validate_ecp_required(opts, mol, "x2c-tzvpall")


def test_validate_ecp_required_refuses_valence_only_pob_and_3c_bases():
    """pob-TZVP{,-rev2} and def2-m* are valence-only beyond Kr; the guard
    must see that although their names carry no ECP marker."""
    mol = vq.Molecule([vq.Atom(47, [0, 0, 0])], multiplicity=2)
    for name in ("pob-tzvp-rev2", "pob-tzvp", "def2-msvp", "def2-mtzvp", "def2-mtzvpp"):
        with pytest.raises(ValueError, match="no ECP centers were configured"):
            vq.validate_ecp_required(vq.RHFOptions(), mol, name)



# ---------- ecp_centers_from_dict — dict-based convenience -------------------


def test_ecp_centers_from_dict_iodine_dhf():
    """I2 with {'I': 'dhf'} produces 2 ECPCenter objects + correct library."""
    mol = vq.Molecule(
        [vq.Atom(53, [0, 0, 0]), vq.Atom(53, [0, 0, 2.666])],
        multiplicity=1,
    )
    centres, lib = vq.ecp_centers_from_dict(
        {"I": "dhf"}, mol, basis_name="dhf-tzvpp"
    )
    assert len(centres) == 2
    assert all(c.Z == 53 for c in centres)
    # dhf-tzvpp: I has ncore=28 -> ecp28mdf
    assert lib == "ecp28mdf"


def test_ecp_centers_from_dict_mixed_elements():
    """Rb + I with {'Rb': 'dhf', 'I': 'dhf'} works."""
    mol = vq.Molecule(
        [vq.Atom(37, [0, 0, 0]), vq.Atom(53, [0, 0, 3.0])],
        multiplicity=1,
    )
    centres, lib = vq.ecp_centers_from_dict(
        {"Rb": "dhf", "I": "dhf"}, mol, basis_name="dhf-tzvpp"
    )
    assert len(centres) == 2
    # Both Rb (ncore=28) and I (ncore=28) map to ecp28mdf — single library.
    assert lib == "ecp28mdf"


def test_ecp_centers_from_dict_skips_non_specified_atoms():
    """Only atoms in the dict get ECP centres; He is not in the dict."""
    mol = vq.Molecule(
        [vq.Atom(53, [0, 0, 0]), vq.Atom(2, [0, 0, 2.0])],
        multiplicity=2,  # I (53e) + He (2e) = 55e, doublet works
    )
    centres, lib = vq.ecp_centers_from_dict(
        {"I": "dhf"}, mol, basis_name="dhf-tzvpp"
    )
    assert len(centres) == 1
    assert centres[0].Z == 53


def test_ecp_centers_from_dict_unknown_element_raises():
    """Unknown element symbol raises ValueError."""
    mol = vq.Molecule([vq.Atom(6, [0, 0, 0])], multiplicity=3)
    with pytest.raises(ValueError, match="Unknown element symbol"):
        vq.ecp_centers_from_dict({"Xx": "dhf"}, mol, basis_name="dhf-tzvpp")


def test_ecp_centers_from_dict_unknown_label_raises():
    """Unknown ECP label raises ValueError."""
    mol = vq.Molecule([vq.Atom(53, [0, 0, 0])], multiplicity=2)
    with pytest.raises(ValueError, match="unknown ECP label"):
        vq.ecp_centers_from_dict({"I": "unknown_ecp"}, mol, basis_name="dhf-tzvpp")


def test_ecp_centers_from_dict_missing_basis_name_raises():
    """'dhf' label without basis_name raises ValueError."""
    mol = vq.Molecule([vq.Atom(53, [0, 0, 0])], multiplicity=2)
    with pytest.raises(ValueError, match="basis_name is required"):
        vq.ecp_centers_from_dict({"I": "dhf"}, mol)


def test_ecp_centers_from_dict_case_insensitive_symbol():
    """Element symbols are case-insensitive."""
    mol = vq.Molecule([vq.Atom(53, [0, 0, 0])], multiplicity=2)
    centres, lib = vq.ecp_centers_from_dict(
        {"i": "dhf"}, mol, basis_name="dhf-tzvpp"
    )
    assert len(centres) == 1
    assert centres[0].Z == 53
    assert lib == "ecp28mdf"


def test_ecp_centers_from_dict_empty_spec():
    """Empty spec returns empty centres + empty library."""
    mol = vq.Molecule([vq.Atom(53, [0, 0, 0])], multiplicity=2)
    centres, lib = vq.ecp_centers_from_dict({}, mol)
    assert centres == []
    assert lib == ""



def test_ecp_centers_from_dict_lanl2_label():
    """{'Pt': 'lanl2'} with lanl2dz basis resolves to lanl2dz.xml."""
    mol = vq.Molecule([vq.Atom(78, [0, 0, 0])], multiplicity=3)
    centres, lib = vq.ecp_centers_from_dict(
        {"Pt": "lanl2"}, mol, basis_name="lanl2dz"
    )
    assert len(centres) == 1
    assert centres[0].Z == 78
    assert lib == "lanl2dz"


def test_ecp_centers_from_dict_supported_labels_in_error():
    """Error message lists supported labels."""
    mol = vq.Molecule([vq.Atom(53, [0, 0, 0])], multiplicity=2)
    with pytest.raises(ValueError, match="Supported labels"):
        vq.ecp_centers_from_dict({"I": "bad"}, mol, basis_name="dhf-tzvpp")


# ---------- End-to-end SCF -------------------------------------------------


@pytest.mark.parametrize("z, mult, expect_e", [
    # Si UHF triplet. Reference value updated after commit 899ab16a
    # ("fix: ECP ncore propagation + §1 licensing cleanup", 2026-05-22)
    # corrected the rhf/uhf/rks/uks drivers to subtract ECP-replaced
    # core electrons from mol.n_electrons() — the Phase-14d-era
    # ``-3.474802`` was computed with the buggy electron count (Si's
    # 14 atomic electrons fit into the 8-function valence basis but
    # left UHF at a non-physical stationary point). The corrected
    # ³P ground state agrees with PySCF UHF/LANL2DZ to 11 decimals.
    (14, 3, -3.675696),   # Si  UHF ³P  (matches PySCF, post-899ab16a)
    (78, 3, -118.226797), # Pt  UHF
])
def test_uhf_with_auto_ecp_matches_manual_path(z, mult, expect_e):
    """The headline UX win: auto-populated SCF gives the same energy
    as the manual ECPCenter+library setup. Validates that
    auto_ecp_centers is interchangeable with the explicit
    boilerplate from docs/user_guide/ecp.md."""
    mol = vq.Molecule([vq.Atom(z, [0, 0, 0])], multiplicity=mult)
    basis = vq.BasisSet(mol, "lanl2dz")
    opts = vq.UHFOptions()
    opts.ecp_centers, opts.ecp_library = vq.auto_ecp_centers(mol, "lanl2dz")
    opts.max_iter = 200
    result = vq.run_uhf(mol, basis, opts)
    assert result.converged
    assert result.energy == pytest.approx(expect_e, abs=1e-4)
