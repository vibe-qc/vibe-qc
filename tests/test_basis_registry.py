"""The basis registry: one answer per basis for fits, ECP need and coverage.

Pins the per-element ECP decision that replaced the name heuristic
(2026-09), the unified default-auxiliary table shared by the molecular and
periodic callers, and the inline-always sidecar attachment on the molecular
SCF wrappers.  The SCF cases are AgH-sized so the whole file stays fast.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc.basis_registry import (
    basis_replaces_core,
    canonical_basis_name,
    default_aux_basis,
    ecp_requirements,
    element_coverage,
    lookup,
    sidecar_core_electrons,
)


def _agh():
    return vq.Molecule([vq.Atom(47, [0, 0, 0]), vq.Atom(1, [0, 0, 3.1])], 0, 1)


def _electrons_treated(result, basis) -> float:
    S = np.asarray(vq.compute_overlap(basis))
    return float(np.trace(np.asarray(result.density) @ S))


# ---------------------------------------------------------------------------
# Names and records
# ---------------------------------------------------------------------------


def test_canonical_name_lowercases_strips_and_resolves_aliases():
    assert canonical_basis_name("  def2-TZVP ") == "def2-tzvp"
    assert canonical_basis_name("6-311+G(3df,2p)") == "6-311+g3df2p"
    assert canonical_basis_name("STO3G") == "sto-3g"


def test_canonical_name_accepts_a_basisset_object():
    mol = vq.Molecule([vq.Atom(8, [0, 0, 0])], 0, 3)
    assert canonical_basis_name(vq.BasisSet(mol, "STO-3G")) == "sto-3g"


@pytest.mark.parametrize(
    ("name", "family", "role", "all_electron", "threshold"),
    [
        ("x2c-tzvpall", "all-electron-relativistic", "orbital", True, None),
        ("ano-rcc-vtzp", "all-electron-relativistic", "orbital", True, None),
        ("def2-tzvp", "def2", "orbital", False, 36),
        ("def2-mtzvpp", "def2-3c", "orbital", False, 36),
        ("pob-tzvp-rev2", "pob-tzvp-rev2", "orbital", False, 36),
        ("pob-tzvp", "pob-tzvp", "orbital", False, 36),
        ("cc-pvtz-pp", "cc-pp", "orbital", False, 28),
        ("lanl2dz", "sidecar-ecp", "orbital", False, None),
        ("def2-svp-rifit", "fitting", "fitting", False, None),
        ("madeup-zeta-99", "unknown", "orbital", False, None),
    ],
)
def test_lookup_family_rules(name, family, role, all_electron, threshold):
    rec = lookup(name)
    assert rec.family == family
    assert rec.role == role
    assert rec.all_electron is all_electron
    assert rec.ecp_required_above_z == threshold


def test_element_coverage_reads_the_bundled_file():
    assert {1, 36, 47, 86} <= element_coverage("def2-tzvp")
    assert 58 not in element_coverage("def2-tzvpd")  # no lanthanides in the -D sets
    assert 47 in element_coverage("pob-tzvp-rev2")
    assert 47 not in element_coverage("cc-pvdz")
    assert element_coverage("madeup-zeta-99") == frozenset()


def test_sidecar_core_electrons_reads_the_sidecar():
    cores = sidecar_core_electrons("lanl2dz")
    assert cores[78] == 60 and cores[16] == 10 and 1 not in cores
    assert sidecar_core_electrons("x2c-tzvpall") == {}


# ---------------------------------------------------------------------------
# The per-element ECP decision
# ---------------------------------------------------------------------------


def test_light_molecule_in_ecp_family_basis_needs_no_ecp():
    """LANL2DZ and dhf-TZVP are all-electron on H, C, N, O."""
    assert ecp_requirements("lanl2dz", [1, 6, 7, 8]) == []
    assert ecp_requirements("dhf-tzvp", [1, 6, 7, 8]) == []
    assert not basis_replaces_core("lanl2dz", [1, 8])


def test_all_electron_relativistic_basis_never_needs_an_ecp():
    assert ecp_requirements("x2c-tzvpall", [79, 92]) == []
    assert ecp_requirements("ano-rcc-vtzp", [79]) == []


def test_sidecar_element_needs_ecp_and_has_data():
    [req] = ecp_requirements("lanl2dz", [1, 78])
    assert (req.Z, req.symbol, req.n_core, req.has_data) == (78, "Pt", 60, True)


def test_custom_core_sidecar_flags_light_elements_too():
    """vDZP replaces two core electrons on oxygen."""
    [req] = ecp_requirements("vdzp", [1, 8])
    assert (req.symbol, req.n_core, req.has_data) == ("O", 2, True)


def test_valence_only_family_without_data_is_flagged_as_missing():
    """A valence-only family with no block for the element carries no data
    and the guard refuses.  def2-TZVP's iodine block comes with the def2-ECP
    sidecar; the diffuse sets carry no lanthanide block and no ECP for one.

    cc-pVTZ-PP used to be this test's no-data example.  Its 16-member family
    now ships the orbital sets and their sidecars, so it moved to
    ``test_pp_family_reports_shipped_core_counts`` below.
    """
    [req] = ecp_requirements("def2-tzvp", [8, 53])
    assert (req.symbol, req.has_data, req.n_core) == ("I", True, 28)
    [req] = ecp_requirements("def2-tzvpd", [58])
    assert req.has_data is False


# ---------------------------------------------------------------------------
# The Peterson/Figgen PP family (2026-09-08 import)
# ---------------------------------------------------------------------------

PP_BASES = [
    f"{aug}cc-p{core}{zeta}-pp"
    for zeta in ("dz", "tz", "qz", "5z")
    for core in ("v", "wcv")
    for aug in ("", "aug-")
]

# The small-core Stuttgart-Koeln pattern the whole family follows: [Ne] on the
# 3d/4p row, [Ar]3d10 on the 4d/5p row, [Kr]4d10-4f14 on the 5d/6p row.
PP_EXPECTED_CORES = {
    29: 10, 30: 10, 31: 10, 32: 10, 33: 10, 34: 10, 35: 10, 36: 10,
    39: 28, 40: 28, 41: 28, 42: 28, 43: 28, 44: 28, 45: 28, 46: 28,
    47: 28, 48: 28, 49: 28, 50: 28, 51: 28, 52: 28, 53: 28, 54: 28,
    72: 60, 73: 60, 74: 60, 75: 60, 76: 60, 77: 60, 78: 60, 79: 60,
    80: 60, 81: 60, 82: 60, 83: 60, 84: 60, 85: 60, 86: 60,
}


@pytest.mark.parametrize("name", PP_BASES)
def test_pp_family_covers_the_same_39_elements(name):
    """All 16 sets span Cu-Kr, Y-Xe and Hf-Rn -- no member is a short import."""
    assert sorted(element_coverage(name)) == sorted(PP_EXPECTED_CORES)


@pytest.mark.parametrize("name", PP_BASES)
def test_pp_family_reports_shipped_core_counts(name):
    """Every element resolves through the sidecar, with the published core.

    This is the check that would have caught a partial import: a missing
    sidecar block leaves ``has_data=False`` and the SCF wrappers refuse,
    rather than running all-electron in a valence basis.
    """
    assert sidecar_core_electrons(name) == PP_EXPECTED_CORES
    reqs = {r.Z: (r.n_core, r.has_data) for r in ecp_requirements(name, PP_EXPECTED_CORES)}
    assert reqs == {z: (n, True) for z, n in PP_EXPECTED_CORES.items()}


def test_pp_family_threshold_does_not_contradict_the_sidecars():
    """Copper is Z=29 and carries a 10-electron ECP, so the family threshold
    has to sit below it.  It read 30 while no data shipped, which would have
    called Cu and Zn all-electron on any element the sidecar missed."""
    threshold = lookup("cc-pvtz-pp").ecp_required_above_z
    assert threshold == 28
    assert min(PP_EXPECTED_CORES) > threshold


@pytest.mark.parametrize("name", PP_BASES)
def test_pp_family_default_ri_is_its_own_published_fit(name):
    assert default_aux_basis(name, "ri") == f"{name}-rifit"


def test_pp_family_has_no_default_jk():
    """No JK fit is published for this family; asking must refuse rather
    than silently borrow a mismatched one."""
    for name in PP_BASES:
        with pytest.raises(NotImplementedError):
            default_aux_basis(name, "jk")


@pytest.mark.parametrize(
    "name", ["pob-tzvp-rev2", "pob-tzvp", "def2-msvp", "def2-mtzvp", "def2-mtzvpp"]
)
def test_valence_only_bases_with_shipped_sidecars_have_data(name):
    [req] = ecp_requirements(name, [1, 47])
    assert (req.symbol, req.n_core, req.has_data) == ("Ag", 28, True)


def test_fitting_bases_never_report_an_ecp_requirement():
    assert ecp_requirements("def2-svp-rifit", [47, 79]) == []


# ---------------------------------------------------------------------------
# One default-auxiliary table for molecular and periodic callers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("orbital", "jk", "ri"),
    [
        ("def2-sv(p)", "def2-svp-jk", "def2-sv(p)-rifit"),
        ("def2-svpd", "def2-svp-jk", "def2-svpd-rifit"),
        ("def2-tzvppd", "def2-tzvpp-jk", "def2-tzvppd-rifit"),
        ("def2-qzvp", "def2-qzvp-jk", "def2-qzvpp-rifit"),
        ("cc-pvqz", "cc-pvqz-jkfit", "cc-pvqz-ri"),
    ],
)
def test_molecular_and_periodic_defaults_agree(orbital, jk, ri):
    from vibeqc.aux_basis import default_aux_for
    from vibeqc.density_fitting import default_aux_basis_for

    assert default_aux_basis_for(orbital, kind="jk") == jk
    assert default_aux_basis_for(orbital, kind="ri") == ri
    assert default_aux_for(orbital) == jk
    assert default_aux_basis(orbital, "jk") == jk


def test_standin_only_when_the_caller_opts_in():
    from vibeqc.aux_basis import default_aux_for
    from vibeqc.density_fitting import default_aux_basis_for

    with pytest.raises(NotImplementedError, match="pob-"):
        default_aux_basis("pob-tzvp-rev2", "jk")
    assert default_aux_basis("pob-tzvp-rev2", "jk", allow_standin=True) == "def2-tzvp-jk"
    assert default_aux_for("pob-tzvp-rev2") == "def2-tzvp-jk"
    with pytest.raises(NotImplementedError, match="No default"):
        default_aux_basis_for("sto-3g", kind="jk")
    assert default_aux_for("sto-3g") == "def2-svp-jk"
    with pytest.raises(KeyError):
        default_aux_for("madeup-zeta-99")


def test_every_registered_default_fit_ships():
    """A registry entry must name a file libint can open."""
    from vibeqc.basis_registry import _registry

    for name, rec in _registry().bases.items():
        for key in ("default_jk", "default_ri", "standin_jk", "standin_ri"):
            fit = rec.get(key)
            if fit:
                assert element_coverage(fit), f"{name}: {key}={fit} is not bundled"


# ---------------------------------------------------------------------------
# The molecular wrappers attach the sidecar inline, on every basis that has one
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["pob-tzvp-rev2", "pob-tzvp", "def2-msvp", "def2-mtzvp"])
def test_valence_only_bases_attach_their_ecp_on_heavy_atoms(name):
    """Pre-fix these ran 48 electrons into a 19-valence-electron silver
    block without any error (measured 2026-09-05)."""
    mol = _agh()
    basis = vq.BasisSet(mol, name)
    opts = vq.RHFOptions()
    opts.max_iter = 3
    result = vq.run_rhf(mol, basis, opts)
    assert result.ecp_operator_applied and result.ecp_total_ncore == 28
    assert len(opts.ecp_primitive_blocks) == 1 and list(opts.ecp_centers) == []
    assert _electrons_treated(result, basis) == pytest.approx(20.0, abs=1e-8)


def test_vdzp_auto_route_is_the_sidecars_own_potential():
    """The core-count pairing sent Ag/vDZP to the Stuttgart ecp28mdf XML
    library (13 primitives) instead of the sidecar's 9-primitive vDZP
    potential; the two energies differed by 85 mHa on AgH."""
    from vibeqc.ecp_metadata import inline_ecp_data_for

    mol = _agh()
    basis = vq.BasisSet(mol, "vdzp")
    auto = vq.RHFOptions()
    auto.max_iter = 200
    r_auto = vq.run_rhf(mol, basis, auto)
    assert r_auto.ecp_xml_library == "" and len(r_auto.ecp_primitive_blocks) == 1

    blocks, centers, charges, ncore = inline_ecp_data_for(mol, "vdzp")
    explicit = vq.RHFOptions()
    explicit.max_iter = 200
    explicit.ecp_primitive_blocks = blocks
    explicit.ecp_primitive_centers = centers
    explicit.ecp_effective_charges = charges
    explicit.ecp_total_ncore = ncore
    r_explicit = vq.run_rhf(mol, basis, explicit)
    assert r_auto.energy == pytest.approx(r_explicit.energy, abs=1e-10)


def test_two_core_sizes_in_one_molecule_attach_both():
    """CsI/dhf-TZVP was refused as 'mixed-row' by the XML pairing although
    the sidecar carries both potentials."""
    mol = vq.Molecule([vq.Atom(55, [0, 0, 0]), vq.Atom(53, [0, 0, 6.2])], 0, 1)
    basis = vq.BasisSet(mol, "dhf-tzvp")
    opts = vq.RHFOptions()
    opts.max_iter = 1
    vq.run_rhf(mol, basis, opts)
    assert len(opts.ecp_primitive_blocks) == 2
    assert opts.ecp_total_ncore == 46 + 28
    assert list(opts.ecp_effective_charges) == [9.0, 25.0]


def test_standard_core_element_absent_from_the_xml_library_attaches_inline():
    """Si/vDZP has a 10-electron core but ecp10mdf carries no silicon: the
    XML pairing died with libecpint's 'stoi: no conversion'."""
    mol = vq.Molecule(
        [vq.Atom(14, [0, 0, 0])] + [vq.Atom(1, [s * 1.6, t * 1.6, u * 1.6]) for s, t, u in
                                    ((1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1))],
        0, 1,
    )
    basis = vq.BasisSet(mol, "vdzp")
    opts = vq.RHFOptions()
    opts.max_iter = 1
    result = vq.run_rhf(mol, basis, opts)
    assert result.ecp_xml_library == "" and result.ecp_total_ncore == 10


def test_light_molecules_run_all_electron_in_ecp_family_bases(tmp_path):
    """H2O/LANL2DZ is a D95V all-electron calculation and x2c-TZVPall is
    all-electron everywhere; both were refused by the name heuristic."""
    mol = vq.Molecule(
        [vq.Atom(8, [0, 0, 0]), vq.Atom(1, [0, 1.43, -0.98]), vq.Atom(1, [0, -1.43, -0.98])],
        0, 1,
    )
    for name, e_ref in (("lanl2dz", -76.00809631), ("x2c-tzvpall", -76.05436355)):
        result = vq.run_job(mol, basis=name, method="rhf", output=tmp_path / name)
        assert result.converged and not result.ecp_operator_applied
        assert result.energy == pytest.approx(e_ref, abs=2e-6)
