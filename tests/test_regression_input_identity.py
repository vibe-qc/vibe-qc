"""Resolved physical-input qualification for periodic comparisons (#648).

Each code's geometry, basis, ECP cores and XC definition are recorded from
what it built or was submitted and compared; names are not trusted. These are
input checks, not a scientific acceptance of any energy.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from dataclasses import replace

import numpy as np
import pytest
import vibeqc as vq

from examples.regression import run_suite
from examples.regression.core import input_identity as ii
from examples.regression.core import runner_crystal, runner_pyscf, runner_vibeqc
from examples.regression.core.case import CodeRow
from examples.regression.core.compare import apply_input_qualification, paired_wall_ratio
from examples.regression.core.spec import AtomFrac, PeriodicSpec
from examples.regression.methods.catalog import METHODS

needs_pyscf = pytest.mark.skipif(importlib.util.find_spec("pyscf") is None,
                                 reason="PySCF not installed")

LIH = run_suite._load_periodic_spec("lih_rocksalt")
# A hexagonal cell whose row matrix is not symmetric: passing it where
# columns are expected builds a different (sheared) crystal (#445).
HEX = PeriodicSpec(
    id="hex_h2", family="synthetic", space_group="P1",
    lattice_ang=((3.0, 0.0, 0.0), (-1.5, 1.5 * 3 ** 0.5, 0.0), (0.0, 0.0, 5.0)),
    atoms=(AtomFrac("H", 1, (1 / 3, 2 / 3, 0.25)), AtomFrac("H", 1, (2 / 3, 1 / 3, 0.75))),
)


def _requested_geometry(spec):
    rows = np.asarray(spec.lattice_ang) * ii.ANGSTROM_TO_BOHR
    return ii.geometry_identity(rows, [(a.z, *(np.asarray(a.frac) @ rows)) for a in spec.atoms])


def _vibeqc(spec, basis_name="sto-3g", method_id="rhf"):
    system = runner_vibeqc._build_periodic_system(spec)
    basis = vq.make_basis(system.unit_cell_molecule(), basis_name)
    return runner_vibeqc._input_identity(system, basis, basis_name, METHODS[method_id])


def _pyscf(spec, tmp_path, basis_name="sto-3g", method=None):
    return runner_pyscf.resolve_periodic_identity(
        spec=spec, basis_name=basis_name, method=method or METHODS["rhf"],
        kmesh=(1, 1, 1), log_path=tmp_path / "pyscf.log")


def test_canonical_contraction_ignores_scale_and_sign():
    exponents, d = [3.4252509, 0.6239137, 0.1688554], [0.15432897, 0.53532814, 0.44463454]
    ref = ii.canonical_contraction(0, exponents, d)
    assert np.allclose(ii.canonical_contraction(0, exponents, [-7.0 * x for x in d]), ref)
    assert not np.allclose(ii.canonical_contraction(1, exponents, d), ref)


def test_geometry_identity_ignores_cartesian_orientation():
    rows = np.asarray(HEX.lattice_ang)
    angle = 0.7
    rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    atoms = [(1, *(np.asarray(a.frac) @ rows)) for a in HEX.atoms]
    turned = [(1, *(rot @ np.asarray(a[1:]))) for a in atoms]
    assert not ii._compare_geometry(ii.geometry_identity(rows, atoms),
                                    ii.geometry_identity(rows @ rot.T, turned))


def test_vibeqc_builds_the_requested_non_orthogonal_crystal():
    assert not ii._compare_geometry(_vibeqc(HEX)["geometry"], _requested_geometry(HEX))


def test_row_matrix_passed_as_columns_is_caught():
    """The pre-#648 construction: lattice=rows (read as columns), r = L @ frac."""
    rows = np.asarray(HEX.lattice_ang) * ii.ANGSTROM_TO_BOHR
    atoms = [vq.Atom(1, list(rows @ np.asarray(a.frac))) for a in HEX.atoms]
    system = vq.PeriodicSystem(3, rows, atoms)
    built = ii.geometry_identity(np.asarray(system.lattice).T,
                                 [(a.Z, *a.xyz) for a in system.unit_cell])
    assert "lattice metric differs" in ii._compare_geometry(built, _requested_geometry(HEX))


@needs_pyscf
def test_pyscf_builds_the_requested_non_orthogonal_crystal(tmp_path):
    assert not ii._compare_geometry(_pyscf(HEX, tmp_path)["geometry"], _requested_geometry(HEX))


@needs_pyscf
@pytest.mark.parametrize("method_id", ["rhf", "rks-pbe", "rks-b3lyp"])
def test_lih_starter_inputs_qualify_against_pyscf(tmp_path, method_id):
    pair = ii.qualify_pair(_vibeqc(LIH, method_id=method_id),
                           _pyscf(LIH, tmp_path, method=METHODS[method_id]))
    assert pair["status"] == "unverified", pair["fields"]
    assert pair["not_compared"]


@needs_pyscf
def test_b3lyp_vwn_variants_cannot_qualify(tmp_path):
    vwn_rpa = replace(METHODS["rks-b3lyp"], xc="b3lyp/g")
    pair = ii.qualify_pair(_vibeqc(LIH, method_id="rks-b3lyp"),
                           _pyscf(LIH, tmp_path, method=vwn_rpa))
    assert pair["fields"]["xc"]["status"] == "mismatch"
    assert pair["status"] == "mismatch"


@pytest.mark.parametrize("edit, reason", [
    (lambda b: b["0"][0][2].__setitem__(0, b["0"][0][2][0] * 1.000001), "exponents"),
    (lambda b: b["0"][0][3].__setitem__(0, b["0"][0][3][0] + 1e-6), "coefficients"),
    (lambda b: b["0"].pop(), "shell structure"),
])
def test_same_basis_name_with_different_primitives_cannot_qualify(edit, reason):
    reference = _vibeqc(LIH)
    candidate = copy.deepcopy(reference)
    edit(candidate["basis"]["shells_by_atom"])
    field = ii.qualify_pair(candidate, reference)["fields"]["basis"]
    assert field["status"] == "mismatch" and reason in field["reasons"][0]


def test_xc_definitions_are_compared_by_evaluation():
    pbe, b3lyp, hf = (_vibeqc(LIH, method_id=m)["xc"] for m in ("rks-pbe", "rks-b3lyp", "rhf"))
    assert ii._compare_xc(pbe, pbe) == []
    assert "hf_fraction differs" in ii._compare_xc(pbe, b3lyp)
    assert ii._compare_xc(hf, pbe) == ["Hartree-Fock against a density functional"]


def _deck(spec, func_kw="HF"):
    return runner_crystal.build_d12(spec=spec, basis_kw="STO-3G", func_kw=func_kw,
                                    kmesh=(1, 1, 1), conv_tol_energy=1e-7, max_iter=40)


@pytest.mark.parametrize("spec", [LIH, HEX])
def test_crystal_deck_geometry_matches_and_library_inputs_stay_unverified(spec):
    deck = ii.crystal_deck_identity(_deck(spec))
    assert not ii._compare_geometry(deck["geometry"], _requested_geometry(spec))
    pair = ii.qualify_pair(_vibeqc(spec), deck)
    assert pair["fields"]["geometry"]["status"] == "matched"
    assert pair["fields"]["xc"]["status"] == "matched"
    assert pair["fields"]["basis"]["status"] == "unverified"
    assert pair["status"] == "unverified"


@pytest.mark.parametrize("keyword, xc", [("SVWN", "lda"), ("BLYP", "blyp"),
                                         ("PBEXC", "pbe"), ("B3LYP", "b3lyp")])
def test_crystal_documented_xc_matches_the_vibeqc_functional(keyword, xc):
    """The manual's composition, mixed from libxc parts, against vibe-qc's own
    resolution (for B3LYP the composite VWN5 functional, libxc 475)."""
    documented = ii.crystal_documented_xc(keyword)
    assert documented["source"] == ii.CRYSTAL_XC_SOURCE
    assert ii._compare_xc(documented, ii.xc_identity_from_functional(vq.Functional(xc))) == []


def test_crystal_b3lyp_is_not_the_vwn_rpa_variant():
    documented = ii.crystal_documented_xc("B3LYP")
    assert ii._compare_xc(documented, ii.xc_identity_from_functional(vq.Functional("b3lyp/g")))


def test_crystal_deck_dft_keywords_resolve_only_when_documented():
    pair = ii.qualify_pair(_vibeqc(LIH, method_id="rks-b3lyp"),
                           ii.crystal_deck_identity(_deck(LIH, "B3LYP")))
    assert pair["fields"]["xc"]["status"] == "matched"
    assert pair["fields"]["xc"]["source"] == [ii.CRYSTAL_XC_SOURCE]
    bare = ii.crystal_deck_identity(_deck(LIH, "PBE"))
    assert "not a documented standalone" in bare["xc"]["unverified"]
    assert runner_crystal._crystal_functional(METHODS["rks-pbe"]) == "PBEXC"


def test_left_handed_cell_is_not_the_cell_crystal_builds():
    mirror = replace(HEX, lattice_ang=(HEX.lattice_ang[1], HEX.lattice_ang[0], HEX.lattice_ang[2]))
    deck = ii.crystal_deck_identity(_deck(mirror))
    assert "cell handedness differs" in ii._compare_geometry(_requested_geometry(mirror), deck["geometry"])


@needs_pyscf
def test_ecp_lanl2dz_core_counts_and_resolved_operator_terms_match(tmp_path):
    nacl = run_suite._load_periodic_spec("nacl_rocksalt")
    pair = ii.qualify_pair(_vibeqc(nacl, "lanl2dz"), _pyscf(nacl, tmp_path, "lanl2dz"))
    assert pair["fields"]["basis"]["status"] == "matched"
    assert pair["fields"]["ecp"]["status"] == "matched", pair["fields"]["ecp"]
    assert pair["status"] == "unverified"


def _rows(status="pass"):
    return [CodeRow("t", "dev", LIH.id, LIH.family, "sto-3g", "rhf", "1x1x1", "vibeqc", "t",
                    status=status, energy_ha=-1.0, converged=True, wall_s=1.0, ref_code="pyscf"),
            CodeRow("t", "dev", LIH.id, LIH.family, "sto-3g", "rhf", "1x1x1", "pyscf", "t",
                    status="reference", energy_ha=-1.0, converged=True, wall_s=1.0)]


def test_mismatched_inputs_turn_a_passing_energy_into_an_error():
    reference = _vibeqc(LIH)
    candidate = copy.deepcopy(reference)
    candidate["geometry"]["metric_bohr2"][0][0] *= 1.01
    rows = _rows()
    apply_input_qualification(rows, ii.qualify_case({"vibeqc": candidate, "pyscf": reference}, "pyscf"))
    assert rows[0].status == "error" and rows[0].input_qualification == "mismatch"
    assert "lattice metric differs" in rows[0].note
    assert rows[1].input_qualification == "reference"
    assert paired_wall_ratio(rows[0], rows[1]) is None


def test_unverified_inputs_keep_the_energy_verdict_but_not_timing():
    rows = _rows()
    apply_input_qualification(rows, ii.qualify_case({"vibeqc": _vibeqc(LIH), "pyscf": None}, "pyscf"))
    assert rows[0].status == "pass" and rows[0].input_qualification == "unverified"
    assert "no identity recorded" in rows[0].note
    assert paired_wall_ratio(rows[0], rows[1]) is None
    qualified = _rows()
    identity = _vibeqc(LIH)
    apply_input_qualification(qualified, ii.qualify_case({"vibeqc": identity, "pyscf": identity}, "pyscf"))
    assert qualified[0].input_qualification == "unverified"
    assert paired_wall_ratio(qualified[0], qualified[1]) is None


def test_suite_records_the_case_qualification(tmp_path, monkeypatch):
    identity = _vibeqc(LIH)

    def fake_call(**kw):
        if kw["code"] == "vibeqc":
            (kw["artifact_dir"] / "input_identity.json").write_text(json.dumps(identity))
        return CodeRow("t", "dev", LIH.id, LIH.family, "sto-3g", "rhf", "1x1x1", kw["code"], "t",
                       status="pending", energy_ha=-1.0, converged=True)

    monkeypatch.setattr(run_suite, "_isolated_runner_call", fake_call)
    case = run_suite.run_one_periodic_case_dev(
        run_id="t", system_id=LIH.id, basis="sto-3g", method_id="rhf", target="dev", env={},
        run_dir=tmp_path, kmesh=(1, 1, 1), tolerance_ha=1e-6)
    record = json.loads((tmp_path / "cases" / case.label / "input_qualification.json").read_text())
    assert record["pairs"]["vibeqc"]["status"] == "unverified"
    assert [r.input_qualification for r in case.rows] == ["unverified", "reference"]


CRYSTAL_OUT = (run_suite.REGRESSION_ROOT / "crystal_parity" / "crystal_demos"
               / "mgo_sto3g_enecycle.out")
_H = 4.21 / 2
MGO_PRIMITIVE = PeriodicSpec(
    id="mgo_primitive", family="synthetic", space_group="Fm-3m",
    lattice_ang=((0.0, _H, _H), (_H, 0.0, _H), (_H, _H, 0.0)),
    atoms=(AtomFrac("Mg", 12, (0.0, 0.0, 0.0)), AtomFrac("O", 8, (0.5, 0.5, 0.5))),
)
HF = {"kind": "hf", "hf_fraction": 1.0, "omega": 0.0}


def _crystal_output():
    return {**ii.crystal_output_identity(CRYSTAL_OUT.read_text()), "xc": HF}


def test_crystal_output_reports_geometry_basis_and_no_ecp():
    out = _crystal_output()
    assert not ii._compare_geometry(out["geometry"], _requested_geometry(MGO_PRIMITIVE))
    shells = out["basis"]["shells_by_element"]
    assert [s[0] for s in shells["12"]] == [0, 0, 0, 1, 1]
    assert [s[0] for s in shells["8"]] == [0, 0, 1]
    assert out["basis"]["precision"].startswith("CRYSTAL printout")
    assert out["ecp"]["core_electrons"] == {"0": 0, "1": 0}
    assert out["ecp"]["scalar_terms"] == {"0": [], "1": []}


def test_crystal_printed_basis_matches_the_same_basis_at_printed_precision():
    pair = ii.qualify_pair(_vibeqc(MGO_PRIMITIVE, "sto-3g"), _crystal_output())
    assert pair["status"] == "unverified", pair["fields"]
    assert "four significant digits" in pair["fields"]["basis"]["precision"]
    other = ii.qualify_pair(_vibeqc(MGO_PRIMITIVE, "3-21g"), _crystal_output())
    assert other["fields"]["basis"]["status"] == "mismatch"


@pytest.mark.parametrize("scale, status", [(1.01, "mismatch"), (1.00001, "matched")])
def test_printed_precision_bounds_what_a_crystal_basis_can_establish(scale, status):
    mine = _vibeqc(MGO_PRIMITIVE, "sto-3g")
    for shell in mine["basis"]["shells_by_atom"]["0"]:
        shell[2] = [e * scale for e in shell[2]]
    assert ii.qualify_pair(mine, _crystal_output())["fields"]["basis"]["status"] == status


def test_suite_prefers_the_fetched_crystal_output_over_the_deck(tmp_path, monkeypatch):
    def fake_call(**kw):
        if kw["code"] == "vibeqc":
            (kw["artifact_dir"] / "input_identity.json").write_text(json.dumps(_vibeqc(LIH)))
        return CodeRow("t", "dev", LIH.id, LIH.family, "sto-3g", "rhf", "1x1x1", kw["code"],
                       "t", status="pending", energy_ha=-1.0, converged=True)

    def fake_crystal(**kw):
        workdir = kw["workdir"] / f"{LIH.id}__sto-3g__rhf"
        (workdir / "fetched" / "job").mkdir(parents=True)
        (workdir / "INPUT.d12").write_text(_deck(LIH))
        (workdir / "fetched" / "job" / "INPUT.out").write_text(CRYSTAL_OUT.read_text())
        return CodeRow("t", "dev", LIH.id, LIH.family, "sto-3g", "rhf", "1x1x1", "crystal",
                       "t", status="pending", energy_ha=-1.0, converged=True)

    monkeypatch.setattr(run_suite, "_isolated_runner_call", fake_call)
    monkeypatch.setattr(runner_crystal, "run_periodic_case", fake_crystal)
    case = run_suite.run_one_periodic_case_dev(
        run_id="t", system_id=LIH.id, basis="sto-3g", method_id="rhf", target="dev", env={},
        run_dir=tmp_path, kmesh=(1, 1, 1), tolerance_ha=1e-6, include_crystal=True,
        reference_code="crystal", reference_version="CRYSTAL23")
    record = json.loads((tmp_path / "cases" / case.label / "input_qualification.json").read_text())
    fields = record["pairs"]["vibeqc"]["fields"]
    # The deck says LiH; the output (an MgO run) says otherwise and wins.
    assert fields["geometry"]["status"] == "mismatch"
    assert fields["basis"]["status"] == "mismatch"
    vibeqc_row = next(r for r in case.rows if r.code == "vibeqc")
    assert vibeqc_row.input_qualification == "mismatch"


@pytest.mark.parametrize("field, value, reason", [
    (0, 1, "channels"), (1, 1, "radial powers"),
    (2, 1.01, "exponents"), (3, 2.01, "coefficients"),
])
def test_same_core_count_different_ecp_terms_mismatch(field, value, reason):
    original = [[-1, -1, 1.0, 2.0], [0, 0, 0.5, -0.2]]
    modified = copy.deepcopy(original)
    modified[0][field] = value
    a = ii.ecp_identity({"0": 10}, {"0": original})
    b = ii.ecp_identity({"0": 10}, {"0": modified})
    result = ii._qualify_ecp(a, b)
    assert result["status"] == "mismatch" and reason in result["reasons"][0]


def test_ecp_identity_combines_identical_terms_without_radial_normalization():
    a = ii.ecp_identity({"0": 0}, {"0": [[-1, 0, 0.5, 2.0], [-1, 0, 0.5, -0.5],
                                         [0, -2, 1.0, 0.0]]})
    b = ii.ecp_identity({"0": 0}, {"0": [[-1, 0, 0.5, 1.5]]})
    assert ii._qualify_ecp(a, b)["status"] == "matched"
    assert a["scalar_terms"] == {"0": [[-1, 0, 0.5, 1.5]]}
    ae = ii.ecp_identity({"0": 0}, {"0": []})
    assert ii._qualify_ecp(a, ae)["status"] == "mismatch"
    assert ii._qualify_ecp({"0": 0}, {"0": 0})["status"] == "unverified"
    assert ii._qualify_ecp({"0": 0}, {"0": 10})["status"] == "mismatch"
    different_core = ii.ecp_identity({"0": 2}, {"0": [[-1, 0, 0.5, 1.5]]})
    assert ii._qualify_ecp(a, different_core)["reasons"] == ["ECP core electrons differ"]


@pytest.mark.parametrize("term", [[-2, 0, 1.0, 1.0], [-1, -3, 1.0, 1.0],
                                 [-1, 0, float("nan"), 1.0], [-1, 0, 0.0, 1.0],
                                 [-1, 0, 1.0, float("inf")]])
def test_invalid_ecp_primitives_do_not_qualify(term):
    with pytest.raises(ValueError, match="invalid scalar ECP"):
        ii.ecp_identity({"0": 0}, {"0": [term]})


def test_production_resolver_zero_core_potential_is_recorded(monkeypatch):
    from types import SimpleNamespace
    from vibeqc import periodic_runner
    system = runner_vibeqc._build_periodic_system(HEX)
    atoms = list(system.unit_cell)
    blocks = [SimpleNamespace(ams=[0, 1], ns=[1, 2], exponents=[0.5, 0.7],
                              coefficients=[-0.2, 0.3], n_primitive=2)]
    centers = [list(atoms[0].xyz)]
    monkeypatch.setattr(periodic_runner, "_resolve_ecp_data",
                        lambda *_: (blocks, centers, [1.0, 1.0], 0))
    record = runner_vibeqc._resolved_ecp_identity(system, None)
    assert record["core_electrons"] == {"0": 0, "1": 0}
    assert record["scalar_terms"] == {"0": [[-1, 0, 0.7, 0.3], [0, -1, 0.5, -0.2]], "1": []}
    centers[0][0] += 0.1
    assert "center-to-atom" in runner_vibeqc._resolved_ecp_identity(system, None)["unverified"]


@needs_pyscf
def test_reference_resolves_zero_core_and_refuses_spin_orbit_terms(tmp_path):
    # Use the actual worker in a separate interpreter, preserving the reference boundary.
    import subprocess
    script = runner_pyscf._PYSCF_EXTERNAL_SCRIPT.replace('if __name__ == "__main__":', 'if False:')
    script += r'''
from pyscf import gto
mol = gto.M(atom="H1 0 0 0; H2 0 0 1", basis="sto-3g",
            ecp={"H1": [0, [[-1, [[], [], [[0.7, 0.3]]]]]]}, verbose=0)
result = scalar_ecp_identity(mol)
from pyscf.gto import mole
mol._ecpbas[0, mole.SO_TYPE_OF] = 1
print(json.dumps({"scalar": result, "so": scalar_ecp_identity(mol)}))
'''
    result = subprocess.run([runner_pyscf._pyscf_python(), "-c", script],
                            capture_output=True, text=True, check=True)
    records = json.loads(result.stdout.strip().splitlines()[-1])
    scalar = records["scalar"]
    assert scalar["core_electrons"] == {"0": 0, "1": 0}
    assert scalar["scalar_terms"] == {"0": [[-1, 0, 0.7, 0.3]], "1": []}
    assert "spin-orbit" in records["so"]["unverified"]


def test_crystal_scalar_input_preserves_actual_radial_power_and_effective_charge():
    operator = ii.crystal_scalar_ecp_input("INPUT\n7 1 1 0 0 0 0\n2.0 3.0 -1\n4.0 -5.0 2", 17)
    assert operator["core_electrons"] == {"0": 10}
    assert operator["scalar_terms"]["0"] == [[-1, -1, 2.0, 3.0], [0, 2, 4.0, -5.0]]
    assert operator["evidence"] == "submitted"
    pair = ii.qualify_pair({"ecp": operator}, {"ecp": operator})
    assert pair["fields"]["ecp"]["status"] == "unverified"
    changed = copy.deepcopy(operator)
    changed["scalar_terms"]["0"][1][-1] += 1
    assert ii.qualify_pair({"ecp": operator}, {"ecp": changed})["fields"]["ecp"]["status"] == "mismatch"


@pytest.mark.parametrize("block", [
    "HAYWSC", "INPSOC", "INPUT\n7 1 0 0 0 0 0", "INPUT\n18 0 0 0 0 0 0",
    "INPUT\n7 -1 0 0 0 0 0", "INPUT\n7 1 0 0 0 0 0\n-2 1 0",
    "INPUT\n7 1 0 0 0 0 0\n2 nan 0", "INPUT\n7 1 0 0 0 0 0\n2 1 0.5",
])
def test_crystal_scalar_input_refuses_unresolved_or_invalid_operators(block):
    with pytest.raises(ValueError):
        ii.crystal_scalar_ecp_input(block, 17)


def test_crystal_explicit_basis_deck_records_requested_ecp_without_execution_claim():
    deck = """case
CRYSTAL
0 0 0
1
5 5 5 90 90 90
1
217 0 0 0
ENDG
217 1
INPUT
7 1 1 0 0 0 0
2 3 -1
4 -5 2
0 0 1 1 1
1.0 1.0
99 0
END
"""
    record = ii.crystal_deck_identity(deck)
    assert record["ecp"]["core_electrons"] == {"0": 10}
    assert record["ecp"]["scalar_terms"]["0"][1] == [0, 2, 4.0, -5.0]
    assert record["ecp"]["evidence"] == "submitted"
    unknown = ii.crystal_deck_identity(deck.replace("INPUT", "INPSOC"))
    assert "unverified" in unknown["ecp"]


def _observed_conventions():
    return {
        "domain": {"evidence": "resolved", "source": "observed runtime definition", "definition": {
            "dimension": 3, "periodic_axes": [True, True, True],
            "normalization": "per_primitive_cell", "charge_background": "none"}},
        "integration": {"evidence": "resolved", "source": "observed grid arrays", "definition": {
            "points_sha256": "a" * 64, "weights_sha256": "b" * 64,
            "coordinates": "cartesian_bohr", "domain": "primitive_cell", "partition": "partition-v1"}},
        "electrostatics": {"evidence": "resolved", "source": "observed kernel definitions", "definition": {
            "coulomb_kernel": "periodic_1/r", "exchange_kernel": "periodic_1/r",
            "g0_coulomb": "removed", "g0_exchange": "ewald_probe_correction",
            "boundary": "3d_tinfoil"}},
    }


@pytest.mark.parametrize("name,key,value", [
    ("domain", "charge_background", "uniform_neutralizing"),
    ("domain", "normalization", "per_molecule"),
    ("integration", "points_sha256", "c" * 64),
    ("integration", "weights_sha256", "c" * 64),
    ("integration", "domain", "all_space"),
    ("electrostatics", "g0_coulomb", "retained"),
    ("electrostatics", "g0_exchange", "none"),
])
def test_resolved_convention_comparison_refuses_different_definitions(name, key, value):
    original = _observed_conventions()
    changed = copy.deepcopy(original)
    changed[name]["definition"][key] = value
    pair = ii.qualify_pair(original, changed)
    assert pair["fields"][name]["status"] == "mismatch"
    assert pair["status"] == "mismatch" and not pair["full_physical_equivalence"]


@pytest.mark.parametrize("name", ["domain", "integration", "electrostatics"])
def test_requested_convention_aliases_cannot_qualify(name):
    original = _observed_conventions()
    assert ii.qualify_pair(original, original)["fields"][name]["status"] == "matched"
    original[name]["evidence"] = "requested"
    assert ii.qualify_pair(original, original)["fields"][name]["status"] == "unverified"
    original[name]["evidence"] = "resolved"
    original[name]["definition"].pop(next(iter(original[name]["definition"])))
    assert ii.qualify_pair(original, original)["fields"][name]["status"] == "unverified"


def test_invalid_resolved_domain_and_grid_refuse_instead_of_matching():
    original = _observed_conventions()
    original["domain"]["definition"]["dimension"] = 2
    original["integration"]["definition"]["points_sha256"] = "XLGRID"
    pair = ii.qualify_pair(original, original)
    assert pair["fields"]["domain"]["status"] == "unverified"
    assert pair["fields"]["integration"]["status"] == "unverified"
    assert pair["status"] == "unverified" and pair["not_compared"]


def test_crystal_output_matching_basis_does_not_invent_grid_or_zero_mode():
    actual = ii.crystal_output_identity(CRYSTAL_OUT.read_text())
    actual["xc"] = {"kind": "hf", "hf_fraction": 1.0, "omega": 0.0}
    pair = ii.qualify_pair(actual, actual)
    assert pair["fields"]["ecp"]["status"] == "matched"
    assert pair["fields"]["basis"]["status"] == "matched"
    assert pair["status"] == "unverified"
    assert all(pair["fields"][name]["status"] == "unverified"
               for name in ("domain", "integration", "electrostatics"))
    assert not pair["full_physical_equivalence"]
