"""Periodic charge/spin requests in the regression comparison suite (#647).

The state is per primitive cell and the same on every Born-von Karman cell.
These checks cover the schema, the matrix, transport across process
isolation, refusals before execution, the recorded counts and the PySCF
per-cell ``nelec`` pin. They are not a scientific acceptance of any state.
"""
from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from examples.regression import run_suite
from examples.regression.core import _isolated_runner, runner_pyscf, runner_vibeqc
from examples.regression.core.case import CaseRecord, CodeRow
from examples.regression.core.compare import _case_mismatch
from examples.regression.core.periodic_matrix import (
    apply_state_overrides, load_periodic_matrix, periodic_state_request, state_tag,
)
from examples.regression.methods.catalog import METHODS
from examples.regression.systems.periodic.h2_box12 import SPEC as H2
from examples.regression.systems.periodic.ne_fcc import SPEC as NE_FCC


def _row(**changes):
    return {**dict(system_id=H2.id, basis="sto-3g", method_id="uhf", kmesh=[1, 1, 1],
                   tolerance_ha=1e-4, tolerance_rationale="Synthetic test threshold",
                   reference_code="pyscf"), **changes}


def _matrix(tmp_path, rows):
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(dict(schema_version=1, jk_method="bipole", cases=rows)))
    return path


def test_electron_counts_are_per_cell_and_check_parity():
    assert H2.electron_counts() == (1, 1)
    assert replace(H2, multiplicity=3).electron_counts() == (2, 0)
    for mult in (2, 5):
        with pytest.raises(ValueError, match="cannot have multiplicity"):
            replace(H2, multiplicity=mult).electron_counts()


@pytest.mark.parametrize("changes", [
    {"atomic_spins": (1,)}, {"atomic_spins": (1, 2)}, {"atomic_spins": (True, 0)},
    {"magnetic_moments": (1.0,)}, {"multiplicity": 0}, {"multiplicity": 2.0},
    {"charge": True}, {"electrostatic_background": "isolated-ion"},
    {"electrostatic_background": []},
])
def test_spec_rejects_malformed_state_fields(changes):
    with pytest.raises(ValueError):
        replace(H2, **changes)


def test_resolved_state_records_counts_and_convention():
    state = periodic_state_request(replace(H2, multiplicity=3), METHODS["uhf"], code="pyscf")
    assert (state["n_alpha"], state["n_beta"], state["multiplicity"]) == (2, 0, 3)
    assert "Born-von Karman" in state["convention"]
    sad = replace(H2, default_initial_guess="SAD", atomic_spins=(1, -1))
    assert periodic_state_request(sad, METHODS["uhf"], code="vibeqc")["atomic_spins"] == [1, -1]


@pytest.mark.parametrize("changes, method, code, match", [
    ({"charge": 2}, "uhf", "vibeqc", "neutralizing-background"),
    ({"magnetic_moments": (1.0, 0.0)}, "uhf", "vibeqc", "magnetic_moments"),
    ({"multiplicity": 3}, "rhf", "vibeqc", "unrestricted"),
    ({"multiplicity": 3}, "rks-pbe", "pyscf", "unrestricted"),
    ({"default_initial_guess": "SAD", "atomic_spins": (1, -1)}, "rhf", "vibeqc", "unrestricted"),
    ({"is_open_shell": True}, "uhf", "vibeqc", "polarized"),
    ({"atomic_spins": (1, -1)}, "uhf", "vibeqc", "SAD"),
    ({"default_initial_guess": "SAD", "atomic_spins": (1, -1)}, "uhf", "pyscf", "atomic-spin"),
    ({"multiplicity": 2}, "uhf", "vibeqc", "cannot have multiplicity"),
])
def test_unsupported_states_are_refused_before_execution(changes, method, code, match):
    with pytest.raises(ValueError, match=match):
        periodic_state_request(replace(H2, **changes), METHODS[method], code=code)


def test_matrix_rows_carry_distinct_states(tmp_path):
    rows = [_row(method_id="rhf"), _row(multiplicity=1), _row(multiplicity=3),
            _row(kmesh=[2, 1, 1], multiplicity=3)]
    _, cases = load_periodic_matrix(_matrix(tmp_path, rows), load_system=run_suite._load_periodic_spec)
    assert [c.state() for c in cases] == [{}, {"multiplicity": 1}, {"multiplicity": 3},
                                          {"multiplicity": 3}]
    assert [state_tag(c.state()) for c in cases] == ["", "mult1", "mult3", "mult3"]


@pytest.mark.parametrize("rows", [
    [_row(method_id="rhf", multiplicity=3)],
    [_row(charge=1, multiplicity=2)],
    [_row(multiplicity=2)],
    [_row(atomic_spins=[1])],
    [_row(atomic_spins="ud")],
    [_row(multiplicity=3), _row(multiplicity=3)],
])
def test_matrix_refuses_impossible_or_duplicate_states(tmp_path, rows):
    with pytest.raises(ValueError):
        load_periodic_matrix(_matrix(tmp_path, rows), load_system=run_suite._load_periodic_spec)


def test_shipped_open_shell_matrix_pairs_limits_and_triplets():
    path = run_suite.REGRESSION_ROOT / "matrices/open_shell_starter.json"
    route, cases = load_periodic_matrix(path, load_system=run_suite._load_periodic_spec)
    assert route == "bipole"
    states = {(c.method_id, c.kmesh, c.multiplicity) for c in cases}
    for mesh in ((1, 1, 1), (2, 1, 1)):
        assert {("rhf", mesh, None), ("uhf", mesh, 1), ("uhf", mesh, 3)} <= states


def test_state_survives_isolation_and_inherited_state_cannot_leak(tmp_path, monkeypatch):
    captured = {}
    row = CodeRow("t", "dev", H2.id, H2.family, "sto-3g", "uhf", "1x1x1", "vibeqc", "t")

    def fake_process(*args, **kw):
        captured["env"] = kw["env"]
        return SimpleNamespace(returncode=0, stdout=run_suite._RESULT_MARKER
                               + json.dumps(row.__dict__) + "\n", stderr="")

    monkeypatch.setenv("VIBEQC_REGRESSION_PERIODIC_STATE", json.dumps({"multiplicity": 5}))
    monkeypatch.setattr(run_suite.subprocess, "run", fake_process)
    for spec, multiplicity in ((H2, 1), (replace(H2, multiplicity=3), 3)):
        run_suite._isolated_runner_call(
            code="vibeqc", spec_kind="periodic", spec=spec, basis="sto-3g",
            method=METHODS["uhf"], log_path=tmp_path / "case.log", workdir=None,
            artifact_dir=None, run_id="t", target="dev", code_version="t")
        sent = json.loads(captured["env"]["VIBEQC_REGRESSION_PERIODIC_STATE"])
        assert sent == {"charge": 0, "multiplicity": multiplicity, "atomic_spins": None,
                        "electrostatic_background": None}

    monkeypatch.setenv("VIBEQC_REGRESSION_PERIODIC_STATE", captured["env"]["VIBEQC_REGRESSION_PERIODIC_STATE"])
    monkeypatch.setattr(runner_vibeqc, "run_periodic_case",
                        lambda **kw: captured.setdefault("spec", kw["spec"]) and row)
    _isolated_runner._run_periodic("vibeqc", H2, "sto-3g", METHODS["uhf"],
                                   tmp_path / "case.log", "t", "dev", "t")
    assert captured["spec"].multiplicity == 3


def test_suite_refuses_state_before_any_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(run_suite, "_isolated_runner_call",
                        lambda **kw: pytest.fail("runner started for a refused state"))
    with pytest.raises(ValueError, match="unrestricted"):
        run_suite.run_one_periodic_case_dev(
            run_id="t", system_id=H2.id, basis="sto-3g", method_id="rhf",
            target="dev", env={}, run_dir=tmp_path, kmesh=(1, 1, 1),
            tolerance_ha=1e-4, state={"multiplicity": 3})


def test_suite_case_label_and_rows_carry_the_state(tmp_path, monkeypatch):
    seen = []

    def fake_call(**kw):
        seen.append((kw["code"], kw["spec"].multiplicity))
        return CodeRow("t", "dev", H2.id, H2.family, "sto-3g", "uhf", "1x1x1",
                       kw["code"], "t", status="unavailable")

    monkeypatch.setattr(run_suite, "_isolated_runner_call", fake_call)
    case = run_suite.run_one_periodic_case_dev(
        run_id="t", system_id=H2.id, basis="sto-3g", method_id="uhf", target="dev",
        env={}, run_dir=tmp_path, kmesh=(1, 1, 1), tolerance_ha=1e-4,
        state={"multiplicity": 3})
    assert case.label.endswith("__mult3")
    assert seen == [("vibeqc", 3), ("pyscf", 3)]


def test_pyscf_payload_carries_per_cell_counts(tmp_path, monkeypatch):
    sent = {}

    def fake_external(payload, log_path, artifact_dir=None):
        sent.update(payload)
        return {"status": "ok", "energy_ha": -0.5, "converged": True,
                "nelec_per_cell": payload["nelec_per_cell"]}

    monkeypatch.setattr(runner_pyscf, "_run_external_pyscf", fake_external)
    row = runner_pyscf.run_periodic_case(
        run_id="t", target="dev", spec=replace(H2, multiplicity=3), basis_name="sto-3g",
        method=METHODS["uhf"], kmesh=(2, 1, 1), log_path=tmp_path / "case.log")
    assert sent["multiplicity"] == 3 and sent["nelec_per_cell"] == [2, 0]
    assert (row.n_alpha, row.n_beta) == (2, 0)


def test_runners_report_refused_states_as_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_pyscf, "_run_external_pyscf",
                        lambda *a, **k: pytest.fail("PySCF started for a refused state"))
    spins = replace(H2, default_initial_guess="SAD", atomic_spins=(1, -1))
    row = runner_pyscf.run_periodic_case(
        run_id="t", target="dev", spec=spins, basis_name="sto-3g",
        method=METHODS["uhf"], kmesh=(1, 1, 1), log_path=tmp_path / "p.log")
    assert row.status == "unavailable" and "atomic-spin" in row.note
    row = runner_vibeqc.run_periodic_case(
        run_id="t", target="dev", code_version="t", spec=replace(H2, multiplicity=3),
        basis_name="sto-3g", method=METHODS["rhf"], kmesh=(1, 1, 1),
        log_path=tmp_path / "v.log")
    assert row.status == "unavailable" and "unrestricted" in row.note


def _counted(code, n_alpha, n_beta):
    return CodeRow("t", "dev", H2.id, H2.family, "sto-3g", "uhf", "1x1x1", code, "t",
                   n_electrons=2, n_alpha=n_alpha, n_beta=n_beta)


def test_comparison_requires_matching_confirmed_spin_counts():
    triplet = _counted("vibeqc", 2, 0)
    assert not _case_mismatch(triplet, _counted("pyscf", 2, 0))
    assert _case_mismatch(triplet, _counted("pyscf", 1, 1))
    assert _case_mismatch(triplet, _counted("crystal", None, None))
    assert not _case_mismatch(_counted("vibeqc", 1, 1), _counted("crystal", None, None))


def test_legacy_case_identity_is_unchanged():
    assert CaseRecord(NE_FCC.id, NE_FCC.family, "sto-3g", "rhf", (1, 1, 1)).label == \
        "ne_fcc__sto-3g__rhf__1x1x1"
    assert apply_state_overrides(NE_FCC, {}) == NE_FCC


@pytest.mark.skipif(importlib.util.find_spec("pyscf") is None, reason="PySCF not installed")
def test_pyscf_multik_triplet_is_the_uniform_per_cell_state(tmp_path):
    """Without the nelec pin, cell.spin alone fills (3, 1) over the 2-cell
    supercell, 142.9 mHa below this state (measured 2026-10-22)."""
    row = runner_pyscf.run_periodic_case(
        run_id="t", target="dev", spec=replace(H2, multiplicity=3), basis_name="sto-3g",
        method=METHODS["uhf"], kmesh=(2, 1, 1), log_path=tmp_path / "case.log")
    assert row.converged and (row.n_alpha, row.n_beta) == (2, 0)
    # E_PYSCF_H2_KUHF_TRIPLET_211 in tests/test_bipole_fock_ewald_exchange.py
    assert row.energy_ha == pytest.approx(-0.5337291457, abs=1e-8)


@pytest.mark.parametrize("method", ["uhf", "uks-pbe"])
def test_charged_state_requires_explicit_supported_route(method):
    spec = replace(H2, charge=1, multiplicity=2,
                   electrostatic_background="uniform-jellium")
    for code in ("vibeqc", "pyscf"):
        state = periodic_state_request(spec, METHODS[method], code=code, jk_method="bipole")
        assert (state["charge"], state["n_alpha"], state["n_beta"]) == (1, 1, 0)
        assert state["electrostatic_background"] == "uniform-jellium"
    for code, route in (("vibeqc", "gdf"), ("vibeqc", None),
                        ("crystal", "bipole"), ("cp2k", "bipole")):
        with pytest.raises(ValueError, match="BIPOLE versus PySCF"):
            periodic_state_request(spec, METHODS[method], code=code, jk_method=route)
    with pytest.raises(ValueError, match="restricted per-cell occupations"):
        periodic_state_request(replace(spec, charge=2, multiplicity=1), METHODS["rhf"],
                               code="pyscf")


def test_matrix_carries_charged_convention_and_refuses_other_routes(tmp_path):
    path = _matrix(tmp_path, [_row(charge=1, multiplicity=2,
                                  electrostatic_background="uniform-jellium")])
    route, cases = load_periodic_matrix(path, load_system=run_suite._load_periodic_spec)
    assert route == "bipole"
    assert cases[0].tag() == "qp1_mult2_bgjellium"
    spec = apply_state_overrides(H2, cases[0].state())
    assert spec.charge == 1 and spec.electrostatic_background == "uniform-jellium"
    # A neutral isolated request must clear a background inherited in its base spec.
    assert apply_state_overrides(spec, {"charge": 0, "multiplicity": 1,
                                       "electrostatic_background": None}).electrostatic_background is None
    data = json.loads(path.read_text())
    data["jk_method"] = "gdf"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="BIPOLE versus PySCF"):
        load_periodic_matrix(path, load_system=run_suite._load_periodic_spec)


def test_charged_suite_dispatch_preserves_background(tmp_path, monkeypatch):
    seen = []
    def fake_call(**kw):
        seen.append((kw["spec"].charge, kw["spec"].electrostatic_background))
        return CodeRow("t", "dev", H2.id, H2.family, "sto-3g", "uhf", "2x1x1",
                       kw["code"], "t", status="unavailable")
    monkeypatch.setattr(run_suite, "_isolated_runner_call", fake_call)
    case = run_suite.run_one_periodic_case_dev(
        run_id="t", system_id=H2.id, basis="sto-3g", method_id="uhf", target="dev",
        env={}, run_dir=tmp_path, kmesh=(2, 1, 1), tolerance_ha=1e-4,
        periodic_jk_method="bipole",
        state={"charge": 1, "multiplicity": 2, "electrostatic_background": "uniform-jellium"})
    assert case.label.endswith("__qp1_mult2_bgjellium")
    assert seen == [(1, "uniform-jellium"), (1, "uniform-jellium")]


@pytest.mark.parametrize("changes", [
    {"charge": None}, {"charge": 0}, {"charge": -1},
    {"electrostatic_background": None}, {"electrostatic_background": "isolated-ion"},
    {"n_alpha": None, "n_beta": None},
])
def test_charged_comparison_rejects_missing_or_contradictory_state(changes):
    row = replace(_counted("vibeqc", 1, 0), n_electrons=1, charge=1,
                  electrostatic_background="uniform-jellium")
    ref = replace(row, code="pyscf")
    assert not _case_mismatch(row, ref)
    assert _case_mismatch(row, replace(ref, **changes))
    assert _case_mismatch(replace(ref, **changes), row)


def test_charged_payload_and_confirmed_result_state(tmp_path, monkeypatch):
    spec = replace(H2, charge=1, multiplicity=2,
                   electrostatic_background="uniform-jellium")
    def fake_external(payload, log_path, artifact_dir=None):
        assert payload["charge"] == 1
        assert payload["electrostatic_background"] == "uniform-jellium"
        assert payload["nelec_per_cell"] == [1, 0]
        return {"status": "ok", "charge": 1, "electrostatic_background": "uniform-jellium",
                "energy_ha": -0.5, "converged": True, "nelec_per_cell": [1, 0]}
    monkeypatch.setattr(runner_pyscf, "_run_external_pyscf", fake_external)
    row = runner_pyscf.run_periodic_case(
        run_id="t", target="dev", spec=spec, basis_name="sto-3g", method=METHODS["uhf"],
        kmesh=(2, 1, 1), log_path=tmp_path / "case.log")
    assert row.charge == 1 and row.electrostatic_background == "uniform-jellium"
    assert (row.n_alpha, row.n_beta) == (1, 0)


@pytest.mark.skipif(importlib.util.find_spec("pyscf") is None, reason="PySCF not installed")
@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1)])
def test_pyscf_charged_helium_retains_per_cell_charge_and_spin(tmp_path, mesh):
    from vibeqc_spec import AtomFrac
    # One occupied alpha AO and an empty beta channel. On two cells, PySCF's
    # default total_electrons/spin would instead give (2, 1), not (2, 0).
    ang = 8 * 0.529177210903
    spec = replace(H2, id="he_ion", atoms=(AtomFrac("He", 2, (0., 0., 0.)),),
                   lattice_ang=((ang, 0., 0.), (0., ang, 0.), (0., 0., ang)),
                   charge=1, multiplicity=2, electrostatic_background="uniform-jellium")
    row = runner_pyscf.run_periodic_case(
        run_id="t", target="dev", spec=spec, basis_name="sto-3g", method=METHODS["uhf"],
        kmesh=mesh, log_path=tmp_path / "case.log", artifact_dir=tmp_path / "artifacts")
    assert row.converged and row.n_electrons == 1
    assert row.charge == 1 and row.electrostatic_background == "uniform-jellium"
    assert (row.n_alpha, row.n_beta) == (1, 0)
    # Independent AFTDF 101^3 reference (#689). GDF is a different J/K
    # approximation: retain the suite's 0.1 mHa comparison tolerance.
    reference = {(1, 1, 1): -2.117738604293459, (2, 1, 1): -2.1155674148922383}
    assert row.energy_ha == pytest.approx(reference[mesh], abs=1e-4)


@pytest.mark.parametrize("changes", [
    {"electrostatic_background": None}, {"dimension": 2},
    {"method": {"scf": "rhf"}}, {"nelec_per_cell": None}, {"charge": True},
])
def test_external_worker_rejects_unsupported_charged_payload_before_cell(changes, monkeypatch):
    pytest.importorskip("pyscf")
    from pyscf.pbc import gto
    namespace = {"__name__": "worker_test"}
    exec(runner_pyscf._PYSCF_EXTERNAL_SCRIPT, namespace)
    monkeypatch.setattr(gto, "M", lambda **kw: pytest.fail("cell built for invalid payload"))
    payload = {"charge": 1, "electrostatic_background": "uniform-jellium",
               "method": {"scf": "uhf"}, "nelec_per_cell": [1, 0], **changes}
    with pytest.raises(ValueError, match="charge"):
        namespace["run_periodic"](payload)


def test_charged_vibeqc_request_reaches_public_driver(tmp_path, monkeypatch):
    captured = {}
    def fake_job(system, basis, **kw):
        captured["state"] = (system.charge, system.multiplicity, system.n_electrons())
        captured.update(kw)
        return SimpleNamespace(energy=-1., converged=True, n_iter=2, backend="pbc-bipole")
    monkeypatch.setattr(runner_vibeqc.vq, "run_periodic_job", fake_job)
    spec = replace(H2, charge=1, multiplicity=2, electrostatic_background="uniform-jellium")
    row = runner_vibeqc.run_periodic_case(
        run_id="t", target="dev", code_version="t", spec=spec, basis_name="sto-3g",
        method=METHODS["uhf"], kmesh=(2, 1, 1), jk_method="bipole",
        log_path=tmp_path / "case.log", artifact_dir=tmp_path / "artifacts")
    assert row.converged and captured["state"] == (1, 2, 1)
    assert captured["jk_method"] == "bipole" and captured["kpoints"] == (2, 1, 1)
    assert row.charge == 1 and row.electrostatic_background == "uniform-jellium"
    recorded = json.loads((tmp_path / "artifacts/input.json").read_text())
    assert recorded["state"]["electrostatic_background"] == "uniform-jellium"


def test_charged_state_serialization_survives_process_boundary(tmp_path, monkeypatch):
    captured = {}
    spec = replace(H2, charge=1, multiplicity=2, electrostatic_background="uniform-jellium")
    row = replace(_counted("vibeqc", 1, 0), charge=1, electrostatic_background="uniform-jellium")
    def fake_process(*args, **kw):
        captured.update(kw["env"])
        return SimpleNamespace(returncode=0, stdout=run_suite._RESULT_MARKER
                               + json.dumps(row.__dict__) + "\n", stderr="")
    monkeypatch.setattr(run_suite.subprocess, "run", fake_process)
    run_suite._isolated_runner_call(
        code="vibeqc", spec_kind="periodic", spec=spec, basis="sto-3g",
        method=METHODS["uhf"], log_path=tmp_path / "case.log", workdir=None,
        artifact_dir=None, run_id="t", target="dev", code_version="t")
    sent = captured["VIBEQC_REGRESSION_PERIODIC_STATE"]
    assert json.loads(sent)["electrostatic_background"] == "uniform-jellium"
    monkeypatch.setenv("VIBEQC_REGRESSION_PERIODIC_STATE", sent)
    monkeypatch.setenv("VIBEQC_REGRESSION_PERIODIC_JK_METHOD", "bipole")
    def receive(**kw):
        assert (kw["spec"].charge, kw["spec"].multiplicity) == (1, 2)
        assert kw["spec"].electrostatic_background == "uniform-jellium"
        return row
    monkeypatch.setattr(runner_vibeqc, "run_periodic_case", receive)
    _isolated_runner._run_periodic("vibeqc", H2, "sto-3g", METHODS["uhf"],
                                   tmp_path / "case.log", "t", "dev", "t")


def _cl2_ecp_spec():
    from vibeqc_spec import AtomFrac
    a = 20 * 0.529177210903
    return replace(H2, id="cl2_ecp", atoms=(
        AtomFrac("Cl", 17, (0., 0., -0.995 / a)),
        AtomFrac("Cl", 17, (0., 0., 0.995 / a))),
        lattice_ang=((a, 0., 0.), (0., a, 0.), (0., 0., a)),
        default_cutoff_bohr=12., default_nuclear_cutoff_bohr=20.,
        default_omega=0.39987715934900675, default_damping=0.)


def test_ecp_comparison_row_uses_valence_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_vibeqc.vq, "run_periodic_job", lambda *a, **kw:
                        SimpleNamespace(energy=-29., converged=True, n_iter=2,
                                        ecp_total_ncore=20, backend="pbc-bipole"))
    row = runner_vibeqc.run_periodic_case(
        run_id="t", target="dev", code_version="t", spec=_cl2_ecp_spec(),
        basis_name="lanl2dz", method=METHODS["uhf"], kmesh=(1, 1, 1), jk_method="bipole",
        log_path=tmp_path / "case.log")
    assert row.converged
    assert (row.n_electrons, row.n_alpha, row.n_beta) == (14, 7, 7)


def test_comparison_rejects_internally_inconsistent_spin_totals():
    ref = replace(_counted("pyscf", 17, 17), n_electrons=14)
    assert _case_mismatch(ref, replace(ref, code="vibeqc"))
    assert _case_mismatch(replace(ref, n_alpha=7, n_beta=7), ref)


@pytest.mark.skipif(importlib.util.find_spec("pyscf") is None, reason="PySCF not installed")
@pytest.mark.parametrize("method", ["rhf", "uhf"])
@pytest.mark.parametrize("mesh", [(1, 1, 1), (3, 1, 1)])
def test_ecp_reference_comparison_preserves_valence_state(tmp_path, method, mesh):
    spec = _cl2_ecp_spec()
    shared = dict(run_id="t", target="dev", spec=spec, basis_name="lanl2dz",
                  method=METHODS[method], kmesh=mesh)
    ref = runner_pyscf.run_periodic_case(
        **shared, log_path=tmp_path / "pyscf.log", artifact_dir=tmp_path / "pyscf")
    assert ref.converged, ref.note
    assert (ref.n_electrons, ref.n_alpha, ref.n_beta) == (14, 7, 7)
    row = runner_vibeqc.run_periodic_case(
        **shared, code_version="t", jk_method="bipole", log_path=tmp_path / "vibeqc.log",
        artifact_dir=tmp_path / "vibeqc")
    assert row.converged, row.note
    assert (row.n_electrons, row.n_alpha, row.n_beta) == (14, 7, 7)
    assert not _case_mismatch(row, ref)
    # Keep #658's original direct-RSJK reference and 1e-6 Ha BIPOLE gate.
    direct = {(1, 1, 1): -29.3455846237986, (3, 1, 1): -29.340156129936254}
    assert row.energy_ha == pytest.approx(direct[mesh], abs=1e-6)
    # Default GDF misses the predeclared 0.1 mHa gate on this ECP fixture.
    # Correct counts must expose that numerical failure, not a state mismatch.
    from examples.regression.core.compare import annotate_against_reference
    from vibeqc_spec import ExpectedRef
    expected = ExpectedRef(system_id=spec.id, basis="lanl2dz", method_id=method,
                           kmesh=mesh, primary_source="pyscf_at_runtime",
                           tolerance_ha=1e-4, tolerance_rationale="Existing GDF gate")
    annotate_against_reference([row, ref], "pyscf", expected)
    assert row.status == "fail"
    assert abs(row.delta_ha_vs_ref) > 1e-4


@pytest.mark.parametrize("receipt", [None, 0, 18])
def test_ecp_comparison_refuses_missing_or_wrong_bipole_core_receipt(tmp_path, monkeypatch, receipt):
    monkeypatch.setattr(runner_vibeqc.vq, "run_periodic_job", lambda *a, **kw:
                        SimpleNamespace(energy=-29., converged=True, n_iter=2,
                                        ecp_total_ncore=receipt, backend="pbc-bipole"))
    row = runner_vibeqc.run_periodic_case(
        run_id="t", target="dev", code_version="t", spec=_cl2_ecp_spec(),
        basis_name="lanl2dz", method=METHODS["uhf"], kmesh=(1, 1, 1), jk_method="bipole",
        log_path=tmp_path / "case.log")
    assert row.status == "error" and row.energy_ha is None
    assert "ECP core count" in row.note
