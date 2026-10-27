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
    {"charge": True},
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
        assert sent == {"charge": 0, "multiplicity": multiplicity, "atomic_spins": None}

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
