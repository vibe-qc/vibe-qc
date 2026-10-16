"""Bounded direct-HF terminal contracts; no native chemistry acceptance here.

Array helpers are extracted from canonical source without importing the native
extension. Installed-main scientific checks separately qualify spin J/K and
support invariance.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "python/vibeqc/pbc_bipole_common.py"


def _array_helpers():
    names = {
        "bipole_unshifted_orbitals", "bipole_electron_count",
        "bipole_hf_acceptance", "bipole_level_shift_inputs",
    }
    tree = ast.parse(COMMON.read_text())
    selected = ast.Module(body=[node for node in tree.body
                                if isinstance(node, ast.FunctionDef) and node.name in names],
                          type_ignores=[])
    # The actual pure-Python schedule type, also without a native import.
    schedule_source = ROOT / "python/vibeqc/level_shift_schedule.py"
    namespace = {"__name__": "chi_direct_contract", "np": np}
    import runpy
    namespace["LevelShiftSchedule"] = runpy.run_path(str(schedule_source))["LevelShiftSchedule"]
    exec(compile(selected, str(COMMON), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names},
                           LevelShiftSchedule=namespace["LevelShiftSchedule"])


helpers = _array_helpers()


@pytest.mark.parametrize("occupations", [[1., 1., 0., 0.], [1., .5, .5, 0.], [2., 2., 0., 0.]])
def test_final_orbitals_preserve_density_and_remove_shifted_spectrum(occupations):
    rng = np.random.default_rng(918)
    unitary, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    metric = np.diag([.8, 1.2, 1.7, 2.1])
    coeffs = np.diag(1 / np.sqrt(np.diag(metric))) @ unitary
    physical_mo = np.array([[-2, .3, .2, 0], [.3, -1, .4, 0],
                            [.2, .4, 1, .2], [0, 0, .2, 2]], dtype=complex)
    fock = metric @ coeffs @ physical_mo @ coeffs.conj().T @ metric
    original = coeffs.copy()
    occupation = np.asarray(occupations)
    final, eps = helpers.bipole_unshifted_orbitals([fock], [coeffs], [occupation])
    before = (coeffs * occupation) @ coeffs.conj().T
    after = (final[0] * occupation) @ final[0].conj().T
    np.testing.assert_allclose(after, before, atol=2e-15)
    np.testing.assert_array_equal(coeffs, original)
    np.testing.assert_allclose(final[0].conj().T @ metric @ final[0], np.eye(4), atol=2e-15)
    for occ in np.unique(occupation):
        idx = np.flatnonzero(occupation == occ)
        expected = np.linalg.eigvalsh(physical_mo[np.ix_(idx, idx)])
        np.testing.assert_allclose(eps[0][idx], expected, atol=2e-15)
    # A physical occupied/virtual coupling survives: canonicalization must not
    # falsely turn a nonstationary shifted iterate into an eigenstate.
    residual = fock @ final[0] - (metric @ final[0]) * eps[0]
    assert np.linalg.norm(residual) > .1


def test_empty_spin_has_unshifted_complete_virtual_spectrum():
    fock = np.array([[1., .3j], [-.3j, 2.]])
    coeffs, energies = helpers.bipole_unshifted_orbitals([fock], [np.eye(2)], [np.zeros(2)])
    np.testing.assert_allclose(energies[0], np.linalg.eigvalsh(fock))
    np.testing.assert_allclose(fock @ coeffs[0], coeffs[0] * energies[0], atol=1e-15)


def test_orbital_input_mismatch_rejected():
    with pytest.raises(ValueError, match="same k-points"):
        helpers.bipole_unshifted_orbitals([], [np.eye(2)], [np.ones(2)])
    with pytest.raises(ValueError, match="occupations"):
        helpers.bipole_unshifted_orbitals([np.eye(2)], [np.eye(2)], [[1.]])


def test_spin_counts_use_metric_and_character_weights():
    overlaps = [np.diag([2., 4.]), np.diag([3., 5.])]
    densities = [np.diag([.5, .25]), np.diag([1. / 3, 0.])]
    assert helpers.bipole_electron_count(densities, overlaps, [.25, .75]) == pytest.approx(1.25)


def _acceptance(**overrides):
    values = dict(converged=True, iterations=5, max_iter=10,
                  energy_residual=1e-12, density_residual=1e-10,
                  orbital_residual=1e-10, electron_count_errors=[0., 0.],
                  conv_tol_energy=1e-8, conv_tol_grad=1e-6)
    values.update(overrides)
    return helpers.bipole_hf_acceptance(**values)


def test_stationary_is_explicitly_not_stability():
    result = _acceptance()
    assert result["stationary"] and result["final_rebuild"]
    assert result["stability"] == "not-assessed"
    assert result["termination_reason"] == "stationary"
    assert result["orbital_tolerance"] == 1e-6


@pytest.mark.parametrize("changes,reason", [
    ({"energy_residual": 1e-3}, "terminal_state_not_stationary"),
    ({"density_residual": 1e-3}, "terminal_state_not_stationary"),
    ({"orbital_residual": 1e-3}, "stagnation"),
    ({"electron_count_errors": [0., 1e-3]}, "electron_count_mismatch"),
    ({"orbital_residual": float("nan")}, "nonfinite_final_state"),
    ({"converged": False, "iterations": 10}, "iteration_limit"),
])
def test_final_checks_demote_false_convergence(changes, reason):
    result = _acceptance(**changes)
    assert not result["stationary"]
    assert result["termination_reason"] == reason


def test_level_shift_inputs_preserve_options_and_keyword_precedence():
    opts = SimpleNamespace(level_shift=.7, level_shift_schedule=[.5, .2, 0.])
    assert helpers.bipole_level_shift_inputs(opts, None) == (.7, (.5, .2, 0.))
    explicit = helpers.LevelShiftSchedule([.4, 0.])
    assert helpers.bipole_level_shift_inputs(opts, explicit) == (.7, (.4, 0.))
    assert opts.level_shift_schedule == [.5, .2, 0.] and opts.level_shift == .7


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -.1, "bad"])
def test_invalid_shift_fails_before_integrals(value):
    with pytest.raises(ValueError, match="finite non-negative"):
        helpers.bipole_level_shift_inputs(SimpleNamespace(level_shift=value), None)
    with pytest.raises(ValueError, match="finite non-negative"):
        helpers.bipole_level_shift_inputs(SimpleNamespace(level_shift=0., level_shift_schedule=[value]), None)


def _driver_tree(filename):
    tree = ast.parse((ROOT / "python/vibeqc" / filename).read_text())
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name.startswith("run_pbc_bipole_"))


def test_uhf_records_actual_both_spin_domains_and_native_nuclear_energy():
    tree = _driver_tree("pbc_bipole_uhf.py")
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    recorded = next(node for node in calls if isinstance(node.func, ast.Name)
                    and node.func.id == "_support_recorder")
    keys = {keyword.arg: keyword.value for keyword in recorded.keywords}
    for field in ("spin_density", "spin_fock"):
        assert [key.value for key in keys[field].keys] == ["alpha", "beta"]
    nuclear = next(node for node in calls if isinstance(node.func, ast.Name)
                   and node.func.id == "ewald_nuclear_repulsion_with_support")
    preflight = next(node for node in calls if isinstance(node.func, ast.Attribute)
                     and node.func.attr == "preflight")
    integrals = next(node for node in calls if isinstance(node.func, ast.Name)
                     and node.func.id == "build_bipole_one_electron_lattice")
    assert preflight.lineno < integrals.lineno < nuclear.lineno < recorded.lineno
    shifts = [node for node in calls if isinstance(node.func, ast.Name)
              and node.func.id == "apply_level_shift_k"]
    assert len(shifts) == 2 and all(node.args[-1].attr == "SPIN" for node in shifts)


@pytest.mark.parametrize("filename", ["pbc_bipole.py", "pbc_bipole_uhf.py"])
def test_both_drivers_resolve_schedule_and_report_rebuilt_acceptance(filename):
    tree = _driver_tree(filename)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)]
    names = {node.func.id for node in calls}
    assert {"level_shift_at_iter", "bipole_level_shift_inputs",
            "bipole_unshifted_orbitals", "bipole_hf_acceptance"} <= names
    rebuilds = [node for node in calls if node.func.id == "_build_fock_for_density"]
    assert sum(any(k.arg == "use_incremental" and isinstance(k.value, ast.Constant)
                   and k.value.value is False for k in node.keywords) for node in rebuilds) >= 2
