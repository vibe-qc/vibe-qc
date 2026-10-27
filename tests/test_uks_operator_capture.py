"""Opt-in failure observation must preserve the native UKS contract."""
import numpy as np
import pytest
from vibeqc import _vibeqc_core as core


def witness():
    mol = core.Molecule([core.Atom(1, [0.0, 0.0, 0.0])], multiplicity=2)
    basis = core.BasisSet(mol, "sto-3g")
    opts = core.UKSOptions()
    opts.functional = "pbe"
    opts.max_iter = 2
    opts.stability_check = False
    return mol, basis, opts


def test_operator_capture_success_preserves_result():
    mol, basis, opts = witness()
    expected = core.run_uks(mol, basis, opts)
    calls = []
    actual = core._run_uks_with_operator_capture(mol, basis, opts, calls.append)
    assert calls == []
    assert actual.energy == expected.energy
    assert actual.n_iter == expected.n_iter
    assert actual.converged == expected.converged
    for name in ("density_alpha", "density_beta", "fock_alpha", "fock_beta"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


def test_operator_capture_preserves_early_failure_and_restores_scope():
    mol, basis, opts = witness()
    opts.functional = "not-a-functional-for-capture-test"
    with pytest.raises(Exception) as original:
        core.run_uks(mol, basis, opts)
    calls = []
    with pytest.raises(type(original.value)) as observed:
        core._run_uks_with_operator_capture(mol, basis, opts, calls.append)
    assert str(observed.value) == str(original.value)
    assert len(calls) == 1
    assert calls[0]["error"] == str(original.value)
    assert not calls[0]["failing_solve_captured"]
    assert calls[0]["matrices"] == {}
    # A failed observer scope cannot poison a subsequent calculation.
    opts.functional = "pbe"
    core._run_uks_with_operator_capture(mol, basis, opts, calls.append)
    assert len(calls) == 1


def test_operator_capture_callback_failure_cannot_replace_native_error(monkeypatch):
    import sys

    mol, basis, opts = witness()
    opts.functional = "not-a-functional-for-capture-test"
    with pytest.raises(Exception) as original:
        core.run_uks(mol, basis, opts)
    unraisable = []
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)

    def broken_observer(record):
        raise RuntimeError("observer deliberately failed")

    with pytest.raises(type(original.value)) as observed:
        core._run_uks_with_operator_capture(mol, basis, opts, broken_observer)
    assert str(observed.value) == str(original.value)
    assert len(unraisable) == 1
    assert str(unraisable[0].exc_value) == "observer deliberately failed"
