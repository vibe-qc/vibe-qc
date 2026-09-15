"""Molecular SCF ``max_iter`` boundary regressions (issue #392).

``max_iter=0`` is the documented ORCA ``NOITER`` compatibility path. Every
molecular SCF route must return one coherent initial-guess evaluation without
taking an SCF iteration. Negative, boolean, and non-integer budgets must fail
at the Python boundary before SCF work; positive and default budgets retain
their existing behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq

_MOLECULAR_SCF_ROUTES = ("rhf", "uhf", "rks", "uks", "rohf", "roks")
_CLOSED_SHELL_ROUTES = {"rhf", "rks"}
_RESTRICTED_ROUTES = {"rhf", "rks", "rohf", "roks"}
_DFT_ROUTES = {"rks", "uks", "roks"}


def _h2() -> vq.Molecule:
    return vq.Molecule(
        [
            vq.Atom(1, [0.0, 0.0, -0.7]),
            vq.Atom(1, [0.0, 0.0, 0.7]),
        ],
        charge=0,
        multiplicity=1,
    )


def _h() -> vq.Molecule:
    return vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0])], charge=0, multiplicity=2
    )


def _run_molecular_scf_route(route: str, max_iter):
    molecule = _h2() if route in _CLOSED_SHELL_ROUTES else _h()
    basis = vq.BasisSet(molecule, "sto-3g")
    options_type = getattr(vq, f"{route.upper()}Options")
    options = options_type()
    options.max_iter = max_iter
    if route in _DFT_ROUTES:
        options.functional = "lda"
    result = getattr(vq, f"run_{route}")(molecule, basis, options)
    return molecule, basis, result


@pytest.mark.parametrize("route", _MOLECULAR_SCF_ROUTES)
@pytest.mark.parametrize("value", [-1, -100])
def test_molecular_scf_rejects_negative_iteration_budget(route, value) -> None:
    with pytest.raises(ValueError, match="max_iter must be a non-negative"):
        _run_molecular_scf_route(route, value)


@pytest.mark.parametrize("route", _MOLECULAR_SCF_ROUTES)
@pytest.mark.parametrize("value", [True, False])
def test_molecular_scf_rejects_boolean_iteration_budget(route, value) -> None:
    with pytest.raises(ValueError, match="max_iter"):
        _run_molecular_scf_route(route, value)


@pytest.mark.parametrize("route", _MOLECULAR_SCF_ROUTES)
@pytest.mark.parametrize("value", [1.5, "5"])
def test_molecular_scf_rejects_noninteger_iteration_budget(route, value) -> None:
    with pytest.raises(ValueError, match="max_iter"):
        _run_molecular_scf_route(route, value)


@pytest.mark.parametrize("route", _MOLECULAR_SCF_ROUTES)
def test_zero_budget_returns_initial_guess_evaluation(route) -> None:
    molecule, basis, result = _run_molecular_scf_route(route, 0)
    nbf = basis.nbasis

    assert result.n_iter == 0
    assert result.converged is False
    assert len(result.scf_trace) == 0
    assert np.isfinite(result.energy)
    assert result.energy != 0.0

    if route in _RESTRICTED_ROUTES:
        matrix_names = ("density", "fock", "mo_coeffs")
        vector_names = ("mo_energies",)
    else:
        matrix_names = (
            "density_alpha",
            "density_beta",
            "fock_alpha",
            "fock_beta",
            "mo_coeffs_alpha",
            "mo_coeffs_beta",
        )
        vector_names = ("mo_energies_alpha", "mo_energies_beta")

    for name in matrix_names:
        matrix = np.asarray(getattr(result, name))
        assert matrix.shape == (nbf, nbf), name
        assert np.all(np.isfinite(matrix)), name
    for name in vector_names:
        vector = np.asarray(getattr(result, name))
        assert vector.shape == (nbf,), name
        assert np.all(np.isfinite(vector)), name

    overlap = np.asarray(vq.compute_overlap(basis))
    if route in _RESTRICTED_ROUTES:
        density = np.asarray(result.density)
    else:
        density = np.asarray(result.density_alpha) + np.asarray(
            result.density_beta
        )
    assert np.trace(density @ overlap) == pytest.approx(
        molecule.n_electrons(), abs=1.0e-10
    )


@pytest.mark.parametrize("route", _MOLECULAR_SCF_ROUTES)
def test_one_iteration_budget_still_runs_one_cycle(route) -> None:
    _molecule, _basis, result = _run_molecular_scf_route(route, 1)

    assert result.n_iter == 1
    assert result.converged is False
    assert len(result.scf_trace) == 1


@pytest.mark.parametrize("route", ("uhf", "uks"))
def test_open_shell_zero_budget_does_not_run_spin_schedule(route) -> None:
    molecule = _h2()
    basis = vq.BasisSet(molecule, "sto-3g")
    options_type = getattr(vq, f"{route.upper()}Options")

    plain_options = options_type()
    plain_options.max_iter = 0
    scheduled_options = options_type()
    scheduled_options.max_iter = 0
    scheduled_options.spinlock_mode = vq.SpinlockMode.SPIN_SCHEDULE
    scheduled_options.spinlock_value = 2
    scheduled_options.spinlock_iterations = 2
    if route == "uks":
        plain_options.functional = "lda"
        scheduled_options.functional = "lda"

    runner = getattr(vq, f"run_{route}")
    plain = runner(molecule, basis, plain_options)
    scheduled = runner(molecule, basis, scheduled_options)

    assert scheduled.n_iter == 0
    assert scheduled.converged is False
    assert len(scheduled.scf_trace) == 0
    assert scheduled.energy == pytest.approx(plain.energy, abs=1.0e-12)
    assert np.asarray(scheduled.density_alpha) == pytest.approx(
        np.asarray(plain.density_alpha), abs=1.0e-12
    )
    assert np.asarray(scheduled.density_beta) == pytest.approx(
        np.asarray(plain.density_beta), abs=1.0e-12
    )


def test_default_iteration_budget_still_converges() -> None:
    result = vq.run_rhf(_h2(), "sto-3g")
    assert result.n_iter >= 1
    assert result.converged
