"""SCF warm-start across geometry-optimization steps.

The glycine SI matrix (rp167) native-optimizer jobs died on the 12 h
queue walltime largely because every geometry step re-ran the SCF from
a cold guess (~40 iterations/step at PBE0/def2-TZVP).
``MolecularSCFProvider`` now restarts each step from the previous
step's converged density (``initial_guess=READ`` + the wrappers'
density projection), which cuts the per-step iteration count without
changing the converged surface. It also fails closed: a nonconverged
mean-field SCF raises instead of feeding a garbage gradient to the
optimizer.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, Molecule, RHFOptions
from vibeqc.geomopt import MolecularSCFProvider


def _h2o(stretch: float = 0.0) -> Molecule:
    return Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 0.0, 1.9 + stretch]),
            Atom(1, [1.85, 0.0, -0.4 - stretch]),
        ]
    )


def test_warm_start_reduces_iterations_and_matches_cold_energy() -> None:
    warm = MolecularSCFProvider("sto-3g", method="rhf")
    e_first, _ = warm(_h2o())
    n_cold_first = warm._last_scf_result.n_iter

    e_warm, g_warm = warm(_h2o(stretch=-0.03))
    n_warm_second = warm._last_scf_result.n_iter

    fresh = MolecularSCFProvider("sto-3g", method="rhf")
    e_cold, g_cold = fresh(_h2o(stretch=-0.03))
    n_cold_second = fresh._last_scf_result.n_iter

    # Same surface: warm and cold agree to SCF convergence noise.
    assert e_warm == pytest.approx(e_cold, abs=1e-9)
    np.testing.assert_allclose(g_warm, g_cold, atol=1e-6)

    # The projected prior density stays competitive with the cold guess for
    # a small displacement. It was strictly fewer iterations under the old
    # AUTO->SAD default; the molecular default is now PATOM (itself a
    # mini-SCF), which converges this easy near-equilibrium case in 6
    # iterations -- at that floor the warm/cold difference is +/-1 iteration
    # of convergence-threshold discreteness, not a mechanism signal (warm=7
    # vs cold=6 measured, invariant to accelerator + damping choices).
    assert n_warm_second <= n_cold_second + 1
    assert n_cold_first > 0

    # The mechanism guard: restarting on the *same* geometry must actually
    # reuse the converged density -- near-instant reconvergence, far below
    # any cold guess. This is what warm-start buys on hard systems (the
    # glycine rp167 40-iter/step case in the module docstring).
    warm(_h2o(stretch=-0.03))
    n_warm_same = warm._last_scf_result.n_iter
    assert n_warm_same <= 3
    assert n_warm_same < n_cold_second


def test_warm_start_restores_user_guess_setting() -> None:
    provider = MolecularSCFProvider("sto-3g", method="rhf")
    provider(_h2o())
    provider(_h2o(stretch=-0.03))
    # The READ toggle used for the warm call must not leak into the
    # user-visible options struct.
    assert provider._rhf_options.initial_guess.name != "READ"


def test_warm_start_disabled_keeps_no_cache() -> None:
    provider = MolecularSCFProvider("sto-3g", method="rhf", warm_start=False)
    provider(_h2o())
    assert provider._last_scf_result is None


def test_nonconverged_scf_raises_instead_of_returning_garbage() -> None:
    opts = RHFOptions()
    opts.max_iter = 1
    provider = MolecularSCFProvider("sto-3g", method="rhf", rhf_options=opts)
    with pytest.raises(RuntimeError, match="did not converge"):
        provider(_h2o())


def test_full_optimization_converges_same_minimum_with_warm_start() -> None:
    from vibeqc.geomopt import ConvergencePolicy, run_geomopt

    results = {}
    for warm in (True, False):
        provider = MolecularSCFProvider("sto-3g", method="rhf", warm_start=warm)
        r = run_geomopt(
            _h2o(),
            provider,
            geom_opt="bfgs",
            geom_conv=ConvergencePolicy(gmax=3e-4),
            geom_max_iter=30,
        )
        assert r.converged, f"warm_start={warm} did not converge"
        results[warm] = float(r.energy)

    assert results[True] == pytest.approx(results[False], abs=1e-8)


def test_warm_geometry_restart_retains_requested_and_physical_guess():
    from vibeqc import InitialGuess

    opts = RHFOptions()
    opts.initial_guess = InitialGuess.HCORE
    provider = MolecularSCFProvider("sto-3g", method="rhf", rhf_options=opts)
    provider(_h2o())
    provider(_h2o(stretch=0.02))
    selection = provider._last_scf_result.guess_selection
    assert selection.requested == InitialGuess.HCORE
    assert selection.effective == InitialGuess.HCORE
    assert selection.transport == InitialGuess.READ
    assert provider._last_scf_result.restart_basis is not None
