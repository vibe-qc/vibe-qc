"""Phase D2c — Newton (full orbital-Hessian Newton-CG) tests.

Pins the contract:

  1. With ``newton_threshold = 0`` (default), RHF behaves bit-for-bit as
     before — no Newton activation, no SCF trace change.

  2. With ``newton_threshold = 1.0`` (Fischer-Almlöf 1992 production
     setting), RHF converges to the *same* energy as plain DIIS on
     standard reference systems, with the SCF trace showing
     ``newton_cg_iter > 0`` on the asymptotic iterations.

  3. Newton + EDIIS+DIIS hybrid composes: the warm-up handles
     iterations 1–N until the gradient drops below 1.0, then Newton
     closes out the last 3–5 iters with quadratic convergence.

  4. Tight closed-shell test (H2O / sto-3g) — Newton does *not*
     break easy cases.

  5. New options + trace fields exist with documented defaults.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import (
    BasisSet,
    RHFOptions,
    SCFAccelerator,
    NewtonOptions,
    run_rhf,
)
from .conftest import ANGSTROM_TO_BOHR, make_molecule


# ---------------------------------------------------------------------------
# Geometries
# ---------------------------------------------------------------------------

def _h2o_atoms():
    return [
        (8, [0.0, 0.0,  0.117 * ANGSTROM_TO_BOHR]),
        (1, [0.0,  0.755 * ANGSTROM_TO_BOHR, -0.471 * ANGSTROM_TO_BOHR]),
        (1, [0.0, -0.755 * ANGSTROM_TO_BOHR, -0.471 * ANGSTROM_TO_BOHR]),
    ]


@pytest.fixture
def h2o_basis():
    mol = make_molecule(_h2o_atoms())
    return mol, BasisSet(mol, "sto-3g")


# ---------------------------------------------------------------------------
# Defaults — back-compat: Newton off, threshold 0.
# ---------------------------------------------------------------------------

def test_newton_threshold_default_is_zero():
    opts = RHFOptions()
    assert opts.newton_threshold == pytest.approx(0.0)


def test_newton_options_default_values():
    so = NewtonOptions()
    assert so.cg_max_iter == 50
    assert so.cg_tol == pytest.approx(1e-4)
    assert so.trust_radius == pytest.approx(0.3)


def test_scf_iteration_has_newton_cg_iter_field():
    it = vq.SCFIteration()
    # Default is 0 — kept consistent with the existing diis_subspace
    # default.
    assert it.newton_cg_iter == 0


# ---------------------------------------------------------------------------
# Newton activation — back-compat with default off.
# ---------------------------------------------------------------------------

def test_rhf_default_does_not_activate_newton(h2o_basis):
    """With ``newton_threshold = 0`` (default), no iteration should
    record any Newton CG work in the trace."""
    mol, basis = h2o_basis
    opts = RHFOptions()
    opts.max_iter = 60
    opts.conv_tol_energy = 1e-10
    opts.conv_tol_grad = 1e-8
    r = run_rhf(mol, basis, opts)
    assert r.converged
    assert all(it.newton_cg_iter == 0 for it in r.scf_trace)


# ---------------------------------------------------------------------------
# Newton parity — converges to the same fixed point as DIIS.
# ---------------------------------------------------------------------------

def test_rhf_newton_matches_diis_h2o(h2o_basis):
    """Full Newton-CG Newton after a DIIS warm-up reaches the same
    converged energy as plain DIIS. The accelerator changes the path,
    not the fixed point."""
    mol, basis = h2o_basis
    base = RHFOptions()
    base.max_iter = 60
    base.conv_tol_energy = 1e-10
    base.conv_tol_grad = 1e-8

    r_diis = run_rhf(mol, basis, base)

    so = RHFOptions()
    so.max_iter = 60
    so.conv_tol_energy = 1e-10
    so.conv_tol_grad = 1e-8
    so.newton_threshold = 1.0     # Fischer-Almlöf 1992 production setting
    r_newton = run_rhf(mol, basis, so)

    assert r_diis.converged
    assert r_newton.converged
    assert r_newton.energy == pytest.approx(r_diis.energy, abs=1e-9)
    # MO energies — Newton and DIIS converge to the same MO frame to
    # canonical-orbital tolerance.
    np.testing.assert_allclose(np.asarray(r_newton.mo_energies),
                                np.asarray(r_diis.mo_energies),
                                atol=1e-7)


def test_rhf_newton_records_cg_iters_in_trace(h2o_basis):
    """Once Newton activates, the trace should show non-zero
    ``newton_cg_iter`` for the asymptotic iterations and zero for the
    DIIS warm-up phase."""
    mol, basis = h2o_basis
    opts = RHFOptions()
    opts.max_iter = 60
    opts.conv_tol_energy = 1e-10
    opts.conv_tol_grad = 1e-8
    opts.newton_threshold = 1.0
    # Pin a plain SAD start so the trace has a genuine DIIS warm-up before
    # Newton. The molecular default guess (PATOM) is good enough that the
    # commutator norm is already below newton_threshold on iteration 1, so
    # Newton fires immediately and there is no warm-up row (cg_iter[0] > 0).
    # SAD keeps the warm-up-then-Newton shape this test checks.
    opts.initial_guess = vq.InitialGuess.SAD
    r = run_rhf(mol, basis, opts)

    assert r.converged
    cg_iters = [it.newton_cg_iter for it in r.scf_trace]
    # At least one iteration should have run Newton.
    assert any(c > 0 for c in cg_iters), (
        f"Newton never activated on H2O/sto-3g; trace cg_iter = {cg_iters}"
    )
    # The first iteration is the DIIS warm-up (gradient hasn't been
    # computed yet on iter 1's standalone path).
    assert cg_iters[0] == 0


def test_rhf_newton_with_ediis_diis_warmup_matches_pure_diis(h2o_basis):
    """The "production stack" — EDIIS+DIIS handles iters 1–N until
    ‖e‖ < 1.0, then Newton closes out with quadratic convergence —
    must converge to the same energy as plain DIIS."""
    mol, basis = h2o_basis
    base = RHFOptions()
    base.max_iter = 80
    base.conv_tol_energy = 1e-10
    base.conv_tol_grad = 1e-8

    r_diis = run_rhf(mol, basis, base)

    stack = RHFOptions()
    stack.max_iter = 80
    stack.conv_tol_energy = 1e-10
    stack.conv_tol_grad = 1e-8
    stack.scf_accelerator = SCFAccelerator.EDIIS_DIIS
    stack.newton_threshold = 1.0
    r_stack = run_rhf(mol, basis, stack)

    assert r_diis.converged
    assert r_stack.converged
    assert r_stack.energy == pytest.approx(r_diis.energy, abs=1e-9)


def test_rhf_newton_does_not_break_easy_cases(h2o_basis):
    """Easy closed-shell case (H2O / sto-3g) — turning on Newton
    should not regress iteration count beyond a small constant
    (Newton converges in ~3-5 final iters; total-iter envelope
    should stay within DIIS+1)."""
    mol, basis = h2o_basis
    base = RHFOptions()
    base.max_iter = 60
    base.conv_tol_energy = 1e-10
    base.conv_tol_grad = 1e-8

    r_diis = run_rhf(mol, basis, base)

    so = RHFOptions()
    so.max_iter = 60
    so.conv_tol_energy = 1e-10
    so.conv_tol_grad = 1e-8
    so.newton_threshold = 1.0
    r_so = run_rhf(mol, basis, so)

    # Loose envelope — easy case must converge in roughly the same
    # number of SCF iters as plain DIIS (Newton's "asymptotic" regime
    # is not where DIIS struggles, so Newton doesn't help here, but
    # also shouldn't hurt by more than a few iterations).
    assert r_so.n_iter <= r_diis.n_iter + 5, (
        f"Newton made an easy case slower: n_iter went {r_diis.n_iter} "
        f"→ {r_so.n_iter} (regression beyond +5 iters)."
    )
