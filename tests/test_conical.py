"""Penalty-function conical-intersection optimizer (vibeqc.conical).

The optimizer is method-agnostic — it consumes two tracked states' energies +
gradients.  These tests validate the *mechanics*:

* the Levine–Coe–Martínez penalty value/derivative and the assembled objective
  gradient (analytic vs finite difference);
* a synthetic linear-vibronic cone with a known seam: the optimizer drives the
  state gap to zero and lands on the minimum-energy crossing point;
* robustness through a state crossing (upper/lower assigned by energy);
* the MSINDO end-to-end wiring (FD CIS gradients → optimizer) runs and reduces
  the S1/S0 gap.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc.conical import (
    MECIResult,
    lcm_penalty,
    make_cis_meci_fn,
    optimize_conical_intersection,
    penalty_objective,
)

_A2B = 1.8897259886


# --------------------------------------------------------------------------- #
# Penalty function + objective gradient                                       #
# --------------------------------------------------------------------------- #


def test_lcm_penalty_derivative_matches_fd():
    """σ·G'(ΔE) is the derivative of σ·G(ΔE) w.r.t. the gap."""
    sigma, alpha, dE = 4.0, 0.02, 0.013
    h = 1e-7
    val, dval = lcm_penalty(dE, sigma, alpha)
    vp, _ = lcm_penalty(dE + h, sigma, alpha)
    vm, _ = lcm_penalty(dE - h, sigma, alpha)
    assert dval == pytest.approx((vp - vm) / (2 * h), rel=1e-5)
    # Penalty and its slope vanish quadratically/linearly as the gap closes.
    v0, d0 = lcm_penalty(0.0, sigma, alpha)
    assert v0 == 0.0 and d0 == 0.0


def _cone_state_eg(R0, *, k=1.0, a=1.0, b=0.6):
    """A synthetic linear-vibronic cone as a two-state (E, ∇E) function.

    E_avg(R) = ½k‖R−R0‖²; the two states split by ±r with
    r = sqrt((a·u)² + (b·v)²) in the branch plane (u, v) = (R[0,0], R[0,1]).
    The intersection seam is u = v = 0; the minimum-energy crossing point is the
    minimum of E_avg on that seam — i.e. u = v = 0 with every other coordinate at
    R0.  Gradients are analytic.
    """
    R0 = np.asarray(R0, float)

    def fn(R):
        R = np.asarray(R, float)
        u, v = R[0, 0], R[0, 1]
        r = np.sqrt((a * u) ** 2 + (b * v) ** 2)
        e_avg = 0.5 * k * np.sum((R - R0) ** 2)
        g_avg = k * (R - R0)
        g_r = np.zeros_like(R)
        if r > 1e-12:
            g_r[0, 0] = a * a * u / r
            g_r[0, 1] = b * b * v / r
        E = np.array([e_avg - r, e_avg + r])
        G = np.array([g_avg - g_r, g_avg + g_r])
        return E, G

    return fn


def test_penalty_objective_gradient_matches_fd():
    """The assembled ∇F_σ equals a finite difference of F_σ at a non-degenerate
    geometry (smooth there)."""
    R0 = np.array([[0.30, -0.20, 0.50], [0.10, 0.40, -0.30]])
    fn = _cone_state_eg(R0)
    R = R0 + 0.05                                      # off the seam → smooth
    sigma, alpha = 5.0, 0.02
    e, g = fn(R)
    _f, grad, gap = penalty_objective(e, g, sigma, alpha)
    assert gap > 0
    h = 1e-6
    g_fd = np.zeros_like(R)
    for i in range(R.shape[0]):
        for d in range(3):
            rp = R.copy(); rp[i, d] += h
            rm = R.copy(); rm[i, d] -= h
            fp, _, _ = penalty_objective(*fn(rp), sigma, alpha)
            fm, _, _ = penalty_objective(*fn(rm), sigma, alpha)
            g_fd[i, d] = (fp - fm) / (2 * h)
    np.testing.assert_allclose(grad, g_fd, atol=1e-6)


def test_penalty_objective_continuous_through_state_swap():
    """Swapping which state is higher must not change ½(∇E_I+∇E_J); the
    gap-difference term's prefactor vanishes at the crossing, so ∇F is
    continuous.  Build two evaluations with the states interchanged."""
    g0 = np.array([[[1.0, 0.0, 0.0]]])
    g1 = np.array([[[-1.0, 0.5, 0.0]]])
    E_a = np.array([0.10, 0.10 + 1e-9])               # state 0 barely lower
    E_b = np.array([0.10 + 1e-9, 0.10])               # ... and barely higher
    fa, ga, _ = penalty_objective(E_a, np.array([g0[0], g1[0]]), 6.0, 0.02)
    fb, gb, _ = penalty_objective(E_b, np.array([g1[0], g0[0]]), 6.0, 0.02)
    assert fa == pytest.approx(fb, abs=1e-9)
    np.testing.assert_allclose(ga, gb, atol=1e-6)


# --------------------------------------------------------------------------- #
# Synthetic cone: the optimizer finds the seam                                #
# --------------------------------------------------------------------------- #


def test_optimizer_finds_synthetic_conical_intersection():
    """The penalty optimizer drives the gap to zero and lands on the
    minimum-energy crossing point of a known cone."""
    R0 = np.array([[0.30, -0.20, 0.50], [0.10, 0.40, -0.30]])
    fn = _cone_state_eg(R0)
    start = R0.copy()
    start[0, 0] = 0.25                                 # off the seam (u, v ≠ 0)
    start[0, 1] = -0.15
    res = optimize_conical_intersection(fn, start, gap_tol=1e-4, fmax=1e-3,
                                        sigma0=3.5, max_macro=20)
    assert isinstance(res, MECIResult)
    assert res.converged
    assert res.gap < 1e-4
    # Branch coordinates driven onto the seam (u = v = 0) ...
    assert abs(res.coords_angstrom[0, 0]) < 1e-2
    assert abs(res.coords_angstrom[0, 1]) < 1e-2
    # ... and every non-branch coordinate relaxed to R0 (min E_avg on the seam).
    assert res.coords_angstrom[0, 2] == pytest.approx(R0[0, 2], abs=1e-2)
    np.testing.assert_allclose(res.coords_angstrom[1, :], R0[1, :], atol=1e-2)


def test_optimizer_reports_unconverged_when_starved():
    """A single low-σ macro is not enough to close a wide gap → converged=False
    (the escalation is what does the work)."""
    R0 = np.array([[0.30, -0.20, 0.50], [0.10, 0.40, -0.30]])
    fn = _cone_state_eg(R0)
    start = R0.copy(); start[0, 0] = 0.25; start[0, 1] = -0.15
    res = optimize_conical_intersection(fn, start, gap_tol=1e-6, sigma0=0.1,
                                        sigma_growth=1.0, max_macro=1)
    assert not res.converged


def test_optimizer_verbose_uses_live_progress(capsys):
    R0 = np.array([[0.30, -0.20, 0.50], [0.10, 0.40, -0.30]])
    fn = _cone_state_eg(R0)
    optimize_conical_intersection(
        fn,
        R0,
        max_macro=1,
        max_micro=1,
        verbose=True,
    )
    assert "[meci] macro" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# MSINDO end-to-end mechanics                                                 #
# --------------------------------------------------------------------------- #


def test_make_cis_meci_fn_shape():
    """make_cis_meci_fn returns the (2,) energies / (2, natom, 3) gradients the
    optimizer expects, from a MSINDO CIS energy function."""
    from vibeqc.semiempirical.methods.msindo import make_msindo_cis_energy_fn
    Z = [8, 1, 1]
    xyz = np.array([[0, 0, 0.117], [0, 0.757, -0.467], [0, -0.757, -0.467]])
    fn = make_msindo_cis_energy_fn(Z, spin="singlet", n_states=5)
    eg = make_cis_meci_fn(fn, 0, 1, step=2e-3)
    E, G = eg(xyz)
    assert E.shape == (2,)
    assert G.shape == (2, 3, 3)
    assert E[1] > E[0]                                 # S1 above S0


def test_msindo_meci_closes_s1_s0_gap():
    """End-to-end MSINDO mechanics: the INDO finite-difference CIS gradients
    drive the S1/S0 penalty optimizer from a ~7 eV vertical gap down to a near
    degeneracy.  Validates the wiring, not reference-MSINDO CIS parity (which is
    empirically scaled — msindo_cis docstring)."""
    from vibeqc.semiempirical.methods.msindo import msindo_meci

    Z = [8, 1, 1]
    xyz = np.array([[0, 0, 0.117], [0, 0.757, -0.467], [0, -0.757, -0.467]])
    res = msindo_meci(Z, xyz, lower_state=0, upper_state=1, spin="singlet",
                      step=2e-3, gap_tol=1e-3, max_macro=4, max_micro=10)
    assert isinstance(res, MECIResult)
    assert res.coords_angstrom.shape == (3, 3)
    assert res.n_energy_evals > 0
    assert res.gap >= 0.0
    assert res.converged                               # gap < gap_tol reached
    assert res.gap_ev < 0.05                            # ~7 eV → ~0 (huge drop)
