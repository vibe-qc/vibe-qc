"""The cavity-derivative seam: one term, two constructions.

Every geometry dependence of the solvated energy reaches the cavity through
segment positions and segment areas, so a cavity term can be written once
against per-segment adjoints and each construction supplies its own chain
rule. These tests pin the three properties that make that split trustworthy:

* the adjoints are the *term's* derivative and do not know the construction;
* the Lebedev contraction reproduces, to floating-point reassociation, the
  hand-rolled routine it replaced -- so the refactor moved no numbers;
* both constructions satisfy the two exact translation invariants, which need
  no finite differences and which is how the FINE cavity's missing
  box-translation term was found.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc.solvation.cavity import build_cavity
from vibeqc.solvation.cavity_derivative import (
    FineCavityDerivative,
    LebedevCavityDerivative,
    cavity_A_adjoints,
    cavity_derivative,
)
from vibeqc.solvation.cpcm import (
    CPCM_DIAG_ALPHA,
    GAUSSIAN_CHARGE,
    build_A_matrix,
)
from vibeqc.solvation.fine_cavity import ANG_TO_BOHR, build_fine_cavity

WATER = np.array([[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]])
Z = [8, 1, 1]


def _lebedev(n=110):
    return build_cavity(
        atom_positions_bohr=WATER, atom_numbers=Z, n_points_per_sphere=n
    )


def _fine(spacing=0.4, frame="lab"):
    """A CFC in the laboratory frame by default.

    Not the production default since #769, deliberately: the stages exercised
    here are the construction's own chain rule, which is defined in the frame
    the cavity was built in. ``frame="molecular"`` reaches them through
    ``FrameCavityDerivative``, which composes that chain with the frame's
    derivative, and is covered in ``test_solvation_fine_cavity.py``.
    """
    rho = np.array([1.72, 1.30, 1.30]) * ANG_TO_BOHR
    return build_fine_cavity(
        WATER, np.array(Z), rho, grid_spacing_ang=spacing, frame=frame
    )


def _q(n, seed=0):
    return np.random.default_rng(seed).normal(scale=1e-2, size=n)


# ---------------------------------------------------------------------
# The term's half
# ---------------------------------------------------------------------


@pytest.mark.parametrize("cav_factory", [_lebedev, _fine])
def test_A_adjoints_match_direct_differentiation_of_the_quadratic_form(
    cav_factory,
):
    """``cavity_A_adjoints`` against finite differences of ``q^T A q`` itself.

    The segment positions and areas are perturbed directly, with no atoms
    involved, which is exactly the boundary the seam claims: this half of the
    derivative is a property of ``A``, not of any construction. Running it on
    both cavities is the assertion that it does not secretly depend on one.
    """
    cav = cav_factory()
    q = _q(cav.n_points, seed=1)
    scale = 0.371
    adj_p, adj_w = cavity_A_adjoints(cav.points, cav.weights, q, q, scale)

    def form(points, weights):
        return scale * float(q @ build_A_matrix(points, weights) @ q)

    rng = np.random.default_rng(7)
    for idx in rng.choice(cav.n_points, size=6, replace=False):
        for c in range(3):
            h = 1e-6
            pp, pm = cav.points.copy(), cav.points.copy()
            pp[idx, c] += h
            pm[idx, c] -= h
            fd = (form(pp, cav.weights) - form(pm, cav.weights)) / (2 * h)
            assert abs(fd - adj_p[idx, c]) < 1e-5 * max(abs(fd), 1.0)

        # A relative step, because segment areas span seven decades on a
        # Lebedev cavity: the most switched-out points sit near 5e-7 bohr^2,
        # where A_ii = alpha/sqrt(w) is ~1e4 and an absolute step would be
        # pure cancellation. Even relative, 1e-9 is too aggressive there --
        # it disagrees by 0.7 percent on w = 5.6e-7 while 1e-5 and 1e-3
        # both reproduce the closed form to seven digits.
        hw = 1e-5 * cav.weights[idx]
        wp, wm = cav.weights.copy(), cav.weights.copy()
        wp[idx] += hw
        wm[idx] -= hw
        fd = (form(cav.points, wp) - form(cav.points, wm)) / (2 * hw)
        assert abs(fd - adj_w[idx]) < 1e-5 * max(abs(fd), 1.0)


@pytest.mark.parametrize("cav_factory", [_lebedev, _fine])
def test_gaussian_A_adjoints_match_direct_differentiation(cav_factory):
    """The Gaussian kernel's adjoints, including the term it alone has.

    Under ``A_ij = erf(zeta_ij r_ij)/r_ij`` the exponents depend on the
    segment areas, so the **off-diagonal** elements have an area derivative
    that the point-charge kernel does not have at all. That term is the
    likeliest thing to be missing from a hand derivation and the hardest to
    notice: dropping it leaves the assembled gradient smooth, plausible, and
    wrong only where segments are close.

    Perturbing positions and areas directly, with no atoms in play, isolates
    it from every cavity chain rule.
    """
    cav = cav_factory()
    q = _q(cav.n_points, seed=11)
    scale = 0.371
    zeta = 4.901
    adj_p, adj_w = cavity_A_adjoints(
        cav.points, cav.weights, q, q, scale,
        representation=GAUSSIAN_CHARGE, zeta=zeta,
    )

    def form(points, weights):
        return scale * float(
            q
            @ build_A_matrix(
                points, weights, representation=GAUSSIAN_CHARGE, zeta=zeta
            )
            @ q
        )

    rng = np.random.default_rng(5)
    for idx in rng.choice(cav.n_points, size=6, replace=False):
        for c in range(3):
            h = 1e-6
            pp, pm = cav.points.copy(), cav.points.copy()
            pp[idx, c] += h
            pm[idx, c] -= h
            fd = (form(pp, cav.weights) - form(pm, cav.weights)) / (2 * h)
            assert abs(fd - adj_p[idx, c]) < 1e-5 * max(abs(fd), 1.0)

        hw = 1e-5 * cav.weights[idx]
        wp, wm = cav.weights.copy(), cav.weights.copy()
        wp[idx] += hw
        wm[idx] -= hw
        fd = (form(cav.points, wp) - form(cav.points, wm)) / (2 * hw)
        assert abs(fd - adj_w[idx]) < 1e-5 * max(abs(fd), 1.0)


def test_gaussian_area_adjoint_has_an_off_diagonal_channel():
    """Fail-first guard on the term above: it must not be zero.

    Built to fail if someone reuses the point-charge area adjoint for the
    Gaussian kernel. Two close segments and one far away: for the close pair
    the exponent dependence contributes materially, so the Gaussian and
    point-charge area adjoints must differ by more than rounding.
    """
    pts = np.array([[0.0, 0.0, 0.0], [0.35, 0.0, 0.0], [0.0, 0.0, 6.0]])
    w = np.array([0.6, 0.6, 0.6])
    q = np.array([0.01, -0.008, 0.004])

    _, g_area = cavity_A_adjoints(
        pts, w, q, q, 1.0, representation=GAUSSIAN_CHARGE, zeta=4.901
    )
    # Same call with only the diagonal channel, i.e. what a point-charge-style
    # derivation would give for this kernel.
    from vibeqc.solvation.cpcm import gaussian_exponents

    z = gaussian_exponents(w, None, zeta=4.901)
    diag_only = (q * q) * (-(z * np.sqrt(2.0 / np.pi)) / (2.0 * w))
    close = np.abs(g_area[:2] - diag_only[:2])
    assert np.all(close > 1e-3 * np.abs(diag_only[:2])), (
        "the off-diagonal exponent channel is missing or negligible"
    )
    # The distant segment sees essentially none of it, as it should.
    assert abs(g_area[2] - diag_only[2]) < 1e-12 * abs(diag_only[2])


def test_gaussian_adjoints_refuse_a_switched_cavity():
    """Fail closed where the chain rule was not derived.

    For a switched cavity the area and the switching factor are not
    independent, so ``zeta_i = zeta sqrt(F_i/a_i)`` is switching-invariant
    while ``A_ii`` carries ``1/F_i`` -- a different chain rule. Guessing it
    would be exactly the #546 pattern.
    """
    cav = _lebedev()
    with pytest.raises(NotImplementedError, match="no switching function"):
        cavity_A_adjoints(
            cav.points, cav.weights, _q(cav.n_points), _q(cav.n_points), 1.0,
            representation=GAUSSIAN_CHARGE, switching=cav.switching,
            zeta=4.901,
        )


def test_A_adjoints_take_distinct_left_and_right_vectors():
    """``u^T A v`` with ``u != v``, which Direct COSMO-RS needs.

    The energy gradient only ever asks for ``q^T A q``, so the symmetrised
    ``u_i v_j + u_j v_i`` factor would be indistinguishable from a bare
    ``2 q_i q_j`` there. Pinning the general form now is what stops the
    factor-of-two from being rediscovered when the first distinct-adjoint
    caller arrives -- losing it once already cost a 3.5e-4 Ha/bohr residual.
    """
    cav = _lebedev()
    u, v = _q(cav.n_points, seed=2), _q(cav.n_points, seed=3)
    adj_p, adj_w = cavity_A_adjoints(cav.points, cav.weights, u, v, 1.0)

    def form(points, weights):
        return float(u @ build_A_matrix(points, weights) @ v)

    for idx in (0, 41, 137):
        for c in range(3):
            h = 1e-6
            pp, pm = cav.points.copy(), cav.points.copy()
            pp[idx, c] += h
            pm[idx, c] -= h
            fd = (form(pp, cav.weights) - form(pm, cav.weights)) / (2 * h)
            assert abs(fd - adj_p[idx, c]) < 1e-5 * max(abs(fd), 1.0)

    # And it is genuinely asymmetric: swapping the vectors changes the area
    # adjoint nowhere (u_i v_i is symmetric) but the same call with u = v
    # would not reproduce it.
    same, _ = cavity_A_adjoints(cav.points, cav.weights, u, u, 1.0)
    assert not np.allclose(same, adj_p)


# ---------------------------------------------------------------------
# The cavity's half
# ---------------------------------------------------------------------


def test_lebedev_contraction_reproduces_the_hand_rolled_routine():
    """The refactor must not move a number.

    Before the seam existed, the A-matrix term was one routine that fused the
    term's derivative with the Lebedev chain rule. This reproduces it through
    the new path; agreement at floating-point reassociation level is what
    licenses believing the Lebedev results are untouched.
    """
    cav = _lebedev()
    q = _q(cav.n_points, seed=4)
    f = 0.9832

    adj_p, adj_w = cavity_A_adjoints(
        cav.points, cav.weights, q, q, 1.0 / (2.0 * f)
    )
    new = LebedevCavityDerivative(cav, 0.5).contract(adj_p, adj_w)
    reference = _legacy_A_term(cav, q, f, len(Z), 0.5)
    assert np.max(np.abs(new - reference)) < 1e-12 * max(
        float(np.max(np.abs(reference))), 1.0
    )


def _legacy_A_term(cavity, q, f_dielectric, n_atoms, sigma_bohr):
    """The pre-refactor routine, verbatim, as an independent oracle.

    Kept here rather than imported because the point is to compare against
    code that is *not* the code under test. It reads the same cavity fields
    and hard-codes the same rigid-parent assumption the original did.
    """
    import math

    pts, weights = cavity.points, cavity.weights
    parent, R_atoms, r_atoms = (
        cavity.point_atom,
        cavity.atom_positions,
        cavity.atom_radii,
    )
    grad = np.zeros((n_atoms, 3))
    diff = pts[:, None, :] - pts[None, :, :]
    r2 = np.einsum("ijk,ijk->ij", diff, diff)
    r2[r2 == 0] = 1.0
    inv_r3 = 1.0 / (r2 * np.sqrt(r2))
    np.fill_diagonal(inv_r3, 0.0)
    coef = -(q[:, None] * q[None, :]) * inv_r3
    per_point = np.einsum("ij,ijk->ik", coef, diff)
    for ia in range(n_atoms):
        sel = parent == ia
        if np.any(sel):
            grad[ia] += per_point[sel].sum(axis=0)
    grad *= 1.0 / f_dielectric

    diag = np.zeros((n_atoms, 3))
    sigma_i = cavity.switching
    A_diag = CPCM_DIAG_ALPHA / np.sqrt(weights)
    dA_dsigma = -A_diag / (2.0 * sigma_i)
    erf_vec = np.frompyfunc(math.erf, 1, 1)
    for ia in range(n_atoms):
        sel_i = parent == ia
        pts_i, sig_a = pts[sel_i], sigma_i[sel_i]
        dA_a, q_a = dA_dsigma[sel_i], q[sel_i]
        for jb in range(n_atoms):
            if jb == ia:
                continue
            d_ib = pts_i - R_atoms[jb][None, :]
            d = np.linalg.norm(d_ib, axis=1)
            arg = (d - r_atoms[jb]) / sigma_bohr
            grad_f = (
                -(0.5 * (2.0 / math.sqrt(math.pi)) * np.exp(-arg * arg))
                / (sigma_bohr * d)
            )[:, None] * d_ib
            f_b = 0.5 * (1.0 + erf_vec(arg).astype(np.float64))
            f_b = np.where(f_b > 1e-12, f_b, 1e-12)
            contrib = (q_a**2 * dA_a)[:, None] * (
                (sig_a / f_b)[:, None] * grad_f
            )
            diag[jb] += contrib.sum(axis=0)
            diag[ia] -= contrib.sum(axis=0)
    return grad + diag / (2.0 * f_dielectric)


@pytest.mark.parametrize("cav_factory", [_lebedev, _fine])
def test_both_constructions_satisfy_the_translation_invariants(cav_factory):
    """``sum_A dp_S/dR_A = I`` and ``sum_A dw_S/dR_A = 0``, exactly.

    Neither needs a finite difference: translating every atom together
    translates the cavity rigidly. A violation is a net force on an isolated
    molecule, and this is the check that exposed the FINE cavity's missing
    box-translation term -- it read 0.34 against components of order 15.
    """
    deriv = cavity_derivative(cav_factory())
    pos_res, area_res = deriv.translation_residuals()
    assert pos_res < 1e-12
    assert area_res < 1e-12


def test_dispatch_refuses_an_unknown_construction():
    """A third cavity must not silently inherit another one's chain rule."""

    class Bogus:
        atom_positions = WATER

    with pytest.raises(NotImplementedError, match="no nuclear derivative"):
        cavity_derivative(Bogus())


def test_dispatch_picks_the_construction_specific_derivative():
    from vibeqc.solvation.cavity_derivative import FrameCavityDerivative

    assert isinstance(cavity_derivative(_lebedev()), LebedevCavityDerivative)
    assert isinstance(cavity_derivative(_fine()), FineCavityDerivative)
    # A CFC built in its molecule-fixed frame -- the default since #769 -- needs
    # the frame's derivative too, and the dispatcher reads that off the record
    # rather than off a flag.
    assert isinstance(
        cavity_derivative(_fine(frame="molecular")), FrameCavityDerivative
    )
