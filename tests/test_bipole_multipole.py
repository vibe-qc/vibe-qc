"""Tests for the BIPOLE multipole-multipole interaction tensor.

Validates the Phase 2 piece of the BIPOLE multipole-far-pair branch:
the analytic tensor ``T_{l1,m1; l2,m2}(R)`` and the closed-form
``multipole_pair_energy`` against known textbook results.

Reference closed forms used here:

* Monopole-monopole: ``E = q1·q2 / |R|``
* Dipole-monopole: ``E = (μ⃗·R̂) · q / R²`` (sign depends on direction convention)
* Dipole-dipole: ``E = [μ⃗_A·μ⃗_B − 3(μ⃗_A·R̂)(μ⃗_B·R̂)] / R³``
* Quadrupole-monopole: standard Stone-Tough form
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from vibeqc.bipole_multipole import (
    irregular_real_solid_harmonic,
    lm_index,
    multipole_interaction_tensor,
    multipole_pair_energy,
    n_components,
    real_solid_harmonic,
)


# ---------------------------------------------------------------------
# Component-index conventions
# ---------------------------------------------------------------------
def test_n_components():
    assert n_components(0) == 1
    assert n_components(1) == 4
    assert n_components(2) == 9
    assert n_components(3) == 16


def test_lm_index_layout():
    assert lm_index(0, 0) == 0
    assert lm_index(1, -1) == 1
    assert lm_index(1, 0) == 2
    assert lm_index(1, +1) == 3
    assert lm_index(2, -2) == 4
    assert lm_index(2, +2) == 8


# ---------------------------------------------------------------------
# Real solid harmonics: spot-check known values
# ---------------------------------------------------------------------
def test_Z_00_constant():
    """Z_{0,0}(r) = 1 for all r (our normalisation)."""
    for r in [(1, 0, 0), (0, 0, 1), (1.5, 2.7, -0.3)]:
        assert math.isclose(real_solid_harmonic(0, 0, np.array(r)), 1.0)


def test_Z_1m_dipole_components():
    """l=1 real solid harmonics in the Stone convention used throughout
    BIPOLE (Schmidt-semi-normalised, identical to
    ``cartesian_to_spherical_matrix``)::

        Z_{1,0} = z,   Z_{1,+1} = √2·x,   Z_{1,-1} = −√2·y

    The √2 on the |m|=1 components is the Stone normalisation. It must be
    consistent between the multipole moments and the interaction tensor —
    both built from ``cartesian_to_spherical_matrix`` — which the
    direct-Coulomb calibration tests below verify end-to-end. (The earlier
    bare-(x, y, z) expectation here predated the Stone-convention refactor
    of ``_cart_to_sph``.)
    """
    r = np.array([0.7, -0.4, 1.1])
    z10 = real_solid_harmonic(1, 0, r)
    z1p = real_solid_harmonic(1, +1, r)
    z1m = real_solid_harmonic(1, -1, r)
    assert math.isclose(z10, r[2], rel_tol=1e-12)
    assert math.isclose(z1p, math.sqrt(2.0) * r[0], rel_tol=1e-12)
    assert math.isclose(z1m, -math.sqrt(2.0) * r[1], rel_tol=1e-12)


def test_I_00_is_1_over_R():
    """I_{0,0}(R) = 1/|R|."""
    for R_vec in [(2.0, 0, 0), (1.0, 1.0, 1.0), (-0.5, 3.7, 2.1)]:
        R = np.array(R_vec)
        expected = 1.0 / np.linalg.norm(R)
        actual = irregular_real_solid_harmonic(0, 0, R)
        assert math.isclose(actual, expected, rel_tol=1e-12)


def test_I_singular_at_origin_raises():
    with pytest.raises(ValueError):
        irregular_real_solid_harmonic(0, 0, np.zeros(3))


# ---------------------------------------------------------------------
# Monopole-monopole — the simplest correctness check
# ---------------------------------------------------------------------
def test_monopole_monopole_pair_energy():
    """E = q_A · q_B / |R|."""
    q_A, q_B = 2.5, -1.3
    R = np.array([3.0, 0.0, 0.0])  # |R| = 3.0
    M_A = np.array([q_A])
    M_B = np.array([q_B])
    E = multipole_pair_energy(M_A, M_B, R, L_max_A=0, L_max_B=0)
    expected = q_A * q_B / np.linalg.norm(R)
    assert math.isclose(E, expected, rel_tol=1e-12), f"got {E}, expected {expected}"


def test_monopole_monopole_random_R():
    """Sample R-orientations to confirm rotational invariance."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        R = rng.normal(size=3)
        if np.linalg.norm(R) < 0.5:
            continue
        E = multipole_pair_energy(
            np.array([1.0]), np.array([1.0]), R, L_max_A=0, L_max_B=0
        )
        assert math.isclose(E, 1.0 / np.linalg.norm(R), rel_tol=1e-12)


# ---------------------------------------------------------------------
# Numerical reference: integrate two point distributions via
# Gaussian quadrature and compare to multipole_pair_energy.
# ---------------------------------------------------------------------
def _gaussian_charge_distribution_multipoles(
    centers: np.ndarray,
    charges: np.ndarray,
    O: np.ndarray,
    L_max: int,
) -> np.ndarray:
    """Compute multipole moments of a set of point charges, around
    expansion center O, in the Z_{l,m} = r^l S_{l,m} convention.

    M_{l,m} = Σ_i q_i · Z_{l,m}(r_i − O)
    """
    M = np.zeros(n_components(L_max), dtype=float)
    for q, c in zip(charges, centers):
        r = np.asarray(c) - np.asarray(O)
        for l in range(L_max + 1):
            for m in range(-l, l + 1):
                M[lm_index(l, m)] += q * real_solid_harmonic(l, m, r)
    return M


def _exact_pair_energy(centers_A, charges_A, centers_B, charges_B):
    """Exact pairwise 1/|r_i − r_j| sum between two point sets."""
    E = 0.0
    for q_A, c_A in zip(charges_A, centers_A):
        for q_B, c_B in zip(charges_B, centers_B):
            R = np.asarray(c_A) - np.asarray(c_B)
            E += q_A * q_B / np.linalg.norm(R)
    return E


def test_two_point_charges_multipole_vs_exact_far_field():
    """Direct-Coulomb calibration for the dipole-dipole channel.

    Two z-aligned dipoles in a head-to-tail arrangement, well separated
    along z: the leading interaction is the analytic dipole-dipole
    ``E = −2·μ_A·μ_B / R³`` (with ``μ = 2a`` here). The L_max=2 multipole
    energy must reproduce both the analytic leading term (sign +
    magnitude) and the exact pairwise Coulomb sum (to within the
    multipole-truncation error).

    A *non-degenerate* geometry is essential. The earlier version used an
    x-dipole vs a z-dipole separated along z, for which ``E ≡ 0`` by
    symmetry — a ``0 == 0`` check that passes even with a badly
    mis-normalised tensor (and indeed hid the L≥2 convention bug).
    """
    a = 0.05
    R_sep = 12.0  # well-separated
    centers_A = np.array([[0, 0, +a], [0, 0, -a]])
    charges_A = np.array([+1.0, -1.0])
    O_A = np.zeros(3)
    centers_B = np.array([[0, 0, R_sep + a], [0, 0, R_sep - a]])
    charges_B = np.array([+1.0, -1.0])
    O_B = np.array([0, 0, R_sep])

    M_A = _gaussian_charge_distribution_multipoles(centers_A, charges_A, O_A, L_max=2)
    M_B = _gaussian_charge_distribution_multipoles(centers_B, charges_B, O_B, L_max=2)

    E_mpole = multipole_pair_energy(M_A, M_B, O_B - O_A, L_max_A=2, L_max_B=2)
    E_exact = _exact_pair_energy(centers_A, charges_A, centers_B, charges_B)

    mu = 2 * a  # dipole magnitude of each ±a pair
    E_lead = -2.0 * mu * mu / R_sep**3  # head-to-tail dipole-dipole

    # Right sign + magnitude (the pre-2026-05-31 tensor mis-normalised this):
    assert math.isclose(E_mpole, E_lead, rel_tol=1e-3), (
        f"dipole-dipole E={E_mpole:.6e} vs analytic lead {E_lead:.6e}"
    )
    # Matches exact Coulomb to the multipole-truncation error (~(a/R)²):
    assert math.isclose(E_mpole, E_exact, rel_tol=3e-4), (
        f"multipole E={E_mpole:.10e}, exact E={E_exact:.10e}, "
        f"rel err={(E_mpole - E_exact) / E_exact:.3e}"
    )


def test_quadrupole_with_monopole_well_separated():
    """Direct-Coulomb calibration for the quadrupole-monopole channel.

    A neutral, dipole-free *linear quadrupole* along z (charges +1, +1, −2)
    interacts with a point charge placed OFF the symmetry axis (so the
    interaction is genuinely non-zero). The multipole energy must match the
    exact pairwise Coulomb sum to within the multipole-truncation error,
    which scales as ``(size/R)²`` for this (octupole-free) distribution.

    This is the calibration that pins the L=2 convention end-to-end: the
    pre-2026-05-31 interaction tensor (single-scalar ``K_L`` correction)
    was off by a clean −1/3 here while the monopole/dipole channels looked
    fine. A residual *systematic* error would survive the size→0 limit; a
    genuine truncation error shrinks as ``(size/R)²``, which the
    convergence check below distinguishes.
    """
    charges_A = np.array([+1.0, +1.0, -2.0])  # neutral, dipole-free
    O_A = np.zeros(3)
    charges_B = np.array([+0.7])

    def _quad_mono_relerr(a, R):
        centers_A = np.array([[0, 0, +a], [0, 0, -a], [0, 0, 0.0]])
        O_B = np.array([0.6 * R, 0.8 * R, 0.0])  # |O_B| = R, off the z-axis
        centers_B = np.array([O_B])
        E_exact = _exact_pair_energy(centers_A, charges_A, centers_B, charges_B)
        out = []
        for L_max in (2, 3):
            M_A = _gaussian_charge_distribution_multipoles(
                centers_A, charges_A, O_A, L_max
            )
            M_B = _gaussian_charge_distribution_multipoles(
                centers_B, charges_B, O_B, L_max
            )
            E_mpole = multipole_pair_energy(M_A, M_B, O_B - O_A, L_max, L_max)
            out.append(abs((E_mpole - E_exact) / E_exact))
        return out  # [L_max=2, L_max=3]

    # Well-separated: tight agreement with exact Coulomb (truncation only).
    # The old −1/3 bug gave relerr ≈ 0.33 here — ~4000× the bound below.
    rel_close = _quad_mono_relerr(a=0.1, R=10.0)
    assert rel_close[0] < 5e-4, f"L_max=2 quad-mono relerr {rel_close[0]:.2e} too large"
    # Octupole vanishes for this symmetric quadrupole, so L=3 adds nothing.
    assert rel_close[1] <= rel_close[0] + 1e-12

    # Multipole convergence: error must shrink ~(size/R)² as the
    # distribution shrinks relative to R (a residual systematic error would
    # not). Quartering size/R drops the error ~16×; require at least 3×.
    rel_far = _quad_mono_relerr(a=0.05, R=20.0)
    assert rel_far[0] < rel_close[0] / 3.0, (
        f"quad-mono error not converging: {rel_close[0]:.2e} -> {rel_far[0]:.2e}"
    )


# ---------------------------------------------------------------------
# Interaction tensor: shape + symmetry
# ---------------------------------------------------------------------
def test_T_shape():
    R = np.array([1.0, 2.0, 3.0])
    T = multipole_interaction_tensor(L_max_A=2, L_max_B=3, R=R)
    assert T.shape == (9, 16)


def test_T_T00_equals_1_over_R():
    R = np.array([2.0, 0.0, 0.0])
    T = multipole_interaction_tensor(L_max_A=0, L_max_B=0, R=R)
    assert math.isclose(T[0, 0], 1.0 / 2.0, rel_tol=1e-12)


# ---------------------------------------------------------------------
# Octupole (L=3) end-to-end validation
# ---------------------------------------------------------------------
def test_octupole_improves_convergence_on_asymmetric_distribution():
    """L=3 octupole corrections measurably improve the multipole energy
    for an asymmetric charge distribution. A tetrahedral arrangement with
    a displaced fourth charge creates non-zero octupole moments."""
    # Asymmetric charge arrangement: three charges in a plane + one offset.
    # This creates genuine octupole (L=3) moments.
    # A single off-centre charge has non-zero multipole moments of all
    # orders. Place a compact charge cluster at a corner of a small cube.
    charges_A = np.array([+1.0, +1.0, +1.0])  # net +3 charge
    O_A = np.array([0.0, 0.0, 0.0])
    centers_A = np.array(
        [
            [0.4, 0.0, 0.0],
            [0.0, 0.4, 0.0],
            [0.0, 0.0, 0.4],
        ]
    )

    # Test charge well-separated to check multipole convergence.
    charges_B = np.array([+0.5])
    O_B = np.array([3.0, 3.0, 3.0])
    centers_B = np.array([O_B])

    E_exact = _exact_pair_energy(centers_A, charges_A, centers_B, charges_B)
    relerrs = []
    for L_max in (1, 2, 3):
        M_A = _gaussian_charge_distribution_multipoles(centers_A, charges_A, O_A, L_max)
        M_B = _gaussian_charge_distribution_multipoles(centers_B, charges_B, O_B, L_max)
        E_mpole = multipole_pair_energy(M_A, M_B, O_B - O_A, L_max, L_max)
        relerrs.append(abs((E_mpole - E_exact) / E_exact))

    # L=3 should improve over L=2 for this asymmetric distribution.
    # (L=1 and L=2 may be similar if the distribution's quadrupole
    # is small — the convergence test focuses on the octupole step).
    assert relerrs[1] <= relerrs[0] + 1e-12, (
        f"L=2 ({relerrs[1]:.2e}) not <= L=1 ({relerrs[0]:.2e})"
    )
    assert relerrs[2] < relerrs[1], (
        f"L=3 ({relerrs[2]:.2e}) should beat L=2 ({relerrs[1]:.2e})"
    )


def test_octupole_interaction_tensor_l3_shape():
    """L_max=3 interaction tensor has shape (16, 16)."""
    R = np.array([1.0, 2.0, 3.0])
    T = multipole_interaction_tensor(L_max_A=3, L_max_B=3, R=R)
    assert T.shape == (16, 16)
    # Must be finite.
    assert np.all(np.isfinite(T))


def test_pure_octupole_monopole_absolute_energy():
    """Absolute normalization pin for the L=3 interaction tensor.

    Eight alternating point charges q = sign(x·y·z) at the corners of a
    cube (side 2s) form a *pure octupole*: monopole, dipole and
    quadrupole moments vanish identically (Td symmetry; the next
    non-zero multipole after L=3 is L=7).  Its interaction with a
    distant monopole is therefore carried entirely by the L=3 channel,
    so ``multipole_pair_energy(..., L=3)`` must match the exact Coulomb
    pair sum *absolutely* — any error in the Wigner-3j / solid-harmonic
    normalization of the interaction tensor shows up as a constant
    factor here (a convergence-ordering test cannot see it).

    The residual relative error is the L=7 term, scaling as (s/R)⁴:
    doubling R must shrink it by ~16×.
    """
    s = 0.25
    corners = np.array(
        [[sx, sy, sz] for sx in (-s, s) for sy in (-s, s) for sz in (-s, s)]
    )
    charges = np.array([np.prod(np.sign(c)) for c in corners])
    O_A = np.zeros(3)
    M_A = _gaussian_charge_distribution_multipoles(corners, charges, O_A, 3)
    # Pure octupole: all l < 3 moments vanish identically.
    assert np.abs(M_A[: lm_index(3, -3)]).max() < 1e-14

    q_B = np.array([1.0])
    rels = []
    for R in (6.0, 12.0):
        O_B = np.array([R / np.sqrt(3.0)] * 3)
        E_exact = _exact_pair_energy(corners, charges, np.array([O_B]), q_B)
        M_B = _gaussian_charge_distribution_multipoles(
            np.array([O_B]), q_B, O_B, 3
        )
        E_mpole = multipole_pair_energy(M_A, M_B, O_B - O_A, 3, 3)
        rels.append(abs((E_mpole - E_exact) / E_exact))
    # Absolute match at the (s/R)⁴ level (measured 1.8e-5 / 1.1e-6).
    assert rels[0] < 5e-5, f"L=3 absolute energy off: rel={rels[0]:.3e}"
    assert rels[1] < 5e-6, f"L=3 absolute energy off: rel={rels[1]:.3e}"
    # Next term is L=7 → error ∝ R⁻⁴: doubling R shrinks it ~16×.
    ratio = rels[0] / rels[1]
    assert 8.0 < ratio < 32.0, (
        f"residual scaling {ratio:.1f} inconsistent with the L=7 tail "
        "(expected ≈16)"
    )


# ---------------------------------------------------------------------
# Screened (erf/erfc) interaction tensors -- BIPOLE-EXACT-ZONE increment
# 2a foundations (bipolar far field in the Ewald-split gauge).
# ---------------------------------------------------------------------
def test_erf_derivative_ladder_matches_finite_differences():
    """The Boys-ladder Cartesian derivatives of erf(sqrt(mu) r)/r match
    central finite differences to |gamma| = 3 -- pins the sign and order
    conventions of the McMurchie-Davidson recursion mechanically."""
    from vibeqc.bipole_multipole import _cartesian_derivative_erf_r

    mu = 0.37
    R0 = np.array([1.7, -0.9, 2.3])
    h = 1e-3

    def f(R):
        r = float(np.linalg.norm(R))
        return math.erf(math.sqrt(mu) * r) / r

    def fd(gamma):
        # Central differences applied recursively per axis.
        def deriv(g, R):
            for ax in range(3):
                if g[ax] > 0:
                    g2 = list(g)
                    g2[ax] -= 1
                    ep = np.array(R, dtype=float)
                    em = np.array(R, dtype=float)
                    ep[ax] += h
                    em[ax] -= h
                    return (deriv(tuple(g2), ep) - deriv(tuple(g2), em)) / (
                        2 * h
                    )
            return f(R)

        return deriv(gamma, R0)

    for gamma in [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (2, 0, 0),
        (1, 1, 0),
        (0, 1, 1),
        (3, 0, 0),
        (2, 0, 1),
        (1, 1, 1),
    ]:
        exact = _cartesian_derivative_erf_r(gamma, R0, mu)
        approx = fd(gamma)
        assert exact == pytest.approx(approx, rel=2e-4, abs=5e-7), gamma


def test_screened_tensor_monopole_is_erf_over_r():
    from vibeqc.bipole_multipole import screened_multipole_interaction_tensor

    mu, R = 0.52, np.array([0.0, 0.0, 9.0])
    T = screened_multipole_interaction_tensor(2, 2, R, mu)
    assert T[0, 0] == pytest.approx(math.erf(math.sqrt(mu) * 9.0) / 9.0, rel=1e-12)


def test_screened_tensor_large_mu_limit_is_bare_tensor():
    """erf -> 1 pointwise as mu -> infinity, so at fixed R the screened
    tensor converges to the bare Coulomb tensor (all components)."""
    from vibeqc.bipole_multipole import (
        multipole_interaction_tensor,
        screened_multipole_interaction_tensor,
    )

    R = np.array([2.0, -3.0, 5.0])
    T_bare = multipole_interaction_tensor(3, 3, R)
    T_scr = screened_multipole_interaction_tensor(3, 3, R, 4.0e2)
    np.testing.assert_allclose(T_scr, T_bare, rtol=1e-10, atol=1e-12)


def test_sr_tensor_linearity_identity_and_decay():
    """T_bare == T_erf(mu) + T_SR(mu) exactly (linearity of the Taylor
    assembly), and the SR tensor decays like the erfc kernel: at
    sqrt(mu)·R >> 1 it is exponentially small while the bare tensor is
    only algebraically small."""
    from vibeqc.bipole_multipole import (
        multipole_interaction_tensor,
        screened_multipole_interaction_tensor,
        sr_multipole_interaction_tensor,
    )

    mu = 0.8
    R = np.array([1.1, 2.2, -0.7])
    T_bare = multipole_interaction_tensor(2, 2, R)
    T_scr = screened_multipole_interaction_tensor(2, 2, R, mu)
    T_sr = sr_multipole_interaction_tensor(2, 2, R, mu)
    np.testing.assert_allclose(T_sr + T_scr, T_bare, rtol=0, atol=1e-13)

    R_far = np.array([0.0, 0.0, 14.0])
    T_sr_far = sr_multipole_interaction_tensor(2, 2, R_far, mu)
    T_bare_far = multipole_interaction_tensor(2, 2, R_far)
    # erfc(sqrt(0.8)*14) ~ 5e-70: the SR tensor is numerically zero
    # while the bare monopole term is 1/14.
    assert abs(T_sr_far[0, 0]) < 1e-40
    assert T_bare_far[0, 0] == pytest.approx(1.0 / 14.0, rel=1e-12)


def test_screened_pair_energy_two_gaussians_reduced_exponent_identity():
    """Physics anchor: two well-separated NORMALIZED s-Gaussians with
    exponents (a, b) interacting through the bare Coulomb kernel have
    the exact energy erf(sqrt(m)·R)/R with 1/m = 1/a + 1/b (the
    reduced-exponent identity behind the SR pad and QQR screening).
    Point-multipole monopoles through the SCREENED tensor with mu = m
    must therefore reproduce the exact two-Gaussian bare-kernel energy:
    the smearing of the distributions and the screening of the kernel
    are the same convolution."""
    from vibeqc.bipole_multipole import screened_multipole_interaction_tensor

    a, b, Rd = 0.9, 0.4, 8.0
    m = 1.0 / (1.0 / a + 1.0 / b)
    exact = math.erf(math.sqrt(m) * Rd) / Rd
    T = screened_multipole_interaction_tensor(0, 0, np.array([0, 0, Rd]), m)
    assert T[0, 0] == pytest.approx(exact, rel=1e-12)
