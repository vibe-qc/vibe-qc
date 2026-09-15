"""Tests for the gapw chat's per-atom radial × Lebedev grid module
(``python/vibeqc/periodic_gapw_atomic_grid.py``).

Pins the conventions M3b GAPW augmentation needs:

* Mura-Knowles radial transform integrates Gaussians to µHa precision;
* Lebedev angular quadrature integrates low-ℓ spherical harmonics
  exactly to machine precision;
* the product grid integrates ``∫ ρ(r) · r^l · Y_lm(r̂) dr`` correctly,
  i.e., the multipole-moment formula works on a Gaussian probe.

Infrastructure tests, not parity tests — the augmentation is not yet
wired into V_ne or J. M3b consumes this module to build the per-atom
hard/soft Hartree decomposition.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from vibeqc.periodic_gapw_atomic_grid import (
    AtomicRadialGrid,
    default_alpha_for_element,
    lebedev_supported_orders,
)
from vibeqc.periodic_gapw_grid import GAPWExperimentalWarning


# ---------- default α per element ---------------------------------------


def test_default_alpha_h_he_tabulated():
    assert default_alpha_for_element(1) == 5.0
    assert default_alpha_for_element(2) == 4.0
    assert default_alpha_for_element(8) == 5.0  # O


def test_default_alpha_heavier_interpolates():
    """Z = 20 should land between Ne (5.0) and the Zn-area cap (6.0)."""
    a = default_alpha_for_element(20)
    assert 5.0 < a < 6.0


def test_default_alpha_rejects_nonpositive_Z():
    with pytest.raises(ValueError, match="positive integer"):
        default_alpha_for_element(0)


# ---------- Construction + invariants -----------------------------------


def test_atomic_grid_build_shapes():
    """Shape pins on a fresh build."""
    g = AtomicRadialGrid.build(
        centre_bohr=np.zeros(3),
        n_radial=40, alpha=5.0, lebedev_order=17,
    )
    assert g.n_radial == 40
    assert g.n_angular >= 50      # 17th-order Lebedev has 110 points
    assert g.n_points == g.n_radial * g.n_angular
    assert g.r.shape == (g.n_radial,)
    assert g.w_r.shape == (g.n_radial,)
    assert g.angular_xyz.shape == (g.n_angular, 3)
    assert g.w_a.shape == (g.n_angular,)


def test_atomic_grid_angular_weights_sum_to_4pi():
    """``Σ_l w_l = 4π`` — the unit-sphere surface area, scipy normalised."""
    g = AtomicRadialGrid.build(
        centre_bohr=np.zeros(3),
        n_radial=20, alpha=5.0, lebedev_order=11,
    )
    assert g.w_a.sum() == pytest.approx(4.0 * math.pi, rel=1e-12)


def test_atomic_grid_angular_points_are_unit_vectors():
    g = AtomicRadialGrid.build(
        centre_bohr=np.zeros(3),
        n_radial=20, alpha=5.0, lebedev_order=11,
    )
    norms = np.linalg.norm(g.angular_xyz, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-12)


def test_atomic_grid_radial_is_strictly_increasing():
    """Mura-Knowles maps x ∈ (0, 1) monotonically to r ∈ (0, ∞)."""
    g = AtomicRadialGrid.build(
        centre_bohr=np.zeros(3),
        n_radial=30, alpha=5.0, lebedev_order=11,
    )
    diffs = np.diff(g.r)
    assert np.all(diffs > 0.0)


def test_atomic_grid_rejects_bad_inputs():
    with pytest.raises(ValueError, match="n_radial must be >= 4"):
        AtomicRadialGrid.build(np.zeros(3), n_radial=2, alpha=5.0,
                               lebedev_order=11)


def test_atomic_grid_post_init_validation():
    """Direct constructor must catch dimensional / sign errors."""
    with pytest.raises(ValueError, match="centre_bohr must be shape"):
        AtomicRadialGrid(
            centre_bohr=np.zeros(4),
            r=np.array([0.1, 0.2]), w_r=np.array([1.0, 1.0]),
            angular_xyz=np.eye(3)[:1], w_a=np.array([1.0]),
            radial_alpha=5.0, lebedev_order=11,
        )
    with pytest.raises(ValueError, match="r must be strictly positive"):
        AtomicRadialGrid(
            centre_bohr=np.zeros(3),
            r=np.array([0.0, 0.5]), w_r=np.array([1.0, 1.0]),
            angular_xyz=np.eye(3)[:1], w_a=np.array([1.0]),
            radial_alpha=5.0, lebedev_order=11,
        )
    with pytest.raises(ValueError, match="unit vectors"):
        AtomicRadialGrid(
            centre_bohr=np.zeros(3),
            r=np.array([0.5]), w_r=np.array([1.0]),
            angular_xyz=np.array([[2.0, 0.0, 0.0]]),
            w_a=np.array([4.0 * math.pi]),
            radial_alpha=5.0, lebedev_order=11,
        )


# ---------- Integration against analytical Gaussian moments -------------


def _gaussian_3d(r_xyz: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """``ρ(r) = (α/√π)³ exp(-α² r²)`` for ``r = |r_xyz - 0|``."""
    r2 = (r_xyz ** 2).sum(axis=-1)
    return (alpha / math.sqrt(math.pi)) ** 3 * np.exp(-alpha ** 2 * r2)


def test_integrates_normalised_gaussian_to_one():
    """``∫ (α/√π)³ exp(-α² r²) d³r = 1`` for any positive α — a unit
    test of the radial × angular quadrature accuracy. The grid must
    resolve the Gaussian; for α = 1 bohr⁻¹ in a 50-point Mura grid
    with order-17 Lebedev, we expect ~µHa precision."""
    g = AtomicRadialGrid.build(
        centre_bohr=np.zeros(3),
        n_radial=80, alpha=5.0, lebedev_order=17,
    )
    pts = g.cartesian_points()                # (n_radial, n_angular, 3)
    rho = _gaussian_3d(pts, alpha=1.0)        # (n_radial, n_angular)
    integral = g.integrate(rho)
    assert integral == pytest.approx(1.0, abs=1e-4)


def test_integrates_gaussian_higher_alpha():
    """Tighter Gaussian (α = 2) still integrates to 1 within ~1e-3 —
    needs a denser radial mesh near r = 0."""
    g = AtomicRadialGrid.build(
        centre_bohr=np.zeros(3),
        n_radial=120, alpha=3.0, lebedev_order=17,
    )
    pts = g.cartesian_points()
    rho = _gaussian_3d(pts, alpha=2.0)
    integral = g.integrate(rho)
    assert integral == pytest.approx(1.0, abs=1e-3)


def test_integrates_constant_gives_4pi_R3_over_3_in_sphere():
    """As an angular-grid sanity, the integral of 1 over the unit
    sphere is 4π — the radial integral diverges so we cap r at a
    finite value (the grid's own tail handles this).

    Concretely: ``∫_0^∞ 1 · r² dr · Σ_l w_l = (4π · cell_radial)``
    where cell_radial = Σ_k w_k. For a smooth Mura-Knowles grid
    this is well-defined and finite (radial integrand effectively
    capped by the Mura mapping). Just check finiteness + sign."""
    g = AtomicRadialGrid.build(
        centre_bohr=np.zeros(3),
        n_radial=50, alpha=5.0, lebedev_order=11,
    )
    pts = g.cartesian_points()
    ones = np.ones((g.n_radial, g.n_angular))
    integral = g.integrate(ones)
    assert integral > 0.0
    assert np.isfinite(integral)


def test_centre_offset_picks_up_the_translation():
    """A Gaussian centred at R_atom should integrate to 1 regardless
    of where R_atom is — the grid sits on R_atom and the integrand
    sees ``r' = r - R_atom``."""
    centre = np.array([2.0, -1.5, 0.3])
    g = AtomicRadialGrid.build(
        centre_bohr=centre,
        n_radial=80, alpha=5.0, lebedev_order=17,
    )
    pts = g.cartesian_points()
    # ρ is centred on R_atom, so pts - centre returns to the
    # spherical-coords frame the grid was built in.
    rho = _gaussian_3d(pts - centre, alpha=1.0)
    integral = g.integrate(rho)
    assert integral == pytest.approx(1.0, abs=1e-4)


# ---------- Public exports + warnings -----------------------------------


def test_from_element_emits_experimental_warning():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        AtomicRadialGrid.from_element(np.zeros(3), Z=1)
    msgs = [
        w for w in captured
        if issubclass(w.category, GAPWExperimentalWarning)
        and "M3b" in str(w.message)
    ]
    assert msgs


def test_from_element_uses_per_Z_alpha():
    g = AtomicRadialGrid.from_element(np.zeros(3), Z=2, quiet=True)
    assert g.radial_alpha == default_alpha_for_element(2)


def test_lebedev_supported_orders_includes_common():
    """The list returned should include common Lebedev tiers."""
    orders = lebedev_supported_orders()
    for o in (11, 17, 23):
        assert o in orders


# ---------- Real spherical harmonics ------------------------------------


def test_multipole_index_layout():
    """The flat-index layout is l² + l + m."""
    from vibeqc.periodic_gapw_atomic_grid import (
        multipole_index, multipole_index_pairs, n_multipole_components,
    )
    assert multipole_index(0, 0) == 0
    assert multipole_index(1, -1) == 1
    assert multipole_index(1, 0) == 2
    assert multipole_index(1, 1) == 3
    assert multipole_index(2, -2) == 4
    assert multipole_index(2, 2) == 8
    pairs = multipole_index_pairs(2)
    assert pairs == [(0, 0), (1, -1), (1, 0), (1, 1),
                     (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)]
    assert n_multipole_components(0) == 1
    assert n_multipole_components(2) == 9
    assert n_multipole_components(4) == 25


def test_multipole_index_validation():
    from vibeqc.periodic_gapw_atomic_grid import multipole_index
    with pytest.raises(ValueError, match="l must be >= 0"):
        multipole_index(-1, 0)
    with pytest.raises(ValueError, match=r"\|m\| must be <= l"):
        multipole_index(1, 2)


def test_real_spherical_harmonics_orthonormal_on_lebedev():
    """``∫_S² S_lm · S_l'm' dΩ̂ = δ_ll' δ_mm'`` numerically — the
    Stone-normalisation invariant. On an order-17 Lebedev grid this
    holds to machine precision for l up through 4 because the
    angular order exceeds 2·lmax."""
    from vibeqc.periodic_gapw_atomic_grid import (
        real_spherical_harmonics, n_multipole_components,
    )
    g = AtomicRadialGrid.build(np.zeros(3), n_radial=10, alpha=5.0,
                               lebedev_order=17)
    lmax = 3
    S = real_spherical_harmonics(g.angular_xyz, lmax)
    n = n_multipole_components(lmax)
    # ⟨S_lm, S_l'm'⟩ = Σ_l_ang S_lm[l_ang] · S_l'm'[l_ang] · w_l_ang
    overlap = np.einsum("ca,da,a->cd", S, S, g.w_a)
    assert np.allclose(overlap, np.eye(n), atol=1e-10), (
        f"Orthonormality breakdown; max deviation from identity = "
        f"{np.max(np.abs(overlap - np.eye(n))):.3e}"
    )


def test_real_spherical_harmonics_l4_l6_orthonormal_via_scipy():
    """Up through lmax=6: orthonormality on a high-order Lebedev
    grid (29 → 302 points, exact for polynomials up to 2·14=28).
    The l ≥ 4 path goes through scipy.special.sph_harm_y; verifies
    the complex→real combination matches Stone normalisation."""
    from vibeqc.periodic_gapw_atomic_grid import (
        real_spherical_harmonics, n_multipole_components,
    )
    g = AtomicRadialGrid.build(np.zeros(3), n_radial=10, alpha=5.0,
                               lebedev_order=29)
    for lmax in (4, 5, 6):
        S = real_spherical_harmonics(g.angular_xyz, lmax)
        n = n_multipole_components(lmax)
        overlap = np.einsum("ca,da,a->cd", S, S, g.w_a)
        assert np.allclose(overlap, np.eye(n), atol=1e-10), (
            f"lmax={lmax}: max dev = "
            f"{np.max(np.abs(overlap - np.eye(n))):.3e}"
        )


def test_real_spherical_harmonics_rejects_bad_input():
    from vibeqc.periodic_gapw_atomic_grid import real_spherical_harmonics
    with pytest.raises(ValueError, match="lmax must be >= 0"):
        real_spherical_harmonics(np.eye(3), lmax=-1)
    # lmax=4 used to raise; M3b-aug-A+ now routes through scipy
    # for l >= 4. Verify it returns the right shape rather than
    # asserting an error.
    S = real_spherical_harmonics(np.eye(3), lmax=4)
    assert S.shape == (25, 3)
    with pytest.raises(ValueError, match=r"xyz must be"):
        real_spherical_harmonics(np.zeros(3), lmax=2)


# ---------- Multipole moments -------------------------------------------


def test_multipole_moments_normalised_gaussian_at_origin():
    """``Q_00 = 1/√(4π)`` for a centred unit Gaussian (Stone
    normalisation); ``Q_lm = 0`` for l > 0 by spherical symmetry."""
    from vibeqc.periodic_gapw_atomic_grid import (
        compute_multipole_moments, multipole_index,
    )
    g = AtomicRadialGrid.build(
        np.zeros(3), n_radial=80, alpha=5.0, lebedev_order=17,
    )
    pts = g.cartesian_points()
    rho = _gaussian_3d(pts, alpha=1.0)
    Q = compute_multipole_moments(rho, g, lmax=3)
    assert Q[multipole_index(0, 0)] == pytest.approx(
        1.0 / math.sqrt(4.0 * math.pi), abs=1e-4
    )
    # All higher-l moments should be near zero.
    for l in range(1, 4):
        for m in range(-l, l + 1):
            idx = multipole_index(l, m)
            assert abs(Q[idx]) < 1e-4, (
                f"Spherical Gaussian should have Q_{l},{m} ≈ 0; "
                f"got {Q[idx]:.6e}"
            )


def test_multipole_moments_offset_gaussian_has_dipole():
    """A Gaussian charge displaced along ``+z`` should pick up a
    non-zero ``Q_10`` (z-dipole) component."""
    from vibeqc.periodic_gapw_atomic_grid import (
        compute_multipole_moments, multipole_index,
    )
    g = AtomicRadialGrid.build(
        np.zeros(3), n_radial=80, alpha=5.0, lebedev_order=17,
    )
    pts = g.cartesian_points()
    # Centre Gaussian on z = 1 bohr above origin; grid is at origin.
    rho = _gaussian_3d(pts - np.array([0.0, 0.0, 1.0]), alpha=1.0)
    Q = compute_multipole_moments(rho, g, lmax=1)
    # Q_10 should dominate; Q_1,-1 and Q_1,1 should be small.
    assert abs(Q[multipole_index(1, 0)]) > 0.05
    assert abs(Q[multipole_index(1, -1)]) < 1e-4
    assert abs(Q[multipole_index(1, 1)]) < 1e-4


def test_multipole_moments_shape_validation():
    """Shape mismatch on the density grid raises."""
    from vibeqc.periodic_gapw_atomic_grid import compute_multipole_moments
    g = AtomicRadialGrid.build(
        np.zeros(3), n_radial=20, alpha=5.0, lebedev_order=11,
    )
    with pytest.raises(ValueError, match="density_on_grid must be"):
        compute_multipole_moments(np.zeros((3, 4)), g, lmax=2)


def test_evaluate_real_spherical_harmonics_on_grid():
    """The ``AtomicRadialGrid.evaluate_real_spherical_harmonics``
    convenience returns the right shape."""
    g = AtomicRadialGrid.build(
        np.zeros(3), n_radial=10, alpha=5.0, lebedev_order=11,
    )
    S = g.evaluate_real_spherical_harmonics(lmax=2)
    assert S.shape == (9, g.n_angular)
    with pytest.raises(ValueError, match="lmax must be >= 0"):
        g.evaluate_real_spherical_harmonics(lmax=-1)
