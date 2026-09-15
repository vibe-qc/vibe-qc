"""Phase 12e-c-3a: FFTW3-based periodic Poisson solver.

Tests the reciprocal-space solver for both the unscreened Coulomb
kernel 1/r and the Ewald long-range erf(ωr)/r kernel. Key contracts:

- The G=0 gauge is pinned to zero, so every returned V has zero mean.
- The two limits of the ω-screened kernel behave correctly:
  * ω → ∞ (no screening) collapses to the unscreened Coulomb result.
  * ω → 0 (full screening) collapses to zero everywhere.
- Linearity in ρ: V(ρ₁ + ρ₂) = V(ρ₁) + V(ρ₂).
- Symmetric ρ gives symmetric V.
- The potential of an isolated Gaussian in a large box matches the
  analytical erf-shape up to finite-box corrections.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


@pytest.fixture
def cubic_grid():
    """Small cubic grid for fast tests. Box large enough that a
    centered Gaussian (α = 2) doesn't overlap its periodic image."""
    a = 12.0
    n = 24
    lat = np.diag([a, a, a])
    xs = np.linspace(0, a, n, endpoint=False)
    return a, n, lat, xs


def _gaussian_density(a, n, xs, alpha, center=None):
    if center is None:
        center = [a / 2, a / 2, a / 2]
    x, y, z = np.meshgrid(xs, xs, xs, indexing="ij")
    r2 = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2
    rho = (alpha / np.pi) ** 1.5 * np.exp(-alpha * r2)
    return rho


# ---------------------------------------------------------------------------
# Gauge and shape
# ---------------------------------------------------------------------------

def test_V_has_zero_mean_coulomb(cubic_grid):
    """V(G=0) is pinned to zero, so V.mean() is zero at numerical
    precision."""
    a, n, lat, xs = cubic_grid
    rho = _gaussian_density(a, n, xs, alpha=2.0)
    V = vq.solve_poisson_coulomb(rho, lat)
    assert V.shape == (n, n, n)
    assert abs(V.mean()) < 1e-12


def test_V_has_zero_mean_erf_screened(cubic_grid):
    a, n, lat, xs = cubic_grid
    rho = _gaussian_density(a, n, xs, alpha=2.0)
    V = vq.solve_poisson_erf_screened(rho, lat, 0.5)
    assert abs(V.mean()) < 1e-12


def test_zero_omega_rejected(cubic_grid):
    """ω must be strictly positive — ω = 0 is an ill-defined limit
    (use ``solve_poisson_coulomb`` for the unscreened case)."""
    a, n, lat, xs = cubic_grid
    rho = _gaussian_density(a, n, xs, alpha=2.0)
    with pytest.raises((ValueError, RuntimeError)):
        vq.solve_poisson_erf_screened(rho, lat, 0.0)
    with pytest.raises((ValueError, RuntimeError)):
        vq.solve_poisson_erf_screened(rho, lat, -1.0)


# ---------------------------------------------------------------------------
# Limits of the ω-screened kernel
# ---------------------------------------------------------------------------

def test_large_omega_recovers_unscreened_coulomb(cubic_grid):
    """erf(ωr)/r → 1/r as ω → ∞, so the long-range V must match the
    full Coulomb V at large ω."""
    a, n, lat, xs = cubic_grid
    rho = _gaussian_density(a, n, xs, alpha=2.0)
    V_full = vq.solve_poisson_coulomb(rho, lat)
    V_lr = vq.solve_poisson_erf_screened(rho, lat, omega=1000.0)
    # ω = 1000 filters only G² > ~4 × 10⁶ — well outside the grid's
    # Nyquist range (max |G|² ~ 350 on a 24-point / 12-bohr grid), so
    # the filter is effectively 1 everywhere. ω = 100 leaves a ~1 %
    # suppression at the highest G, which prints as a mismatch in the
    # 7th decimal — unnecessarily tight for a limit test.
    assert np.allclose(V_lr, V_full, atol=1e-10)


def test_small_omega_gives_zero_long_range(cubic_grid):
    """erf(ωr)/r → 0 as ω → 0, so the long-range V collapses to zero
    everywhere (to within the chosen tolerance)."""
    a, n, lat, xs = cubic_grid
    rho = _gaussian_density(a, n, xs, alpha=2.0)
    V_lr = vq.solve_poisson_erf_screened(rho, lat, omega=1e-4)
    assert abs(V_lr).max() < 1e-4


# ---------------------------------------------------------------------------
# Linearity
# ---------------------------------------------------------------------------

def test_linearity_in_density(cubic_grid):
    """V is linear in ρ: V(ρ₁ + ρ₂) = V(ρ₁) + V(ρ₂)."""
    a, n, lat, xs = cubic_grid
    rho1 = _gaussian_density(a, n, xs, alpha=2.0, center=[a/3, a/2, a/2])
    rho2 = _gaussian_density(a, n, xs, alpha=3.0, center=[2*a/3, a/2, a/2])

    V1 = vq.solve_poisson_coulomb(rho1, lat)
    V2 = vq.solve_poisson_coulomb(rho2, lat)
    V_sum = vq.solve_poisson_coulomb(rho1 + rho2, lat)
    assert np.allclose(V_sum, V1 + V2, atol=1e-12)


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------

def test_centred_gaussian_gives_symmetric_potential(cubic_grid):
    """Density centered at the box center is invariant under inversion
    through the center; V must share that symmetry."""
    a, n, lat, xs = cubic_grid
    rho = _gaussian_density(a, n, xs, alpha=2.0)
    V = vq.solve_poisson_coulomb(rho, lat)
    # Inversion: V[i,j,k] must equal V[n-i, n-j, n-k] (with the
    # convention that index 0 maps to itself, not to n).
    # In an FFT grid, true inversion symmetry is V[i,j,k] = V[-i,-j,-k]
    # with wrap-around, i.e. V[i,j,k] = V[(n-i) % n, (n-j) % n, (n-k) % n].
    i, j, k = np.arange(n), np.arange(n), np.arange(n)
    V_mirror = V[(n - i) % n][:, (n - j) % n][:, :, (n - k) % n]
    assert np.allclose(V, V_mirror, atol=1e-10)


# ---------------------------------------------------------------------------
# Quantitative Gaussian check
# ---------------------------------------------------------------------------

def test_gaussian_self_potential_peak():
    """The potential of an isolated Gaussian (α, unit charge) at its
    center is V_iso(0) = 2 sqrt(α/π). In a periodic cell with G=0
    removed, V at the center minus V at the corner converges to
    V_iso(0) - erf(sqrt(α) r_corner)/r_corner as the box grows.
    Check that the *shape* matches the isolated expectation to within
    a few percent at a reasonable box size (a = 16 bohr).
    """
    a = 16.0
    n = 48
    lat = np.diag([a, a, a])
    xs = np.linspace(0, a, n, endpoint=False)
    alpha = 2.0
    rho = _gaussian_density(a, n, xs, alpha=alpha)

    V = vq.solve_poisson_coulomb(rho, lat)

    from math import erf, sqrt, pi
    V_centre = V[n // 2, n // 2, n // 2]
    V_corner = V[0, 0, 0]
    V_iso_centre = 2.0 * sqrt(alpha / pi)
    r_corner = sqrt(3.0) * a / 2.0
    V_iso_corner = erf(sqrt(alpha) * r_corner) / r_corner
    delta_grid = V_centre - V_corner
    delta_iso = V_iso_centre - V_iso_corner
    # Within ~5% of the isolated answer at this box size.
    assert abs(delta_grid - delta_iso) / abs(delta_iso) < 0.05


# ---------------------------------------------------------------------------
# Hartree-energy helper
# ---------------------------------------------------------------------------

def test_hartree_energy_is_half_integral(cubic_grid):
    """The helper must equal (1/2) Σ ρ V · dV manually."""
    a, n, lat, xs = cubic_grid
    rho = _gaussian_density(a, n, xs, alpha=2.0)
    V = vq.solve_poisson_coulomb(rho, lat)
    V_cell = float(np.linalg.det(lat))
    E_h = vq.hartree_energy_on_grid(rho, V, V_cell)
    dV = V_cell / rho.size
    E_manual = 0.5 * (rho * V).sum() * dV
    assert E_h == pytest.approx(E_manual, rel=1e-12)


# ---------------------------------------------------------------------------
# Skew-cell support
# ---------------------------------------------------------------------------

def test_skew_lattice_supported_zero_mean():
    n = 16
    rho = np.zeros((n, n, n))
    rho[8, 8, 8] = 1.0
    # Non-orthogonal: a_2 has a component along a_1.
    lat = np.array([
        [10.0, 1.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
    ])
    V = vq.solve_poisson_coulomb(rho, lat)
    assert V.shape == rho.shape
    assert np.all(np.isfinite(V))
    assert abs(float(V.mean())) < 1e-12
