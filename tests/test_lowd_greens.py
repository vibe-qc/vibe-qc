"""Mixed-boundary (wire/slab) Green's-function kernels (D3 M2).

A dependency-free correctness gate for the low-D kernels of
``docs/aiccm2026dev_a_lowd_greens.md``: each solves Poisson's equation with the
source confined to the charge axis / plane, so **away from the source it is
harmonic** — ``∇²G^{1D}=0`` for ``ρ>0`` and ``∇²G^{2D}=0`` for ``z≠0`` — and it
respects the lattice periodicity and transverse-reflection symmetry. These pin
the kernel form with no external reference; the four-center routing and the M3
reduction to ``v_E^{3D}`` build on them.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc.periodic.lowd_greens import (
    slab_greens,
    slab_self_energy,
    wire_greens,
    wire_self_energy,
)


# --------------------------------------------------------------------------- #
# 1-D wire.
# --------------------------------------------------------------------------- #
def _wire_laplacian(rho, z, L, h=2e-3, n_recip=400):
    """Cylindrical (axisymmetric) ∇²G = ∂_ρρ + (1/ρ)∂_ρ + ∂_zz, by finite diff."""
    def g(r, zz):
        return wire_greens(r, zz, period=L, n_recip=n_recip)
    g0 = g(rho, z)
    d_rr = (g(rho + h, z) - 2 * g0 + g(rho - h, z)) / h**2
    d_r = (g(rho + h, z) - g(rho - h, z)) / (2 * h)
    d_zz = (g(rho, z + h) - 2 * g0 + g(rho, z - h)) / h**2
    return d_rr + d_r / rho + d_zz


@pytest.mark.parametrize("rho,z", [(0.4, 0.2), (0.6, 0.0), (0.35, 0.45), (0.5, -0.3)])
def test_wire_is_harmonic_off_axis(rho, z):
    """G^{1D} solves Laplace's equation off the wire axis (ρ > 0)."""
    L = 1.0
    lap = _wire_laplacian(rho, z, L)
    # The central-difference truncation floor (h=2e-3) on this kernel is ~1e-4;
    # the individual second-derivative terms are O(10) (e.g. the 2-D line term
    # ∂_ρρ(−2ln ρ) = 2/ρ² and (1/ρ)∂_ρ(−2ln ρ) = −2/ρ² cancel), so the harmonic
    # cancellation is genuine to ~5 orders of magnitude.
    assert abs(lap) < 1e-3, f"∇²G^1D = {lap:.3e} at (ρ={rho}, z={z}) — not harmonic"


def test_wire_periodic_in_z():
    L = 4.0
    a = wire_greens(0.7, 0.9, period=L)
    b = wire_greens(0.7, 0.9 + L, period=L)
    c = wire_greens(0.7, 0.9 - 2 * L, period=L)
    assert a == pytest.approx(b, abs=1e-10)
    assert a == pytest.approx(c, abs=1e-10)


def test_wire_even_in_z():
    L = 3.0
    assert wire_greens(0.5, 0.8, period=L) == pytest.approx(
        wire_greens(0.5, -0.8, period=L), abs=1e-12)


def test_wire_on_axis_raises():
    with pytest.raises(ValueError):
        wire_greens(0.0, 0.1, period=1.0)


def test_wire_unit_source_strength():
    """G^{1D} carries exactly a unit charge: G − 1/ρ → a finite self-energy ξ
    (not ±∞), i.e. ρ·G → 1 as ρ → 0. A non-unit source would make G − 1/ρ diverge.
    """
    L = 1.0
    xi = {ro: wire_greens(ro, 0.0, period=L, n_recip=int(30 * L / ro)) - 1.0 / ro
          for ro in (0.01, 0.005)}
    # Finite, converging self-energy ⇒ unit source.
    assert abs(xi[0.005] - xi[0.01]) < 1e-3
    assert abs(xi[0.005]) < 10.0                       # order-1, not diverging
    # ρ·G → 1, and closer at the smaller ρ.
    s = {ro: ro * wire_greens(ro, 0.0, period=L, n_recip=int(30 * L / ro))
         for ro in (0.01, 0.005)}
    assert abs(s[0.005] - 1.0) < abs(s[0.01] - 1.0) < 0.02


# --------------------------------------------------------------------------- #
# 2-D slab.
# --------------------------------------------------------------------------- #
def _slab_laplacian(rp, z, lat, h=2e-3, n_shell=60):
    """Cartesian ∇²G = ∂_xx + ∂_yy + ∂_zz, by finite difference (z ≠ 0)."""
    def g(x, y, zz):
        return slab_greens([x, y], zz, lattice2d=lat, n_shell=n_shell)
    x, y = rp
    g0 = g(x, y, z)
    d_xx = (g(x + h, y, z) - 2 * g0 + g(x - h, y, z)) / h**2
    d_yy = (g(x, y + h, z) - 2 * g0 + g(x, y - h, z)) / h**2
    d_zz = (g(x, y, z + h) - 2 * g0 + g(x, y, z - h)) / h**2
    return d_xx + d_yy + d_zz


@pytest.mark.parametrize("x,y,z", [(0.3, 0.2, 0.5), (0.1, 0.4, -0.7), (0.45, 0.15, 0.9)])
def test_slab_is_harmonic_off_plane(x, y, z):
    """G^{2D} solves Laplace's equation off the source plane (z ≠ 0)."""
    lat = np.array([[1.0, 0.0], [0.0, 1.0]])  # square, a = 1
    lap = _slab_laplacian((x, y), z, lat)
    assert abs(lap) < 1e-3, f"∇²G^2D = {lap:.3e} at ({x},{y},{z}) — not harmonic"


def test_slab_periodic_in_plane():
    lat = np.array([[1.3, 0.0], [0.4, 1.1]])  # oblique
    g0 = slab_greens([0.2, 0.3], 0.6, lattice2d=lat)
    # shift by a lattice vector a1
    g1 = slab_greens([0.2 + 1.3, 0.3 + 0.0], 0.6, lattice2d=lat)
    g2 = slab_greens([0.2 + 0.4, 0.3 + 1.1], 0.6, lattice2d=lat)  # shift by a2
    assert g0 == pytest.approx(g1, abs=1e-9)
    assert g0 == pytest.approx(g2, abs=1e-9)


def test_slab_even_in_z():
    lat = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert slab_greens([0.3, 0.2], 0.7, lattice2d=lat) == pytest.approx(
        slab_greens([0.3, 0.2], -0.7, lattice2d=lat), abs=1e-12)


def test_wire_self_energy_value_and_convergence():
    """ξ_wire = lim_{ρ→0}[G^{1D}−1/ρ] is a finite, reproducible Madelung-analog
    constant; the r² Richardson extraction is stable across the sampling radius."""
    xi = wire_self_energy(1.0)
    assert xi == pytest.approx(-0.23186, abs=2e-4)     # the measured self-energy
    # Stable under halving the Richardson sampling radius (the limit is well-posed).
    assert wire_self_energy(1.0, _rel=0.005) == pytest.approx(xi, abs=2e-4)
    # Distinct period → distinct self-energy (not a constant artefact).
    assert abs(wire_self_energy(2.0) - xi) > 0.1


def test_wire_self_energy_gauge_shift():
    """The ρ₀ gauge enters only through the line term −(2/L)ln(ρ/ρ₀): changing ρ₀
    shifts ξ by exactly (2/L)·ln(ρ₀'/ρ₀)."""
    L = 1.0
    shift = wire_self_energy(L, rho0=float(np.e)) - wire_self_energy(L, rho0=1.0)
    assert shift == pytest.approx((2.0 / L) * np.log(np.e / 1.0), abs=1e-6)


def test_slab_self_energy_value():
    xi = slab_self_energy(np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert xi == pytest.approx(-3.90061, abs=2e-3)
    assert slab_self_energy(np.array([[1.0, 0.0], [0.0, 1.0]]), _rel=0.005) == pytest.approx(
        xi, abs=2e-3)


def test_slab_unit_source_strength():
    """G^{2D} carries exactly a unit charge: G − 1/z → a finite self-energy as
    z → 0 along the normal, i.e. z·G → 1. A non-unit source would diverge.
    """
    a = 1.0
    lat = np.array([[a, 0.0], [0.0, a]])

    def g(z):
        return slab_greens([0.0, 0.0], z, lattice2d=lat, n_shell=int(2.0 / z))

    xi = {z: g(z) - 1.0 / z for z in (0.01, 0.005)}
    assert abs(xi[0.005] - xi[0.01]) < 2e-3            # finite, converging ⇒ unit source
    assert abs(xi[0.005]) < 10.0
    # z·G → 1 (error ≈ z·|ξ|, larger here since |ξ_slab| ≈ 3.9): converging, and
    # within z·|ξ| at the smallest z.
    s = {z: z * g(z) for z in (0.01, 0.005)}
    assert abs(s[0.005] - 1.0) < abs(s[0.01] - 1.0)    # converging toward 1
    assert abs(s[0.005] - 1.0) < 0.03
