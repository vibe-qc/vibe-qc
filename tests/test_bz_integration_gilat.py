"""Standalone tests for the Gilat-Raubenheimer BZ-integration kernel.

These exercise the kernel in isolation (no SCF / driver wiring): the cell
occupied-fraction closed form vs Monte-Carlo, electron-count conservation, the
insulator integer-occupation limit, and the defining property of the method --
that GR integrates the BZ more accurately than naive step counting on a metal.

GR method + formula: G. Gilat & L. J. Raubenheimer, Phys. Rev. 144, 390 (1966).
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc.bz_integration import (
    gilat_dos,
    gilat_raubenheimer_occupations,
    grid_band_slopes,
    occupied_fraction,
)


def _trapz(y, x):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


# Fixed slope tuples covering generic + degenerate (1/2/3 zero slopes) cells.
_SLOPE_CASES = [
    (1.0, 1.0, 1.0),
    (2.0, 1.0, 0.5),
    (1.5, -0.7, 0.2),
    (-2.0, 1.3, 0.9),
    (3.0, 0.0, 0.0),     # one direction dispersive (2 flat)
    (1.0, 2.0, 0.0),     # planar (1 flat)
    (-1.7, 0.0, 0.6),
    (0.0, 0.0, 0.0),     # fully flat cell
]


def _fraction_monte_carlo(delta, g1, g2, g3, rng, n=400_000):
    """Direct Monte-Carlo estimate of the occupied cell fraction."""
    t = rng.uniform(-0.5, 0.5, size=(n, 3))
    e = g1 * t[:, 0] + g2 * t[:, 1] + g3 * t[:, 2]
    return float(np.mean(e < delta))


def test_occupied_fraction_matches_monte_carlo():
    rng = np.random.default_rng(20260616)
    for (g1, g2, g3) in _SLOPE_CASES:
        spread = 0.5 * (abs(g1) + abs(g2) + abs(g3))
        # sample delta across (and a little beyond) the cell's energy spread
        for delta in np.linspace(-spread - 0.4, spread + 0.4, 9):
            if abs(delta) < 1e-6:
                continue  # skip the exact-edge ambiguity of a flat cell
            f_formula = float(occupied_fraction(delta, g1, g2, g3))
            f_mc = _fraction_monte_carlo(delta, g1, g2, g3, rng)
            assert abs(f_formula - f_mc) < 5e-3, (
                f"slopes={(g1, g2, g3)} delta={delta}: "
                f"formula={f_formula:.4f} mc={f_mc:.4f}"
            )
            assert 0.0 <= f_formula <= 1.0


def test_occupied_fraction_endpoints_and_monotonicity():
    g1, g2, g3 = 1.3, 0.8, 0.4
    spread = 0.5 * (g1 + g2 + g3)
    assert occupied_fraction(-spread - 1.0, g1, g2, g3) == pytest.approx(0.0)
    assert occupied_fraction(+spread + 1.0, g1, g2, g3) == pytest.approx(1.0)
    # at delta = 0 (Fermi level through the cell centre) the linear band splits
    # the cell exactly in half
    assert occupied_fraction(0.0, g1, g2, g3) == pytest.approx(0.5, abs=1e-12)
    # monotone increasing in delta
    deltas = np.linspace(-spread, spread, 25)
    fr = np.array([float(occupied_fraction(d, g1, g2, g3)) for d in deltas])
    assert np.all(np.diff(fr) >= -1e-12)


def test_flat_cell_is_a_step():
    # asum == 0 -> exact step at delta = 0
    assert occupied_fraction(-0.1, 0.0, 0.0, 0.0) == pytest.approx(0.0)
    assert occupied_fraction(+0.1, 0.0, 0.0, 0.0) == pytest.approx(1.0)


@pytest.mark.parametrize("spin_degeneracy", [1.0, 2.0])
def test_electron_count_conserved(spin_degeneracy):
    rng = np.random.default_rng(7)
    eps = rng.uniform(-1.0, 1.0, size=(6, 6, 6, 4))
    n_cells = 6 * 6 * 6
    g = spin_degeneracy
    for nelec in (g * 1.0, g * 2.3, g * 3.7):
        occ, ef = gilat_raubenheimer_occupations(eps, nelec, g)
        total = float(occ.sum()) / n_cells
        assert total == pytest.approx(nelec, abs=1e-6)
        assert occ.min() >= -1e-9
        assert occ.max() <= g + 1e-9
        assert np.isfinite(ef)


def test_insulator_integer_occupations():
    # two well-separated, nearly flat bands -> a clean gap; filling the lower
    # band must give integer occupations (2 below, 0 above) and E_F in the gap.
    rng = np.random.default_rng(11)
    lo = -1.0 + 0.02 * rng.standard_normal((5, 5, 5, 1))
    hi = +1.0 + 0.02 * rng.standard_normal((5, 5, 5, 1))
    eps = np.concatenate([lo, hi], axis=3)  # (5,5,5,2)
    occ, ef = gilat_raubenheimer_occupations(eps, n_electrons_per_cell=2.0)
    assert np.max(np.abs(occ[..., 0] - 2.0)) < 1e-6
    assert np.max(np.abs(occ[..., 1] - 0.0)) < 1e-6
    # E_F sits in the gap; a step-counting bisection over the count=2 plateau
    # converges to the gap edge (top of the lower band), so allow the whole gap.
    b0_max = float(eps[..., 0].max())
    b1_min = float(eps[..., 1].min())
    assert b0_max - 0.1 <= ef <= b1_min + 0.1


def _tight_binding_band(n, t=1.0):
    """Simple-cubic nearest-neighbour band eps(k) = -2t sum_i cos(k_i)."""
    idx = np.arange(n)
    c = np.cos(2.0 * np.pi * idx / n)
    eps = -2.0 * t * (c[:, None, None] + c[None, :, None] + c[None, None, :])
    return eps[..., None]  # (n,n,n,1)


def _gr_count(eps_grid, ef, g=1.0):
    g1, g2, g3 = grid_band_slopes(eps_grid)
    frac = occupied_fraction(ef - eps_grid, g1, g2, g3)
    return g * float(frac.sum()) / (eps_grid.shape[0] * eps_grid.shape[1] * eps_grid.shape[2])


def test_metal_more_accurate_than_step_counting():
    # Integrated electron-count error over a Fermi-level sweep: GR (cell
    # linearisation) must beat naive step counting against a dense reference.
    coarse = _tight_binding_band(6)
    dense = _tight_binding_band(60)
    dense_vals = dense.ravel()

    efs = np.linspace(-5.0, 5.0, 41)
    err_gr = 0.0
    err_step = 0.0
    for ef in efs:
        truth = float(np.mean(dense_vals < ef))
        step = float(np.mean(coarse.ravel() < ef))      # naive step counting
        gr = _gr_count(coarse, ef)                       # Gilat-Raubenheimer
        err_step += abs(step - truth)
        err_gr += abs(gr - truth)

    assert err_gr < err_step, (
        f"GR integrated error {err_gr:.4f} not better than step "
        f"counting {err_step:.4f}"
    )
    # GR should be a substantial improvement, not a marginal one
    assert err_gr < 0.7 * err_step


def test_gilat_dos_integrates_to_total_states():
    # integral of D(E) dE over all energies = total states per cell = g * nband
    eps = _tight_binding_band(10)  # (10,10,10,1) -> 1 band
    g = 2.0
    energies = np.linspace(float(eps.min()) - 1.0, float(eps.max()) + 1.0, 800)
    dos = gilat_dos(eps, energies, spin_degeneracy=g)
    assert np.all(dos >= 0.0)  # DOS is clamped non-negative
    assert _trapz(dos, energies) == pytest.approx(g * 1, rel=0.02)


def test_gilat_dos_zero_in_gap_positive_in_band():
    rng = np.random.default_rng(2)
    lo = -1.0 + 0.05 * rng.standard_normal((6, 6, 6, 1))
    hi = +1.0 + 0.05 * rng.standard_normal((6, 6, 6, 1))
    eps = np.concatenate([lo, hi], axis=3)  # gapped two-band
    dos_gap = float(gilat_dos(eps, np.array([0.0]))[0])     # mid-gap
    dos_band = float(gilat_dos(eps, np.array([-1.0]))[0])   # centre of lower band
    assert dos_gap < 1e-2
    assert dos_band > dos_gap
