"""Well-tempered metadynamics (vibeqc.metadynamics).

Validated on cheap analytic potentials (no SCF): the collective-variable
values + gradients are checked against finite differences, the well-tempered
bias bookkeeping is checked directly, and the headline test fills a 1-D double
well and recovers the underlying free-energy profile from the accumulated
bias.

The 1-D test uses two atoms whose x-separation sits in a double well (minima
at dc±w) with stiff perpendicular confinement, so the distance CV is
effectively the 1-D separation and F(r) ≈ U(r) — letting us compare the
reconstructed free energy directly to the known potential.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc.metadynamics import (
    AngleCV,
    DistanceCV,
    WellTemperedBias,
    run_metadynamics,
)

_KB = 1.380649e-23 / 4.3597447222071e-18  # Ha/K


def _fd_cv_gradient(cv, c, h=1e-6):
    g = np.zeros_like(c)
    for i in range(c.shape[0]):
        for d in range(3):
            cp = c.copy(); cp[i, d] += h
            cm = c.copy(); cm[i, d] -= h
            g[i, d] = (cv.value(cp) - cv.value(cm)) / (2 * h)
    return g


# ---------------------------------------------------------------------------
# Collective variables
# ---------------------------------------------------------------------------


def test_distance_cv_value_and_gradient():
    c = np.array([[0.0, 0, 0], [3.0, 4.0, 0.0], [1, 1, 1]], dtype=float)
    cv = DistanceCV(0, 1)
    assert cv.value(c) == pytest.approx(5.0)
    assert np.max(np.abs(cv.gradient(c) - _fd_cv_gradient(cv, c))) < 1e-7


def test_angle_cv_value_and_gradient():
    # Right angle at the origin vertex.
    c = np.array([[0.0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float)
    c[0] = [1, 0, 0]   # i
    c[1] = [0, 0, 0]   # j (vertex)
    c[2] = [0, 1, 0]   # k
    cv = AngleCV(0, 1, 2)
    assert cv.value(c) == pytest.approx(np.pi / 2)
    rng = np.random.default_rng(0)
    c2 = rng.standard_normal((3, 3))
    assert np.max(np.abs(cv.gradient(c2) - _fd_cv_gradient(cv, c2))) < 1e-7


def test_cv_constructor_validation():
    with pytest.raises(ValueError):
        DistanceCV(1, 1)
    with pytest.raises(ValueError):
        AngleCV(0, 1, 0)


# ---------------------------------------------------------------------------
# Well-tempered bias bookkeeping
# ---------------------------------------------------------------------------


def test_bias_potential_and_force_match_finite_difference():
    bias = WellTemperedBias([DistanceCV(0, 1)], sigma=0.2, hill_height=1e-3,
                            bias_factor=10.0, temperature_K=300.0)
    for center in (2.0, 2.5, 3.1):
        bias.centers.append(np.array([center]))
        bias.heights.append(1e-3)
    # force_on_cv = -dV/ds; check against FD of potential.
    s0 = np.array([2.6])
    h = 1e-5
    dV = (bias.potential(s0 + [h]) - bias.potential(s0 - [h])) / (2 * h)
    assert bias.force_on_cv(s0)[0] == pytest.approx(-dV, abs=1e-6)


def test_bias_atomic_gradient_matches_finite_difference():
    bias = WellTemperedBias([DistanceCV(0, 1), AngleCV(0, 1, 2)], sigma=[0.2, 0.1],
                            hill_height=1e-3, bias_factor=8.0, temperature_K=300.0)
    rng = np.random.default_rng(1)
    for _ in range(4):
        c = rng.standard_normal((3, 3))
        bias.deposit(bias.cv_values(c))
    c0 = rng.standard_normal((3, 3))
    s0 = bias.cv_values(c0)
    ana = bias.atomic_gradient(c0, s0)
    # FD of V_bias(s(R)) wrt the Cartesian coordinates.
    fd = np.zeros_like(c0)
    h = 1e-6
    for i in range(3):
        for d in range(3):
            cp = c0.copy(); cp[i, d] += h
            cm = c0.copy(); cm[i, d] -= h
            fd[i, d] = (bias.potential(bias.cv_values(cp))
                        - bias.potential(bias.cv_values(cm))) / (2 * h)
    assert np.max(np.abs(ana - fd)) < 1e-6


def test_well_tempered_height_decreases_as_bias_accumulates():
    bias = WellTemperedBias([DistanceCV(0, 1)], sigma=0.2, hill_height=1e-3,
                            bias_factor=5.0, temperature_K=300.0)
    s = np.array([2.5])
    h_first = bias.deposit(s)
    # Depositing repeatedly at the same point: each hill is shorter than the
    # last (well-tempered damping by the accumulated bias).
    heights = [h_first] + [bias.deposit(s) for _ in range(5)]
    assert all(heights[k + 1] < heights[k] for k in range(len(heights) - 1))
    assert heights[0] == pytest.approx(1e-3)  # first hill is the full height


def test_free_energy_is_scaled_negative_bias():
    bias = WellTemperedBias([DistanceCV(0, 1)], sigma=0.3, hill_height=1e-3,
                            bias_factor=6.0, temperature_K=300.0)
    for _ in range(10):
        bias.deposit(np.array([2.5]))
    grid = np.linspace(2.0, 3.0, 11)
    f = bias.free_energy_on_grid(grid)
    v = bias.potential_on_grid(grid)
    gamma = 6.0
    expected = -(gamma / (gamma - 1.0)) * v
    expected -= expected.min()
    assert np.allclose(f, expected)
    assert f.min() == pytest.approx(0.0)


def test_bias_constructor_validation():
    with pytest.raises(ValueError):
        WellTemperedBias([], sigma=0.1, hill_height=1e-3, bias_factor=5.0,
                         temperature_K=300.0)
    with pytest.raises(ValueError):  # bias_factor must be > 1
        WellTemperedBias([DistanceCV(0, 1)], sigma=0.1, hill_height=1e-3,
                         bias_factor=1.0, temperature_K=300.0)
    with pytest.raises(ValueError):  # one sigma per CV
        WellTemperedBias([DistanceCV(0, 1)], sigma=[0.1, 0.2], hill_height=1e-3,
                         bias_factor=5.0, temperature_K=300.0)


# ---------------------------------------------------------------------------
# Headline: fill a 1-D double well, recover the free-energy profile
# ---------------------------------------------------------------------------


def _double_well_force_fn(A=0.02, dc=3.0, w=0.8, kperp=0.6):
    """Two atoms: U is a double well in their x-separation (minima dc±w) plus
    stiff perpendicular confinement, so DistanceCV(0,1) is ≈ the 1-D separation
    and F(r) ≈ U(r)."""

    def force_fn(R):
        R = np.asarray(R, float)
        dx = R[1, 0] - R[0, 0]
        s = (dx - dc) ** 2 - w ** 2
        e = A * s * s
        g = np.zeros_like(R)
        d_e_d_dx = A * 4.0 * s * (dx - dc)
        g[1, 0] += d_e_d_dx
        g[0, 0] -= d_e_d_dx
        for a in range(2):
            for d in (1, 2):
                e += 0.5 * kperp * R[a, d] ** 2
                g[a, d] += kperp * R[a, d]
        return e, g

    return force_fn


def test_metadynamics_fills_double_well_and_recovers_free_energy():
    A, dc, w = 0.02, 3.0, 0.8
    force_fn = _double_well_force_fn(A=A, dc=dc, w=w)
    x0 = np.array([[0.0, 0, 0], [2.2, 0, 0]])  # start in the left (dc-w) well
    res = run_metadynamics(
        force_fn, x0, [DistanceCV(0, 1)], masses_amu=np.full(2, 12.0),
        bias_factor=8.0, hill_height=8e-4, hill_sigma=0.15,
        deposition_stride=10, timestep_fs=0.5, n_steps=12000,
        temperature_K=300.0, thermostat="berendsen", thermostat_tau_fs=30.0,
        remove_com=False, seed=1, record_stride=20,
    )

    cv = res.cv_trajectory[:, 0]
    # The bias drives the system over the barrier into *both* wells.
    assert cv.min() < 2.5 and cv.max() > 3.5
    assert res.n_hills > 100

    grid = np.linspace(1.9, 4.1, 221)
    f = res.free_energy(grid)
    u = A * ((grid - dc) ** 2 - w ** 2) ** 2
    u -= u.min()
    # Reconstructed minima land near the two wells (2.2 / 3.8 bohr).
    mlo = grid[grid < dc][np.argmin(f[grid < dc])]
    mhi = grid[grid >= dc][np.argmin(f[grid >= dc])]
    assert 1.9 < mlo < 2.6
    assert 3.3 < mhi < 4.1
    # The central barrier height is recovered within a factor of ~2.
    barrier = f[(grid > 2.85) & (grid < 3.15)].max()
    assert 0.5 * A * w ** 4 < barrier < 2.5 * A * w ** 4
    # The whole profile matches the underlying potential to a few k_B T.
    sel = (grid > 2.0) & (grid < 4.0)
    rmsd = np.sqrt(np.mean((f - u)[sel] ** 2))
    assert rmsd < 6.0 * _KB * 300.0


def test_metadynamics_supports_multiple_cvs():
    # Two CVs (two distances) — just exercise the machinery end-to-end.
    force_fn = _double_well_force_fn()
    x0 = np.array([[0.0, 0, 0], [2.2, 0, 0], [0.0, 2.0, 0.0]])
    res = run_metadynamics(
        force_fn, x0, [DistanceCV(0, 1), DistanceCV(0, 2)],
        masses_amu=np.full(3, 12.0), bias_factor=6.0, hill_height=5e-4,
        hill_sigma=[0.15, 0.2], deposition_stride=20, timestep_fs=0.5,
        n_steps=400, temperature_K=300.0, thermostat="berendsen",
        remove_com=False, seed=0, record_stride=20,
    )
    assert res.cv_trajectory.shape[1] == 2
    assert res.bias.n_cv == 2
    # A 2-D free-energy read-out on a small grid of CV points.
    pts = np.array([[2.2, 2.0], [3.8, 2.5]])
    fe = res.free_energy(pts)
    assert fe.shape == (2,) and fe.min() == pytest.approx(0.0)
