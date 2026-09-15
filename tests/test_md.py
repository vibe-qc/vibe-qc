"""General velocity-Verlet MD driver (vibeqc.md).

These tests exercise the *integrator + thermostats* on cheap analytic
potentials — no SCF — so they are fast and validate the physics directly:

* NVE: the symplectic velocity-Verlet conserves energy (bounded
  oscillation, no secular drift; amplitude scales ∝ Δt²).
* NVT: Berendsen and Nosé-Hoover both drive the mean temperature to the
  target; the Nosé-Hoover extended-system energy H_NH is conserved.

The NVT temperature tests use a *quartic-anharmonic* well: a single
Nosé-Hoover thermostat on purely harmonic modes is famously non-ergodic
(Martyna-Klein-Tuckerman 1992), so a touch of anharmonicity makes the
sampling robust across seeds.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc.md import (
    BerendsenThermostat,
    NoseHooverThermostat,
    kinetic_energy,
    maxwell_boltzmann_velocities,
    remove_com_velocity,
    run_md,
    run_nve,
    run_nvt,
    temperature_from_kinetic,
)

_KB_HARTREE_PER_K = 1.380649e-23 / 4.3597447222071e-18
_AMU_TO_ME = 1822.888486209


def _harmonic(centers, k=0.10):
    """Anchored harmonic well U = ½k Σ|x−x0|² (force_fn returns the gradient)."""

    def force_fn(x):
        d = x - centers
        return 0.5 * k * float(np.sum(d * d)), k * d

    return force_fn


def _anharmonic(centers, k=0.10, lam=0.20):
    """Quartic-anharmonic well U = Σ[½k d² + ¼λ d⁴] — ergodic under a single
    Nosé-Hoover thermostat."""

    def force_fn(x):
        d = x - centers
        e = 0.5 * k * float(np.sum(d * d)) + 0.25 * lam * float(np.sum((d * d) ** 2))
        grad = k * d + lam * d ** 3
        return e, grad

    return force_fn


# ---------------------------------------------------------------------------
# Helpers (mass / kinetic / temperature)
# ---------------------------------------------------------------------------


def test_kinetic_energy_and_temperature_helpers():
    # One atom of mass m moving at speed v: KE = ½ m v².
    m_amu = 4.0
    m_e = m_amu * _AMU_TO_ME
    v = np.array([[0.3, -0.2, 0.1]])
    ke = kinetic_energy(v, np.array([m_e]))
    assert ke == pytest.approx(0.5 * m_e * (0.3**2 + 0.2**2 + 0.1**2))
    # T from equipartition with n_dof = 3.
    t = temperature_from_kinetic(ke, 3)
    assert t == pytest.approx(2.0 * ke / (3 * _KB_HARTREE_PER_K))
    assert temperature_from_kinetic(ke, 0) == 0.0


def test_remove_com_velocity_zeroes_momentum():
    rng = np.random.default_rng(0)
    masses = np.array([1.0, 12.0, 16.0]) * _AMU_TO_ME
    v = rng.standard_normal((3, 3))
    v0 = remove_com_velocity(v, masses)
    p = np.sum(masses[:, None] * v0, axis=0)
    # Momenta are O(m·v) ~ 1e4, so judge against that scale, not absolute 0.
    assert np.max(np.abs(p)) / np.sum(masses) < 1e-14


def test_mass_resolution_from_atomic_numbers_and_explicit():
    centers = np.zeros((2, 3))
    # H2: lookup masses from Z; just check it runs and gives sane KE/T.
    t = run_md(_harmonic(centers), centers + 0.1, atomic_numbers=[1, 1],
               timestep_fs=0.2, n_steps=5, temperature_K=200.0, seed=1)
    assert t.positions.shape == (6, 2, 3)
    # Unknown Z (no tabulated mass) must raise.
    with pytest.raises(ValueError):
        run_md(_harmonic(centers), centers, atomic_numbers=[200, 200],
               timestep_fs=0.2, n_steps=1)
    # Neither masses nor atomic numbers → error.
    with pytest.raises(ValueError):
        run_md(_harmonic(centers), centers, timestep_fs=0.2, n_steps=1)


# ---------------------------------------------------------------------------
# Maxwell-Boltzmann velocity initialisation
# ---------------------------------------------------------------------------


def test_maxwell_boltzmann_hits_target_temperature():
    rng = np.random.default_rng(3)
    masses = np.full(20, 12.0) * _AMU_TO_ME
    # exact_temperature rescales so the instantaneous T is exactly the target.
    v = maxwell_boltzmann_velocities(masses, 350.0, rng=rng, remove_com=True)
    n_dof = 3 * 20 - 3
    t = temperature_from_kinetic(kinetic_energy(v, masses), n_dof)
    assert t == pytest.approx(350.0, rel=1e-9)
    # COM momentum removed.
    assert np.allclose(np.sum(masses[:, None] * v, axis=0), 0.0, atol=1e-10)
    # T = 0 → zero velocities.
    assert np.allclose(maxwell_boltzmann_velocities(masses, 0.0), 0.0)


def test_velocity_seed_is_reproducible():
    centers = np.random.default_rng(0).standard_normal((4, 3))
    kw = dict(masses_amu=np.full(4, 12.0), timestep_fs=0.5, n_steps=10,
              temperature_K=300.0, thermostat=None, remove_com=False, seed=42)
    a = run_md(_harmonic(centers), centers.copy(), **kw)
    b = run_md(_harmonic(centers), centers.copy(), **kw)
    assert np.allclose(a.velocities, b.velocities)
    assert np.allclose(a.total_energy, b.total_energy)


# ---------------------------------------------------------------------------
# NVE — energy conservation
# ---------------------------------------------------------------------------


def test_nve_conserves_energy_no_secular_drift():
    rng = np.random.default_rng(0)
    n = 8
    centers = rng.standard_normal((n, 3))
    x0 = centers + 0.3 * rng.standard_normal((n, 3))
    traj = run_nve(_harmonic(centers), x0, masses_amu=np.full(n, 12.0),
                   timestep_fs=0.5, n_steps=400, temperature_K=300.0,
                   remove_com=False, seed=1)
    e = traj.conserved_energy
    # Bounded oscillation, small.
    assert traj.energy_drift() / abs(e[0]) < 2e-3
    # No secular drift: mean of last quarter ≈ mean of first quarter.
    q = e.size // 4
    secular = abs(e[-q:].mean() - e[:q].mean()) / abs(e[0])
    assert secular < 5e-4
    # NVE bath energy is identically zero (conserved == total).
    assert np.allclose(traj.conserved_energy, traj.total_energy)
    assert traj.thermostat == "nve"


def test_nve_drift_scales_quadratically_with_timestep():
    rng = np.random.default_rng(0)
    n = 8
    centers = rng.standard_normal((n, 3))
    x0 = centers + 0.3 * rng.standard_normal((n, 3))

    def drift(dt, nsteps):
        t = run_nve(_harmonic(centers), x0.copy(), masses_amu=np.full(n, 12.0),
                    timestep_fs=dt, n_steps=nsteps, temperature_K=300.0,
                    remove_com=False, seed=1)
        return t.energy_drift()

    d1 = drift(1.0, 200)
    d2 = drift(0.5, 400)
    # Halving Δt should cut the symplectic oscillation amplitude ~4× (∝ Δt²).
    assert 3.0 < d1 / d2 < 5.5


# ---------------------------------------------------------------------------
# NVT — Berendsen + Nosé-Hoover temperature control
# ---------------------------------------------------------------------------


def test_berendsen_controls_temperature():
    rng = np.random.default_rng(7)
    n = 6
    centers = rng.standard_normal((n, 3))
    traj = run_md(_anharmonic(centers), centers.copy(), masses_amu=np.full(n, 12.0),
                  timestep_fs=0.5, n_steps=4000, temperature_K=300.0,
                  thermostat="berendsen", thermostat_tau_fs=25.0,
                  remove_com=False, seed=2)
    assert traj.thermostat == "berendsen"
    # Strong-coupling thermostat: mean T locks onto the target.
    assert traj.mean_temperature(0.4) == pytest.approx(300.0, abs=15.0)


def test_nose_hoover_controls_temperature_and_conserves_h_nh():
    rng = np.random.default_rng(7)
    n = 6
    centers = rng.standard_normal((n, 3))
    traj = run_nvt(_anharmonic(centers), centers.copy(), masses_amu=np.full(n, 12.0),
                   timestep_fs=0.5, n_steps=6000, temperature_K=300.0,
                   thermostat="nose_hoover", thermostat_tau_fs=25.0,
                   remove_com=False, seed=3)
    assert traj.thermostat == "nose_hoover"
    # Canonical mean temperature near target (ergodic on the anharmonic well).
    assert traj.mean_temperature(0.4) == pytest.approx(300.0, abs=40.0)
    # The extended-system energy H_NH = KE + PE + ½Qζ² + N k_B T η is the
    # genuine conserved quantity — a correct integrator keeps it fixed.
    h = traj.conserved_energy
    assert np.max(np.abs(h - h[0])) < 1e-3
    # ... and it is *not* just the bare total energy (the bath does work).
    assert np.ptp(traj.total_energy) > 10.0 * np.ptp(h)


def test_target_temperature_scales_kinetic_energy():
    # Berendsen at 600 K should equilibrate to ~2× the kinetic energy of 300 K.
    rng = np.random.default_rng(1)
    n = 6
    centers = rng.standard_normal((n, 3))

    def mean_ke(temp):
        t = run_md(_anharmonic(centers), centers.copy(), masses_amu=np.full(n, 12.0),
                   timestep_fs=0.5, n_steps=4000, temperature_K=temp,
                   thermostat="berendsen", thermostat_tau_fs=25.0,
                   remove_com=False, seed=5)
        return t.kinetic_energy[t.kinetic_energy.size // 2:].mean()

    ratio = mean_ke(600.0) / mean_ke(300.0)
    assert 1.7 < ratio < 2.3


# ---------------------------------------------------------------------------
# Thermostat objects + dispatch
# ---------------------------------------------------------------------------


def test_thermostat_string_and_instance_dispatch():
    centers = np.zeros((3, 3))
    common = dict(masses_amu=np.full(3, 12.0), timestep_fs=0.3, n_steps=20,
                  temperature_K=300.0, remove_com=False, seed=0)
    assert run_md(_harmonic(centers), centers + 0.1, thermostat=None, **common).thermostat == "nve"
    assert run_md(_harmonic(centers), centers + 0.1, thermostat="berendsen", **common).thermostat == "berendsen"
    inst = NoseHooverThermostat(300.0, tau_fs=30.0)
    assert run_md(_harmonic(centers), centers + 0.1, thermostat=inst, **common).thermostat == "nose_hoover"
    with pytest.raises(ValueError):
        run_md(_harmonic(centers), centers + 0.1, thermostat="bogus", **common)


def test_thermostat_constructors_reject_bad_tau():
    with pytest.raises(ValueError):
        BerendsenThermostat(300.0, tau_fs=0.0)
    with pytest.raises(ValueError):
        NoseHooverThermostat(300.0, tau_fs=-1.0)


def test_force_provider_bad_gradient_shape_raises():
    centers = np.zeros((3, 3))

    def bad(x):
        return 0.0, np.zeros((2, 3))  # wrong shape

    with pytest.raises(ValueError):
        run_md(bad, centers, masses_amu=np.full(3, 12.0), timestep_fs=0.2,
               n_steps=1, remove_com=False)


@pytest.mark.parametrize(
    "record_stride",
    [
        pytest.param(0, id="zero"),
        pytest.param(-2, id="negative"),
        pytest.param(np.int64(0), id="numpy-zero"),
        pytest.param(np.int64(-2), id="numpy-negative"),
        pytest.param(False, id="false"),
        pytest.param(True, id="true"),
        pytest.param(1.0, id="float"),
        pytest.param("1", id="string"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
    ],
)
def test_record_stride_rejects_invalid_values_before_force_work(record_stride):
    def unexpected_force_work(coords):
        pytest.fail("force provider ran before record_stride validation")

    with pytest.raises(
        ValueError,
        match="run_md: record_stride must be a positive integer",
    ):
        run_md(
            unexpected_force_work,
            np.zeros((1, 3)),
            masses_amu=np.ones(1),
            velocities=np.zeros((1, 3)),
            n_steps=1,
            remove_com=False,
            record_stride=record_stride,
        )


@pytest.mark.parametrize("record_stride", [1, 2, np.int64(2)])
def test_record_stride_and_frame_count(record_stride):
    centers = np.zeros((2, 3))
    traj = run_md(_harmonic(centers), centers + 0.1, masses_amu=np.full(2, 1.0),
                  timestep_fs=0.2, n_steps=10, temperature_K=100.0,
                  thermostat=None, remove_com=False, seed=0,
                  record_stride=record_stride)
    expected_steps = [0] + [
        step
        for step in range(1, 11)
        if step % int(record_stride) == 0 or step == 10
    ]
    assert traj.positions.shape[0] == len(expected_steps)
    assert traj.times_fs == pytest.approx(np.array(expected_steps) * 0.2)
