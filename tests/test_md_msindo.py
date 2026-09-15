"""MSINDO wired into the general MD driver (vibeqc.md via msindo_force_provider).

Short trajectories only — each MD step costs 6N+1 MSINDO SCFs (the
finite-difference gradient), so these tests deliberately run a handful of
steps on tiny molecules. The integrator physics itself is validated cheaply
on analytic potentials in ``test_md.py``; here we check the *wiring*: the
provider's energy/gradient agree with the engine, an NVE run conserves energy
on the real INDO surface, a thermostat run is labelled correctly, and the
citation surface fires the MD + thermostat papers.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc.md import MDTrajectory, run_md
from vibeqc.metadynamics import DistanceCV, MetadynamicsResult
from vibeqc.semiempirical.methods.msindo import (
    ANGSTROM_TO_BOHR,
    msindo_force_provider,
    run_msindo,
    run_msindo_md,
    run_msindo_metadynamics,
)

_ANG2BOHR = ANGSTROM_TO_BOHR


def _h2o_angstrom():
    return [8, 1, 1], np.array(
        [[0.0, 0.0, 0.0],
         [0.0, 0.7572, 0.5865],
         [0.0, -0.7572, 0.5865]]
    )


def test_force_provider_matches_engine_energy():
    z, xyz = _h2o_angstrom()
    provider = msindo_force_provider(z)
    e, g = provider(xyz * _ANG2BOHR)
    # Energy is exactly run_msindo at this geometry.
    assert e == pytest.approx(run_msindo(z, xyz).total_energy, abs=1e-9)
    # Gradient: right shape, finite, in Ha/bohr.
    assert g.shape == (3, 3)
    assert np.all(np.isfinite(g))


def test_msindo_nve_conserves_energy():
    z, xyz = _h2o_angstrom()
    traj = run_msindo_md(z, xyz, timestep_fs=0.3, n_steps=8,
                         temperature_K=300.0, thermostat=None, seed=1)
    assert isinstance(traj, MDTrajectory)
    assert traj.positions.shape == (9, 3, 3)
    assert traj.thermostat == "nve"
    # FD gradients are consistent with the energy surface, so the symplectic
    # integrator conserves the total energy to well under a milli-Hartree.
    assert traj.energy_drift() / abs(traj.total_energy[0]) < 1e-4


def test_msindo_md_runs_through_general_driver_directly():
    # The general driver consumes the provider with no MSINDO knowledge.
    z, xyz = _h2o_angstrom()
    provider = msindo_force_provider(z)
    traj = run_md(provider, xyz * _ANG2BOHR, atomic_numbers=z,
                  timestep_fs=0.3, n_steps=4, temperature_K=250.0,
                  thermostat="berendsen", thermostat_tau_fs=20.0, seed=0)
    assert traj.thermostat == "berendsen"
    assert traj.positions.shape[0] == 5


def test_msindo_md_emits_citations(tmp_path):
    z, xyz = _h2o_angstrom()
    stem = tmp_path / "md_nh"
    run_msindo_md(z, xyz, timestep_fs=0.3, n_steps=2,
                  thermostat="nose_hoover", seed=1, output=str(stem))
    refs = (tmp_path / "md_nh.references").read_text()
    # MSINDO + velocity-Verlet + Nosé + Hoover; libint suppressed (INDO).
    assert "swope_velocity_verlet_1982" in refs
    assert "nose_thermostat_1984" in refs
    assert "hoover_thermostat_1985" in refs
    assert "ahlswede_jug_msindo_1_1999" in refs
    assert "berendsen_thermostat_1984" not in refs
    assert "valeev_libint" not in refs


def test_msindo_metadynamics_runs_and_emits_citations(tmp_path):
    # Short run on H2O biasing an O-H distance — exercises the wiring + the
    # metadynamics citation surface (well-filling is validated analytically in
    # test_metadynamics.py).
    z, xyz = _h2o_angstrom()
    stem = tmp_path / "mtd"
    res = run_msindo_metadynamics(
        z, xyz, [DistanceCV(0, 1)], hill_height=5e-4, hill_sigma=0.15,
        deposition_stride=3, timestep_fs=0.3, n_steps=12, temperature_K=300.0,
        thermostat="berendsen", seed=1, output=str(stem))
    assert isinstance(res, MetadynamicsResult)
    assert res.n_hills >= 1
    assert res.cv_trajectory.shape == (res.trajectory.positions.shape[0], 1)
    refs = (tmp_path / "mtd.references").read_text()
    for key in ("swope_velocity_verlet_1982", "berendsen_thermostat_1984",
                "laio_parrinello_metadynamics_2002", "barducci_well_tempered_2008",
                "ahlswede_jug_msindo_1_1999"):
        assert key in refs
    assert "valeev_libint" not in refs
