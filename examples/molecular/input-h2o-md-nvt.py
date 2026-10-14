"""Water, 20 steps of Born-Oppenheimer NVT dynamics on the RHF/STO-3G surface.

System: H2O at the fixed geometry shared by these examples (bohr), started
from Maxwell-Boltzmann velocities at 300 K (``seed=7`` makes them
reproducible).
Method: ``vibeqc.md.run_nvt(force_fn, coords, ...)`` with a 0.5 fs step,
20 steps, the default Nose-Hoover thermostat (relaxation time 50 fs) and
centre-of-mass motion removed (6 degrees of freedom). The integrator is
method-agnostic: it only needs ``force_fn(coords_bohr) -> (energy_Ha,
gradient_Ha_per_bohr)``, and the callable below builds a Molecule at the
new coordinates, converges RHF with ``vibeqc.run_rhf`` and returns the
analytic gradient from ``vibeqc.compute_gradient`` (forces are minus the
gradient; the driver applies the sign).

Files (next to this script):
    input-h2o-md-nvt.out        per-step table: time, T, E_pot, E_kin, E_tot, conserved quantity
    input-h2o-md-nvt.traj.xyz   multi-frame XYZ trajectory in angstrom
(run_nvt writes no files; both are written by this script.)

Success: 21 force evaluations (frame 0 plus 20 steps); the instantaneous
temperature swings between 130 and 420 K as the thermostat and the stiff
O-H stretches exchange energy, the potential energy stays within 1.6 mHa
of its start, and the Nose-Hoover conserved quantity drifts by 2.0e-5 Ha
over 10 fs, which is the integration error of a 0.5 fs step on 4000 cm^-1
modes, not a bug. Tolerance: drift below 5e-5 Ha (vibe-qc 0.17.x
development build, Apple M5 Max, about 10 s including start-up).
"""

from pathlib import Path

import numpy as np

import vibeqc as vq
from vibeqc.md import run_nvt

HERE = Path(__file__).parent
STEM = Path(__file__).stem
BOHR = 0.52917721067
BASIS = "sto-3g"

atomic_numbers = [8, 1, 1]
symbols = ["O", "H", "H"]
coords0 = np.array([[0.000, 0.000, 0.000],
                    [0.000, 1.499, -1.160],
                    [0.000, -1.499, -1.160]])

n_calls = 0


def force_fn(coords):
    """RHF/STO-3G energy and analytic gradient at ``coords`` (bohr)."""
    global n_calls
    n_calls += 1
    mol = vq.Molecule([vq.Atom(z, list(xyz)) for z, xyz in zip(atomic_numbers, coords)])
    basis = vq.BasisSet(mol, BASIS)
    scf = vq.run_rhf(mol, BASIS)
    grad = vq.compute_gradient(mol, basis, scf)
    return float(scf.energy), np.asarray(grad, dtype=float)


traj = run_nvt(
    force_fn, coords0,
    atomic_numbers=atomic_numbers,
    timestep_fs=0.5,
    n_steps=20,
    temperature_K=300.0,
    seed=7,
)

lines = [f"NVT MD of water on RHF/{BASIS}: thermostat {traj.thermostat}, "
         f"dt = {traj.timestep_fs} fs, N_dof = {traj.n_dof}, {n_calls} force evaluations",
         f"{'step':>4s} {'t (fs)':>7s} {'T (K)':>8s} {'E_pot (Ha)':>14s} {'E_kin (Ha)':>12s} "
         f"{'E_tot (Ha)':>14s} {'conserved (Ha)':>15s}"]
for k in range(traj.times_fs.size):
    lines.append(f"{k:4d} {traj.times_fs[k]:7.2f} {traj.temperature[k]:8.1f} "
                 f"{traj.potential_energy[k]:14.8f} {traj.kinetic_energy[k]:12.8f} "
                 f"{traj.total_energy[k]:14.8f} {traj.conserved_energy[k]:15.8f}")
lines += [
    f"mean T over the second half: {traj.mean_temperature():.1f} K",
    f"max |E_pot - E_pot(0)| = {np.abs(traj.potential_energy - traj.potential_energy[0]).max():.2e} Ha",
    f"drift of the conserved quantity: {traj.energy_drift():.2e} Ha",
]
text = "\n".join(lines)
print("\n" + "\n".join("  " + ln for ln in lines))
(HERE / f"{STEM}.out").write_text(text + "\n")

with open(HERE / f"{STEM}.traj.xyz", "w") as fh:
    for k in range(traj.times_fs.size):
        fh.write(f"{len(symbols)}\n")
        fh.write(f"t = {traj.times_fs[k]:.2f} fs  E_pot = {traj.potential_energy[k]:.8f} Ha  "
                 f"T = {traj.temperature[k]:.1f} K\n")
        for sym, xyz in zip(symbols, traj.positions[k]):
            fh.write(f"{sym} {xyz[0] * BOHR:12.6f} {xyz[1] * BOHR:12.6f} {xyz[2] * BOHR:12.6f}\n")
