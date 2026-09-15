#!/usr/bin/env python3
"""GFN2-xTB molecular dynamics via ASE.

Mirrors the xtb workshop § "MD". Demonstrates:
  1. NVE (microcanonical) -- total-energy conservation check
  2. NVT (canonical) -- temperature control with Berendsen thermostat
  3. Force-driven MD using GFN2Model wrapped as an ASE Calculator

Run:
    .venv/bin/python examples/semiempirical/35_gfn2_molecular_dynamics.py
Requires: pip install ase
"""

from __future__ import annotations

import numpy as np
from vibeqc import Molecule, Atom
from vibeqc.semiempirical.methods.gfn2 import GFN2Model
from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

params = load_gfn2_params()

print("=" * 72)
print("GFN2-xTB Molecular Dynamics")

try:
    from ase import Atoms, units
    from ase.calculators.calculator import Calculator
    from ase.md.verlet import VelocityVerlet
    from ase.md.nvtberendsen import NVTBerendsen
    from ase.io import Trajectory
except ImportError:
    print("ASE not installed. Install with: pip install ase")
    raise SystemExit(1)


class GFN2Calculator(Calculator):
    """Minimal ASE Calculator backed by GFN2Model.

    Converts between ASE's angstrom/eV units and vibe-qc's bohr/Hartree.
    """
    implemented_properties = ["energy", "forces"]

    def __init__(self, params):
        super().__init__()
        self._params = params

    def calculate(self, atoms, properties, system_changes):
        super().calculate(atoms, properties, system_changes)
        mol = Molecule(
            [Atom(int(z), (pos / units.Bohr).tolist())
             for z, pos in zip(atoms.numbers, atoms.positions)],
            charge=0, multiplicity=1,
        )
        m = GFN2Model(mol, params=self._params, warn=False)
        energy_ha = m.energy()
        gradient_ha_bohr = np.asarray(m.gradient())
        self.results["energy"] = energy_ha * units.Hartree
        self.results["forces"] = -gradient_ha_bohr * (units.Hartree / units.Bohr)


# -- Build water molecule --------------------------------------------------
theta = np.deg2rad(104.5 / 2)
r_oh_ang = 1.81 * units.Bohr  # bohr -> angstrom

atoms = Atoms('OH2', positions=np.array([
    [0.0, 0.0, 0.0],
    [r_oh_ang * np.sin(theta), r_oh_ang * np.cos(theta), 0.0],
    [-r_oh_ang * np.sin(theta), r_oh_ang * np.cos(theta), 0.0],
]))

calc = GFN2Calculator(params)
atoms.calc = calc

print(f"\nSystem: {atoms.get_chemical_formula()}")
print(f"Initial potential energy: {atoms.get_potential_energy():.4f} eV")

# -- 1. NVE -- total-energy conservation ----------------------------------
print("\n" + "-" * 48)
print("1. NVE (microcanonical) -- 100 steps, 0.5 fs timestep")

# Initial velocities from Maxwell-Boltzmann at 300 K
np.random.seed(42)
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

dyn_nve = VelocityVerlet(atoms, timestep=0.5 * units.fs)

e_tot_traj = []
n_steps = 100
for step in range(n_steps):
    dyn_nve.run(1)
    e_pot = atoms.get_potential_energy()
    e_kin = atoms.get_kinetic_energy()
    e_tot = e_pot + e_kin
    e_tot_traj.append(e_tot)

e_tot_traj = np.array(e_tot_traj)
e_tot_mean = np.mean(e_tot_traj)
drift = np.std(e_tot_traj) / np.abs(e_tot_mean) if abs(e_tot_mean) > 1e-12 else 0

print(f"   Mean total energy:  {e_tot_mean:.6f} eV")
print(f"   Std dev:            {np.std(e_tot_traj):.4e} eV")
print(f"   Relative drift:     {drift:.2e}")
print(f"   Max |delta-E| per step:  {np.max(np.abs(np.diff(e_tot_traj))):.4e} eV")
if drift < 1e-4:
    print("   ✓ Total energy conserved (stable integrator)")
else:
    print("   ⚠ Energy drift above 1e-4 -- check timestep")

# -- 2. NVT -- temperature control ----------------------------------------
print("\n" + "-" * 48)
print("2. NVT (canonical) -- Berendsen thermostat, 300 K")

# Reset geometry and velocities
atoms_nvt = atoms.copy()
atoms_nvt.calc = GFN2Calculator(params)

np.random.seed(123)
MaxwellBoltzmannDistribution(atoms_nvt, temperature_K=300)

dyn_nvt = NVTBerendsen(
    atoms_nvt,
    timestep=0.5 * units.fs,
    temperature_K=300,
    taut=50 * units.fs,
)

temps = []
n_nvt = 200
for step in range(n_nvt):
    dyn_nvt.run(1)
    temps.append(atoms_nvt.get_temperature())

temps = np.array(temps)
print(f"   Target temperature:  300 K")
print(f"   Mean temperature:    {np.mean(temps):.1f} K")
print(f"   Std dev:             {np.std(temps):.1f} K")
print(f"   Equilibrium (last 50): {np.mean(temps[-50:]):.1f} +/- {np.std(temps[-50:]):.1f} K")

# Print a few temperature snapshots
print(f"\n   Temperature profile (every 50 steps):")
for i in range(0, n_nvt, 50):
    print(f"      step {i:4d}: T = {temps[i]:6.1f} K")

# -- 3. Trajectory I/O ----------------------------------------------------
print("\n" + "-" * 48)
print("3. Trajectory I/O")

traj = Trajectory("h2o_gfn2_nvt.traj", "w", atoms_nvt)
# Re-run a few more steps to capture in trajectory
for step in range(50):
    dyn_nvt.run(1)
    traj.write()
traj.close()
print(f"   Wrote 50 frames to h2o_gfn2_nvt.traj")
print(f"   Visualise with: ase gui h2o_gfn2_nvt.traj")

print("\nDone.")
