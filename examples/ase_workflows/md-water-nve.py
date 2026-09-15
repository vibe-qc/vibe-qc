"""NVE molecular dynamics on H2O via vibe-qc forces.

Velocity-Verlet integration in the microcanonical (NVE) ensemble.
The conserved quantity is the **total energy** E = Etot + Ekin; if
the integrator is right and the forces are consistent with the
energy gradient (no drift, no noise), E should stay flat over the
trajectory to within the integrator's truncation error.

This script is the textbook MD-driver-validation: vibe-qc supplies
analytic forces, ASE's VelocityVerlet integrates, and we plot E(t)
to verify conservation. Drift > 1 meV/step typically means a force-
energy inconsistency bug; drift ~1e-7 eV/step is what a well-behaved
HF/STO-3G calculation produces.

Run:
    .venv/bin/python examples/ase_workflows/md-water-nve.py

Wall time: ~30 s (50 steps × 0.5 fs × HF/STO-3G water, single core).

Produces:
    output-md-water-nve.csv   — t, E_tot, E_kin, E_pot per step
    output-md-water-nve.png   — energy-conservation plot
    output-md-water-nve.traj  — ASE trajectory (every step)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.io import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

from vibeqc.ase import VibeQC

HERE = Path(__file__).resolve().parent

N_STEPS = 50
DT_FS = 0.5  # 0.5 fs is conservative for H-containing systems; 1 fs
             # is also fine at HF/STO-3G but 0.5 keeps drift low.
T_INIT_K = 300.0


def main() -> None:
    print("=" * 68)
    print(" ASE NVE MD:  H2O / RHF / STO-3G  via VelocityVerlet")
    print("=" * 68)
    print(f"  steps = {N_STEPS}, dt = {DT_FS} fs, T(init) = {T_INIT_K} K")
    print()

    # Slightly off-equilibrium starting geometry — non-zero forces +
    # MB-distributed velocities give a non-trivial trajectory.
    atoms = Atoms(
        symbols=["O", "H", "H"],
        positions=[
            (0.000, 0.000,  0.000),
            (0.755, 0.000, -0.585),
            (-0.755, 0.000, -0.585),
        ],
    )
    atoms.calc = VibeQC(basis="sto-3g")

    # Initialise velocities at T_INIT_K. ``MaxwellBoltzmannDistribution``
    # samples each component from a Gaussian with σ = sqrt(kT/m).
    MaxwellBoltzmannDistribution(atoms, temperature_K=T_INIT_K)

    traj_path = HERE / "output-md-water-nve.traj"
    dyn = VelocityVerlet(atoms, timestep=DT_FS * units.fs,
                         trajectory=str(traj_path), logfile=None)

    # Per-step diagnostics
    times: list[float] = []
    e_tot: list[float] = []
    e_pot: list[float] = []
    e_kin: list[float] = []
    temp_K: list[float] = []

    def _record() -> None:
        times.append(dyn.get_number_of_steps() * DT_FS)
        e_kin_eV = atoms.get_kinetic_energy()
        e_pot_eV = atoms.get_potential_energy()
        e_tot.append(e_pot_eV + e_kin_eV)
        e_pot.append(e_pot_eV)
        e_kin.append(e_kin_eV)
        # Instantaneous temperature: 2·Ekin / (3N·k_B) for an
        # unconstrained system. Standard formula; ASE has the same in
        # ``ase.md.md.MolecularDynamics.get_temperature`` if you'd
        # rather not write it out.
        n_dof = 3 * len(atoms)
        T = 2.0 * e_kin_eV / (n_dof * units.kB)
        temp_K.append(T)

    _record()
    for step in range(N_STEPS):
        dyn.run(1)
        _record()
        if step % 10 == 0:
            print(f"  step {step+1:3d}:  E_tot = {e_tot[-1]:.6f} eV   "
                  f"T = {temp_K[-1]:6.1f} K   "
                  f"ΔE_tot = {e_tot[-1] - e_tot[0]:+.3e} eV")

    # Energy conservation diagnostic
    drift_eV = e_tot[-1] - e_tot[0]
    drift_per_step = drift_eV / N_STEPS
    print()
    print(f"  ΔE_tot over {N_STEPS} steps:   {drift_eV:+.3e} eV")
    print(f"  drift per step:               {drift_per_step:+.3e} eV/step")
    print(f"  ΔE_tot vs σ(E_tot):           "
          f"{abs(drift_eV) / np.std(e_tot):.2f}  σ")

    # CSV
    csv_path = HERE / "output-md-water-nve.csv"
    with csv_path.open("w") as f:
        f.write("t_fs,E_tot_eV,E_kin_eV,E_pot_eV,T_K\n")
        for t, et, ek, ep, T in zip(times, e_tot, e_kin, e_pot, temp_K):
            f.write(f"{t},{et},{ek},{ep},{T}\n")
    print(f"\n  CSV: {csv_path.relative_to(HERE.parent.parent)}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    e_tot_arr = np.asarray(e_tot)
    e_kin_arr = np.asarray(e_kin)
    e_pot_arr = np.asarray(e_pot)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # Top: energy components vs time
    ax1.plot(times, e_pot_arr - e_pot_arr[0], label="ΔE_pot",
             color="C0")
    ax1.plot(times, e_kin_arr - e_kin_arr[0], label="ΔE_kin",
             color="C1")
    ax1.plot(times, e_tot_arr - e_tot_arr[0], label="ΔE_tot (conserved!)",
             color="C3", linewidth=2)
    ax1.set_ylabel("ΔE / eV  (relative to step 0)")
    ax1.set_title("H2O NVE / HF/STO-3G — energy conservation")
    ax1.legend(loc="best")
    ax1.grid(alpha=0.3)

    # Bottom: instantaneous temperature
    ax2.plot(times, temp_K, color="C2")
    ax2.axhline(T_INIT_K, linestyle="--", color="grey",
                label=f"T(init) = {T_INIT_K:.0f} K")
    ax2.set_xlabel("t / fs")
    ax2.set_ylabel("T / K")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    png_path = HERE / "output-md-water-nve.png"
    fig.savefig(png_path, dpi=150)
    print(f"  Plot: {png_path.relative_to(HERE.parent.parent)}")

    # Sanity check: the integrator should conserve E_tot to within
    # something like 1 meV over 50 steps at 0.5 fs / HF / STO-3G water.
    # This is loose enough not to trip on a one-off SCF blip but
    # tight enough to catch a real force-energy inconsistency.
    assert abs(drift_eV) < 0.01, (
        f"NVE drift over {N_STEPS} steps = {drift_eV:+.3e} eV — "
        f"force/energy inconsistency or integrator bug"
    )
    print("\n✓ NVE conservation: |ΔE_tot| < 10 meV over 50 steps. "
          "Forces and energy are mutually consistent.")


if __name__ == "__main__":
    main()
