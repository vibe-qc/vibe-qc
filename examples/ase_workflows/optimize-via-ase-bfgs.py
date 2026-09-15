"""H2O geometry optimization via ASE's BFGS, vibe-qc as the backend.

Demonstrates the ASE-native optimization idiom — the same pattern that
ASE tutorials use for any other calculator. Vibe-qc plugs in via the
``VibeQC`` Calculator (Phase A); ASE's ``BFGS`` driver does the rest.

Run:
    .venv/bin/python examples/ase_workflows/optimize-via-ase-bfgs.py

Produces:
    output-optimize-via-ase-bfgs.traj   — full per-step trajectory
                                          (open with ``ase gui``)
    output-optimize-via-ase-bfgs.png    — energy + |F|max convergence

Compare this to ``examples/molecular/input-h2o-opt.py`` which does the same
thing through ``run_job(optimize=True)`` — same chemistry, same
trajectory, different driver. ``run_job`` is the higher-level
convenience; ``BFGS(...).run()`` is the ASE-native way and slots
into any larger ASE workflow (multi-step protocols, constraints,
nudging, etc.).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write
from ase.optimize import BFGS

from vibeqc.ase import VibeQC

HERE = Path(__file__).resolve().parent


def main() -> None:
    print("=" * 68)
    print(" ASE BFGS geometry optimization:  H2O / RHF / 6-31G*")
    print("=" * 68)

    # Slightly distorted starting geometry — gives BFGS something to do.
    atoms = Atoms(
        symbols=["O", "H", "H"],
        positions=[
            (0.0, 0.0,  0.10),
            (0.0, 0.80, -0.55),
            (0.0, -0.80, -0.55),
        ],
    )
    atoms.calc = VibeQC(basis="6-31g*")

    traj_path = HERE / "output-optimize-via-ase-bfgs.traj"
    log_path = HERE / "output-optimize-via-ase-bfgs.log"

    # ASE's BFGS — energy + forces from `atoms.calc`, no per-step
    # boilerplate. `trajectory=` writes every accepted step.
    opt = BFGS(atoms, trajectory=str(traj_path), logfile=str(log_path))
    opt.run(fmax=0.01, steps=50)

    print(f"\n  Final energy:   {atoms.get_potential_energy():.6f} eV")
    print(f"  Final |F|max:    {np.linalg.norm(atoms.get_forces(), axis=1).max():.4e} eV/Å")
    print(f"  Final geometry (bohr-ish, Å in ASE):")
    for sym, pos in zip(atoms.symbols, atoms.positions):
        print(f"    {sym}  {pos[0]:9.4f}  {pos[1]:9.4f}  {pos[2]:9.4f}")
    print(f"\n  Trajectory: {traj_path.relative_to(HERE.parent.parent)}")
    print(f"  Log:        {log_path.relative_to(HERE.parent.parent)}")

    # Plot the per-step convergence if matplotlib's available.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    from ase.io.trajectory import Trajectory
    energies = []
    fmax_per_step = []
    for img in Trajectory(str(traj_path)):
        e = img.get_potential_energy()
        f = img.get_forces()
        energies.append(e)
        fmax_per_step.append(float(np.linalg.norm(f, axis=1).max()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(range(len(energies)), np.asarray(energies) - energies[0],
             marker="o", color="C0")
    ax1.set_xlabel("BFGS step")
    ax1.set_ylabel("E − E₀ / eV")
    ax1.set_title("Energy convergence")
    ax1.grid(alpha=0.3)
    ax2.semilogy(range(len(fmax_per_step)), fmax_per_step,
                 marker="s", color="C1")
    ax2.axhline(0.01, linestyle="--", color="grey",
                label="fmax target (0.01 eV/Å)")
    ax2.set_xlabel("BFGS step")
    ax2.set_ylabel("|F|max / (eV/Å)  [log]")
    ax2.set_title("Force convergence")
    ax2.legend()
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    png_path = HERE / "output-optimize-via-ase-bfgs.png"
    fig.savefig(png_path, dpi=150)
    print(f"  Plot:       {png_path.relative_to(HERE.parent.parent)}")


if __name__ == "__main__":
    main()
