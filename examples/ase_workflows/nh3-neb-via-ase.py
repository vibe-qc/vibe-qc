"""NH3 umbrella inversion via ASE's NEB, vibe-qc as the only backend.

The pure-ASE counterpart to ``examples/ase_compare/compare-nh3-neb-
barrier.py`` — same chemistry, same NEB protocol, but no external
QC code in the loop. Demonstrates that vibe-qc + ASE alone is enough
for a complete reaction-path study.

Run:
    .venv/bin/python examples/ase_workflows/nh3-neb-via-ase.py

Wall time: ~3-5 minutes (5 NEB images × ~25 FIRE iterations × HF/STO-3G
SCF on 4-atom system).

Produces:
    output-nh3-neb-mep.csv   — per-image energies along the MEP
    output-nh3-neb-mep.png   — MEP plot with TS marked
    output-nh3-neb.traj      — full trajectory (every NEB image)

Method: HF / STO-3G — qualitative only. The barrier comes out
~10 kcal/mol vs. experimental ~5.8 kcal/mol; HF + minimal basis
overshoots inversion barriers by ~2× systematically. For
quantitative work, use MP2 or hybrid DFT in def2-TZVP+ basis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import Trajectory
from ase.mep import NEB
from ase.optimize import BFGSLineSearch, FIRE

from vibeqc.ase import VibeQC

HERE = Path(__file__).resolve().parent
EV_PER_KCAL = 1.0 / 23.06054783


def nh3_endpoints() -> tuple[Atoms, Atoms]:
    """Two enantiomers of pyramidal NH3."""
    n_z, h_z = 0.06, -0.31
    initial = Atoms(
        symbols=["N", "H", "H", "H"],
        positions=[
            (0.0,  0.0,  n_z),
            (0.940, 0.0, h_z),
            (-0.470, 0.814, h_z),
            (-0.470, -0.814, h_z),
        ],
    )
    final = initial.copy()
    final.positions[:, 2] *= -1.0
    return initial, final


def main() -> None:
    print("=" * 68)
    print(" ASE NEB:  NH3 umbrella inversion via vibe-qc HF/STO-3G")
    print("=" * 68)
    print()

    initial, final = nh3_endpoints()

    print("Relaxing the two endpoints…")
    initial.calc = VibeQC(basis="sto-3g")
    BFGSLineSearch(initial, logfile=None).run(fmax=0.05, steps=30)
    final.calc = VibeQC(basis="sto-3g")
    BFGSLineSearch(final, logfile=None).run(fmax=0.05, steps=30)
    print(f"  E(initial) = {initial.get_potential_energy():.6f} eV")
    print(f"  E(final)   = {final.get_potential_energy():.6f} eV")

    n_intermediates = 5
    images = [initial]
    for _ in range(n_intermediates):
        img = initial.copy()
        img.calc = VibeQC(basis="sto-3g")
        images.append(img)
    images.append(final)

    neb = NEB(images, climb=True)
    neb.interpolate()
    # Re-attach calculators after interpolate (ASE drops them).
    for img in images[1:-1]:
        img.calc = VibeQC(basis="sto-3g")

    print(f"\nRunning climbing-image NEB on {n_intermediates} intermediates…")
    traj_path = HERE / "output-nh3-neb.traj"
    fire = FIRE(neb, trajectory=str(traj_path), logfile=None)
    fire.run(fmax=0.10, steps=60)

    energies = np.array([img.get_potential_energy() for img in images])
    e_ref = energies[0]
    barrier_eV = energies.max() - e_ref
    barrier_kcal = barrier_eV / EV_PER_KCAL
    ts_idx = int(np.argmax(energies))

    print(f"\n  TS index along MEP: {ts_idx}")
    print(f"  TS energy:          {energies.max():.6f} eV")
    print(f"  Barrier:            {barrier_eV*1000:.3f} meV "
          f"({barrier_kcal:.3f} kcal/mol)")
    print(f"\n  (HF/STO-3G overshoots experimental ~5.8 kcal/mol by ~2× — "
          f"qualitative agreement only.)")

    # CSV
    csv_path = HERE / "output-nh3-neb-mep.csv"
    with csv_path.open("w") as f:
        f.write("image,E_eV,DeltaE_eV,DeltaE_kcal_mol\n")
        for i, e in enumerate(energies):
            de = e - e_ref
            f.write(f"{i},{e},{de},{de / EV_PER_KCAL}\n")
    print(f"\n  CSV: {csv_path.relative_to(HERE.parent.parent)}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    de_meV = (energies - e_ref) * 1000
    ax.plot(range(len(energies)), de_meV, marker="o", color="C0")
    ax.scatter(
        [ts_idx], [de_meV[ts_idx]],
        color="C3", s=120, zorder=5,
        label=f"TS  (Δ = {barrier_kcal:.2f} kcal/mol)",
    )
    ax.set_xlabel("NEB image index")
    ax.set_ylabel("ΔE / meV")
    ax.set_title("NH3 umbrella inversion (HF/STO-3G via vibe-qc)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png_path = HERE / "output-nh3-neb-mep.png"
    fig.savefig(png_path, dpi=150)
    print(f"  Plot: {png_path.relative_to(HERE.parent.parent)}")


if __name__ == "__main__":
    main()
