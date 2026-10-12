"""Peierls dimerisation energy curve — tutorial 17 figure.

Re-runs the same SCF scan as ``examples/periodic/input-h-chain-peierls.py`` but
finer (δ = 0, 0.1, 0.2, ..., 1.2 bohr) and writes
``docs/_static/plots/h-chain-peierls-energy.png``.

Run:
    .venv/bin/python examples/plots/h-chain-peierls-energy.py

Like the scan it repeats, the chain is a ``dim=3`` cell with
``PAD`` bohr of vacuum on the two transverse axes, run with k-point GDF:
vibe-qc has no route with a consistent 1-D Coulomb gauge for a
``PeriodicSystem(dim=1, ...)``. Each SCF on the 2-H/cell pob-tzvp chain
with k-mesh [8,1,1] takes about two minutes with 4 threads, so the 13
points take roughly half an hour.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vibeqc import (
    Atom,
    BasisSet,
    PeriodicRHFOptions,
    PeriodicSystem,
    run_krhf_periodic_gdf,
)

HERE = Path(__file__).resolve().parent
PLOT_OUT = HERE.parent.parent / "docs" / "_static" / "plots" / "h-chain-peierls-energy.png"

A = 5.0
PAD = 15.0                       # transverse vacuum (y, z), bohr
KMESH = (8, 1, 1)
DELTAS = np.arange(0.0, 1.21, 0.1)


def build_system(delta: float) -> PeriodicSystem:
    return PeriodicSystem(
        dim=3,
        lattice=np.diag([A, PAD, PAD]),
        unit_cell=[
            Atom(1, [0.0, 0.0, 0.0]),
            Atom(1, [A / 2.0 - delta, 0.0, 0.0]),
        ],
    )


def scf_opts() -> PeriodicRHFOptions:
    opts = PeriodicRHFOptions()
    opts.conv_tol_energy = 1e-8
    opts.conv_tol_grad = 1e-3
    opts.max_iter = 80
    return opts


def main() -> None:
    energies = []
    converged = []
    for delta in DELTAS:
        sysp = build_system(delta)
        basis = BasisSet(sysp.unit_cell_molecule(), "pob-tzvp")
        r = run_krhf_periodic_gdf(sysp, basis, KMESH, scf_opts(),
                                  progress=False)
        energies.append(r.energy)
        converged.append(r.converged)
        print(f"  δ={delta:.2f}  E={r.energy:+.6f} Ha  "
              f"converged={'yes' if r.converged else 'NO'}  ({r.n_iter} iters)")

    energies = np.array(energies)
    converged = np.array(converged)
    deltas = np.array(DELTAS)

    PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.8, 3.8), dpi=150)

    # The reference is the uniform (δ=0) energy.
    e_ref = energies[0]
    de_kcal = (energies - e_ref) * 627.5095  # Hartree → kcal/mol

    ax.plot(deltas, de_kcal, color="#7f7f7f", linewidth=1.5, zorder=1)
    ax.scatter(deltas[converged], de_kcal[converged],
               color="#1f77b4", s=70, zorder=3, edgecolor="white",
               linewidth=1.0, label="converged SCF")
    if (~converged).any():
        ax.scatter(deltas[~converged], de_kcal[~converged],
                   color="#d62728", s=70, marker="x", linewidth=2.0,
                   zorder=3, label="SCF not converged")

    # Annotate the H2-equilibrium point: short bond = 1.4 bohr at δ = 1.1.
    idx_h2 = int(np.argmin(np.abs(deltas - 1.1)))
    ax.annotate(
        f"short bond = 1.4 bohr\n(isolated H$_2$ equilibrium)",
        xy=(deltas[idx_h2], de_kcal[idx_h2]),
        xytext=(0.02, -58),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8, alpha=0.7),
    )

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlabel(r"Dimerisation amplitude $\delta$ (bohr)")
    ax.set_ylabel(r"$\Delta E$ per unit cell (kcal mol$^{-1}$)")
    ax.set_title("Peierls dimerisation of a 1D H-chain (HF/pob-TZVP)")
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=300, bbox_inches="tight")

    print(f"\nWrote {PLOT_OUT.relative_to(HERE.parent.parent)}")
    if converged.any():
        de_min = de_kcal[converged].min()
        d_min = deltas[converged][int(np.argmin(de_kcal[converged]))]
        print(f"  Lowest converged ΔE = {de_min:.2f} kcal/mol at δ = {d_min:.2f} bohr")


if __name__ == "__main__":
    main()
