"""Bands + DOS for the 1D H2 molecular crystal — tutorial 12 figure.

Re-runs the same Hcore calculation as ``examples/periodic/input-h-chain-bands.py``
(non-interacting bands on a Γ → X path, Gaussian-broadened DOS on a
80×1×1 k-mesh) and writes
``docs/_static/plots/h-chain-bands-dos.png`` — the figure embedded
in tutorial 12. Pure Hcore, no SCF, runs in well under a second.

Run:
    .venv/bin/python examples/plots/h-chain-bands-dos.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import vibeqc as vq

HERE = Path(__file__).resolve().parent
PLOT_OUT = HERE.parent.parent / "docs" / "_static" / "plots" / "h-chain-bands-dos.png"

HARTREE_TO_EV = 27.211386


def main() -> None:
    a = 6.0  # bohr
    atoms = [
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [1.4, 0.0, 0.0]),
    ]
    system = vq.PeriodicSystem(
        1,
        [[a, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]],
        atoms,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    kpath = vq.kpath_from_segments(
        system,
        [((0.0, 0.0, 0.0), "Γ", (0.5, 0.0, 0.0), "X")],
        points_per_segment=120,
    )
    bands = vq.band_structure_hcore(
        system, basis, kpath, n_electrons_per_cell=2,
    )
    dos = vq.density_of_states_hcore(
        system, basis, [200, 1, 1],
        sigma=0.01, n_electrons_per_cell=2,
    )

    PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)

    # Convert energies to eV relative to E_F for the conventional view.
    e_fermi = bands.e_fermi
    e_bands = (bands.energies - e_fermi) * HARTREE_TO_EV
    e_dos = (dos.energies - e_fermi) * HARTREE_TO_EV

    fig, (ax_b, ax_d) = plt.subplots(
        1, 2, sharey=True, figsize=(7.0, 4.0), dpi=150,
        gridspec_kw={"width_ratios": [3.0, 1.0], "wspace": 0.05},
    )

    # --- Bands panel ---------------------------------------------------------
    x = bands.kpath.distances
    band_colors = ["#1f77b4", "#d62728"]
    band_labels = ["bonding", "antibonding"]
    for i in range(e_bands.shape[1]):
        ax_b.plot(x, e_bands[:, i], color=band_colors[i % len(band_colors)],
                  linewidth=2.0, label=band_labels[i] if i < 2 else None)

    ax_b.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
    tick_pos, tick_txt = zip(*bands.kpath.labels)
    ax_b.set_xticks(tick_pos)
    ax_b.set_xticklabels(tick_txt, fontsize=11)
    ax_b.set_xlim(x[0], x[-1])
    ax_b.set_ylabel(r"$E - E_F$ (eV)")
    ax_b.set_xlabel("k-path")
    ax_b.grid(axis="y", alpha=0.3, linestyle=":")
    ax_b.legend(loc="center right", frameon=True, fontsize=9)

    # --- DOS panel -----------------------------------------------------------
    ax_d.fill_betweenx(e_dos, dos.dos, color="#ff7f0e", alpha=0.55,
                       linewidth=0)
    ax_d.plot(dos.dos, e_dos, color="#ff7f0e", linewidth=1.4)
    ax_d.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
    ax_d.set_xlabel("DOS (states / Ha)")
    ax_d.set_xlim(left=0)
    ax_d.grid(axis="y", alpha=0.3, linestyle=":")

    # Common y-window: tight around both bands with a little headroom.
    y_lo = e_bands.min() - 1.0
    y_hi = e_bands.max() + 1.0
    ax_b.set_ylim(y_lo, y_hi)

    fig.suptitle(r"H$_2$ molecular crystal — bands + DOS (Hcore/STO-3G)",
                 fontsize=12, y=0.995)

    fig.savefig(PLOT_OUT, dpi=300, bbox_inches="tight")

    gap_eV = (bands.energies[:, 1].min() - bands.energies[:, 0].max()) * HARTREE_TO_EV
    print(f"Wrote {PLOT_OUT.relative_to(HERE.parent.parent)}")
    print(f"  E_F = {e_fermi:.4f} Ha")
    print(f"  Bonding bandwidth:    "
          f"{(e_bands[:, 0].max() - e_bands[:, 0].min()):.3f} eV")
    print(f"  Antibonding bandwidth: "
          f"{(e_bands[:, 1].max() - e_bands[:, 1].min()):.3f} eV")
    print(f"  Direct gap:            {gap_eV:.3f} eV")


if __name__ == "__main__":
    main()
