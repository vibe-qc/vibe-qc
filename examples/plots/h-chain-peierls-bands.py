"""Peierls gap opening — uniform vs. dimerised bands, tutorial 17 figure.

Builds the same 2-H/cell H-chain at two dimerisation amplitudes and
plots the Hcore (non-interacting, no SCF) bands side-by-side. At
δ=0 the two folded-back bands are degenerate at X — the band-folding
artefact of the doubled unit cell. At δ=1.1 (short bond = 1.4 bohr,
the H₂ equilibrium) the bands have separated cleanly: a deep, flat
bonding band and a high antibonding band, with a wide gap at every k.

This is the gap-opening half of the Peierls story: the energy drop
in ``h-chain-peierls-energy.py`` is *because* this gap opens, lowering
the occupied bonding band relative to the metallic reference.

Output:
    docs/_static/plots/h-chain-peierls-bands.png

Run:
    .venv/bin/python examples/plots/h-chain-peierls-bands.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import vibeqc as vq

HERE = Path(__file__).resolve().parent
PLOT_OUT = HERE.parent.parent / "docs" / "_static" / "plots" / "h-chain-peierls-bands.png"

A = 5.0
HARTREE_TO_EV = 27.211386


def build_system(delta: float) -> vq.PeriodicSystem:
    return vq.PeriodicSystem(
        dim=1,
        lattice=np.diag([A, 30.0, 30.0]),
        unit_cell=[
            vq.Atom(1, [0.0, 0.0, 0.0]),
            vq.Atom(1, [A / 2.0 - delta, 0.0, 0.0]),
        ],
    )


def hcore_bands(delta: float) -> tuple:
    sysp = build_system(delta)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "pob-tzvp")
    kpath = vq.kpath_from_segments(
        sysp,
        [((0.0, 0.0, 0.0), "Γ", (0.5, 0.0, 0.0), "X")],
        points_per_segment=120,
    )
    bands = vq.band_structure_hcore(sysp, basis, kpath, n_electrons_per_cell=2)
    return bands


def main() -> None:
    bands_uniform = hcore_bands(0.0)
    bands_dimerised = hcore_bands(1.1)

    PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_u, ax_d) = plt.subplots(
        1, 2, sharey=True, figsize=(7.5, 4.0), dpi=150,
    )

    def plot_panel(ax, bands, title):
        x = bands.kpath.distances
        e = (bands.energies - bands.e_fermi) * HARTREE_TO_EV
        # Plot lowest few bands; the bonding/antibonding pair is the
        # interesting one near E_F. pob-TZVP has many higher virtual
        # bands; clip to the lowest 6 for visual clarity.
        n_plot = min(6, e.shape[1])
        for i in range(n_plot):
            ax.plot(x, e[:, i], color="#1f77b4", linewidth=2.0)
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
        tick_pos, tick_txt = zip(*bands.kpath.labels)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_txt, fontsize=11)
        ax.set_xlim(x[0], x[-1])
        ax.set_xlabel("k-path")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3, linestyle=":")

    plot_panel(ax_u, bands_uniform,
               r"Uniform: $\delta = 0$  (R$_{\mathrm{HH}}$ = 2.5 bohr)")
    plot_panel(ax_d, bands_dimerised,
               r"Dimerised: $\delta = 1.1$  (1.4 / 3.6 bohr alternation)")

    ax_u.set_ylabel(r"$E - E_F$ (eV)")

    # Highlight the gap at the dimerised X point with a colored arrow.
    e_d = (bands_dimerised.energies - bands_dimerised.e_fermi) * HARTREE_TO_EV
    x_X = bands_dimerised.kpath.distances[-1]
    e_homo = e_d[-1, 0]
    # Bonding (homo) is band index 0 since pob-tzvp orders by energy.
    # Find LUMO at X.
    idx_lumo = int(np.argmax(e_d[-1] > 0))
    e_lumo = e_d[-1, idx_lumo]
    ax_d.annotate(
        "", xy=(x_X * 0.95, e_lumo), xytext=(x_X * 0.95, e_homo),
        arrowprops=dict(arrowstyle="<->", color="#d62728", lw=2.0),
    )
    ax_d.text(x_X * 0.55, (e_homo + e_lumo) / 2,
              f"gap at X\n{e_lumo - e_homo:.1f} eV",
              fontsize=10, color="#d62728", ha="left", va="center",
              fontweight="bold")

    ax_u.set_ylim(-25, 25)

    fig.suptitle(
        r"Peierls gap opening in the 1D H-chain — Hcore bands (pob-TZVP)",
        fontsize=12, y=0.995,
    )
    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=300, bbox_inches="tight")

    gap_uniform_eV = (bands_uniform.energies[-1, 1] -
                      bands_uniform.energies[-1, 0]) * HARTREE_TO_EV
    gap_dimerised_eV = e_lumo - e_homo
    print(f"Wrote {PLOT_OUT.relative_to(HERE.parent.parent)}")
    print(f"  Gap at X, uniform   (δ=0.0): {gap_uniform_eV:.3f} eV")
    print(f"  Gap at X, dimerised (δ=1.1): {gap_dimerised_eV:.3f} eV")


if __name__ == "__main__":
    main()
