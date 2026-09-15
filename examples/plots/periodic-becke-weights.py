"""Periodic Becke partition vs molecular partition — tutorial 23 figure.

Compares the total DFT integration weight produced by

  (a) the molecular Becke partition (vq.build_grid) — assumes each
      atom is isolated, lets its 'territory' extend out to infinity;
  (b) the periodic Becke partition (vq.build_periodic_becke_grid) —
      includes image atoms so neighboring cells correctly cap the
      atomic territories at the cell boundary,

across a sweep of cubic-H2 cell sizes a ∈ {4, 5, 6, 8, 10, 12, 15, 20}
bohr. The exact total weight equals the cell volume V = a³. The
molecular partition over-counts by ~100× in the tight-cell regime
(a = 5 bohr); the periodic partition recovers V to within ~2%.

Output: ``docs/_static/plots/periodic-becke-weights.png``.

Run:
    .venv/bin/python examples/plots/periodic-becke-weights.py
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

import vibeqc as vq

HERE = Path(__file__).resolve().parent
PLOT_OUT = HERE.parent.parent / "docs" / "_static" / "plots" / "periodic-becke-weights.png"


def make_h2_cubic_cell(a: float) -> vq.PeriodicSystem:
    """Cubic cell of side ``a`` bohr containing one H2 (1.4 bohr bond)."""
    return vq.PeriodicSystem(
        3,
        np.diag([a, a, a]).tolist(),
        [vq.Atom(1, [0.0, 0.0, 0.0]),
         vq.Atom(1, [0.0, 0.0, 1.4])],
    )


def main() -> None:
    t0 = time.perf_counter()

    cell_sizes = np.array([4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0])
    w_mol = []
    w_per = []
    for a in cell_sizes:
        sysp = make_h2_cubic_cell(a)
        # Molecular path: build_grid sees only the unit-cell atoms.
        g_mol = vq.build_grid(sysp.unit_cell_molecule())
        # Periodic Becke: include image atoms within image_radius of any
        # home-cell atom. 10 bohr is the default and is plenty for a
        # cubic cell up to ~10 bohr; bump to 1.2*a for the larger cells.
        image_r = max(10.0, 1.2 * a)
        g_per = vq.build_periodic_becke_grid(sysp, image_radius_bohr=image_r)

        w_mol.append(float(g_mol.weights.sum()))
        w_per.append(float(g_per.weights.sum()))
        print(f"  a = {a:5.1f} bohr   V = {a**3:8.1f}   "
              f"w_mol = {w_mol[-1]:10.2f}   "
              f"w_per = {w_per[-1]:8.2f}   "
              f"per/V = {w_per[-1] / a**3:6.3f}")

    w_mol = np.array(w_mol)
    w_per = np.array(w_per)
    V = cell_sizes ** 3

    PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10.5, 4.5), dpi=150)

    # --- LEFT: total weight vs cell size on log-y --------------------
    ax_l.semilogy(cell_sizes, w_mol, "o-", color="#d62728",
                  markersize=7, lw=1.6, label="molecular partition")
    ax_l.semilogy(cell_sizes, w_per, "s-", color="#1f77b4",
                  markersize=7, lw=1.6, label="periodic Becke")
    ax_l.semilogy(cell_sizes, V, "k--", lw=1.0,
                  label=r"exact: $V_\mathrm{cell} = a^3$")
    ax_l.set_xlabel("Cubic cell parameter $a$ (bohr)")
    ax_l.set_ylabel(r"Total DFT integration weight $\sum_g w_g$  (bohr³)")
    ax_l.set_title("(a) Weight vs cell size")
    ax_l.legend(loc="upper left", fontsize=9)
    ax_l.grid(alpha=0.3, linestyle=":")

    # --- RIGHT: weight / V_cell ratio --------------------------------
    ax_r.semilogy(cell_sizes, w_mol / V, "o-", color="#d62728",
                  markersize=7, lw=1.6, label="molecular partition")
    ax_r.semilogy(cell_sizes, w_per / V, "s-", color="#1f77b4",
                  markersize=7, lw=1.6, label="periodic Becke")
    ax_r.axhline(1.0, color="k", linestyle="--", lw=1.0,
                 label=r"target ratio = 1")
    ax_r.set_xlabel("Cubic cell parameter $a$ (bohr)")
    ax_r.set_ylabel(r"$\sum_g w_g \,/\, V_\mathrm{cell}$")
    ax_r.set_title("(b) Over-counting ratio")
    ax_r.legend(loc="upper right", fontsize=9)
    ax_r.grid(alpha=0.3, which="both", linestyle=":")
    ax_r.set_ylim(0.5, 1e3)

    fig.suptitle(
        "Periodic Becke partition recovers the correct integration "
        "weight; molecular partition fails for tight cells",
        fontsize=11, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=300, bbox_inches="tight")

    print(f"\nWrote {PLOT_OUT.relative_to(HERE.parent.parent)}  "
          f"({time.perf_counter()-t0:.1f} s)")


if __name__ == "__main__":
    main()
