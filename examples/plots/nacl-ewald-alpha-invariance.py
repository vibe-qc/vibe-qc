"""NaCl Madelung sum — Ewald alpha-invariance figure for tutorial 06.

Computes the NaCl Madelung constant via Ewald summation across a sweep of
alpha values on a fixed (large) real- and reciprocal-space cutoff, plus
the same sweep with a deliberately *too small* real cutoff to show the
breakdown. Writes
``docs/_static/plots/nacl-ewald-alpha-invariance.png`` — the figure
embedded in tutorial 06. Pure point-charge Ewald, sub-second.

Run:
    .venv/bin/python examples/plots/nacl-ewald-alpha-invariance.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vibeqc import EwaldOptions, ewald_point_charge_energy

HERE = Path(__file__).resolve().parent
PLOT_OUT = HERE.parent.parent / "docs" / "_static" / "plots" / "nacl-ewald-alpha-invariance.png"

A_TO_BOHR = 1.0 / 0.529177210903
M_LIT = 1.7475645946  # NaCl Madelung constant (Sherman 1932; high-precision Nijboer-De Wette 1957)


def madelung(alpha: float, real_cut: float, recip_cut: float) -> float:
    a = 5.64 * A_TO_BOHR
    lattice = 0.5 * a * np.array([[0, 1, 1],
                                  [1, 0, 1],
                                  [1, 1, 0]], dtype=float).T
    positions = np.column_stack([[0.0, 0.0, 0.0],
                                 [0.5 * a, 0.0, 0.0]])
    charges = np.array([+1.0, -1.0])

    opts = EwaldOptions()
    opts.alpha = alpha
    opts.real_cutoff_bohr = real_cut
    opts.recip_cutoff_bohr_inv = recip_cut
    energy = ewald_point_charge_energy(lattice, positions, charges, opts)
    r_nn = 0.5 * a
    return -energy * r_nn


def main() -> None:
    alphas = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
                       0.70, 1.00, 1.50])

    converged = np.array([madelung(a, real_cut=40.0, recip_cut=12.0)
                          for a in alphas])
    too_short = np.array([madelung(a, real_cut=8.0, recip_cut=12.0)
                          for a in alphas])

    PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=150)

    ax.axhline(M_LIT, color="#888", ls="--", lw=1.0,
               label=fr"literature $M = {M_LIT:.10f}$")
    ax.plot(alphas, converged, "o-", color="#1f77b4", ms=7, lw=1.4,
            label=r"converged: $r_{\rm cut}=40\,\mathrm{bohr},\ G_{\rm cut}=12\,\mathrm{bohr}^{-1}$")
    ax.plot(alphas, too_short, "s--", color="#d62728", ms=6, lw=1.2,
            label=r"too-short real cutoff: $r_{\rm cut}=8\,\mathrm{bohr}$")

    ax.set_xlabel(r"Ewald splitting parameter $\alpha$ (bohr$^{-1}$)")
    ax.set_ylabel(r"computed Madelung constant $M$")
    ax.set_xscale("log")
    ax.grid(alpha=0.3, linestyle=":", which="both")
    ax.legend(loc="lower center", fontsize=9, frameon=True)
    ax.set_title(r"NaCl rocksalt — Ewald $\alpha$-invariance test",
                 fontsize=11)

    # Inset annotation: max deviation from literature on converged sweep
    dev = np.max(np.abs(converged - M_LIT))
    ax.text(0.04, 0.94,
            fr"max deviation (blue): {dev:.2e}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#bbbbbb"))

    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=300, bbox_inches="tight")

    print(f"Wrote {PLOT_OUT.relative_to(HERE.parent.parent)}")
    print(f"  Max |M_converged - M_lit| = {dev:.3e}")
    print(f"  Spread on too-short sweep: "
          f"{too_short.max() - too_short.min():.3e}")


if __name__ == "__main__":
    main()
