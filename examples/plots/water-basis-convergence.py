"""H₂O HF basis-set convergence + Helgaker CBS extrapolation — tutorial 13.

Runs HF on water at cc-pVDZ / cc-pVTZ / cc-pVQZ, fits the Helgaker-
Klopper-Koch exponential E(X) = E_CBS + A · exp(-α·X), and writes
``docs/_static/plots/water-basis-convergence.png`` — a two-panel figure:

  Top:    E(X) data + fit curve + horizontal CBS asymptote.
  Bottom: |E(X) - E_CBS| vs X on a semi-log axis (straight line is the
          signature of exponential convergence).

Run:
    .venv/bin/python examples/plots/water-basis-convergence.py

cc-pVQZ is the slow point (~30 s on 4 threads); D and T finish in a
couple of seconds each.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from vibeqc import Atom, BasisSet, Molecule, run_rhf

HERE = Path(__file__).resolve().parent
PLOT_OUT = HERE.parent.parent / "docs" / "_static" / "plots" / "water-basis-convergence.png"

CARDINAL = {"cc-pvdz": 2, "cc-pvtz": 3, "cc-pvqz": 4}


def helgaker(X, e_cbs, A, alpha):
    """E(X) = E_CBS + A * exp(-alpha * X) — Helgaker-Klopper-Koch HF form."""
    return e_cbs + A * np.exp(-alpha * X)


def main() -> None:
    mol = Molecule([
        Atom(8, [0.0,  0.00,  0.00]),
        Atom(1, [0.0,  1.43, -0.98]),
        Atom(1, [0.0, -1.43, -0.98]),
    ])

    rows = []  # (name, X, n_basis, energy)
    for name, X in CARDINAL.items():
        basis = BasisSet(mol, name)
        r = run_rhf(mol, basis)
        rows.append((name, X, basis.nbasis, r.energy))
        print(f"  {name:10s}  X={X}  nbf={basis.nbasis:3d}  "
              f"E = {r.energy:.8f} Ha")

    Xs = np.array([row[1] for row in rows], dtype=float)
    Es = np.array([row[3] for row in rows])

    # Fit E_CBS, A, alpha. Initial guesses: extrapolated minimum, the
    # spread, and Helgaker's α ≈ 1.63.
    p0 = (Es.min() - 0.005, 1.0, 1.6)
    popt, _ = curve_fit(helgaker, Xs, Es, p0=p0)
    e_cbs, A_fit, alpha_fit = popt
    print(f"\nFit:  E_CBS = {e_cbs:.6f} Ha   A = {A_fit:+.4f}   α = {alpha_fit:.3f}")

    # Dense fit curve for visualization
    X_dense = np.linspace(1.5, 6.0, 200)
    E_dense = helgaker(X_dense, *popt)

    PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6.0, 5.5), dpi=150,
        gridspec_kw={"height_ratios": [3.0, 2.0], "hspace": 0.35},
    )

    # --- Top: E(X) with data + fit + CBS asymptote ---------------------------
    ax_top.plot(X_dense, E_dense, color="#7f7f7f", linestyle="--",
                linewidth=1.4,
                label=fr"Helgaker fit: $E_\mathrm{{CBS}} + A e^{{-\alpha X}}$, "
                      fr"$\alpha = {alpha_fit:.2f}$")
    ax_top.scatter(Xs, Es, color="#1f77b4", s=80, zorder=3,
                   edgecolor="white", linewidth=1.0,
                   label="HF / cc-pVXZ")
    ax_top.axhline(e_cbs, color="#d62728", linewidth=1.2, linestyle=":",
                   label=fr"$E_\mathrm{{CBS}} = {e_cbs:.4f}$ Ha")

    for X, E, (name, _, nbf, _) in zip(Xs, Es, rows):
        ax_top.annotate(f"{name}\n({nbf} bf)", (X, E),
                        xytext=(8, 6), textcoords="offset points",
                        fontsize=9)

    ax_top.set_xlabel("Cardinal number X")
    ax_top.set_ylabel(r"$E_\mathrm{HF}$ (Ha)")
    ax_top.set_title("H$_2$O HF total energy vs. cc-pVXZ basis")
    ax_top.set_xlim(1.5, 5.5)
    ax_top.set_xticks([2, 3, 4, 5])
    ax_top.grid(alpha=0.3, linestyle=":")
    ax_top.legend(loc="upper right", frameon=True, fontsize=9)

    # --- Bottom: |E - E_CBS| on semi-log -------------------------------------
    deviations = np.abs(Es - e_cbs)
    ax_bot.semilogy(Xs, deviations, "o-", color="#1f77b4",
                    markersize=8, linewidth=1.5)
    # The fit curve as a line for comparison
    dev_dense = np.abs(E_dense - e_cbs)
    ax_bot.semilogy(X_dense, dev_dense, color="#7f7f7f",
                    linestyle="--", linewidth=1.0)

    ax_bot.set_xlabel("Cardinal number X")
    ax_bot.set_ylabel(r"$|E - E_\mathrm{CBS}|$ (Ha)")
    ax_bot.set_title("Exponential convergence (semi-log)")
    ax_bot.set_xlim(1.5, 5.5)
    ax_bot.set_xticks([2, 3, 4, 5])
    ax_bot.grid(alpha=0.3, which="both", linestyle=":")

    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=300, bbox_inches="tight")

    print(f"\nWrote {PLOT_OUT.relative_to(HERE.parent.parent)}")
    print(f"  Convergence: cc-pVDZ → cc-pVQZ improves by "
          f"{(Es[0] - Es[-1]) * 1000:+.2f} mHa")
    print(f"  Remaining cc-pVQZ → CBS gap: "
          f"{(Es[-1] - e_cbs) * 1000:+.2f} mHa")


if __name__ == "__main__":
    main()
