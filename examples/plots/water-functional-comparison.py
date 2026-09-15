"""DFT functional comparison on H2O — tutorial 15 figure.

Runs HF, LDA (SVWN), GGA (PBE, BLYP) and hybrid (B3LYP, PBE0) on water at
the experimental geometry with 6-31G* and writes a 3-panel figure
``docs/_static/plots/water-functional-comparison.png`` showing dipole,
HOMO level, and HOMO-LUMO gap across the Jacob's-ladder rungs, with
experimental references where available. ~5 s wall.

Run:
    .venv/bin/python examples/plots/water-functional-comparison.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import vibeqc as vq

HERE = Path(__file__).resolve().parent
PLOT_OUT = HERE.parent.parent / "docs" / "_static" / "plots" / "water-functional-comparison.png"

HARTREE_TO_EV = 27.211386

# (label, libxc spec, ladder rung, color)
METHODS = [
    ("HF",     None,    "HF",     "#444444"),
    ("SVWN",   "SVWN",  "LDA",    "#1f77b4"),
    ("PBE",    "PBE",   "GGA",    "#2ca02c"),
    ("BLYP",   "BLYP",  "GGA",    "#17becf"),
    ("B3LYP",  "B3LYP", "hybrid", "#ff7f0e"),
    ("PBE0",   "406",   "hybrid", "#d62728"),
]

# Experimental references (gas phase)
EXP_DIPOLE_D = 1.855       # Clough et al. 1973
EXP_HOMO_EV = -12.621      # vertical ionisation potential (NIST)


def main() -> None:
    mol = vq.Molecule([
        vq.Atom(8, [0.0,  0.00,  0.00]),
        vq.Atom(1, [0.0,  1.43, -0.98]),
        vq.Atom(1, [0.0, -1.43, -0.98]),
    ])
    basis = vq.BasisSet(mol, "6-31g*")

    labels: list[str] = []
    rungs: list[str] = []
    colors: list[str] = []
    dipoles: list[float] = []
    homos_ev: list[float] = []
    gaps_ev: list[float] = []

    n_occ = mol.n_electrons() // 2

    for label, functional, rung, color in METHODS:
        if functional is None:
            r = vq.run_rhf(mol, basis)
        else:
            opts = vq.RKSOptions()
            opts.functional = functional
            r = vq.run_rks(mol, basis, opts)
        mu = vq.dipole_moment(r, basis, mol)
        eps = np.asarray(r.mo_energies)
        homo_eV = float(eps[n_occ - 1]) * HARTREE_TO_EV
        lumo_eV = float(eps[n_occ]) * HARTREE_TO_EV

        labels.append(label)
        rungs.append(rung)
        colors.append(color)
        dipoles.append(float(mu.total_debye))
        homos_ev.append(homo_eV)
        gaps_ev.append(lumo_eV - homo_eV)

    PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), dpi=150)
    x = np.arange(len(labels))

    # Panel 1: dipole
    ax = axes[0]
    ax.bar(x, dipoles, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(EXP_DIPOLE_D, color="#d62728", ls="--", lw=1.0,
               label=f"exp. {EXP_DIPOLE_D} D")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(r"$|\mu|$ (D)")
    ax.set_title("Dipole moment", fontsize=11)
    ax.set_ylim(0, max(dipoles) * 1.20)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.legend(loc="upper right", fontsize=9, frameon=True)

    # Panel 2: HOMO level (Koopmans)
    ax = axes[1]
    ax.bar(x, homos_ev, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(EXP_HOMO_EV, color="#d62728", ls="--", lw=1.0,
               label=fr"$-\mathrm{{IP}}_\mathrm{{exp}} = {EXP_HOMO_EV:.2f}$ eV")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(r"$\varepsilon_{\mathrm{HOMO}}$ (eV)")
    ax.set_title("HOMO level (Koopmans)", fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.legend(loc="lower right", fontsize=9, frameon=True)

    # Panel 3: HOMO-LUMO gap
    ax = axes[2]
    ax.bar(x, gaps_ev, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("HOMO-LUMO gap (eV)")
    ax.set_title("HOMO-LUMO gap", fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle=":")

    fig.suptitle(r"H$_2$O / 6-31G* — functional comparison across Jacob's ladder",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=300, bbox_inches="tight")

    print(f"Wrote {PLOT_OUT.relative_to(HERE.parent.parent)}")
    print(f"{'method':10s} {'rung':8s} {'mu (D)':>8s}  "
          f"{'HOMO (eV)':>10s}  {'gap (eV)':>9s}")
    for label, rung, mu, h, g in zip(labels, rungs, dipoles, homos_ev, gaps_ev):
        print(f"  {label:8s} {rung:8s} {mu:8.3f}  {h:10.3f}  {g:9.3f}")
    print(f"  exp. dipole = {EXP_DIPOLE_D} D,  -IP_exp = {EXP_HOMO_EV} eV")


if __name__ == "__main__":
    main()
