"""H2O harmonic frequencies — tutorial 09 figure.

Optimises water at HF/STO-3G via vibe-qc through ASE, runs the ASE
finite-difference Hessian, and produces a stick-spectrum-style
comparison of raw harmonic frequencies, scaled (Pople 0.89) frequencies,
and gas-phase experimental values. Writes
``docs/_static/plots/water-vibrational-frequencies.png`` — the figure
embedded in tutorial 09. ~30 s wall (geometry + 18 force calls).

Run:
    .venv/bin/python examples/plots/water-vibrational-frequencies.py
"""

from pathlib import Path
import shutil
import tempfile

from ase import Atoms
from ase.optimize import BFGSLineSearch
from ase.vibrations import Vibrations
import matplotlib.pyplot as plt
import numpy as np

from vibeqc.ase import VibeQC

HERE = Path(__file__).resolve().parent
PLOT_OUT = HERE.parent.parent / "docs" / "_static" / "plots" / "water-vibrational-frequencies.png"

POPLE_HF_SCALE = 0.89  # Pople / Wong 1996 scaling factor for HF/STO-3G-class methods

# Experimental gas-phase fundamentals from NIST WebBook / Benedict et al. 1956.
EXP_FREQS = {
    "HOH bend":              1595.0,
    "symmetric O-H stretch": 3657.0,
    "antisymmetric O-H stretch": 3756.0,
}


def main() -> None:
    atoms = Atoms(
        symbols=["O", "H", "H"],
        positions=[
            [0.0,  0.00,  0.00],
            [0.0,  0.76,  0.59],
            [0.0, -0.76,  0.59],
        ],
    )
    atoms.calc = VibeQC(basis="sto-3g")
    BFGSLineSearch(atoms, logfile=None).run(fmax=0.01)

    # ASE Vibrations writes per-displacement JSON files; isolate them
    # in a temp dir so re-runs are clean.
    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path.cwd()
        try:
            import os
            os.chdir(tmp)
            vib = Vibrations(atoms, name="h2o_vib")
            vib.run()
            freqs_complex = vib.get_frequencies()
            zpe = vib.get_zero_point_energy()
        finally:
            os.chdir(cwd)

    # Keep real internal modes only (drop translations/rotations + tiny
    # imaginary numerical noise).
    real_mask = np.abs(freqs_complex.imag) < 1.0
    real_freqs = np.sort(np.real(freqs_complex[real_mask]))
    # The three highest real frequencies are bend + 2 stretches.
    internal = real_freqs[-3:]
    bend, sym_stretch, asym_stretch = internal

    raw = np.array([bend, sym_stretch, asym_stretch])
    scaled = raw * POPLE_HF_SCALE
    exp = np.array(list(EXP_FREQS.values()))
    labels = list(EXP_FREQS.keys())

    PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=150)
    x = np.arange(len(labels))
    width = 0.27

    bars_raw = ax.bar(x - width, raw, width,
                      color="#1f77b4", edgecolor="black", linewidth=0.6,
                      label="HF/STO-3G harmonic")
    bars_scaled = ax.bar(x, scaled, width,
                         color="#2ca02c", edgecolor="black", linewidth=0.6,
                         label=fr"scaled $\times {POPLE_HF_SCALE}$ (Pople)")
    bars_exp = ax.bar(x + width, exp, width,
                      color="#d62728", edgecolor="black", linewidth=0.6,
                      label="experiment")

    # Annotate each bar with its value
    for group in (bars_raw, bars_scaled, bars_exp):
        for bar in group:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 60,
                    f"{h:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(r"frequency (cm$^{-1}$)")
    ax.set_title(rf"H$_2$O harmonic frequencies — HF/STO-3G vs Pople scaling vs experiment "
                 rf"(ZPE = {zpe:.3f} eV)",
                 fontsize=10)
    ax.set_ylim(0, max(raw.max(), exp.max()) * 1.18)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.legend(loc="upper left", fontsize=9, frameon=True)

    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=300, bbox_inches="tight")

    print(f"Wrote {PLOT_OUT.relative_to(HERE.parent.parent)}")
    for label, r, s, e in zip(labels, raw, scaled, exp):
        print(f"  {label:30s}  raw={r:7.1f}  scaled={s:7.1f}  "
              f"exp={e:7.1f}  Δ_scaled={s - e:+6.1f}")
    print(f"  ZPE = {zpe:.3f} eV")


if __name__ == "__main__":
    main()
