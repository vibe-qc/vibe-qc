#!/usr/bin/env python3
"""Semiempirical preoptimization workflow — DFTB0 → DFT.

Demonstrates the recommended preoptimization pattern:
1. Fast DFTB0 structure optimization
2. Refinement with DFT (PBE/def2-SVP)

Run:
    .venv/bin/python examples/semiempirical/09_dftb_preoptimize.py
"""

from __future__ import annotations

import numpy as np
from vibeqc import Atom, Molecule
from vibeqc.semiempirical import DFTB0Model


def main():
    # ── Water molecule (initial guess, slightly distorted) ──
    theta = np.deg2rad(104.5 / 2)
    r_oh = 1.90  # slightly stretched
    mol = Molecule(
        [
            Atom(8, [0.00, 0.00, 0.00]),
            Atom(1, [r_oh * np.sin(theta), r_oh * np.cos(theta), 0.00]),
            Atom(1, [-r_oh * np.sin(theta), r_oh * np.cos(theta), 0.00]),
        ],
        charge=0,
        multiplicity=1,
    )

    # ── Step 1: DFTB0 energy at initial geometry ──
    model = DFTB0Model(mol)
    e0 = model.energy()
    print(f"Initial DFTB0 energy: {e0:12.6f} Ha")

    # ── Step 2: Run DFTB0 geometry optimization via ASE ──
    print("\nOptimizing with DFTB0 (analytic gradients)...")
    try:
        from ase.optimize import BFGS

        mol_ase = mol.to_ase()
        mol_ase.calc = model.to_ase_calculator()
        opt = BFGS(mol_ase)
        opt.run(fmax=0.005)
        e_opt = model.energy()
        print(f"Optimized DFTB0 energy: {e_opt:12.6f} Ha")
        print(f"Optimized geometry (bohr):")
        for atom in mol.atoms:
            print(
                f"  {atom.Z:2d}  {atom.xyz[0]:10.6f} {atom.xyz[1]:10.6f} {atom.xyz[2]:10.6f}"
            )
    except ImportError:
        print("ASE not available — skipping optimization")
        print("Install: pip install ase")

    # ── Step 3: Refine with DFT ──
    print("\n┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈")
    print("Next: refine with DFT (PBE/def2-SVP)")
    print("  from vibeqc import run_job")
    print("  run_job(mol, method='rks', functional='PBE',")
    print("          basis='def2-svp', optimize=True, output='h2o_dft')")

    # ── Performance ──
    print(f"\nDFTB0 cost: ~0.01 s per energy evaluation (vs DFT: ~1–10 s)")


if __name__ == "__main__":
    main()
