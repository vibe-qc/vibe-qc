"""v0.10.0 D4-refinement bridge: Hirshfeld charges and the dftd4 wall.

This driver runs the new Hirshfeld charge calculator alongside the
production ``vibeqc.compute_d4`` (dftd4-backed, EEQ-internal) on the
same molecule, then states the dftd4-wrapper limitation explicitly
and prints what the Phase-D4b native backend will need to consume.

Once Phase D4b lands (native D4 dispersion energy in
``cpp/src/dispersion_d4.cpp`` with a ``partial_charges`` argument),
this driver becomes the regression seed: same Hirshfeld charges, but
fed into ``compute_d4(..., charge_source='hirshfeld')`` to produce
the D4(Hirshfeld) energy that the v0.10.0 D2b roadmap entry promises.

Usage:
    python studies/d4-hirshfeld-spike/d4_with_hirshfeld_charges.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow importing the sibling spike module without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import vibeqc as vq
from vibeqc import Atom, BasisSet, Molecule, RHFOptions, run_rhf
from vibeqc.dispersion_d4 import compute_d4, dftd4_available

from hirshfeld_charges import hirshfeld_charges


A2B = 1.0 / 0.529177210903


def water_dimer():
    """Linear-ish H₂O···H₂O — donor + acceptor (~2.95 Å O···O).

    Small enough to converge fast; large enough that D4 contributes a
    measurable fraction of the binding energy.
    """
    return Molecule([
        # Acceptor
        Atom(8, [0.000,        0.000,        0.000]),
        Atom(1, [0.000,        0.757 * A2B, -0.587 * A2B]),
        Atom(1, [0.000,       -0.757 * A2B, -0.587 * A2B]),
        # Donor, ~2.95 Å along z
        Atom(8, [0.000,        0.000,        2.95 * A2B]),
        Atom(1, [0.000,        0.000,        1.99 * A2B]),
        Atom(1, [0.940 * A2B,  0.000,        3.20 * A2B]),
    ])


def main() -> None:
    print("v0.10.0 D4 refinement spike — Hirshfeld charges + dftd4 wall")
    print(f"vibe-qc {vq.__version__}, dftd4 available: {dftd4_available()}")
    print()

    mol = water_dimer()
    basis = BasisSet(mol, "def2-svp")

    opts = RHFOptions()
    opts.conv_tol_energy = 1e-9
    result = run_rhf(mol, basis, opts)
    print(f"H2O dimer HF/def2-SVP: E = {result.energy:.8f} Ha, "
          f"n_iter = {result.n_iter}")
    print()

    # The two charge models on the same density.
    hirsh = hirshfeld_charges(result, basis, mol)
    eeq   = np.asarray(vq.eeq_charges(mol, total_charge=0.0).charges)

    print(f"{'atom':>4}  {'Z':>2}  {'EEQ (D4 default)':>16}  "
          f"{'Hirshfeld (D2b)':>16}  {'Δ':>10}")
    print("-" * 60)
    for i, atom in enumerate(mol.atoms):
        print(f"{i:>4}  {int(atom.Z):>2}  "
              f"{eeq[i]:>16.6f}  {float(hirsh.charges[i]):>16.6f}  "
              f"{float(hirsh.charges[i]) - eeq[i]:>+10.6f}")
    print()
    print(f"Σ q_EEQ        = {eeq.sum():+.6e}")
    print(f"Σ q_Hirshfeld  = {hirsh.charges.sum():+.6e}")
    print()

    # The production D4 path — EEQ-internal.
    if dftd4_available():
        d4 = compute_d4(mol, functional="b3lyp", charge=0.0)
        print(f"compute_d4(B3LYP-D4) [EEQ internal]: "
              f"E_disp = {d4.energy:+.8f} Ha")
    else:
        print("compute_d4 unavailable — install with `pip install dftd4`.")

    print()
    print("=" * 72)
    print("Phase D4b gap:")
    print("=" * 72)
    print(
        "  The dftd4 Python *and* C ABI accept only a total-system charge.\n"
        "  There is no `partial_charges=…` argument anywhere in the stack,\n"
        "  so the Hirshfeld charges computed above CANNOT yet be plumbed\n"
        "  into the D4 C6 scaling step. The v0.10.0 D2b refinement is\n"
        "  blocked on Phase D4b (native D4 dispersion energy w/ external\n"
        "  charges). When D4b lands, expect:\n"
        "\n"
        "      compute_d4(mol, functional='b3lyp',\n"
        "                 partial_charges=hirsh.charges)\n"
        "\n"
        "  on the native backend — same damping parameters and the same\n"
        "  Caldeweyher reference C6 grid, but EEQ → Hirshfeld at the\n"
        "  charge-input step. The Hirshfeld charges above are the\n"
        "  regression input.\n"
    )


if __name__ == "__main__":
    main()
