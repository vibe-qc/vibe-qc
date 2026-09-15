"""Smallest-system comparison: Hirshfeld vs Mulliken vs Löwdin vs EEQ.

Driver for the v0.10.0 D4-refinement spike. Runs RHF on a handful of
closed-shell test molecules with the same basis, then prints each
charge model's per-atom values side-by-side. The comparison surfaces
the qualitative + quantitative gap between EEQ (D4's built-in
electronegativity-equalisation surrogate) and Hirshfeld (the SCF-
derived partition the v0.10.0 D2b roadmap calls for).

Reference targets are from ORCA 5.x ``! HF def2-SVP HIRSHFELD``
output on the same geometries (Bohr coordinates copied directly into
ORCA inputs for byte-equal geometries).

Usage:
    python studies/d4-hirshfeld-spike/compare_charges.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow importing the sibling spike module without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import vibeqc as vq
from vibeqc import Atom, BasisSet, Molecule, RHFOptions, run_rhf
from vibeqc.properties import loewdin_charges, mulliken_charges

from hirshfeld_charges import hirshfeld_charges


A2B = 1.0 / 0.529177210903   # Angstrom → bohr


def water():
    """Equilibrium-ish H₂O (rOH ≈ 0.96 Å, ∠HOH ≈ 104.5°)."""
    return Molecule([
        Atom(8, [0.0,           0.0,           0.0]),
        Atom(1, [0.0,           0.757 * A2B,  -0.587 * A2B]),
        Atom(1, [0.0,          -0.757 * A2B,  -0.587 * A2B]),
    ])


def ammonia():
    """NH₃, planar-ish C3v geometry (rNH ≈ 1.012 Å, ∠HNH ≈ 106.7°)."""
    return Molecule([
        Atom(7, [0.0,           0.0,           0.0]),
        Atom(1, [ 0.937 * A2B,  0.0,          -0.382 * A2B]),
        Atom(1, [-0.469 * A2B,  0.812 * A2B,  -0.382 * A2B]),
        Atom(1, [-0.469 * A2B, -0.812 * A2B,  -0.382 * A2B]),
    ])


def hydrogen_fluoride():
    """HF, rHF ≈ 0.917 Å."""
    return Molecule([
        Atom(9, [0.0,  0.0,  0.0]),
        Atom(1, [0.0,  0.0,  0.917 * A2B]),
    ])


def ammonium():
    """NH₄⁺, regular Td (rNH ≈ 1.025 Å). Sanity for Σ q_A = +1."""
    d = 1.025 * A2B / np.sqrt(3.0)
    return Molecule(
        [
            Atom(7, [0.0,  0.0,  0.0]),
            Atom(1, [+d, +d, +d]),
            Atom(1, [+d, -d, -d]),
            Atom(1, [-d, +d, -d]),
            Atom(1, [-d, -d, +d]),
        ],
        charge=+1,
    )


def hydroxide():
    """OH⁻, rOH ≈ 0.964 Å. Sanity for Σ q_A = −1."""
    return Molecule(
        [
            Atom(8, [0.0,  0.0,  0.0]),
            Atom(1, [0.0,  0.0,  0.964 * A2B]),
        ],
        charge=-1,
    )


CASES = [
    ("H2O",   water,              0),
    ("NH3",   ammonia,            0),
    ("HF",    hydrogen_fluoride,  0),
    ("NH4+",  ammonium,          +1),
    ("OH-",   hydroxide,         -1),
]

BASIS_NAME = "def2-svp"


def fmt_row(label: str, charges: np.ndarray, atoms) -> str:
    """One-line per-atom display: `Hirshfeld:  O −0.318  H +0.159  H +0.159  Σ=+0.000`."""
    Z_SYMBOL = {1: "H ", 6: "C ", 7: "N ", 8: "O ", 9: "F "}
    parts = [f"{label:>10s}:"]
    for atom, q in zip(atoms, charges):
        sym = Z_SYMBOL.get(int(atom.Z), f"Z{int(atom.Z):>2d}")
        parts.append(f"{sym}{q:+0.4f}")
    parts.append(f"  Σ={charges.sum():+0.4f}")
    return "  ".join(parts)


def run_case(name: str, mol_factory, expected_total_q: int) -> None:
    mol = mol_factory()
    basis = BasisSet(mol, BASIS_NAME)

    opts = RHFOptions()
    opts.conv_tol_energy = 1e-9
    result = run_rhf(mol, basis, opts)

    mulliken = mulliken_charges(result, basis, mol)
    loewdin  = loewdin_charges(result, basis, mol)
    hirsh    = hirshfeld_charges(result, basis, mol)
    eeq      = vq.eeq_charges(mol, total_charge=float(mol.charge)).charges

    print(f"\n## {name}  (HF/{BASIS_NAME}, E = {result.energy:.8f} Ha, "
          f"n_iter = {result.n_iter})")
    print(f"     molecular charge = {mol.charge:+d}  "
          f"(expected Σ q_A = {expected_total_q:+d})")
    print(f"     Hirshfeld grid:  Ng = {hirsh.n_grid_points}, "
          f"∫ρ_mol = {hirsh.molecule_norm:.4f}, "
          f"∫ρ_pro = {hirsh.promolecule_norm:.4f}")
    print()
    print(fmt_row("Mulliken",  np.asarray(mulliken), mol.atoms))
    print(fmt_row("Loewdin",   np.asarray(loewdin),  mol.atoms))
    print(fmt_row("EEQ",       np.asarray(eeq),      mol.atoms))
    print(fmt_row("Hirshfeld", np.asarray(hirsh.charges), mol.atoms))

    # Sanity: the SCF density integrates to n_e on the grid;
    # Σ q_A integrates to molecular charge.
    n_e_expected = sum(int(a.Z) for a in mol.atoms) - mol.charge
    print(f"     sanity:  ∫ρ_mol vs n_e: "
          f"{hirsh.molecule_norm:.4f} / {n_e_expected:d} "
          f"= {hirsh.molecule_norm / n_e_expected:.6f}")
    sum_q = float(hirsh.charges.sum())
    print(f"     sanity:  Σ q^Hirshfeld − q_mol = "
          f"{sum_q - expected_total_q:+.2e}")


def main() -> None:
    print(f"Hirshfeld vs Mulliken vs Löwdin vs EEQ — HF/{BASIS_NAME}")
    print(f"vibe-qc {vq.__version__}")
    for name, factory, q_total in CASES:
        run_case(name, factory, q_total)
    print()
    print("Notes:")
    print(" * Hirshfeld values should sit between Mulliken and EEQ in")
    print("   magnitude (EEQ tends to under-polarise charges on small")
    print("   molecules; Mulliken depends strongly on basis diffuseness).")
    print(" * For H2O / NH3 / HF the ORCA classical-Hirshfeld reference")
    print("   values are q(O)≈−0.32, q(N)≈−0.42, q(F)≈−0.21 (HF/def2-SVP).")
    print(" * NH4+ and OH- exercise Σ q_A = molecular charge to grid")
    print("   precision (the Hirshfeld weight identity is exact; only")
    print("   the integration grid introduces error).")


if __name__ == "__main__":
    main()
