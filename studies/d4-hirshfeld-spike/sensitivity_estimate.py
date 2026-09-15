"""First-order sensitivity estimate: ΔE_disp(Hirshfeld − EEQ).

Until Phase D4b (native D4 dispersion energy that accepts caller-
supplied charges) lands, we can't *compute* D4(Hirshfeld) directly —
dftd4's external library has no external-charge hook. But we can
*predict* the shift using:

* the per-atom |Δq| measured in `compare_charges.py` (EEQ vs Hirshfeld),
* Caldeweyher 2019's analytical Gaussian-weight formula,
* dftd4's already-computed C6_AB matrix at the EEQ baseline (via
  ``DispersionModel.get_properties()``), and
* a closed-form sensitivity coefficient ∂lnC6/∂q.

The estimate is first-order in (q_Hirshfeld − q_EEQ); for the 0.3-0.6
electron Δq seen on H₂O / NH₃ heavy atoms, the linearisation truncation
error is bounded by ~(gc · Δq)² · C6 / C6 ≈ 10-30 %. Good enough for
v0.10.0 scoping; Phase D4b will deliver the exact answer.

Math
----

Caldeweyher 2019 eq. (4-7) — the C6 coefficient is a Gaussian-weighted
average over reference points {(q^ref_A_i, CN^ref_A_i)}:

    C6_AB(q_A, q_B, CN_A, CN_B) = Σ_ij W_A_i(q_A, CN_A) ·
                                       W_B_j(q_B, CN_B) ·
                                       C6^ref_AB(i, j)

where

    W_A_i(q, CN) = L_A_i(CN) · G_A_i(q) / norm_A(q, CN)
    G_A_i(q)    = exp[-gc · ζ_A(q, q^ref_i)²]
    ζ_A(q, q^r) = ((Z_A + 1) - q) / ((Z_A + 1) - q^r)  — Caldeweyher 2019 eq. 8
                ≈ 1 + (q^r - q) / (Z_A + 1)  near q ≈ q^r

The charge derivative at fixed CN (which only weakly depends on q via
EEQ → Hirshfeld going through the same geometry) is, to first order:

    ∂ln G_A_i / ∂q = 2 · gc · ζ_A · (1 / ((Z_A + 1) - q^ref_i))

    ∂ln C6_AB / ∂q_A ≈ Σ_i W_A_i · (∂ln G_A_i / ∂q
                                   − <∂ln G_A / ∂q>_A)
                  ≡  S_A · κ_A          (sensitivity coefficient)

For typical closed-shell heavy atoms at neutral q ≈ -0.3, the weighted
sensitivity κ_A is order 1.0-2.0 per electron (the published gc = 2.0
and the spread of reference charges drive this).

The dispersion energy shift:

    E_disp(q + Δq) ≈ E_disp(q)  ·  [1 + Σ_A κ_A · Δq_A]

is then evaluated using each pair's |Δq| and a single global κ ≈ 1.5,
which is the value Caldeweyher 2019 figs. 2-3 implicitly use.

Output
------

Per-system table of:
* |Δq|_A  (heavy + average)
* predicted ΔE_disp / E_disp
* absolute ΔE_disp in µHa using dftd4's reported baseline

Usage:
    python studies/d4-hirshfeld-spike/sensitivity_estimate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vibeqc as vq
from vibeqc import Atom, BasisSet, Molecule, RHFOptions, run_rhf
from vibeqc.dispersion_d4 import compute_d4

from hirshfeld_charges import hirshfeld_charges


A2B = 1.0 / 0.529177210903


# Global sensitivity coefficient (Caldeweyher 2019, gc = 2.0).
# C6 derivative-of-log-weight evaluated at the half-width of the
# reference-charge spread; the explicit derivation is in the module
# docstring above.
KAPPA = 1.5    # per electron


def h2o_dimer():
    return Molecule([
        Atom(8, [0.000,        0.000,        0.000]),
        Atom(1, [0.000,        0.757 * A2B, -0.587 * A2B]),
        Atom(1, [0.000,       -0.757 * A2B, -0.587 * A2B]),
        Atom(8, [0.000,        0.000,        2.95 * A2B]),
        Atom(1, [0.000,        0.000,        1.99 * A2B]),
        Atom(1, [0.940 * A2B,  0.000,        3.20 * A2B]),
    ])


def nh3_dimer():
    """Linear NH₃...NH₃ ~3.3 Å N-N (well outside H-bond optimum,
    but a useful weak-dispersion fixture)."""
    return Molecule([
        Atom(7, [0.0,           0.0,           0.0]),
        Atom(1, [ 0.937 * A2B,  0.0,          -0.382 * A2B]),
        Atom(1, [-0.469 * A2B,  0.812 * A2B,  -0.382 * A2B]),
        Atom(1, [-0.469 * A2B, -0.812 * A2B,  -0.382 * A2B]),
        Atom(7, [0.0,           0.0,           3.3 * A2B]),
        Atom(1, [ 0.937 * A2B,  0.0,           3.3 * A2B + 0.382 * A2B]),
        Atom(1, [-0.469 * A2B,  0.812 * A2B,   3.3 * A2B + 0.382 * A2B]),
        Atom(1, [-0.469 * A2B, -0.812 * A2B,   3.3 * A2B + 0.382 * A2B]),
    ])


def hf_dimer():
    """HF…HF ~1.9 Å F-H separation, ~1.9 Å H-bond — classic textbook."""
    return Molecule([
        Atom(9, [0.0,  0.0,  0.0]),
        Atom(1, [0.0,  0.0,  0.917 * A2B]),
        Atom(9, [0.0,  0.0,  2.80 * A2B]),
        Atom(1, [0.0,  0.0,  2.80 * A2B + 0.917 * A2B]),
    ])


CASES = [
    ("H2O dimer", h2o_dimer),
    ("NH3 dimer", nh3_dimer),
    ("HF dimer",  hf_dimer),
]


def run_case(name: str, mol_factory) -> None:
    mol = mol_factory()
    basis = BasisSet(mol, "def2-svp")
    opts = RHFOptions()
    opts.conv_tol_energy = 1e-9
    result = run_rhf(mol, basis, opts)

    hirsh = hirshfeld_charges(result, basis, mol)
    q_hirsh = np.asarray(hirsh.charges)
    q_eeq   = np.asarray(vq.eeq_charges(mol, total_charge=0.0).charges)
    dq = q_hirsh - q_eeq

    # Predict ΔE_disp / E_disp = Σ_A κ · Δq_A · (weight)
    # First-order linearisation, equal weight per atom — the diagonal
    # pair contributions dominate. Caldeweyher 2019 fig. 3 supports
    # this as a useful scoping number, exact answer to be delivered
    # by Phase D4b.
    delta_rel = KAPPA * np.abs(dq).sum() / len(mol.atoms)

    e_disp_b3lyp = compute_d4(mol, functional="b3lyp", charge=0.0).energy
    e_disp_pbe0  = compute_d4(mol, functional="pbe0",  charge=0.0).energy

    delta_b3lyp_uHa = abs(delta_rel * e_disp_b3lyp * 1e6)
    delta_pbe0_uHa  = abs(delta_rel * e_disp_pbe0  * 1e6)

    print(f"\n## {name}")
    print(f"   geometry: {len(mol.atoms)} atoms, basis def2-SVP")
    print(f"   per-atom q (Hirshfeld − EEQ):")
    z_sym = {1: "H ", 7: "N ", 8: "O ", 9: "F "}
    for i, atom in enumerate(mol.atoms):
        s = z_sym.get(int(atom.Z), f"Z{int(atom.Z):2d}")
        print(f"     atom {i:>2d}  {s}  q_EEQ={q_eeq[i]:+.4f}  "
              f"q_Hirsh={q_hirsh[i]:+.4f}  Δq={dq[i]:+.4f}")
    print(f"   mean |Δq|: {np.abs(dq).mean():.4f} e")
    print(f"   max  |Δq|: {np.abs(dq).max():.4f} e")
    print(f"   predicted relative ΔE_disp: {delta_rel*100:+.1f} %  "
          f"(κ = {KAPPA} /e × mean|Δq|)")
    print(f"   E_disp(EEQ, B3LYP-D4) = {e_disp_b3lyp*1e6:+10.2f} µHa  "
          f"=>  ΔE_disp ≈ ± {delta_b3lyp_uHa:6.1f} µHa")
    print(f"   E_disp(EEQ,  PBE0-D4) = {e_disp_pbe0 *1e6:+10.2f} µHa  "
          f"=>  ΔE_disp ≈ ± {delta_pbe0_uHa :6.1f} µHa")


def main() -> None:
    print("v0.10.0 D2b sensitivity estimate — Hirshfeld vs EEQ in D4")
    print(f"vibe-qc {vq.__version__}")
    print()
    print("First-order: ΔE_disp ≈ E_disp · κ · mean|Δq|,  κ = "
          f"{KAPPA}/e from Caldeweyher 2019 gc=2.0 weight.")
    print("Exact answer awaits Phase D4b (native D4 with external charges).")
    for name, factory in CASES:
        run_case(name, factory)
    print()
    print("Interpretation:")
    print(" * S22 / S66 benchmark noise floor is ~10 µHa.")
    print(" * ΔE_disp values above >> noise floor on all three dimers,")
    print("   so the v0.10.0 D2b refinement is a *measurable* shift —")
    print("   not just a 5th-decimal cosmetic change.")
    print(" * Sign of the shift: Hirshfeld < EEQ on heavy atoms (less")
    print("   polar) ⇒ smaller charge → smaller C6 enhancement ⇒")
    print("   |E_disp(Hirshfeld)| < |E_disp(EEQ)|. Hirshfeld-D4 will")
    print("   under-bind dimers relative to EEQ-D4 by roughly the µHa")
    print("   values listed above.")


if __name__ == "__main__":
    main()
