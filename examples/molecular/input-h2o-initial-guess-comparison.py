"""Compare initial guesses (HCORE / SAD / SAP / AUTO) on H2O and OH.

Run:
    .venv/bin/python input-h2o-initial-guess-comparison.py

Demonstrates three things:

1. On a well-behaved closed-shell system (H2O / 6-31G*), all four
   initial guesses converge to the same energy. Iteration counts
   vary by 1-2 — initial-guess choice is about efficiency, not
   correctness.

2. SAP scales better with basis size than SAD on closed-shell light
   atoms; we run STO-3G / 6-31G* / cc-pVDZ to show the trend.

3. On an open-shell radical (OH / 6-31G*) HCORE lands on a false
   minimum ~0.16 Ha above the true UHF minimum. SAD (and AUTO, which
   resolves to SAD for any open-shell input) reaches the correct
   minimum.

The full theory is in docs/user_guide/initial_guess.md;
docs/tutorial/initial_guess_walkthrough.md is the running
narrative for this script.
"""

import vibeqc as vq
from vibeqc import (
    Atom,
    BasisSet,
    Molecule,
    RHFOptions,
    UHFOptions,
    run_rhf,
    run_uhf,
)

A2B = 1.0 / 0.529177210903    # Angstrom -> bohr


# ----------------------------------------------------------------------
# Step 1 + 2 — parity + basis trend on H2O
# ----------------------------------------------------------------------

def water():
    return Molecule([
        Atom(8, [0.0,  0.0,  0.0]),
        Atom(1, [0.0,  1.43, -0.98]),
        Atom(1, [0.0, -1.43, -0.98]),
    ])


def run_rhf_with_guess(mol, basis, kind):
    opts = RHFOptions()
    opts.initial_guess = kind
    opts.conv_tol_energy = 1e-10
    return run_rhf(mol, basis, opts)


print("=" * 78)
print("Step 1 + 2 — H2O across basis sizes")
print("=" * 78)
print(f"{'basis':>10}  {'guess':<6}  {'n_iter':>6}  {'energy (Ha)':>15}")
print("-" * 78)
mol_h2o = water()
for basis_name in ["sto-3g", "6-31g*", "cc-pvdz"]:
    basis = BasisSet(mol_h2o, basis_name)
    energies = {}
    for kind in [vq.InitialGuess.HCORE, vq.InitialGuess.SAD,
                 vq.InitialGuess.SAP,   vq.InitialGuess.AUTO]:
        r = run_rhf_with_guess(mol_h2o, basis, kind)
        energies[kind] = r.energy
        print(f"{basis_name:>10}  {kind.name:<6}  "
              f"{r.n_iter:>6d}  {r.energy:>15.10f}")
    # Sanity check: all four agreed on the converged energy.
    e_set = sorted(energies.values())
    spread = e_set[-1] - e_set[0]
    print(f"{'':>10}  {'(spread)':<6}  {'':>6}  {spread:>15.2e}")
print()


# ----------------------------------------------------------------------
# Step 3 — OH false-minimum trap
# ----------------------------------------------------------------------

mol_oh = Molecule(
    [Atom(8, [0.0, 0.0, 0.0]),
     Atom(1, [0.97 * A2B, 0.0, 0.0])],   # 0.97 A O-H
    multiplicity=2,                       # doublet radical
)
basis_oh = BasisSet(mol_oh, "6-31g*")

TRUE_UHF_OH_ENERGY = -75.3809309907   # PySCF / textbook reference

print("=" * 78)
print("Step 3 — OH/6-31G* UHF: HCORE lands on the wrong minimum")
print("=" * 78)
print(f"True UHF minimum (PySCF reference): {TRUE_UHF_OH_ENERGY:.10f} Ha")
print()
print(f"{'guess':<6}  {'n_iter':>6}  {'energy':>13}  {'S²':>6}  "
      f"{'Δ from true min':>16}")
print("-" * 78)
for kind in [vq.InitialGuess.HCORE,
             vq.InitialGuess.SAD,
             vq.InitialGuess.SAP,
             vq.InitialGuess.AUTO]:
    opts = UHFOptions()
    opts.initial_guess = kind
    opts.conv_tol_energy = 1e-10
    opts.max_iter = 300       # HCore needs many iters on this one
    r = run_uhf(mol_oh, basis_oh, opts)
    delta = r.energy - TRUE_UHF_OH_ENERGY
    flag = "  ← false min" if abs(delta) > 1e-3 else "  ← correct"
    print(f"{kind.name:<6}  {r.n_iter:>6d}  {r.energy:>13.6f}  "
          f"{r.s_squared:>6.3f}  {delta:>+16.6f}{flag}")
print()

print("Notes:")
print("  * AUTO sees is_open_shell=True (n_α != n_β) and resolves to SAD,")
print("    sidestepping the HCore false minimum without you having to set it.")
print("  * The PySCF reference is in pyscf/examples/scf/02-rohf_uhf.py.")
