"""H₂ dissociation curve: RHF vs Selected-CI vs FCI.

Scans the H-H bond length from 0.6 to 6.0 bohr and compares:
  * Restricted Hartree–Fock (size-consistent but wrong at dissociation)
  * Selected-CI (systematically improvable, exact-in-basis at large ndet)
  * Full CI in the STO-3G basis (exact reference)

The STO-3G basis has only 2 spatial orbitals for H₂, so the FCI
space is 2 determinants and Selected-CI with target_size=2 recovers
the exact-in-basis result.

Run:
    .venv/bin/python examples/wavefunction/h2_dissociation.py
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from vibeqc import Atom, BasisSet, Molecule
from vibeqc._vibeqc_core import RHFOptions, run_rhf
from vibeqc.solvers import (
    Hamiltonian,
    SelectedCIOptions,
    V2RDMOptions,
    build_hamiltonian_matrix,
    build_hamiltonian_mo,
    generate_closed_shell_determinants,
    get_hf_orbital_provider,
    solve_selected_ci,
    solve_v2rdm,
)


def compute_energies(r: float) -> dict:
    """Compute RHF, Selected-CI, and FCI energies for H₂ at bond length r."""
    mol = Molecule(
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, r])],
        charge=0,
        multiplicity=1,
    )
    basis = BasisSet(mol, "sto-3g")
    C = get_hf_orbital_provider(mol, basis)
    ham = build_hamiltonian_mo(mol, basis, C)

    # RHF
    rhf_opts = RHFOptions()
    rhf_opts.conv_tol_energy = 1e-12
    rhf_opts.conv_tol_grad = 1e-10
    rhf_result = run_rhf(mol, basis, rhf_opts)
    e_rhf = rhf_result.energy

    # Selected-CI (FCI limit: target_size=2 recovers exact-in-basis)
    ci_opts = SelectedCIOptions(
        target_size=5,
        max_iter=20,
        conv_tol_energy=1e-10,
        do_pt2_correction=True,
        verbose=0,
    )
    ci_result = solve_selected_ci(ham, ci_opts)
    e_ci = ci_result.energy

    # Full CI
    all_dets = generate_closed_shell_determinants(ham.norb, ham.nelec // 2)
    H_fci = build_hamiltonian_matrix(all_dets, ham.h1e, ham.h2e)
    evals, _ = eigh(H_fci)
    e_fci = evals[0] + ham.nuclear_repulsion

    return {"R": r, "RHF": e_rhf, "Sel-CI": e_ci, "FCI": e_fci}


# ── Scan bond lengths ───────────────────────────────────────────────────
print("=" * 72)
print(" H₂ dissociation curve  —  STO-3G")
print("=" * 72)
print(
    f" {'R/bohr':>8s}  {'E(RHF)':>14s}  {'E(Sel-CI)':>14s}  "
    f"{'E(FCI)':>14s}  {'Δ(RHF−FCI)':>14s}"
)
print("-" * 72)

results = []
for r in [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]:
    res = compute_energies(r)
    results.append(res)
    delta = res["RHF"] - res["FCI"]
    print(
        f" {res['R']:>8.2f}  {res['RHF']:>14.10f}  {res['Sel-CI']:>14.10f}  "
        f"{res['FCI']:>14.10f}  {delta:>14.10f}"
    )

# ── Summary ─────────────────────────────────────────────────────────────
print(f"\nEquilibrium bond length (RHF minimum):")
r_vals = np.array([r["R"] for r in results])
e_rhf_vals = np.array([r["RHF"] for r in results])
imin = np.argmin(e_rhf_vals)
print(f"  R_eq(RHF) ≈ {r_vals[imin]:.2f} bohr")

print(f"\nDissociation limit (R = 6.0 bohr):")
r_diss = results[-1]
print(f"  E(RHF)   = {r_diss['RHF']:.10f} Ha")
print(f"  E(FCI)   = {r_diss['FCI']:.10f} Ha")
print(f"  RHF error = {(r_diss['RHF'] - r_diss['FCI']) * 627.509:.2f} kcal/mol")
print(
    f"  Selected-CI error = {(r_diss['Sel-CI'] - r_diss['FCI']) * 627.509:.6f} kcal/mol"
)
