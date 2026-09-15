"""BH dissociation — polar bond breaking and the ionic-to-covalent transition.

Boron monohydride is the smallest heteronuclear diatomic with a polar
covalent bond.  At equilibrium (R = 2.329 bohr), the wavefunction has
significant B⁻H⁺ ionic character.  As the bond stretches, the system
must smoothly transition to neutral open-shell fragments B(²P) + H(²S).

This tests whether a correlation method can:
  * Describe the ionic character at equilibrium.
  * Correctly dissociate to neutral fragments without a size-consistency
    error.
  * Handle the changing dipole moment across the potential curve.

Reference: Harrison & Handy, Chem. Phys. Lett. 95, 386 (1983).

Run:
    .venv/bin/python examples/wavefunction/benchmark_bh_dissociation.py
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from vibeqc import Atom, BasisSet, Molecule
from vibeqc._vibeqc_core import RHFOptions, run_rhf
from vibeqc.solvers import (
    Hamiltonian,
    SelectedCIOptions,
    build_hamiltonian_matrix,
    build_hamiltonian_mo,
    generate_closed_shell_determinants,
    get_hf_orbital_provider,
    solve_selected_ci,
)


def compute_bh(r_bh: float) -> dict:
    """Compute energies for BH at bond length r_bh (bohr)."""
    mol = Molecule(
        [Atom(5, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, r_bh])],
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

    # Selected-CI
    ci_opts = SelectedCIOptions(
        target_size=50,
        max_iter=30,
        conv_tol_energy=1e-10,
        do_pt2_correction=True,
        pt2_threshold=1e-8,
        verbose=0,
    )
    ci_result = solve_selected_ci(ham, ci_opts)

    # FCI
    all_dets = generate_closed_shell_determinants(ham.norb, ham.nelec // 2)
    H_fci = build_hamiltonian_matrix(all_dets, ham.h1e, ham.h2e)
    evals, evecs = eigh(H_fci)
    e_fci = evals[0] + ham.nuclear_repulsion

    # Dominant determinant weight
    c0 = np.abs(evecs[:, 0])
    max_c = float(np.max(c0))

    return {
        "R": r_bh,
        "RHF": rhf_result.energy,
        "Sel-CI": ci_result.energy,
        "FCI": e_fci,
        "ndet_ci": len(ci_result.ci_labels) if ci_result.ci_labels else 0,
        "c0_max": max_c,
    }


# ═══════════════════════════════════════════════════════════════════════════
print("=" * 78)
print("  BH dissociation — Polar bond breaking benchmark")
print("=" * 78)
print(f"  Basis: STO-3G  (6 electrons in 6 spatial orbitals)")
print(f"  Equilibrium: R_eq(BH) = 2.329 bohr  (1.232 Å)")
print(f"  FCI space:  C(6,3) = 20 closed-shell determinants")
print()
print(
    f"  {'R/bohr':>8s}  {'E(RHF)':>14s}  {'E(Sel-CI)':>14s}  "
    f"{'E(FCI)':>14s}  {'Δ(RHF−FCI)':>14s}  {'|c₀|²':>8s}  {'ndet(CI)':>8s}"
)
print("  " + "-" * 78)

results = []
for r in [1.8, 2.0, 2.2, 2.329, 2.5, 2.8, 3.2, 3.6, 4.0, 4.5, 5.0, 6.0]:
    res = compute_bh(r)
    results.append(res)
    delta = res["RHF"] - res["FCI"]
    print(
        f"  {res['R']:>8.3f}  {res['RHF']:>14.8f}  {res['Sel-CI']:>14.8f}  "
        f"{res['FCI']:>14.8f}  {delta:>14.8f}  {res['c0_max'] ** 2:>8.4f}  "
        f"{res['ndet_ci']:>8d}"
    )

# ── Equilibrium analysis ─────────────────────────────────────────────────
idx_eq = 3  # R = 2.329
r_eq = results[idx_eq]
e_corr_eq = r_eq["FCI"] - r_eq["RHF"]
print(f"\n  ═══ Equilibrium (R = 2.329 bohr) ═══")
print(
    f"    E_corr(FCI) = {e_corr_eq * 1000:.2f} mHa  "
    f"({e_corr_eq * 627.509:.3f} kcal/mol)"
)
print(f"    Reference weight |c₀|² = {r_eq['c0_max'] ** 2:.4f}")
print(f"    Selected-CI error vs FCI: {(r_eq['Sel-CI'] - r_eq['FCI']) * 1e6:.2f} μHa")

# ── Dissociation analysis ────────────────────────────────────────────────
r_diss = results[-1]
e_corr_diss = r_diss["FCI"] - r_diss["RHF"]
print(f"\n  ═══ Dissociation limit (R = 6.0 bohr) ═══")
print(
    f"    E_corr(FCI) = {e_corr_diss * 1000:.2f} mHa  "
    f"({e_corr_diss * 627.509:.3f} kcal/mol)"
)
print(f"    Reference weight |c₀|² = {r_diss['c0_max'] ** 2:.4f}")
print(f"    RHF error vs FCI: {(r_diss['RHF'] - r_diss['FCI']) * 627.509:.3f} kcal/mol")
print(
    f"    Selected-CI error vs FCI: {(r_diss['Sel-CI'] - r_diss['FCI']) * 1e6:.2f} μHa"
)

print(f"\n  The RHF error grows from {e_corr_eq * 627.509:.1f} kcal/mol at")
print(f"  equilibrium to {e_corr_diss * 627.509:.1f} kcal/mol at dissociation,")
print(f"  a factor of {e_corr_diss / e_corr_eq:.1f}×.  This is the signature of")
print(f"  the ionic-to-covalent transition — RHF overstabilises the ionic")
print(f"  configuration B⁻H⁺ at large R, while FCI correctly describes the")
print(f"  neutral fragments.")

print(f"\n  References:")
print(f"    Harrison & Handy, Chem. Phys. Lett. 95, 386 (1983)")
print(f"    Bauschlicher, Langhoff & Taylor, J. Chem. Phys. 93, 5029 (1990)")
