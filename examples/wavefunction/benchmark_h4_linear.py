"""Linear H₄ chain — the minimal model of strong correlation.

Four equally-spaced hydrogen atoms in a line is the archetypal
strong-correlation benchmark.  At compressed geometries (R ≈ 1.4 bohr),
the system is weakly correlated and RHF is a good starting point.  At
stretched geometries (R ≥ 2.5 bohr), the HOMO and LUMO become
near-degenerate and the exact wavefunction is an equal superposition
of many determinants — RHF fails qualitatively.

This example scans the nearest-neighbour distance R and compares:
  * RHF energy
  * Selected-CI energy (systematically improvable)
  * Exact FCI energy (only 4 electrons in 4 orbitals = 6 determinants)

Key references:
  * Paldus & Čížek, Phys. Rev. A 2, 2268 (1970)
  * Jankowski & Paldus, Int. J. Quantum Chem. 18, 1243 (1980)
  * Chan & Head-Gordon, J. Chem. Phys. 116, 4462 (2002)

Run:
    .venv/bin/python examples/wavefunction/benchmark_h4_linear.py
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


def compute_h4_linear(r_nn: float) -> dict:
    """Compute energies for linear H₄ with nearest-neighbour distance R.

    H atoms at z = −1.5R, −0.5R, +0.5R, +1.5R.
    """
    mol = Molecule(
        [
            Atom(1, [0.0, 0.0, -1.5 * r_nn]),
            Atom(1, [0.0, 0.0, -0.5 * r_nn]),
            Atom(1, [0.0, 0.0, +0.5 * r_nn]),
            Atom(1, [0.0, 0.0, +1.5 * r_nn]),
        ],
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
        target_size=20,
        max_iter=15,
        conv_tol_energy=1e-10,
        do_pt2_correction=False,
        verbose=0,
    )
    ci_result = solve_selected_ci(ham, ci_opts)

    # FCI
    all_dets = generate_closed_shell_determinants(ham.norb, ham.nelec // 2)
    H_fci = build_hamiltonian_matrix(all_dets, ham.h1e, ham.h2e)
    evals, evecs = eigh(H_fci)
    e_fci = evals[0] + ham.nuclear_repulsion

    # Leading coefficients
    c0 = np.abs(evecs[:, 0])

    return {
        "R": r_nn,
        "RHF": rhf_result.energy,
        "Sel-CI": ci_result.energy,
        "FCI": e_fci,
        "ndet_ci": len(ci_result.ci_labels) if ci_result.ci_labels else 0,
        "c0_max": float(np.max(c0)),
        "n_sig": int(np.sum(c0 > 0.1)),
    }


# ═══════════════════════════════════════════════════════════════════════════
print("=" * 78)
print("  Linear H₄ chain — Strong-correlation benchmark")
print("=" * 78)
print(f"  Geometry: H₁—H₂—H₃—H₄  equally spaced on z-axis")
print(f"  Basis: STO-3G  (4 electrons in 4 spatial orbitals)")
print()
print(
    f"  {'R/bohr':>7s}  {'E(RHF)':>14s}  {'E(Sel-CI)':>14s}  "
    f"{'E(FCI)':>14s}  {'Δ(RHF−FCI)':>14s}  {'max|c|':>8s}  {'#|c|>0.1':>9s}"
)
print("  " + "-" * 76)

results = []
for r in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.3, 2.6, 3.0, 3.5, 4.0, 5.0]:
    res = compute_h4_linear(r)
    results.append(res)
    delta = res["RHF"] - res["FCI"]
    print(
        f"  {res['R']:>7.2f}  {res['RHF']:>14.8f}  {res['Sel-CI']:>14.8f}  "
        f"{res['FCI']:>14.8f}  {delta:>14.8f}  {res['c0_max']:>8.4f}  "
        f"{res['n_sig']:>9d}"
    )

# ── Summary ─────────────────────────────────────────────────────────────
print(f"\n  ═══ Analysis ═══")

# Near-equilibrium
r_eq = results[2]  # R = 1.4
print(f"\n  At R = 1.4 bohr (near-equilibrium):")
print(f"    Leading determinant weight: {r_eq['c0_max']:.4f}")
print(f"    RHF error vs FCI: {(r_eq['RHF'] - r_eq['FCI']) * 627.509:.3f} kcal/mol")
print(f"    Selected-CI error: {(r_eq['Sel-CI'] - r_eq['FCI']) * 627.509:.6f} kcal/mol")

# Dissociation limit
r_diss = results[-1]  # R = 5.0
print(f"\n  At R = 5.0 bohr (dissociation limit):")
print(f"    Leading determinant weight: {r_diss['c0_max']:.4f}")
print(f"    Number of significant determinants (|c| > 0.1): {r_diss['n_sig']}")
print(f"    RHF error vs FCI: {(r_diss['RHF'] - r_diss['FCI']) * 627.509:.3f} kcal/mol")
print(
    f"    Selected-CI error: {(r_diss['Sel-CI'] - r_diss['FCI']) * 627.509:.6f} kcal/mol"
)

# Correlation growth
r_near = results[0]  # R = 1.0
r_far = results[-1]  # R = 5.0
e_corr_near = r_near["FCI"] - r_near["RHF"]
e_corr_far = r_far["FCI"] - r_far["RHF"]
print(f"\n  Correlation energy growth:")
print(
    f"    R = 1.0:  E_corr = {e_corr_near * 1000:.2f} mHa  "
    f"({e_corr_near * 627.509:.3f} kcal/mol)"
)
print(
    f"    R = 5.0:  E_corr = {e_corr_far * 1000:.2f} mHa  "
    f"({e_corr_far * 627.509:.3f} kcal/mol)"
)
print(f"    Growth factor: {e_corr_far / e_corr_near:.1f}×")
print(f"\n  This growth in correlation energy is the hallmark of strong")
print(f"  (static) correlation — RHF fails because a single Slater")
print(f"  determinant cannot describe the near-degenerate configurations")
print(f"  that become important as the chain is stretched.")

print(f"\n  References:")
print(f"    Paldus & Čížek, Phys. Rev. A 2, 2268 (1970)")
print(f"    Jankowski & Paldus, Int. J. Quantum Chem. 18, 1243 (1980)")
print(f"    Chan & Head-Gordon, J. Chem. Phys. 116, 4462 (2002)")
