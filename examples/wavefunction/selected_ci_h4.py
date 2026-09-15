"""Selected-CI example: H4 square / STO-3G — multi-reference system.

Demonstrates Selected-CI convergence for a strongly correlated
stretched H4 square at R = 2.0 bohr.

Run:
    .venv/bin/python examples/wavefunction/selected_ci_h4.py
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from vibeqc import Atom, BasisSet, Molecule
from vibeqc.solvers import (
    Hamiltonian,
    SelectedCIOptions,
    build_hamiltonian_matrix,
    build_hamiltonian_mo,
    generate_closed_shell_determinants,
    get_hf_orbital_provider,
    solve_selected_ci,
)

# ── Build H4 square ─────────────────────────────────────────────────────
r = 2.0  # bohr — stretched to induce correlation
mol = Molecule(
    [
        Atom(1, [0.0, 0.0, 0.0]),
        Atom(1, [r, 0.0, 0.0]),
        Atom(1, [0.0, r, 0.0]),
        Atom(1, [r, r, 0.0]),
    ],
    charge=0,
    multiplicity=1,
)
basis = BasisSet(mol, "sto-3g")
C = get_hf_orbital_provider(mol, basis)
ham = build_hamiltonian_mo(mol, basis, C)
print(f"System: H4 square / STO-3G, R={r:.1f} bohr")
print(f"  norb={ham.norb}, nelec={ham.nelec}, E_nuc={ham.nuclear_repulsion:.8f} Ha")

# ── Selected-CI ─────────────────────────────────────────────────────────
opts = SelectedCIOptions(
    target_size=50,
    max_iter=15,
    conv_tol_energy=1e-8,
    do_pt2_correction=True,
    verbose=1,
)
result = solve_selected_ci(ham, opts)

print(f"\nSelected-CI result:")
print(f"  E_var = {result.energy - (result.pt2_correction or 0.0):.10f} Ha")
print(f"  E_pt2 = {result.pt2_correction or 0.0:.10f} Ha")
print(f"  E_tot = {result.energy:.10f} Ha")
print(f"  ndet  = {len(result.ci_labels)}")
print(f"  converged = {result.converged}")

# ── Leading determinant coefficients ────────────────────────────────────
coeffs = np.abs(result.ci_coeffs)
idx = np.argsort(-coeffs)
print(f"\nLeading determinants:")
for rank in range(min(5, len(idx))):
    i = idx[rank]
    label = result.ci_labels[i]
    print(f"  |c_{rank}| = {coeffs[i]:.6f}  occ = {label}")

# ── Full CI in the same 4-orbital space ──────────────────────────────────
all_dets = generate_closed_shell_determinants(ham.norb, ham.nelec // 2)
H_fci = build_hamiltonian_matrix(all_dets, ham.h1e, ham.h2e)
evals, _ = eigh(H_fci)
e_fci = evals[0] + ham.nuclear_repulsion
print(f"\nFull CI ({len(all_dets)} dets): E = {e_fci:.10f} Ha")
print(f"Selected-CI error vs FCI: {result.energy - e_fci:.2e} Ha")
