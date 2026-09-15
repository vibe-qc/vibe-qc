"""Full Configuration Interaction example: H₂ / STO-3G.

Diagonalises the Hamiltonian in the complete determinant space — the
exact solution within the finite one-electron basis.  H₂ / STO-3G
has only 2 spatial orbitals and 2 electrons, so the FCI space is
just 2 closed-shell determinants (the σ_g² ground state and the
σ_u² doubly-excited state).

This is the gold-standard reference against which approximate
solvers (Selected-CI, DMRG, v2RDM) are validated.

Run:
    .venv/bin/python examples/wavefunction/fci_h2.py
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from vibeqc import Atom, BasisSet, Molecule
from vibeqc.solvers import (
    Hamiltonian,
    build_hamiltonian_matrix,
    build_hamiltonian_mo,
    generate_closed_shell_determinants,
    get_hf_orbital_provider,
)

# ── Build system ────────────────────────────────────────────────────────
mol = Molecule(
    [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])],
    charge=0,
    multiplicity=1,
)
basis = BasisSet(mol, "sto-3g")
C = get_hf_orbital_provider(mol, basis)
ham = build_hamiltonian_mo(mol, basis, C)

norb = ham.norb
nelec = ham.nelec
nocc = nelec // 2

print("=" * 60)
print(" Full CI on H₂ / STO-3G")
print("=" * 60)
print(f"  norb={norb}, nelec={nelec}, nocc={nocc}")
print(f"  E_nuc = {ham.nuclear_repulsion:.10f} Ha")

# ── Generate determinant space ──────────────────────────────────────────
all_dets = generate_closed_shell_determinants(norb, nocc)
print(f"\nDeterminant space: {len(all_dets)} closed-shell determinants")
for i, det in enumerate(all_dets):
    # Binary occupation string
    occ_str = "".join("1" if j in det else "0" for j in range(norb))
    print(f"  |{occ_str}⟩  =  {det}")

# ── Build and diagonalise H ─────────────────────────────────────────────
H = build_hamiltonian_matrix(all_dets, ham.h1e, ham.h2e)
print(f"\nHamiltonian matrix ({H.shape[0]}×{H.shape[1]}):")
for i in range(H.shape[0]):
    row = "  ".join(f"{H[i, j]:12.8f}" for j in range(H.shape[1]))
    print(f"  [{row}]")

evals, evecs = eigh(H)
print(f"\nEigenvalues (electronic only):")
for i, ev in enumerate(evals):
    print(f"  E_{i} = {ev:14.10f} Ha")

# ── Ground state ────────────────────────────────────────────────────────
e_fci = evals[0] + ham.nuclear_repulsion
c0 = evecs[:, 0]
print(f"\nGround state:")
print(f"  E(FCI)  = {e_fci:.10f} Ha")
print(f"  Coefficients:")
for i, (det, c) in enumerate(zip(all_dets, c0)):
    print(f"    |{' '.join(str(d) for d in det)}⟩  c = {c:12.10f}  |c|² = {c**2:.10f}")

# ── Compare with RHF ────────────────────────────────────────────────────
from vibeqc._vibeqc_core import RHFOptions, run_rhf

rhf_opts = RHFOptions()
rhf_opts.conv_tol_energy = 1e-12
rhf_opts.conv_tol_grad = 1e-10
rhf_result = run_rhf(mol, basis, rhf_opts)

print(f"\nComparison:")
print(f"  E(RHF)  = {rhf_result.energy:.10f} Ha")
print(f"  E(FCI)  = {e_fci:.10f} Ha")
print(
    f"  Δ(FCI−RHF) = {e_fci - rhf_result.energy:+.10f} Ha  "
    f"({(e_fci - rhf_result.energy) * 627.509:.4f} kcal/mol)"
)
print(
    f"\nCorrelation energy captured: {(e_fci - rhf_result.energy) * 627.509:.4f} kcal/mol"
)
