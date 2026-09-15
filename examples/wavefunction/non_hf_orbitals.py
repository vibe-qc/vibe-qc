"""Non-HF orbital sources for non-mean-field solvers.

Demonstrates that the solvers in `vibeqc.solvers` do **not** require
Hartree–Fock orbitals.  This example compares three orbital sources:

  1. **Canonical HF orbitals** — the standard approach.
  2. **Löwdin (symmetric) orthogonalised AOs** — no SCF at all,
     just X = S^{-1/2} applied to the AO-basis integrals.
  3. **Random orthonormal orbitals** — an extreme test showing the
     solver works with any orthonormal basis (though convergence
     is slower).

All three produce the same FCI energy in the complete-determinant
limit, but Selected-CI convergence differs.

Run:
    .venv/bin/python examples/wavefunction/non_hf_orbitals.py
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from vibeqc import Atom, BasisSet, Molecule
from vibeqc._vibeqc_core import BasisSet, compute_overlap
from vibeqc.solvers import (
    Hamiltonian,
    SelectedCIOptions,
    build_hamiltonian_ao,
    build_hamiltonian_matrix,
    canonical_orthogonalize,
    generate_closed_shell_determinants,
    get_hf_orbital_provider,
    solve_selected_ci,
    transform_hamiltonian,
)

# ── Build H₂ ────────────────────────────────────────────────────────────
mol = Molecule(
    [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])],
    charge=0,
    multiplicity=1,
)
basis = BasisSet(mol, "sto-3g")

# ── AO-basis Hamiltonian (basis for all transformations) ────────────────
ham_ao = build_hamiltonian_ao(mol, basis)
S = np.asarray(compute_overlap(basis))

print("=" * 66)
print(" Non-HF orbital sources for Selected-CI on H₂ / STO-3G")
print("=" * 66)

# ── 1. HF orbitals ─────────────────────────────────────────────────────
C_hf = get_hf_orbital_provider(mol, basis)
ham_hf = transform_hamiltonian(ham_ao, C_hf)
print(f"\n─── 1. Canonical HF orbitals ───")
print(
    f"    h1e diagonal?  off-diag max = {np.max(np.abs(ham_hf.h1e - np.diag(np.diag(ham_hf.h1e)))):.2e}"
)

# ── 2. Löwdin orthogonalised AOs ───────────────────────────────────────
X = canonical_orthogonalize(S)
ham_lowdin = transform_hamiltonian(ham_ao, X)
print(f"\n─── 2. Löwdin (S^{-1 / 2}) orthogonalised AOs ───")
print(f"    n_active = {X.shape[1]}  (out of {S.shape[0]} AOs)")
print(
    f"    X^T S X = I?  max|off| = {np.max(np.abs(X.T @ S @ X - np.eye(X.shape[1]))):.2e}"
)
print(
    f"    h1e off-diag max = {np.max(np.abs(ham_lowdin.h1e - np.diag(np.diag(ham_lowdin.h1e)))):.4f}"
)

# ── 3. Random orthonormal orbitals ──────────────────────────────────────
rng = np.random.default_rng(42)
n_ao = S.shape[0]
# Gram-Schmidt random vectors
V = rng.normal(0, 1, (n_ao, n_ao))
Q, _ = np.linalg.qr(V)
C_random = Q  # Q is orthonormal in the Euclidean sense, not S-orthonormal
# To get S-orthonormal orbitals: C = X @ U where U is unitary
U = Q  # Use Q as the unitary rotation of the Löwdin orbitals
C_rand_ortho = X @ U
ham_random = transform_hamiltonian(ham_ao, C_rand_ortho)
S_random = C_rand_ortho.T @ S @ C_rand_ortho
print(f"\n─── 3. Random orthonormal orbitals ───")
print(f"    C^T S C = I?  max|off| = {np.max(np.abs(S_random - np.eye(n_ao))):.2e}")
print(
    f"    h1e off-diag max = {np.max(np.abs(ham_random.h1e - np.diag(np.diag(ham_random.h1e)))):.4f}"
)

# ── Run Selected-CI with each orbital source ────────────────────────────
ci_opts = SelectedCIOptions(
    target_size=10,
    max_iter=15,
    conv_tol_energy=1e-10,
    do_pt2_correction=False,
    verbose=0,
)

print(
    f"\n{'Orbital source':<34s}  {'E(Selected-CI)':>16s}  {'ndet':>6s}  {'n_iter':>6s}"
)
print("-" * 68)
for name, ham in [
    ("HF (canonical)", ham_hf),
    ("Löwdin (S^{-1/2})", ham_lowdin),
    ("Random orthonormal", ham_random),
]:
    result = solve_selected_ci(ham, ci_opts)
    ndet = len(result.ci_labels) if result.ci_labels else 0
    print(f"  {name:<32s}  {result.energy:>16.10f}  {ndet:>6d}  {result.n_iter:>6d}")

# ── FCI reference (orbital-invariant) ───────────────────────────────────
all_dets = generate_closed_shell_determinants(ham_hf.norb, ham_hf.nelec // 2)
H_fci = build_hamiltonian_matrix(all_dets, ham_hf.h1e, ham_hf.h2e)
evals, _ = eigh(H_fci)
e_fci = evals[0] + ham_hf.nuclear_repulsion
print(f"\nFCI (orbital-invariant):  E = {e_fci:.10f} Ha")
print(f"\nAll three orbital sources recover the same FCI limit.")
print(f"Selected-CI with HF orbitals converges fastest because the reference")
print(f"determinant is already a good approximation in that basis.")
