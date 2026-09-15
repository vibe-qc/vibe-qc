"""v2RDM example: H2 / STO-3G — variational 2-RDM minimisation.

Demonstrates the v2RDM solver with PQG constraints and shows
convergence of the trace constraint and PSD residual.

Run:
    .venv/bin/python examples/wavefunction/v2rdm_h2.py
"""

from __future__ import annotations

import numpy as np
from vibeqc import Atom, BasisSet, Molecule
from vibeqc.solvers import (
    Hamiltonian,
    V2RDMOptions,
    build_hamiltonian_mo,
    get_hf_orbital_provider,
    solve_v2rdm,
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

print("=" * 60)
print(" v2RDM (PQG) on H2 / STO-3G")
print("=" * 60)
print(f"  norb={ham.norb}, nelec={ham.nelec}")

# ── v2RDM solve ─────────────────────────────────────────────────────────
opts = V2RDMOptions(
    outer_max_iter=500,
    mu=10.0,
    mu_factor=1.2,
    conv_tol_primal=1e-4,
    verbose=1,
)
result = solve_v2rdm(ham, opts)

print(f"\nFinal result:")
print(f"  E(v2RDM)       = {result.energy:.10f} Ha")
print(f"  Tr(¹D)          = {np.trace(result.rdm1):.8f}")
print(f"  trace residual  = {result.constraint_residual:.2e}")
print(f"  converged       = {result.converged}")
print(f"  n_iter          = {result.n_iter}")

# ── Compare with RHF ────────────────────────────────────────────────────
from vibeqc._vibeqc_core import RHFOptions, run_rhf

rhf_opts = RHFOptions()
rhf_opts.conv_tol_energy = 1e-12
rhf_opts.conv_tol_grad = 1e-10
rhf_result = run_rhf(mol, basis, rhf_opts)
print(f"\nComparison:")
print(f"  E(RHF)          = {rhf_result.energy:.10f} Ha")
print(f"  E(v2RDM) - E(RHF) = {result.energy - rhf_result.energy:+.6e} Ha")
