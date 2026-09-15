"""DMRG example: H2 / STO-3G — MPS-based solver.

Demonstrates the two-site DMRG solver with bond-dimension scheduling.

NOTE: The Python-level DMRG implementation is a teaching tool for
small active spaces. For production DMRG, use block2 or a C++ backend.

Run:
    .venv/bin/python examples/wavefunction/dmrg_h2.py
"""

from __future__ import annotations

import numpy as np
from vibeqc import Atom, BasisSet, Molecule
from vibeqc.solvers import (
    DMRGOptions,
    Hamiltonian,
    build_hamiltonian_mo,
    get_hf_orbital_provider,
    solve_dmrg,
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
print(" DMRG on H2 / STO-3G  (M = 4 → 32)")
print("=" * 60)
print(f"  norb={ham.norb}, nelec={ham.nelec}")

# ── DMRG solve ──────────────────────────────────────────────────────────
opts = DMRGOptions(
    bond_dim_schedule=[4, 8, 16, 32],
    n_sweeps=8,
    conv_tol_energy=1e-6,
    verbose=1,
)
result = solve_dmrg(ham, opts)

print(f"\nFinal result:")
print(f"  E(DMRG)    = {result.energy:.10f} Ha")
print(f"  bond_dim   = {result.bond_dim}")
print(f"  trunc_err  = {result.truncation_error}")
print(f"  converged  = {result.converged}")

# ── Compare with RHF ────────────────────────────────────────────────────
from vibeqc._vibeqc_core import RHFOptions, run_rhf

rhf_opts = RHFOptions()
rhf_opts.conv_tol_energy = 1e-12
rhf_opts.conv_tol_grad = 1e-10
rhf_result = run_rhf(mol, basis, rhf_opts)
print(f"\nComparison:")
print(f"  E(RHF)     = {rhf_result.energy:.10f} Ha")
