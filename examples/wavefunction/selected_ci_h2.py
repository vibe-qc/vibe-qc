"""Selected-CI example: H2 / STO-3G convergence study.

Compares:
  * RHF energy (target_size=1, no PT2)
  * Selected-CI with increasing determinant count
  * Full CI in the minimal basis (exact)

Run:
    .venv/bin/python examples/wavefunction/selected_ci_h2.py
"""

from __future__ import annotations

import time

import numpy as np
from vibeqc import Atom, BasisSet, Molecule
from vibeqc.solvers import (
    Hamiltonian,
    SelectedCIOptions,
    build_hamiltonian_mo,
    get_hf_orbital_provider,
    solve_selected_ci,
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
print(f"System: H2 / STO-3G, R=1.4 bohr")
print(f"  norb={ham.norb}, nelec={ham.nelec}, E_nuc={ham.nuclear_repulsion:.8f} Ha")

# ── RHF (1 determinant) ─────────────────────────────────────────────────
opts_hf = SelectedCIOptions(
    target_size=1, max_iter=1, do_pt2_correction=False, verbose=0
)
result_hf = solve_selected_ci(ham, opts_hf)
print(f"\nRHF (1 det):     E = {result_hf.energy:.10f} Ha")

# ── Selected-CI convergence ─────────────────────────────────────────────
print(f"\n{'target':>8s}  {'ndet':>6s}  {'E_var':>16s}  {'ΔE':>10s}  {'pt2':>12s}")
print("-" * 60)
prev_e = float("inf")
for target in [2, 5, 10, 20, 50]:
    t0 = time.perf_counter()
    opts = SelectedCIOptions(
        target_size=target,
        max_iter=20,
        conv_tol_energy=1e-10,
        do_pt2_correction=True,
        verbose=0,
    )
    result = solve_selected_ci(ham, opts)
    dt = time.perf_counter() - t0
    delta = result.energy - prev_e if prev_e != float("inf") else 0.0
    prev_e = result.energy
    ndet = len(result.ci_labels) if result.ci_labels else 0
    pt2 = result.pt2_correction or 0.0
    print(
        f"{target:>8d}  {ndet:>6d}  {result.energy - pt2:>16.10f}"
        f"  {delta:>10.2e}  {pt2:>12.2e}"
    )

# ── Full CI (all determinants in 4-orbital space) ────────────────────────
from scipy.linalg import eigh
from vibeqc.solvers import build_hamiltonian_matrix, generate_closed_shell_determinants

all_dets = generate_closed_shell_determinants(ham.norb, ham.nelec // 2)
H_fci = build_hamiltonian_matrix(all_dets, ham.h1e, ham.h2e)
evals, _ = eigh(H_fci)
e_fci = evals[0] + ham.nuclear_repulsion
print(f"\nFull CI ({len(all_dets)} dets): E = {e_fci:.10f} Ha")
print(f"Selected-CI error vs FCI: {result.energy - e_fci:.2e} Ha")
