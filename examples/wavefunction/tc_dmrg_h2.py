"""Transcorrelated + DMRG workflow: H₂ / STO-3G.

Demonstrates the combined transcorrelated-Hamiltonian + DMRG pipeline:
  1. Build the bare Hamiltonian in the MO basis.
  2. Apply the transcorrelated similarity transformation (Gaussian
     correlator with varying strength γ).
  3. Solve the transformed Hamiltonian with the DMRG solver.
  4. Compare energies and convergence against the untransformed DMRG
     and the exact FCI result.

The transcorrelated Hamiltonian has modified two-electron integrals
that partially absorb short-range correlation, potentially improving
basis-set convergence.  For H₂ / STO-3G the effect is small; larger
bases and multi-electron systems show the benefit.

Run:
    .venv/bin/python examples/wavefunction/tc_dmrg_h2.py
"""

from __future__ import annotations

import numpy as np
from vibeqc import Atom, BasisSet, Molecule
from vibeqc.solvers import (
    DMRGOptions,
    Hamiltonian,
    SelectedCIOptions,
    TranscorrelatedOptions,
    build_hamiltonian_mo,
    build_transcorrelated_hamiltonian,
    get_hf_orbital_provider,
    solve_dmrg,
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
ham_bare = build_hamiltonian_mo(mol, basis, C)

print("=" * 72)
print(" Transcorrelated + DMRG on H₂ / STO-3G")
print("=" * 72)
print(f"  norb={ham_bare.norb}, nelec={ham_bare.nelec}")

# ── Untransformed DMRG ──────────────────────────────────────────────────
dmrg_opts = DMRGOptions(
    bond_dim_schedule=[4, 8, 16],
    n_sweeps=6,
    conv_tol_energy=1e-6,
    verbose=1,
)
print(f"\n─── Bare Hamiltonian + DMRG ───")
result_bare = solve_dmrg(ham_bare, dmrg_opts)
print(f"  E(DMRG, bare) = {result_bare.energy:.10f} Ha")

# ── Transcorrelated + DMRG (varying gamma) ──────────────────────────────
print(
    f"\n{'gamma':>8s}  {'E(DMRG)':>16s}  {'Δ vs bare':>14s}  "
    f"{'norm(h2e)':>14s}  {'M':>6s}"
)
print("-" * 66)
for gamma in [0.01, 0.05, 0.1, 0.2, 0.5]:
    tc_opts = TranscorrelatedOptions(
        form="simple_gaussian",
        gamma=gamma,
        no2b=True,
        symmetrize=True,
        report_diagnostics=False,
    )
    ham_tc = build_transcorrelated_hamiltonian(ham_bare, tc_opts)

    result_tc = solve_dmrg(ham_tc, dmrg_opts)
    delta = result_tc.energy - result_bare.energy
    norm_h2e = np.linalg.norm(ham_tc.h2e)
    print(
        f"{gamma:>8.3f}  {result_tc.energy:>16.10f}  {delta:>+14.2e}  "
        f"{norm_h2e:>14.6f}  {result_tc.bond_dim:>6d}"
    )

# ── Reference: FCI via Selected-CI ──────────────────────────────────────
ci_opts = SelectedCIOptions(
    target_size=10,
    max_iter=10,
    conv_tol_energy=1e-10,
    do_pt2_correction=True,
    verbose=0,
)
result_ci_bare = solve_selected_ci(ham_bare, ci_opts)
print(f"\n─── Reference (Selected-CI on bare H) ───")
print(f"  E(FCI, approx) = {result_ci_bare.energy:.10f} Ha")

print(f"\nNote: For 2 electrons in 2 orbitals the transcorrelated")
print(f"transformation primarily rescales the integrals.  The benefit")
print(f"becomes clearer with larger basis sets where short-range")
print(f"correlation is slow to converge with orbital expansions.")
