"""Transcorrelated + Selected-CI example: H2 / STO-3G.

Demonstrates the transcorrelated workflow:
  1. Build bare Hamiltonian in MO basis (via HF orbitals).
  2. Apply transcorrelated similarity transformation.
  3. Solve the transformed Hamiltonian with Selected-CI.
  4. Compare with the untransformed result.

Run:
    .venv/bin/python examples/wavefunction/transcorrelated_h2.py
"""

from __future__ import annotations

import numpy as np
from vibeqc import Atom, BasisSet, Molecule
from vibeqc.solvers import (
    Hamiltonian,
    SelectedCIOptions,
    TranscorrelatedOptions,
    build_hamiltonian_mo,
    build_transcorrelated_hamiltonian,
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
ham_bare = build_hamiltonian_mo(mol, basis, C)

print("=" * 60)
print(" Transcorrelated Hamiltonian + Selected-CI on H2 / STO-3G")
print("=" * 60)

# ── Bare Hamiltonian ────────────────────────────────────────────────────
ci_opts = SelectedCIOptions(
    target_size=10,
    max_iter=10,
    conv_tol_energy=1e-10,
    do_pt2_correction=True,
    verbose=0,
)
result_bare = solve_selected_ci(ham_bare, ci_opts)
print(f"\nBare Hamiltonian:")
print(f"  E(Selected-CI) = {result_bare.energy:.10f} Ha")

# ── Transcorrelated with varying gamma ──────────────────────────────────
print(
    f"\n{'gamma':>8s}  {'norm_h1e':>14s}  {'norm_h2e':>14s}  {'anti-H':>12s}  {'E(CI)':>16s}"
)
print("-" * 72)
for gamma in [0.01, 0.05, 0.1, 0.2, 0.5]:
    tc_opts = TranscorrelatedOptions(
        form="simple_gaussian",
        gamma=gamma,
        no2b=True,
        symmetrize=True,
        report_diagnostics=True,
    )
    ham_tc = build_transcorrelated_hamiltonian(ham_bare, tc_opts)

    # Extract diagnostics
    # (in full version, these would be on the Hamiltonian object)
    norm_h1e = np.linalg.norm(ham_tc.h1e)
    norm_h2e = np.linalg.norm(ham_tc.h2e)

    # Anti-Hermitian part of h1e
    anti = 0.5 * (ham_tc.h1e - ham_tc.h1e.T)
    anti_norm = np.linalg.norm(anti) if not tc_opts.symmetrize else 0.0

    result_tc = solve_selected_ci(ham_tc, ci_opts)
    print(
        f"{gamma:>8.3f}  {norm_h1e:>14.6f}  {norm_h2e:>14.6f}  "
        f"{anti_norm:>12.2e}  {result_tc.energy:>16.10f}"
    )

print(f"\nBare reference: {result_bare.energy:.10f} Ha")
