"""Timing benchmark: Selected-CI vs FCI for H2O/STO-3G active spaces.

Compares wall-clock performance of Selected-CI (CIPSI-style iterative
determinant selection) against brute-force FCI (construct full matrix +
exact diagonalisation) across different active-space sizes.

Active spaces are obtained by truncating the MO basis to the lowest
*n_act* spatial orbitals while keeping all 10 electrons — the FCI
determinant count grows combinatorially, so this gives a meaningful
range of problem sizes.

Run:
    .venv/bin/python examples/wavefunction/benchmark_timing.py
"""

from __future__ import annotations

import time

import numpy as np
from scipy.linalg import eigh
from vibeqc import Atom, BasisSet, Molecule
from vibeqc.solvers import (
    SelectedCIOptions,
    build_hamiltonian_matrix_unrestricted,
    build_hamiltonian_mo,
    generate_determinants,
    get_hf_orbital_provider,
    solve_selected_ci,
)

# ═══════════════════════════════════════════════════════════════════════════
#  Geometry — Pitzer/Pulay H₂O equilibrium
# ═══════════════════════════════════════════════════════════════════════════
_R_OH = 1.8089  # bohr
_HALF_ANGLE = np.radians(104.52 / 2)
_YH = _R_OH * np.sin(_HALF_ANGLE)
_ZH = -_R_OH * np.cos(_HALF_ANGLE)

mol = Molecule(
    [
        Atom(8, [0.0, 0.0, 0.0]),
        Atom(1, [0.0, +_YH, _ZH]),
        Atom(1, [0.0, -_YH, _ZH]),
    ],
    charge=0,
    multiplicity=1,
)

# ═══════════════════════════════════════════════════════════════════════════
#  Build full MO Hamiltonian
# ═══════════════════════════════════════════════════════════════════════════
basis = BasisSet(mol, "sto-3g")
C_full = get_hf_orbital_provider(mol, basis)
ham_full = build_hamiltonian_mo(mol, basis, C_full)
norb_total = ham_full.norb
nelec = ham_full.nelec
nalpha = (nelec + ham_full.ms2) // 2
nbeta = (nelec - ham_full.ms2) // 2
print(f"H₂O / STO-3G   {norb_total} MOs, {nelec} electrons")
print()

# ═══════════════════════════════════════════════════════════════════════════
#  Vary active space — keep lowest n_act MOs
# ═══════════════════════════════════════════════════════════════════════════
header = (
    f"{'Active':>7s}  {'n_det(FCI)':>10s}  "
    f"{'t(SCI-10)':>10s}  {'t(SCI-50)':>10s}  {'t(SCI-200)':>11s}  "
    f"{'t(FCI-build)':>12s}  {'t(FCI-diag)':>12s}  {'t(total)':>9s}"
)
print(header)
print("-" * len(header))

for n_act in [5, 6, 7]:
    # Truncate MOs to active space
    C_act = np.asarray(C_full, order="C")[:, :n_act]
    ham = build_hamiltonian_mo(mol, basis, C_act)

    norb = ham.norb
    nelec_ham = ham.nelec
    nalpha_ham = (nelec_ham + ham.ms2) // 2
    nbeta_ham = (nelec_ham - ham.ms2) // 2

    # ── FCI: build matrix + diagonalise ──────────────────────────────
    t0 = time.perf_counter()
    all_dets = generate_determinants(norb, nalpha_ham, nbeta_ham)
    H_fci = build_hamiltonian_matrix_unrestricted(all_dets, ham.h1e, ham.h2e)
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    evals, _ = eigh(H_fci)
    t_diag = time.perf_counter() - t0
    e_fci = evals[0] + ham.nuclear_repulsion
    n_det = len(all_dets)

    # ── Selected-CI at different target sizes ────────────────────────
    sci_times = {}
    for tsize in [10, 50, 200]:
        t0 = time.perf_counter()
        opts = SelectedCIOptions(
            target_size=tsize,
            max_iter=30,
            conv_tol_energy=1e-10,
            do_pt2_correction=False,
            verbose=0,
        )
        solve_selected_ci(ham, opts)
        sci_times[tsize] = time.perf_counter() - t0

    t_total = sci_times[200] + t_build + t_diag

    print(
        f"{n_act:>7d}  {n_det:>10d}  "
        f"{sci_times[10] * 1000:>9.2f}ms  "
        f"{sci_times[50] * 1000:>9.2f}ms  "
        f"{sci_times[200] * 1000:>10.2f}ms  "
        f"{t_build * 1000:>11.2f}ms  "
        f"{t_diag * 1000:>11.2f}ms  "
        f"{t_total * 1000:>8.2f}ms"
    )

print()
print(f"FCI energies (reference):")
for n_act in [5, 6, 7]:
    C_act = np.asarray(C_full, order="C")[:, :n_act]
    ham = build_hamiltonian_mo(mol, basis, C_act)
    norb = ham.norb
    nelec_ham = ham.nelec
    nalpha_ham = (nelec_ham + ham.ms2) // 2
    nbeta_ham = (nelec_ham - ham.ms2) // 2
    all_dets = generate_determinants(norb, nalpha_ham, nbeta_ham)
    H_fci = build_hamiltonian_matrix_unrestricted(all_dets, ham.h1e, ham.h2e)
    evals, _ = eigh(H_fci)
    e_fci = evals[0] + ham.nuclear_repulsion
    print(
        f"  {n_act} active orbitals ({len(all_dets)} dets):  E(FCI) = {e_fci:.10f} Ha"
    )
