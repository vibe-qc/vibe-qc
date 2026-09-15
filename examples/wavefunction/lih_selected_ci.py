"""Selected-CI example: LiH / STO-3G — heteronuclear diatomic.

LiH is a 4-electron, 6-orbital system in STO-3G.  The FCI space is
C(6,2) = 15 closed-shell determinants.  This example demonstrates:
  * Selected-CI convergence toward the FCI limit.
  * Leading determinant coefficients and the multi-reference character.
  * Comparison with RHF.

Run:
    .venv/bin/python examples/wavefunction/lih_selected_ci.py
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

# ── Build LiH ───────────────────────────────────────────────────────────
# LiH equilibrium bond length ≈ 3.015 bohr (1.595 Å)
R_LiH = 3.015
mol = Molecule(
    [Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, R_LiH])],
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
print(" Selected-CI on LiH / STO-3G")
print("=" * 60)
print(f"  R(Li−H) = {R_LiH:.3f} bohr")
print(f"  norb = {norb}, nelec = {nelec}, nocc = {nocc}")
print(f"  E_nuc = {ham.nuclear_repulsion:.10f} Ha")

# ── RHF ─────────────────────────────────────────────────────────────────
rhf_opts = RHFOptions()
rhf_opts.conv_tol_energy = 1e-12
rhf_opts.conv_tol_grad = 1e-10
rhf_result = run_rhf(mol, basis, rhf_opts)
print(f"\nRHF energy: {rhf_result.energy:.10f} Ha")

# ── Selected-CI convergence ─────────────────────────────────────────────
print(
    f"\n{'target':>8s}  {'ndet':>6s}  {'E_var':>16s}  "
    f"{'ΔE_var':>12s}  {'E_pt2':>12s}  {'E_tot':>16s}"
)
print("-" * 78)
prev_e = float("inf")
for target in [1, 3, 5, 10, 20, 50]:
    opts = SelectedCIOptions(
        target_size=target,
        max_iter=30,
        conv_tol_energy=1e-10,
        do_pt2_correction=True,
        pt2_threshold=1e-8,
        verbose=0,
    )
    result = solve_selected_ci(ham, opts)
    ndet = len(result.ci_labels) if result.ci_labels else 0
    e_var = result.energy - (result.pt2_correction or 0.0)
    delta = e_var - prev_e if prev_e != float("inf") else 0.0
    prev_e = e_var
    pt2 = result.pt2_correction or 0.0
    print(
        f"{target:>8d}  {ndet:>6d}  {e_var:>16.10f}  "
        f"{delta:>12.2e}  {pt2:>12.2e}  {result.energy:>16.10f}"
    )

# ── Leading determinants ────────────────────────────────────────────────
coeffs = np.abs(result.ci_coeffs)
idx = np.argsort(-coeffs)
print(f"\nLeading determinants in the CI expansion:")
for rank in range(min(6, len(idx))):
    i = idx[rank]
    label = result.ci_labels[i]
    occ_str = "".join("1" if j in label else "0" for j in range(norb))
    print(f"  |c_{rank}| = {coeffs[i]:.6f}  |{occ_str}⟩  {label}")

# ── Full CI ─────────────────────────────────────────────────────────────
all_dets = generate_closed_shell_determinants(norb, nocc)
H_fci = build_hamiltonian_matrix(all_dets, ham.h1e, ham.h2e)
evals, evecs = eigh(H_fci)
e_fci = evals[0] + ham.nuclear_repulsion
print(f"\nFull CI ({len(all_dets)} determinants):")
print(f"  E(FCI)  = {e_fci:.10f} Ha")
print(f"  Δ(CI−FCI) = {result.energy - e_fci:+.2e} Ha")

# ── Correlation energy ──────────────────────────────────────────────────
e_corr_fci = e_fci - rhf_result.energy
e_corr_ci = result.energy - rhf_result.energy
print(f"\nCorrelation energy:")
print(
    f"  E_corr(FCI)       = {e_corr_fci:+.10f} Ha  "
    f"({e_corr_fci * 627.509:.4f} kcal/mol)"
)
print(
    f"  E_corr(Sel-CI)    = {e_corr_ci:+.10f} Ha  ({e_corr_ci * 627.509:.4f} kcal/mol)"
)
print(f"  % captured        = {100 * e_corr_ci / e_corr_fci:.2f}%")
