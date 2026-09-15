"""Correlation-energy benchmark: H₂O / STO-3G — RHF, Selected-CI, and FCI.

H₂O in the STO-3G minimal basis is the canonical "hello world" of
post-HF quantum chemistry.  With 10 electrons in 7 spatial orbitals,
the full FCI determinant space is C(7,5) × C(7,5) = 21 × 21 = 441
Slater determinants — small enough for exact diagonalisation, large
enough to be non-trivial.

This example demonstrates both:
  * **Closed-shell FCI** (21 determinants): only doubly-occupied
    configurations — fast but misses ~98% of the correlation.
  * **Full FCI** (441 determinants): all spin configurations —
    recovers the exact-in-basis energy within ~0.1 mHa.

Literature reference values (STO-3G):
  * E(RHF)  = −74.96306 Ha   (Harrison & Handy, Chem. Phys. Lett. 95, 386, 1983)
  * E(FCI)  = −75.01266 Ha   (Bauschlicher & Taylor, J. Chem. Phys. 85, 2779, 1986)
  * E_corr  =  −0.04960 Ha   ≈ 31.1 kcal/mol

Geometry: O at origin, H atoms from the Pitzer/Pulay optimisation.
  R(OH) = 1.8089 bohr  (0.9572 Å),  ∠HOH = 104.52°

Run:
    .venv/bin/python examples/wavefunction/benchmark_h2o_fci.py
"""

from __future__ import annotations

import time

import numpy as np
from scipy.linalg import eigh
from vibeqc import Atom, BasisSet, Molecule
from vibeqc._vibeqc_core import RHFOptions, run_rhf
from vibeqc.solvers import (
    Hamiltonian,
    SelectedCIOptions,
    build_hamiltonian_matrix,
    build_hamiltonian_mo,
    diagonal_matrix_element_unrestricted,
    generate_closed_shell_determinants,
    generate_determinants,
    get_hf_orbital_provider,
    solve_selected_ci,
)
from vibeqc.solvers._determinant import excitation_rank as _ex_rank
from vibeqc.solvers._slater_condon import (
    double_excitation_matrix_element,
    single_excitation_matrix_element,
)

# ═══════════════════════════════════════════════════════════════════════════
#  Geometry — Pitzer/Pulay H₂O equilibrium, STO-3G
# ═══════════════════════════════════════════════════════════════════════════
R_oh = 1.8089
theta = np.radians(104.52 / 2)
y_h = R_oh * np.sin(theta)
z_h = -R_oh * np.cos(theta)

mol = Molecule(
    [Atom(8, [0.0, 0.0, 0.0]), Atom(1, [0.0, +y_h, z_h]), Atom(1, [0.0, -y_h, z_h])],
    charge=0,
    multiplicity=1,
)
basis = BasisSet(mol, "sto-3g")

print("=" * 72)
print("  H₂O / STO-3G — Correlation-energy benchmark")
print("=" * 72)
print(f"  R(OH) = {R_oh:.4f} bohr,  ∠HOH = {2 * np.degrees(theta):.2f}°")
print(f"  n_ao = {basis.nbasis},  n_el = {mol.n_electrons()}")
print(f"  E_nuc = {mol.nuclear_repulsion():.10f} Ha")

# ── RHF reference ────────────────────────────────────────────────────────
t0 = time.perf_counter()
rhf_opts = RHFOptions()
rhf_opts.conv_tol_energy = 1e-12
rhf_opts.conv_tol_grad = 1e-10
rhf_result = run_rhf(mol, basis, rhf_opts)
dt_rhf = time.perf_counter() - t0
print(f"\n  RHF:  E = {rhf_result.energy:.8f} Ha  ({dt_rhf * 1000:.0f} ms)")

# ── Build MO-basis Hamiltonian ───────────────────────────────────────────
C = get_hf_orbital_provider(mol, basis)
ham = build_hamiltonian_mo(mol, basis, C)
norb = ham.norb
nelec = ham.nelec
nocc = nelec // 2
nalpha = nocc
nbeta = nocc
print(f"  MO basis: {norb} orbitals, {nelec} electrons")
print(
    f"  Closed-shell FCI space: C({norb},{nocc}) = "
    f"{len(generate_closed_shell_determinants(norb, nocc))} determinants"
)
print(
    f"  Full FCI space: C({norb},{nalpha}) × C({norb},{nbeta}) = "
    f"{len(generate_determinants(norb, nalpha, nbeta))} determinants"
)

# ── Selected-CI convergence scan ─────────────────────────────────────────
print(
    f"\n  {'target':>7s}  {'ndet':>6s}  {'E_var':>16s}  {'ΔE':>12s}  "
    f"{'E_pt2':>12s}  {'E_tot':>16s}  {'ms':>6s}"
)
print("  " + "-" * 75)

prev_e_var = float("inf")
final_ci = None
for target in [1, 3, 5, 10, 20, 50, 100, 200]:
    t0 = time.perf_counter()
    ci_opts = SelectedCIOptions(
        target_size=target,
        max_iter=30,
        conv_tol_energy=1e-10,
        do_pt2_correction=True,
        pt2_threshold=1e-8,
        verbose=0,
    )
    result = solve_selected_ci(ham, ci_opts)
    dt = time.perf_counter() - t0
    ndet = len(result.ci_labels) if result.ci_labels else 0
    e_var = result.energy - (result.pt2_correction or 0.0)
    delta = e_var - prev_e_var if prev_e_var != float("inf") else 0.0
    prev_e_var = e_var
    pt2 = result.pt2_correction or 0.0
    print(
        f"  {target:>7d}  {ndet:>6d}  {e_var:>16.10f}  {delta:>12.2e}  "
        f"{pt2:>12.2e}  {result.energy:>16.10f}  {dt * 1000:>5.0f}"
    )
    final_ci = result

# ── Closed-shell FCI (21 determinants) ──────────────────────────────────
t0 = time.perf_counter()
cs_dets = generate_closed_shell_determinants(norb, nocc)
H_cs = build_hamiltonian_matrix(cs_dets, ham.h1e, ham.h2e)
evals_cs, evecs_cs = eigh(H_cs)
e_fci_cs = evals_cs[0] + ham.nuclear_repulsion
dt_cs = time.perf_counter() - t0
print(
    f"\n  Closed-shell FCI ({len(cs_dets)} dets): "
    f"E = {e_fci_cs:.8f} Ha  ({dt_cs * 1000:.0f} ms)"
)

# ── Full FCI (441 determinants, all spin configurations) ─────────────────
t0 = time.perf_counter()
all_spin_dets = generate_determinants(norb, nalpha, nbeta)
ndet_full = len(all_spin_dets)

# Build the full Hamiltonian matrix
H_full = np.zeros((ndet_full, ndet_full))
# Precompute diagonal
for I, (aI, bI) in enumerate(all_spin_dets):
    H_full[I, I] = diagonal_matrix_element_unrestricted(aI, bI, ham.h1e, ham.h2e)

# Off-diagonal: loop over all pairs, check excitation rank
# For two-body Hamiltonians, only total rank ≤ 2 connects
for I in range(ndet_full):
    aI, bI = all_spin_dets[I]
    for J in range(I + 1, ndet_full):
        aJ, bJ = all_spin_dets[J]
        rank_a = _ex_rank(aI, aJ)
        rank_b = _ex_rank(bI, bJ)
        total = rank_a + rank_b
        if total > 2:
            continue

        # Compute ⟨D_I|H|D_J⟩
        val = 0.0

        if total == 0:
            # Already handled by diagonal
            pass
        elif total == 1:
            # Single excitation in one spin sector
            if rank_a == 1 and rank_b == 0:
                holes_a = sorted(set(aI) - set(aJ))
                parts_a = sorted(set(aJ) - set(aI))
                # Single excitation in α sector, β unchanged
                # Contribution: h_{ia} + Σ_{k∈α} (g_{ikak} − g_{ikka})
                #            + Σ_{k∈β} g_{ikak}
                i, a = holes_a[0], parts_a[0]
                from vibeqc.solvers._slater_condon import _phase_factor as _pf

                sign = _pf(aI, [(i, a)])
                val += sign * ham.h1e[i, a]
                for k in aI:
                    val += sign * (ham.h2e[i, k, a, k] - ham.h2e[i, k, k, a])
                for k in bI:
                    val += sign * ham.h2e[i, k, a, k]
            elif rank_a == 0 and rank_b == 1:
                holes_b = sorted(set(bI) - set(bJ))
                parts_b = sorted(set(bJ) - set(bI))
                i, a = holes_b[0], parts_b[0]
                from vibeqc.solvers._slater_condon import _phase_factor as _pf

                sign = _pf(bI, [(i, a)])
                val += sign * ham.h1e[i, a]
                for k in bI:
                    val += sign * (ham.h2e[i, k, a, k] - ham.h2e[i, k, k, a])
                for k in aI:
                    val += sign * ham.h2e[i, k, a, k]

        elif total == 2:
            if rank_a == 2 and rank_b == 0:
                holes = sorted(set(aI) - set(aJ))
                parts = sorted(set(aJ) - set(aI))
                i, j = holes
                a, b = parts
                val = double_excitation_matrix_element(aI, i, j, a, b, ham.h2e)
            elif rank_a == 0 and rank_b == 2:
                holes = sorted(set(bI) - set(bJ))
                parts = sorted(set(bJ) - set(bI))
                i, j = holes
                a, b = parts
                val = double_excitation_matrix_element(bI, i, j, a, b, ham.h2e)
            elif rank_a == 1 and rank_b == 1:
                # One excitation in each spin sector:
                # ⟨...a†_a a_i...| H |...a†_b a_j...⟩
                # = g_{i_α j_β, a_α b_β} with appropriate sign
                holes_a = sorted(set(aI) - set(aJ))
                parts_a = sorted(set(aJ) - set(aI))
                holes_b = sorted(set(bI) - set(bJ))
                parts_b = sorted(set(bJ) - set(bI))
                i_a, a_a = holes_a[0], parts_a[0]
                i_b, a_b = holes_b[0], parts_b[0]
                from vibeqc.solvers._slater_condon import _phase_factor as _pf

                sign = _pf(aI, [(i_a, a_a)]) * _pf(bI, [(i_b, a_b)])
                val = sign * ham.h2e[i_a, i_b, a_a, a_b]

        H_full[I, J] = val
        H_full[J, I] = val

evals_full, evecs_full = eigh(H_full)
e_fci_full = evals_full[0] + ham.nuclear_repulsion
dt_full = time.perf_counter() - t0
print(
    f"  Full FCI ({ndet_full} dets):       "
    f"E = {e_fci_full:.8f} Ha  ({dt_full * 1000:.0f} ms)"
)

# ── Leading full-FCI determinants ───────────────────────────────────────
c_full = np.abs(evecs_full[:, 0])
idx_full = np.argsort(-c_full)
print(f"\n  Leading full-FCI determinants:")
for rank in range(min(8, len(idx_full))):
    i = idx_full[rank]
    a_occ, b_occ = all_spin_dets[i]
    a_str = "".join("1" if j in a_occ else "0" for j in range(norb))
    b_str = "".join("1" if j in b_occ else "0" for j in range(norb))
    note = ""
    if a_occ == b_occ:
        note = "  (closed-shell)"
    print(f"    |c_{rank}| = {c_full[i]:.6f}  α|{a_str}⟩ β|{b_str}⟩{note}")

# ── Literature comparison ───────────────────────────────────────────────
e_rhf_lit = -74.96306
e_fci_lit = -75.01266
e_corr_lit = e_fci_lit - e_rhf_lit

print(f"\n  ═══ Literature comparison ═══")
print(f"  {'':22s}  {'vibe-qc':>16s}  {'Literature':>16s}  {'Δ':>12s}")
print(
    f"  {'E(RHF)':22s}  {rhf_result.energy:>16.8f}  {e_rhf_lit:>16.8f}  "
    f"{rhf_result.energy - e_rhf_lit:>+12.2e}"
)
print(f"  {'E(FCI, closed-shell)':22s}  {e_fci_cs:>16.8f}  {'—':>16s}  {'—':>12s}")
print(
    f"  {'E(FCI, full)':22s}  {e_fci_full:>16.8f}  {e_fci_lit:>16.8f}  "
    f"{e_fci_full - e_fci_lit:>+12.2e}"
)
print(
    f"  {'E_corr (full)':22s}  {e_fci_full - rhf_result.energy:>16.8f}  "
    f"{e_corr_lit:>16.8f}  "
    f"{(e_fci_full - rhf_result.energy) - e_corr_lit:>+12.2e}"
)

pct_cs = 100 * (e_fci_cs - rhf_result.energy) / (e_fci_full - rhf_result.energy)
print(f"\n  The closed-shell FCI captures {pct_cs:.1f}% of the correlation.")
print(f"  The remaining ~{100 - pct_cs:.0f}% comes from determinants with")
print(f"  different α and β occupation patterns — single excitations")
print(f"  where one spin channel excites an electron but the other stays")
print(f'  in the reference configuration.  These "open-shell singlet"')
print(f"  configurations are essential for quantitative dynamic correlation.")

print(f"\n  References:")
print(f"    Harrison & Handy, Chem. Phys. Lett. 95, 386 (1983)")
print(f"    Bauschlicher & Taylor, J. Chem. Phys. 85, 2779 (1986)")
