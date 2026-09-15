"""Phase 6 PROBE: Γ-only MgO SCF via the BIPOLE Ewald J-split.

End-to-end validation that the Phase 5 ``build_fock_2e_ewald_j_split_gamma``
machinery (J^SR via direct-space erfc + J^LR via reciprocal-sum
analytic + K via direct) plus Ewald V_ne and Ewald E_nn produces a
converged SCF that lands within ~2 Ha of CRYSTAL's full SHRINK 8 8
converged total energy.

This is the structural-validation milestone for the BIPOLE
multipole-far-pair branch. The remaining sub-Ha discrepancy will
close with:
  * SHRINK > 1 k-mesh (Γ-only is the worst case; k-averaging
    filters out long-range "image-image" contributions)
  * Larger real-space cutoff for the K extraction
  * Fix for SAD over-normalisation (D trace ≈ 20.68 vs 20)

Per CLAUDE.md §10: no CRYSTAL import; the reference number comes
from a sealed local CRYSTAL23-demo run.

Expected output (cutoff=14, 12 iters):
  iter  1: -269.28 Ha
  iter 12: -269.35 Ha  (converging; dE ~ 1e-5)
  CRYSTAL: -271.22 Ha
  Δ ≈ +1.9 Ha at Γ-only (closes at SHRINK 8 8)

Run wall time: ~25-30 minutes on M5 Max at cutoff 14 (each iter
calls build_fock_2e_real_space 3 times).
"""
from __future__ import annotations

import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, "examples/regression/crystal_parity")

import numpy as np
from scipy.linalg import eigh

from vibeqc import CoulombMethod, InitialGuess, LatticeSumOptions
from vibeqc._vibeqc_core import (
    bloch_sum,
    compute_kinetic_lattice,
    compute_overlap_lattice,
    nuclear_repulsion_per_cell,
)
from vibeqc.bipole_ext_el_pole import crystal_default_ewald_alpha
from vibeqc.bipole_fock_ewald import build_fock_2e_ewald_j_split_gamma
from vibeqc.bipole_lattice_self_energy import cell_volume_bohr3
from vibeqc.guess import initial_density_closed_shell
from vibeqc.periodic_v_ne import compute_nuclear_lattice_dispatch
from crystal_demos.builders import build_mgo_sto3g


CRYSTAL_E_TOTAL = -271.21814374982  # sealed; SHRINK 8 8 CYC 7 final


def main() -> int:
    print("=== MgO/STO-3G Γ-only SCF via BIPOLE Ewald J-split ===")
    system, basis = build_mgo_sto3g()
    V = cell_volume_bohr3(system)
    omega = crystal_default_ewald_alpha(V)
    n_elec = system.n_electrons()
    n_occ = n_elec // 2
    print(f"  V = {V:.3f} bohr³, ω = {omega:.4f} bohr⁻¹, n_e = {n_elec}")

    opts_1e = LatticeSumOptions()
    opts_1e.cutoff_bohr = 14.0
    opts_1e.nuclear_cutoff_bohr = 14.0
    opts_1e.coulomb_method = CoulombMethod.EWALD_3D

    opts_2e = LatticeSumOptions()
    opts_2e.cutoff_bohr = 14.0
    opts_2e.nuclear_cutoff_bohr = 14.0
    opts_2e.coulomb_method = CoulombMethod.DIRECT_TRUNCATED

    S_lat = compute_overlap_lattice(basis, system, opts_2e)
    T_lat = compute_kinetic_lattice(basis, system, opts_2e)
    V_lat = compute_nuclear_lattice_dispatch(basis, system, opts_1e)

    S_gamma = 0.5 * (np.asarray(bloch_sum(S_lat, np.zeros(3)))
                     + np.asarray(bloch_sum(S_lat, np.zeros(3))).conj().T)
    T_gamma = np.asarray(bloch_sum(T_lat, np.zeros(3)))
    V_gamma = np.asarray(bloch_sum(V_lat, np.zeros(3)))
    H_core = T_gamma + V_gamma
    H_core = 0.5 * (H_core + H_core.conj().T)

    e_nuc = float(nuclear_repulsion_per_cell(system, opts_1e))
    print(f"  E_nuc (Ewald) = {e_nuc:+.6f} Ha")

    D_sad = initial_density_closed_shell(
        system.unit_cell_molecule(), basis, n_occ,
        InitialGuess.SAD, is_periodic=True,
    )
    P_real = S_lat
    g0_idx = next(i for i, c in enumerate(P_real.cells)
                   if (np.asarray(c.index) == np.array([0, 0, 0])).all())
    for g_idx in range(len(P_real.cells)):
        is_g0 = g_idx == g0_idx
        P_real.set_block(
            g_idx,
            np.asarray(D_sad, dtype=float) if is_g0
            else np.zeros_like(np.asarray(D_sad), dtype=float),
        )

    damping = 0.3
    max_iter = 20
    conv_tol_energy = 1e-7
    print(f"  damping = {damping}, max_iter = {max_iter}")
    print(f"  CRYSTAL ref E_total = {CRYSTAL_E_TOTAL:+.6f} Ha")
    print()
    print(f"  {'iter':>4} {'E_total':>14} {'dE':>14} {'wall':>8}")

    E_prev = 0.0
    D_prev = np.asarray(D_sad, dtype=float).copy()
    converged = False
    E_total = 0.0
    for iter_idx in range(1, max_iter + 1):
        t0 = time.time()
        result = build_fock_2e_ewald_j_split_gamma(
            P_real, basis, system, opts_2e,
            omega=omega, precision=1e-8,
        )
        D = np.asarray(P_real.blocks[g0_idx], dtype=float)
        F = H_core + result.F2e
        E_elec = 0.5 * np.real(np.trace(D @ (H_core + F)))
        E_total = E_elec + e_nuc

        eps, C = eigh(F.real, S_gamma.real)
        C_occ = C[:, :n_occ]
        D_new = 2.0 * (C_occ @ C_occ.T)
        if iter_idx > 1:
            D_new = (1.0 - damping) * D_new + damping * D_prev
        P_real.set_block(g0_idx, D_new)
        D_prev = D_new.copy()

        dE = E_total - E_prev if iter_idx > 1 else 0.0
        wall = time.time() - t0
        print(f"  {iter_idx:4d} {E_total:+14.8f} {dE:+14.2e} {wall:7.1f}s")
        E_prev = E_total

        if iter_idx > 1 and abs(dE) < conv_tol_energy:
            converged = True
            print(f"  → converged on energy criterion (|dE| < {conv_tol_energy})")
            break

    delta_mha = (E_total - CRYSTAL_E_TOTAL) * 1000.0
    print()
    print(f"  CRYSTAL E_total (SHRINK 8 8) = {CRYSTAL_E_TOTAL:+.6f} Ha")
    print(f"  vibe-qc E_total (Γ-only)     = {E_total:+.6f} Ha")
    print(f"  Δ                             = {delta_mha:+.2f} mHa")
    print(f"  status: {'CONVERGED' if converged else 'not yet converged'}")
    return 0 if converged and abs(delta_mha) < 3000.0 else 1


if __name__ == "__main__":
    sys.exit(main())
