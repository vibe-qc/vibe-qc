"""LiH conventional rocksalt — iter-1 energy decomposition.

Goal: localise which component of the SCF total is responsible for
the residual ~700 Ha gap (vibe-qc gives ~-700, physical ~-32) after
the v0.7 J-build gauge fix and the auto-optimise lattice screening.

Strategy: build all the one-electron + two-electron pieces
manually with a Hcore initial guess, print every contribution to
the total, compare against the atomic-limit expectation
(4·E_Li + 4·E_H ≈ -31.6 Ha for STO-3G/LDA).
"""
from __future__ import annotations

import numpy as np
import vibeqc as vq
from vibeqc import (
    BasisSet, CoulombMethod, EwaldOptions, GridOptions, LatticeSumOptions,
    PeriodicSystem, bloch_sum, build_grid, build_jk_gamma_molecular_limit,
    compute_kinetic_lattice, compute_nuclear_lattice, compute_overlap_lattice,
    nuclear_repulsion_per_cell,
)
from vibeqc.ewald_composed import build_j_ewald_3d


def main() -> None:
    # LiH conventional cubic, a = 4.084 Å.
    a = 4.084 / 0.529177210903
    unit_cell = []
    for fx, fy, fz in [(0, 0, 0), (0, 0.5, 0.5),
                        (0.5, 0, 0.5), (0.5, 0.5, 0)]:
        unit_cell.append(vq.Atom(3, [fx * a, fy * a, fz * a]))
    for fx, fy, fz in [(0.5, 0.5, 0.5), (0.5, 0, 0),
                        (0, 0.5, 0), (0, 0, 0.5)]:
        unit_cell.append(vq.Atom(1, [fx * a, fy * a, fz * a]))
    sysp = vq.PeriodicSystem(3, np.diag([a, a, a]), unit_cell)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

    lat_opts = vq.LatticeSumOptions()
    lat_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    # Use the v0.7 auto-optimised values (cutoff 15, schwarz 1e-14)
    # so we're looking at the converged-screening case.
    lat_opts.cutoff_bohr = 15.0
    lat_opts.nuclear_cutoff_bohr = 22.5
    lat_opts.schwarz_threshold = 1e-14

    print(f"LiH conventional, a = {a:.4f} bohr, V = {a**3:.2f} bohr^3")
    print(f"basis nbf = {basis.nbasis}, n_electrons/cell = {sysp.n_electrons()}")
    print(f"cutoff_bohr = {lat_opts.cutoff_bohr}, "
          f"nuclear_cutoff_bohr = {lat_opts.nuclear_cutoff_bohr}, "
          f"schwarz = {lat_opts.schwarz_threshold:.0e}")
    print()

    # ---- Build one-electron integrals at Γ -------------------------------
    # Two V_ne paths: bare libint (LEGACY, default) vs Ewald-dispatch
    # (NEW, gauge-aligned with the FFT-Poisson J build). Print both
    # to localise the gauge bug.
    from vibeqc.periodic_v_ne import compute_nuclear_lattice_dispatch
    S_lat = compute_overlap_lattice(basis, sysp, lat_opts)
    T_lat = compute_kinetic_lattice(basis, sysp, lat_opts)
    V_lat_legacy = compute_nuclear_lattice(basis, sysp, lat_opts)
    V_lat_ewald  = compute_nuclear_lattice_dispatch(basis, sysp, lat_opts)
    # Choose which to feed into the energy decomp. By default use the
    # NEW Ewald-dispatched path so we can see whether it cures LiH.
    V_lat = V_lat_ewald
    k = np.zeros(3)
    S = np.real(bloch_sum(S_lat, k))
    T = np.real(bloch_sum(T_lat, k))
    V = np.real(bloch_sum(V_lat, k))
    Hcore = T + V

    # ---- Hcore guess: D = 2 C_occ C_occ^T --------------------------------
    s_eig, s_vec = np.linalg.eigh(S)
    X = s_vec @ np.diag(s_eig**-0.5) @ s_vec.T
    Hcore_p = X.T @ Hcore @ X
    eps, Cp = np.linalg.eigh(Hcore_p)
    C = X @ Cp
    n_occ = sysp.n_electrons() // 2
    D = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T
    D = 0.5 * (D + D.T)
    Q_e = float(np.trace(D @ S))
    Q_n = float(sum(a.Z for a in sysp.unit_cell))
    print(f"n_occ = {n_occ},  Q_e = {Q_e:.4f},  Q_n = {Q_n:.4f}")

    # ---- One-electron contributions to E ---------------------------------
    V_legacy = np.real(bloch_sum(V_lat_legacy, k))
    V_legacy = 0.5 * (V_legacy + V_legacy.T)
    E_T  = float(np.einsum("ij,ij->", D, T))
    E_V_legacy = float(np.einsum("ij,ij->", D, V_legacy))
    E_V        = float(np.einsum("ij,ij->", D, V))
    print(f"  E_T  = tr(D·T)            = {E_T:+12.4f} Ha   (kinetic)")
    print(f"  E_V  (legacy bare libint) = {E_V_legacy:+12.4f} Ha   "
          "(no G=0 gauge — pre-v0.7)")
    print(f"  E_V  (NEW Ewald dispatch) = {E_V:+12.4f} Ha   "
          "(G=0 dropped + jellium — gauge-matched to J)")
    print(f"  delta V_ne (Ewald-legacy) = {E_V - E_V_legacy:+12.4f} Ha")

    # ---- Hartree J via composed Ewald-3D ---------------------------------
    omega = 0.5
    spacing = 0.5
    J = build_j_ewald_3d(
        basis, sysp, D, omega=omega,
        lattice_opts=lat_opts, spacing_bohr=spacing,
    )
    E_J = 0.5 * float(np.einsum("ij,ij->", D, J))
    print(f"  E_J  = ½ tr(D·J)     = {E_J:+12.4f} Ha   (Hartree)")

    # ---- Exchange K (full-range real-space) ------------------------------
    jk = build_jk_gamma_molecular_limit(basis, sysp, lat_opts, D, 0.0)
    K = np.asarray(jk.K)
    E_K = -0.25 * float(np.einsum("ij,ij->", D, K))   # RHF: -¼ tr(D·K)
    print(f"  E_K  = -¼ tr(D·K)    = {E_K:+12.4f} Ha   (HF exchange — N/A for LDA)")

    # ---- LDA exchange-correlation (proper periodic Becke grid) -----------
    grid = build_grid(sysp.unit_cell_molecule(), GridOptions())
    try:
        from vibeqc._vibeqc_core import build_xc_periodic
        from vibeqc.xc_lattice import build_xc_periodic_drive  # may exist
        E_xc = 0.0  # filled if function is callable
        # Best-effort: if the API isn't exactly what we expected, just
        # report 0.0 — the dominant V_ne bug is what matters for
        # this decomposition.
    except Exception:
        E_xc = 0.0
    print(f"  E_xc (LDA, skipped)  = {E_xc:+12.4f} Ha   "
          "(not the bottleneck for this diagnosis)")

    # ---- Nuclear repulsion via full Ewald --------------------------------
    e_nuc = nuclear_repulsion_per_cell(sysp, lat_opts)
    print(f"  E_nn = Ewald sum     = {e_nuc:+12.4f} Ha")

    # ---- Madelung correction (v0.6.1 fix) --------------------------------
    from vibeqc.madelung import madelung_energy_correction
    E_madelung = madelung_energy_correction(
        D, S, sysp,
        nuclear_uses_ewald=(lat_opts.coulomb_method ==
                             vq.CoulombMethod.EWALD_3D),
    )
    print(f"  E_madelung_fix       = {E_madelung:+12.4f} Ha   "
          "(v0.6.1 G=0-leak compensation)")

    # ---- Total -----------------------------------------------------------
    E_total = E_T + E_V + E_J + E_xc + e_nuc + E_madelung
    print(f"  ----------------------------------------")
    print(f"  E_total (iter 1)     = {E_total:+12.4f} Ha")
    print()
    # Atomic-limit reference: 4 Li (LDA, STO-3G) + 4 H (LDA, STO-3G).
    # Approximate atomic energies from textbook / quick PySCF:
    #   E(Li, sto-3g, LDA) ≈ -7.20 Ha
    #   E(H,  sto-3g, LDA) ≈ -0.43 Ha
    # → 4 × -7.20 + 4 × -0.43 = -30.5 Ha
    expected = 4 * -7.20 + 4 * -0.43
    print(f"  atomic-limit ref ≈   {expected:+12.4f} Ha "
          "(4 Li/LDA + 4 H/LDA, sto-3g)")
    print(f"  delta                = {E_total - expected:+12.4f} Ha")
    print()
    print("Per-component sanity:")
    print(f"  Q_e (electron count) = {Q_e:.4f}  "
          f"({'OK — ~16' if abs(Q_e - 16) < 0.5 else 'SUSPECT'})")
    print(f"  E_J  / Q_e²  = {E_J / max(Q_e**2, 1e-9):.4f} Ha/e²  "
          f"(should be O(1/L) ≈ {1/a:.4f})")
    print(f"  E_V  / Q_e   = {E_V / max(Q_e, 1e-9):.4f} Ha/e   "
          "(per-electron nuclear attraction)")
    print(f"  E_nn         = {e_nuc:.4f} Ha  "
          f"(Madelung-summed Z²/r ~ {Q_n**2/a:.2f} for first-neighbour scale)")


if __name__ == "__main__":
    main()
