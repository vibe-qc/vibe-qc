"""Iter-1-only F(k) eigenvalue dump for LiH primitive @ kmesh=(8,8,8).

After the F-build, dumps per-k MO eigenvalues at the SAD-derived
density — equivalent to CRYSTAL14's CYC 0 ("after first Fock build
at SAD"). For direct comparison against CRYSTAL14's per-cycle
"TOP OF VALENCE BANDS / BOTTOM OF VIRTUAL BANDS" prints (from
/tmp/crystal_lih_verbose2/out.log).
"""
from __future__ import annotations

import json
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import vibeqc as vq
from vibeqc import (
    CoulombMethod, InitialGuess, LatticeSumOptions,
    attach_symmetry, bloch_sum, compute_kinetic_lattice,
    compute_overlap_lattice, monkhorst_pack, nuclear_repulsion_per_cell,
    real_space_density_from_kpoints,
)
from vibeqc.guess import initial_density_closed_shell
from vibeqc.periodic_fock_multi_k import build_periodic_fock_ewald3d_k
from vibeqc.periodic_rhf_multi_k_ewald import (
    _canonical_orthogonalizer_complex, _diag_in_orth_basis,
)
from vibeqc.periodic_v_ne import compute_nuclear_lattice_dispatch
from vibeqc.ewald_j import auto_grid

ANG2BOHR = 1.0 / 0.529177210903
CUTOFF_BOHR = 18.0


def main() -> int:
    a = 4.084 * ANG2BOHR
    lattice = (a / 2.0) * np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    system = vq.PeriodicSystem(3, lattice, [
        vq.Atom(3, [0.0, 0.0, 0.0]),
        vq.Atom(1, [a / 2.0, a / 2.0, a / 2.0]),
    ])
    attach_symmetry(system)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    n_occ = system.n_electrons() // 2
    print(f"LiH primitive STO-3G, n_occ={n_occ}, cutoff={CUTOFF_BOHR}")

    lat_opts = LatticeSumOptions()
    lat_opts.cutoff_bohr = CUTOFF_BOHR
    lat_opts.nuclear_cutoff_bohr = CUTOFF_BOHR
    lat_opts.coulomb_method = CoulombMethod.EWALD_3D

    kmesh = monkhorst_pack(system, [8, 8, 8], use_symmetry=True)
    k_points = list(kmesh.kpoints)
    weights = list(kmesh.weights)
    n_k = len(k_points)
    print(f"kmesh=(8,8,8), {n_k} irreducible k-points, weights sum = {sum(weights):.4f}")

    # Real-space one-electron integrals
    t0 = time.time()
    S_lat = compute_overlap_lattice(basis, system, lat_opts)
    T_lat = compute_kinetic_lattice(basis, system, lat_opts)
    V_lat = compute_nuclear_lattice_dispatch(basis, system, lat_opts)
    print(f"S/T/V lattice integrals: {time.time()-t0:.1f}s  ({len(S_lat.cells)} cells)")

    # SAD initial density (g=0 only, zeros elsewhere — multi-k driver convention)
    D_engine = initial_density_closed_shell(
        system.unit_cell_molecule(), basis, n_occ,
        InitialGuess.SAD, is_periodic=True,
    )
    cells = list(S_lat.cells)

    # Build D_real with SAD at g=0, zeros elsewhere.
    # We use Hcore-diagonalisation then overwrite g=0 with D_engine.
    S_k_list, Hcore_k_list, X_k_list = [], [], []
    for k_idx, k in enumerate(k_points):
        k_arr = np.asarray(k, dtype=float).reshape(3)
        S_k = np.asarray(bloch_sum(S_lat, k_arr))
        T_k = np.asarray(bloch_sum(T_lat, k_arr))
        V_k = np.asarray(bloch_sum(V_lat, k_arr))
        H_k = T_k + V_k
        S_k = 0.5 * (S_k + S_k.conj().T)
        H_k = 0.5 * (H_k + H_k.conj().T)
        X_k, n_kept = _canonical_orthogonalizer_complex(S_k, 1e-7, normalize_diag_first=True)
        S_k_list.append(S_k)
        Hcore_k_list.append(H_k)
        X_k_list.append(X_k)

    # Hcore-diagonalise per k to seed C, then overwrite D(g=0) with SAD.
    C_per_k = []
    eps_per_k_hcore = []
    for H_k, X_k in zip(Hcore_k_list, X_k_list):
        C_k, eps_k = _diag_in_orth_basis(H_k, X_k)
        C_per_k.append(C_k.astype(complex))
        eps_per_k_hcore.append(eps_k)
    D_real = real_space_density_from_kpoints(C_per_k, [n_occ] * n_k, kmesh, cells)
    for g_idx in range(len(D_real.cells)):
        if (D_real.cells[g_idx].index == np.array([0, 0, 0])).all():
            D_real.set_block(g_idx, D_engine)
        else:
            D_real.set_block(g_idx, np.zeros_like(D_engine, dtype=float))

    # Build F at SAD density.
    grid_shape = auto_grid(np.asarray(system.lattice), 0.3)
    t0 = time.time()
    F_k_list = build_periodic_fock_ewald3d_k(
        basis, system, D_real, omega=0.5,
        k_points_cart=[np.asarray(k) for k in k_points],
        Hcore_k=Hcore_k_list,
        lattice_opts=lat_opts,
        grid_shape=grid_shape, origin=None, spacing_bohr=0.3,
    )
    print(f"F(SAD) build: {time.time()-t0:.1f}s")

    # Diagonalise F per k.
    print()
    print(f"=== Iter 1 F(k) eigenvalues (at SAD density) per irreducible k ===")
    print(f"  CRYSTAL CYC 0 reference (from /tmp/crystal_lih_verbose2/out.log):")
    print(f"    TOP VAL @ K=21 = -0.2253 Ha,  K=1(Γ) = -1.1767 Ha")
    print(f"    BOT VIRT @ K=21 = +0.0055 Ha,  K=1(Γ) = +0.6095 Ha")
    print(f"    E_total/cell = -7.9071 Ha")
    print()
    for k_idx in range(n_k):
        C_k, eps_k = _diag_in_orth_basis(F_k_list[k_idx], X_k_list[k_idx])
        homo = float(np.real(eps_k[n_occ - 1]))
        lumo = float(np.real(eps_k[n_occ])) if n_occ < len(eps_k) else float("inf")
        k_cart = np.asarray(k_points[k_idx], dtype=float)
        eigs_str = "  ".join(f"{e:>+.4f}" for e in eps_k[:6])
        print(f"  k{k_idx:>2}  k={k_cart.round(3).tolist()!r:>32}  w={weights[k_idx]:.4f}")
        print(f"       eigvals: {eigs_str}")
        print(f"       HOMO={homo:+.4f}  LUMO={lumo:+.4f}  gap={lumo-homo:+.4f}")

    # Energy at SAD density
    E_elec = 0.0
    for idx in range(n_k):
        C_occ = C_per_k[idx][:, :n_occ]
        D_k = 2.0 * (C_occ @ C_occ.conj().T)
        # Note: D_k here is from Hcore-diag, NOT SAD.
        # For the "energy at SAD" we need the SAD-derived D_k:
        D_g0 = np.asarray(D_engine, dtype=complex)
        # Trivial Bloch-sum (since D_real is localized at g=0): D(k) = D(g=0) for all k
        E_elec += float(weights[idx]) * 0.5 * np.real(
            np.trace(D_g0 @ (Hcore_k_list[idx] + F_k_list[idx]))
        )
    e_nuc = float(nuclear_repulsion_per_cell(system, lat_opts))
    print()
    print(f"E_elec (at SAD-D, Bloch-trivial)  = {E_elec:+.6f}")
    print(f"E_nuc                              = {e_nuc:+.6f}")
    print(f"E_total                            = {E_elec + e_nuc:+.6f}")
    print(f"CRYSTAL CYC 0 E_total              = -7.907138")
    return 0


if __name__ == "__main__":
    sys.exit(main())
