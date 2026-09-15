"""Pinpoint translation-invariance breakage — compare each one-electron
matrix element between H2 at the origin and H2 centered at L/2.

If T, V_ne, J_LR, J_SR, K are all translation-invariant matrix-by-matrix
(when computed with the molecule's AO basis), then SCF energy must also
be translation-invariant. If any one of them differs, we've localized
the bug.
"""
from __future__ import annotations
import numpy as np

import vibeqc as vq
from vibeqc import (
    BasisSet, CoulombMethod, EwaldOptions, GridOptions, LatticeSumOptions,
    PeriodicSystem, build_grid, bloch_sum, compute_kinetic_lattice,
    compute_nuclear_lattice, compute_nuclear_lattice_ewald,
    compute_overlap_lattice,
)


def setup(L: float, offset):
    R0 = np.asarray(offset, dtype=float)
    atoms = [
        vq.Atom(1, R0.tolist()),
        vq.Atom(1, (R0 + np.array([0.0, 0.0, 1.4])).tolist()),
    ]
    sys_p = vq.PeriodicSystem(
        dim=3, lattice=L * np.eye(3), unit_cell=atoms,
    )
    basis = BasisSet(sys_p.unit_cell_molecule(), "sto-3g")
    return sys_p, basis


def gamma_matrices(sys_p, basis, lat_opts):
    k_gamma = np.zeros(3)
    S_lat = compute_overlap_lattice(basis, sys_p, lat_opts)
    T_lat = compute_kinetic_lattice(basis, sys_p, lat_opts)
    # V_ne via Ewald (the gauge-consistent path)
    grid = build_grid(sys_p.unit_cell_molecule(), GridOptions())
    eopts = EwaldOptions()
    eopts.real_cutoff_bohr = lat_opts.nuclear_cutoff_bohr
    V_lat_ewald = compute_nuclear_lattice_ewald(
        basis, sys_p, grid, lat_opts, eopts,
    )
    # V_ne via bare libint (the legacy path)
    V_lat_bare = compute_nuclear_lattice(basis, sys_p, lat_opts)

    S = np.real(bloch_sum(S_lat, k_gamma))
    T = np.real(bloch_sum(T_lat, k_gamma))
    V_ewald = np.real(bloch_sum(V_lat_ewald, k_gamma))
    V_bare = np.real(bloch_sum(V_lat_bare, k_gamma))
    return dict(S=S, T=T, V_ewald=V_ewald, V_bare=V_bare)


def main():
    L = 30.0
    lat_opts = LatticeSumOptions()
    lat_opts.coulomb_method = CoulombMethod.EWALD_3D
    lat_opts.cutoff_bohr = 12.0
    lat_opts.nuclear_cutoff_bohr = 25.0

    sys_o, basis_o = setup(L, [0.0, 0.0, 0.0])
    sys_c, basis_c = setup(L, [L / 2 - 0.0, L / 2 - 0.0, L / 2 - 0.7])

    M_o = gamma_matrices(sys_o, basis_o, lat_opts)
    M_c = gamma_matrices(sys_c, basis_c, lat_opts)

    print("Frobenius diffs (||M_origin - M_centered||):")
    print(f"  S       = {np.linalg.norm(M_o['S'] - M_c['S']):.3e}")
    print(f"  T       = {np.linalg.norm(M_o['T'] - M_c['T']):.3e}")
    print(f"  V_bare  = {np.linalg.norm(M_o['V_bare'] - M_c['V_bare']):.3e}")
    print(f"  V_ewald = {np.linalg.norm(M_o['V_ewald'] - M_c['V_ewald']):.3e}")
    print()
    print("Trace diffs (tr M_origin - tr M_centered):")
    for k in ["S", "T", "V_bare", "V_ewald"]:
        print(f"  tr {k:8s} = {np.trace(M_o[k]) - np.trace(M_c[k]):+.6e}")
    print()
    print("Eigenvalues (S, ascending):")
    print(f"  origin   = {np.sort(np.linalg.eigvalsh(M_o['S']))}")
    print(f"  centered = {np.sort(np.linalg.eigvalsh(M_c['S']))}")
    print()
    print("Eigenvalues (T+V_ewald, Hcore, ascending):")
    H_o = M_o['T'] + M_o['V_ewald']
    H_c = M_c['T'] + M_c['V_ewald']
    print(f"  origin   = {np.sort(np.linalg.eigvalsh(H_o))}")
    print(f"  centered = {np.sort(np.linalg.eigvalsh(H_c))}")


if __name__ == "__main__":
    main()
