"""Check J / K translation invariance for the same Hcore-derived D."""
from __future__ import annotations
import numpy as np

import vibeqc as vq
from vibeqc import (
    BasisSet, CoulombMethod, EwaldOptions, GridOptions, LatticeSumOptions,
    PeriodicSystem, build_grid, bloch_sum, compute_kinetic_lattice,
    compute_nuclear_lattice_ewald, compute_overlap_lattice,
    build_jk_gamma_molecular_limit,
)
from vibeqc.ewald_composed import build_j_ewald_3d
from vibeqc.ewald_j import auto_grid


def setup(L, offset):
    R0 = np.asarray(offset, dtype=float)
    atoms = [
        vq.Atom(1, R0.tolist()),
        vq.Atom(1, (R0 + np.array([0.0, 0.0, 1.4])).tolist()),
    ]
    sys_p = vq.PeriodicSystem(dim=3, lattice=L * np.eye(3), unit_cell=atoms)
    basis = BasisSet(sys_p.unit_cell_molecule(), "sto-3g")
    return sys_p, basis


def main():
    L = 30.0
    lat_opts = LatticeSumOptions()
    lat_opts.coulomb_method = CoulombMethod.EWALD_3D
    lat_opts.cutoff_bohr = 12.0
    lat_opts.nuclear_cutoff_bohr = 25.0

    sys_o, basis_o = setup(L, [0.0, 0.0, 0.0])
    sys_c, basis_c = setup(L, [L / 2, L / 2, L / 2 - 0.7])

    # Build Hcore at each
    k = np.zeros(3)
    grid_o = build_grid(sys_o.unit_cell_molecule(), GridOptions())
    grid_c = build_grid(sys_c.unit_cell_molecule(), GridOptions())
    eopts = EwaldOptions()
    eopts.real_cutoff_bohr = lat_opts.nuclear_cutoff_bohr

    S_o = np.real(bloch_sum(compute_overlap_lattice(basis_o, sys_o, lat_opts), k))
    T_o = np.real(bloch_sum(compute_kinetic_lattice(basis_o, sys_o, lat_opts), k))
    V_o = np.real(bloch_sum(compute_nuclear_lattice_ewald(basis_o, sys_o, grid_o, lat_opts, eopts), k))
    Hcore_o = T_o + V_o

    S_c = np.real(bloch_sum(compute_overlap_lattice(basis_c, sys_c, lat_opts), k))
    T_c = np.real(bloch_sum(compute_kinetic_lattice(basis_c, sys_c, lat_opts), k))
    V_c = np.real(bloch_sum(compute_nuclear_lattice_ewald(basis_c, sys_c, grid_c, lat_opts, eopts), k))
    Hcore_c = T_c + V_c

    # Same Hcore → same C_occ → same D
    def D_from_Hcore(S, H, n_occ=1):
        # X = S^-1/2
        s_eig, s_vec = np.linalg.eigh(S)
        X = s_vec @ np.diag(s_eig**-0.5) @ s_vec.T
        Hp = X.T @ H @ X
        eps, Cp = np.linalg.eigh(Hp)
        C = X @ Cp
        D = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T
        return D, eps

    D_o, eps_o = D_from_Hcore(S_o, Hcore_o)
    D_c, eps_c = D_from_Hcore(S_c, Hcore_c)
    print(f"D diff: {np.linalg.norm(D_o - D_c):.3e}")
    print(f"eps diff: {eps_o - eps_c}")

    # Build J at each, decomposed into J_SR + J_LR
    from vibeqc.ewald_j import build_j_long_range
    omega = 0.5
    spacing = 0.4
    grid_shape = auto_grid(L * np.eye(3), spacing)

    jk_sr_o = build_jk_gamma_molecular_limit(basis_o, sys_o, lat_opts, D_o, omega)
    jk_sr_c = build_jk_gamma_molecular_limit(basis_c, sys_c, lat_opts, D_c, omega)
    J_SR_o = np.asarray(jk_sr_o.J)
    J_SR_c = np.asarray(jk_sr_c.J)

    # v0.7 fix: pass ``system`` so build_j_long_range uses periodic AOs.
    # Drop the system= kwarg to get the legacy translation-broken path.
    J_LR_o = build_j_long_range(
        basis_o, D_o, sys_o.lattice, omega,
        grid_shape=grid_shape, spacing_bohr=spacing, system=sys_o,
    )
    J_LR_c = build_j_long_range(
        basis_c, D_c, sys_c.lattice, omega,
        grid_shape=grid_shape, spacing_bohr=spacing, system=sys_c,
    )

    print(f"||J_SR diff|| = {np.linalg.norm(J_SR_o - J_SR_c):.6e}  "
          f"  tr(D·J_SR): origin={np.einsum('ij,ij->', D_o, J_SR_o):.6f}  "
          f"centred={np.einsum('ij,ij->', D_c, J_SR_c):.6f}")
    print(f"||J_LR diff|| = {np.linalg.norm(J_LR_o - J_LR_c):.6e}  "
          f"  tr(D·J_LR): origin={np.einsum('ij,ij->', D_o, J_LR_o):.6f}  "
          f"centred={np.einsum('ij,ij->', D_c, J_LR_c):.6f}")
    J_o = J_SR_o + J_LR_o
    J_c = J_SR_c + J_LR_c
    print(f"||J     diff|| = {np.linalg.norm(J_o - J_c):.6e}")

    # K
    jk_o = build_jk_gamma_molecular_limit(basis_o, sys_o, lat_opts, D_o, 0.0)
    jk_c = build_jk_gamma_molecular_limit(basis_c, sys_c, lat_opts, D_c, 0.0)
    K_o = np.asarray(jk_o.K)
    K_c = np.asarray(jk_c.K)
    print(f"||K_o - K_c|| = {np.linalg.norm(K_o - K_c):.6e}")
    print(f"tr(D·K) at origin   = {np.einsum('ij,ij->', D_o, K_o):.6f}")
    print(f"tr(D·K) at centered = {np.einsum('ij,ij->', D_c, K_c):.6f}")


if __name__ == "__main__":
    main()
