"""Confirm the AO permutation works for periodic MgO at Γ with a
sufficient lattice cutoff (cutoff_bohr=18 was shown to converge to
~1e-6 vs PySCF in scratch_basis_exponents.py).

If this shows max|elem diff| < 1e-3 elementwise for S, T, V_ne, then
the same permutation can be used to convert PySCF J/K matrices into
vibe-qc layout and vice versa.
"""
from __future__ import annotations

import time

import numpy as np

import vibeqc as vq
from vibeqc.periodic_v_ne import compute_nuclear_lattice_dispatch

from pyscf.pbc import gto as pbc_gto, scf as pbc_scf
from pyscf.pbc import df as pbc_df


ANG2BOHR = 1.0 / 0.529177210903
A_ANG = 4.211

# Per-atom permutation: vibeqc index → pyscf index.
PERM_MG = [0, 1, 4, 5, 3, 2, 7, 8, 6]
PERM_O = [0, 1, 3, 4, 2]
ATOM_LIST = [("Mg", 9, PERM_MG)] * 4 + [("O", 5, PERM_O)] * 4


def build_permutation(atom_list):
    n = sum(nbf for _, nbf, _ in atom_list)
    P = np.zeros((n, n))
    voff = 0
    poff = 0
    for el, nbf, perm in atom_list:
        for i, p in enumerate(perm):
            P[poff + p, voff + i] = 1.0
        voff += nbf
        poff += nbf
    return P


def build_systems(cutoff_bohr=18.0):
    a = A_ANG * ANG2BOHR
    mg_frac = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    o_frac = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
    cell_atoms = []
    for fx, fy, fz in mg_frac:
        cell_atoms.append(vq.Atom(12, [fx * a, fy * a, fz * a]))
    for fx, fy, fz in o_frac:
        cell_atoms.append(vq.Atom(8, [fx * a, fy * a, fz * a]))
    sys_p = vq.PeriodicSystem(3, np.diag([a, a, a]), cell_atoms)
    basis = vq.BasisSet(sys_p.unit_cell_molecule(), "sto-3g")
    lat_opts = vq.LatticeSumOptions()
    lat_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    lat_opts.cutoff_bohr = cutoff_bohr
    lat_opts.nuclear_cutoff_bohr = max(25.0, 2.0 * cutoff_bohr)
    atom_lines = []
    for fx, fy, fz in mg_frac:
        atom_lines.append(f"Mg {fx*A_ANG:.6f} {fy*A_ANG:.6f} {fz*A_ANG:.6f}")
    for fx, fy, fz in o_frac:
        atom_lines.append(f"O  {fx*A_ANG:.6f} {fy*A_ANG:.6f} {fz*A_ANG:.6f}")
    cell = pbc_gto.M(
        atom="; ".join(atom_lines),
        a=[[A_ANG, 0, 0], [0, A_ANG, 0], [0, 0, A_ANG]],
        basis="sto-3g",
        unit="A",
        verbose=0,
        mesh=[31, 31, 31],
        precision=1e-8,
    )
    return sys_p, basis, lat_opts, cell


def vibeqc_one_e(sys_p, basis, lat_opts):
    k_gamma = np.zeros(3)
    S_lat = vq.compute_overlap_lattice(basis, sys_p, lat_opts)
    T_lat = vq.compute_kinetic_lattice(basis, sys_p, lat_opts)
    V_lat = compute_nuclear_lattice_dispatch(basis, sys_p, lat_opts)
    S = np.real(vq.bloch_sum(S_lat, k_gamma))
    T = np.real(vq.bloch_sum(T_lat, k_gamma))
    V = np.real(vq.bloch_sum(V_lat, k_gamma))
    return 0.5*(S+S.T), 0.5*(T+T.T), 0.5*(V+V.T)


def main():
    sys_p, basis, lat_opts, cell = build_systems(cutoff_bohr=18.0)
    P = build_permutation(ATOM_LIST)

    print("Computing vibe-qc one-electron operators (cutoff_bohr=18)...")
    t0 = time.perf_counter()
    Sv, Tv, Vv = vibeqc_one_e(sys_p, basis, lat_opts)
    print(f"  done in {time.perf_counter()-t0:.1f}s")

    print("Computing PySCF one-electron operators...")
    t0 = time.perf_counter()
    mf = pbc_scf.RHF(cell)
    mf.exxdiv = "ewald"
    mf.with_df = pbc_df.GDF(cell)
    mf.with_df.build()
    Sp = mf.get_ovlp()
    Tp = cell.pbc_intor("int1e_kin", hermi=1)
    Vp = mf.get_hcore() - Tp
    print(f"  done in {time.perf_counter()-t0:.1f}s")

    for name, Mv, Mp in [("S", Sv, Sp), ("T", Tv, Tp), ("V_ne", Vv, Vp)]:
        Mv_in_pyscf = P @ Mv @ P.T
        diff = Mv_in_pyscf - Mp
        max_ad = float(np.max(np.abs(diff)))
        rel = float(np.linalg.norm(diff)/np.linalg.norm(Mp))
        print(f"  {name:>5s}  max|elem|={max_ad:.3e}  ||rel||={rel:.3e}  "
              f"verdict={'OK' if max_ad < 1e-3 else 'MISMATCH'}")


if __name__ == "__main__":
    main()
