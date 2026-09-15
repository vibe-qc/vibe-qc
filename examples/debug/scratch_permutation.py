"""Build the AO permutation between vibe-qc (libint) and PySCF (libcint)
on MgO sto-3g.

Per-atom shell ordering (from scratch_basis_map.py):

  vibe-qc Mg:  [1s, 2s, 2p, 3s, 3p]    → AOs [1s, 2s, 2py, 2pz, 2px, 3s, 3py, 3pz, 3px]
  pyscf  Mg:  [1s, 2s, 3s, 2p, 3p]    → AOs [1s, 2s, 3s, 2py, 2pz, 2px, 3py, 3pz, 3px]

  vibe-qc O:   [1s, 2s, 2p]            → AOs [1s, 2s, 2py, 2pz, 2px]
  pyscf  O:   [1s, 2s, 2p]            → AOs [1s, 2s, 2py, 2pz, 2px]

For each AO i in vibe-qc layout, find its index perm[i] in pyscf layout.
Then verify P^T S_v P ≈ S_p elementwise.

Also try sign flips on each shell to absorb any libint vs libcint
spherical-harmonic sign differences.
"""
from __future__ import annotations

import numpy as np

import vibeqc as vq
from vibeqc.periodic_v_ne import compute_nuclear_lattice_dispatch

from pyscf.pbc import gto as pbc_gto, scf as pbc_scf
from pyscf.pbc import df as pbc_df

ANG2BOHR = 1.0 / 0.529177210903
A_ANG = 4.211


def build_systems(cutoff_bohr=24.0):
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


def vibeqc_S_T_V(sys_p, basis, lat_opts):
    k_gamma = np.zeros(3)
    S_lat = vq.compute_overlap_lattice(basis, sys_p, lat_opts)
    T_lat = vq.compute_kinetic_lattice(basis, sys_p, lat_opts)
    V_lat = compute_nuclear_lattice_dispatch(basis, sys_p, lat_opts)
    S = np.real(vq.bloch_sum(S_lat, k_gamma))
    T = np.real(vq.bloch_sum(T_lat, k_gamma))
    V = np.real(vq.bloch_sum(V_lat, k_gamma))
    return 0.5 * (S + S.T), 0.5 * (T + T.T), 0.5 * (V + V.T)


def pyscf_S_T_V(cell):
    mf = pbc_scf.RHF(cell)
    mf.exxdiv = "ewald"
    mf.with_df = pbc_df.GDF(cell)
    mf.with_df.build()
    S = mf.get_ovlp()
    T = cell.pbc_intor("int1e_kin", hermi=1)
    Hcore = mf.get_hcore()
    return S, T, Hcore - T, mf


# Per-atom permutation: vibeqc index → pyscf index within the atom block.
# Mg (9 AOs):
#   vibeqc layout [1s, 2s, 2py, 2pz, 2px, 3s, 3py, 3pz, 3px]
#   pyscf  layout [1s, 2s, 3s,  2py, 2pz, 2px, 3py, 3pz, 3px]
# So vibeqc[5]=3s lives at pyscf[2]=3s; vibeqc[2]=2py lives at pyscf[3]=2py; etc.
PERM_MG = [0, 1, 3, 4, 5, 2, 6, 7, 8]
# O (5 AOs): identical layout in both codes
PERM_O = [0, 1, 2, 3, 4]

# Atom counts per element (matches build order: 4 Mg, 4 O)
ATOM_LIST = [("Mg", 9, PERM_MG)] * 4 + [("O", 5, PERM_O)] * 4


def build_permutation(atom_list):
    """Return P (n × n) such that for vibeqc-AO vector v, P @ v gives
    the AO vector in pyscf ordering."""
    n = sum(nbf for _, nbf, _ in atom_list)
    P = np.zeros((n, n))
    voff = 0
    poff = 0
    for el, nbf, perm in atom_list:
        for i, p in enumerate(perm):
            # vibeqc index voff+i -> pyscf index poff+p
            P[poff + p, voff + i] = 1.0
        voff += nbf
        poff += nbf
    return P


def main():
    # Use cutoff_bohr=12 to keep this spike fast; the per-element
    # comparison cares about the basis-ordering pattern, not the
    # absolute lattice convergence (handled separately).
    sys_p, basis, lat_opts, cell = build_systems(cutoff_bohr=12.0)
    Sv, Tv, Vv = vibeqc_S_T_V(sys_p, basis, lat_opts)
    Sp, Tp, Vp, _ = pyscf_S_T_V(cell)

    P = build_permutation(ATOM_LIST)
    print(f"Permutation P shape {P.shape}, det={np.linalg.det(P):+.0f}")

    for name, Mv, Mp in [("S", Sv, Sp), ("T", Tv, Tp), ("V_ne", Vv, Vp)]:
        Mv_in_pyscf_order = P @ Mv @ P.T
        diff = Mv_in_pyscf_order - Mp
        max_ad = float(np.max(np.abs(diff)))
        fro_rel = float(np.linalg.norm(diff) / np.linalg.norm(Mp))
        print(f"  {name:>5s}  max|elem diff| = {max_ad:.3e}   "
              f"||Δ||/||M|| = {fro_rel:.3e}")
        if max_ad > 1e-6:
            # Find which (i, j) has the largest difference
            i, j = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
            print(f"          worst element ({i}, {j}): "
                  f"vibeqc(perm)={Mv_in_pyscf_order[i,j]:+.6f}  "
                  f"pyscf={Mp[i,j]:+.6f}")

    # If pure permutation isn't enough, try with per-shell sign flips.
    print()
    print("Trying per-shell sign flips for the p-shells (libint vs libcint "
          "may differ in m ordering or sign):")
    # Within each p-shell, libint's m=(−1,0,+1) and libcint's m=(−1,0,+1)
    # nominally agree, but the real-spherical convention can differ.
    # Try (+1, +1, +1), (+1, −1, +1), etc., on Mg's two p shells and O's
    # one p shell.

    # Collect AO ranges of each p-shell in vibe-qc layout (after applying P
    # to convert to pyscf layout). The per-element mismatches will reveal
    # which p-shell has a sign issue.

    # First, find rows where Mv_in_pyscf_order disagrees with Mp.
    Sv_pyscf_order = P @ Sv @ P.T
    diff = Sv_pyscf_order - Sp
    # Print diagonal differences
    print("  Diag differences in S (after permutation):")
    diag_diff = np.diag(diff)
    for i in range(min(20, len(diag_diff))):
        if abs(diag_diff[i]) > 1e-9:
            print(f"    diag[{i}] = vq:{Sv_pyscf_order[i,i]:+.6f}  py:{Sp[i,i]:+.6f}  Δ={diag_diff[i]:+.3e}")


if __name__ == "__main__":
    main()
