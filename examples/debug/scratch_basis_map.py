"""Spike: establish the AO-ordering permutation between vibe-qc (libint)
and PySCF (libcint) on MgO sto-3g, Γ-only.

Why
---
The handover says vibe-qc S/T/V_ne match PySCF to <1e-5 Ha (basis-
independent traces). To use PySCF's J/K matrices as a reference for
vibe-qc's SCF iter-1 parity check, we need a per-element comparison —
that requires knowing the permutation P such that

    M_vibeqc = P^T  M_pyscf  P     (or vice versa)

Once we know P, we can compare arbitrary matrices elementwise.

Plan
----
1. Build the same MgO sto-3g cell in both codes (identical geometry,
   identical basis, identical Γ).
2. Compute S, T, V_ne in vibe-qc (Bloch-summed at Γ, real part).
3. Compute S, T, V_ne in PySCF (cell.pbc_intor + mf.get_hcore).
4. Eigenvalues should match exactly (basis-independent). Confirm.
5. Pair off shells: same atom, same l, contiguous block. Within an
   l-shell, the m ordering may differ; figure it out by matching a
   single matrix element pattern (ideally on a non-degenerate
   eigenvalue) or by testing a known L=1 convention.

Output
------
- Print eigenvalue match for S, T, V_ne (sanity).
- Print first 8 diagonal elements of S in both bases.
- Print shell list metadata for both codes.
- Save full S, T, V_ne matrices for both codes to npz for inspection.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import vibeqc as vq
from vibeqc.periodic_v_ne import compute_nuclear_lattice_dispatch

from pyscf.pbc import gto as pbc_gto, scf as pbc_scf
from pyscf.pbc import df as pbc_df

ANG2BOHR = 1.0 / 0.529177210903
A_ANG = 4.211

OUT = Path(__file__).resolve().parent / "output" / "scratch"
OUT.mkdir(parents=True, exist_ok=True)


def build_vibeqc():
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
    return sys_p, basis, mg_frac, o_frac, a


def build_pyscf(mg_frac, o_frac, a_ang):
    atom_lines = []
    for fx, fy, fz in mg_frac:
        atom_lines.append(f"Mg {fx*a_ang:.6f} {fy*a_ang:.6f} {fz*a_ang:.6f}")
    for fx, fy, fz in o_frac:
        atom_lines.append(f"O  {fx*a_ang:.6f} {fy*a_ang:.6f} {fz*a_ang:.6f}")
    cell = pbc_gto.M(
        atom="; ".join(atom_lines),
        a=[[a_ang, 0, 0], [0, a_ang, 0], [0, 0, a_ang]],
        basis="sto-3g",
        unit="A",
        verbose=0,
        mesh=[31, 31, 31],
        precision=1e-8,
    )
    return cell


def vibeqc_one_electron(sys_p, basis):
    lat_opts = vq.LatticeSumOptions()
    lat_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    lat_opts.cutoff_bohr = 12.0
    lat_opts.nuclear_cutoff_bohr = 25.0
    k_gamma = np.zeros(3)
    S_lat = vq.compute_overlap_lattice(basis, sys_p, lat_opts)
    T_lat = vq.compute_kinetic_lattice(basis, sys_p, lat_opts)
    V_lat = compute_nuclear_lattice_dispatch(basis, sys_p, lat_opts)
    S = np.real(vq.bloch_sum(S_lat, k_gamma))
    T = np.real(vq.bloch_sum(T_lat, k_gamma))
    V = np.real(vq.bloch_sum(V_lat, k_gamma))
    S = 0.5 * (S + S.T)
    T = 0.5 * (T + T.T)
    V = 0.5 * (V + V.T)
    return S, T, V, lat_opts


def pyscf_one_electron(cell):
    mf = pbc_scf.RHF(cell)
    mf.exxdiv = "ewald"
    mf.with_df = pbc_df.GDF(cell)
    mf.with_df.build()
    S = mf.get_ovlp()
    T = cell.pbc_intor("int1e_kin", hermi=1)
    Hcore = mf.get_hcore()
    V_ne = Hcore - T
    return S, T, V_ne, mf


def shell_metadata_vibeqc(basis, sys_p):
    # Walk the BasisSet and report per-shell (atom_idx, l, contracted size).
    info = []
    sh = list(basis.shells())
    for i, s in enumerate(sh):
        # ShellInfo fields: l, atom_index (or position), nbf
        info.append((i, getattr(s, "l", "?"),
                     getattr(s, "atom_index", "?"),
                     getattr(s, "nbf", "?")))
    return info


def shell_metadata_pyscf(cell):
    info = []
    for ish in range(cell.nbas):
        l = cell.bas_angular(ish)
        atm_id = cell.bas_atom(ish)
        nctr = cell.bas_nctr(ish)
        nbf = nctr * (2 * l + 1)
        info.append((ish, l, atm_id, nbf))
    return info


def main():
    sys_p, basis, mg_frac, o_frac, a_bohr = build_vibeqc()
    cell = build_pyscf(mg_frac, o_frac, A_ANG)

    print(f"vibe-qc nbf = {basis.nbasis}, pyscf nao = {cell.nao}")
    print()

    Sv, Tv, Vv, lat_opts = vibeqc_one_electron(sys_p, basis)
    Sp, Tp, Vp, mf = pyscf_one_electron(cell)

    # Eigenvalue compare (basis-invariant)
    for name, Mv, Mp in [("S", Sv, Sp), ("T", Tv, Tp), ("V_ne", Vv, Vp)]:
        ev_v = np.sort(np.linalg.eigvalsh(Mv))
        ev_p = np.sort(np.linalg.eigvalsh(Mp))
        max_ad = float(np.max(np.abs(ev_v - ev_p)))
        max_rel = float(np.max(np.abs(ev_v - ev_p) /
                               np.maximum(np.abs(ev_p), 1e-12)))
        verdict = "OK" if max_ad < 1e-3 else "MISMATCH"
        print(f"  {name:>5s}  max|Δλ|={max_ad:.3e}  "
              f"max|rel|={max_rel:.3e}  [{verdict}]")
        print(f"          first 6 vibeqc: {ev_v[:6]}")
        print(f"          first 6 pyscf : {ev_p[:6]}")
        print(f"          last  6 vibeqc: {ev_v[-6:]}")
        print(f"          last  6 pyscf : {ev_p[-6:]}")

    print()
    print("Shell metadata vibe-qc (i, l, atom, nbf):")
    for row in shell_metadata_vibeqc(basis, sys_p):
        print(f"  {row}")
    print()
    print("Shell metadata pyscf (i, l, atom, nbf):")
    for row in shell_metadata_pyscf(cell):
        print(f"  {row}")

    print()
    print("Diag S vibeqc (first 16):", np.diag(Sv)[:16])
    print("Diag S pyscf  (first 16):", np.diag(Sp)[:16])

    # Save matrices for offline inspection
    np.savez(OUT / "mgo_one_e.npz",
             Sv=Sv, Tv=Tv, Vv=Vv,
             Sp=Sp, Tp=Tp, Vp=Vp)
    print()
    print(f"Saved → {OUT / 'mgo_one_e.npz'}")


if __name__ == "__main__":
    main()
