"""RHF-only version of mgo_al2o3_parity.py — uses the new GDF driver.

The full parity test still runs RKS-PBE on the LEGACY (broken)
``run_rks_periodic_gamma_ewald3d`` driver, which makes it slow
(~1-2 hours per RKS-PBE arm on MgO/Al2O3). This script restricts
to the RHF arm to give a fast pass/fail for the v0.7.1 milestone.

Output format mirrors mgo_al2o3_parity.py so the verdict is
directly comparable.
"""
from __future__ import annotations

import time

import numpy as np

import vibeqc as vq
from vibeqc.periodic_rhf_gdf import run_rhf_periodic_gamma_gdf

from pyscf.pbc import gto as pbc_gto, scf as pbc_scf, df as pbc_df

ANG2BOHR = 1.0 / 0.529177210903


def mgo_rocksalt(a_ang: float = 4.211):
    a = a_ang * ANG2BOHR
    mg_frac = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    o_frac = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
    cell_atoms = []
    for fx, fy, fz in mg_frac:
        cell_atoms.append(vq.Atom(12, [fx*a, fy*a, fz*a]))
    for fx, fy, fz in o_frac:
        cell_atoms.append(vq.Atom(8, [fx*a, fy*a, fz*a]))
    sys_p = vq.PeriodicSystem(3, np.diag([a, a, a]), cell_atoms)
    pyscf_atoms = []
    for fx, fy, fz in mg_frac:
        pyscf_atoms.append(("Mg", fx*a_ang, fy*a_ang, fz*a_ang))
    for fx, fy, fz in o_frac:
        pyscf_atoms.append(("O", fx*a_ang, fy*a_ang, fz*a_ang))
    return "MgO-rocksalt", sys_p, pyscf_atoms, a_ang


def al2o3_cubic(a_ang: float = 5.05):
    a = a_ang * ANG2BOHR
    al_frac = [(0.25, 0.25, 0.25), (0.75, 0.75, 0.75)]
    o_frac = [(0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]
    cell_atoms = []
    for fx, fy, fz in al_frac:
        cell_atoms.append(vq.Atom(13, [fx*a, fy*a, fz*a]))
    for fx, fy, fz in o_frac:
        cell_atoms.append(vq.Atom(8, [fx*a, fy*a, fz*a]))
    sys_p = vq.PeriodicSystem(3, np.diag([a, a, a]), cell_atoms)
    pyscf_atoms = []
    for fx, fy, fz in al_frac:
        pyscf_atoms.append(("Al", fx*a_ang, fy*a_ang, fz*a_ang))
    for fx, fy, fz in o_frac:
        pyscf_atoms.append(("O", fx*a_ang, fy*a_ang, fz*a_ang))
    return "Al2O3-cubic", sys_p, pyscf_atoms, a_ang


def run_vibeqc(label, system):
    opts = vq.PeriodicRHFOptions()
    opts.max_iter = 60
    opts.use_diis = True
    opts.damping = 0.85
    opts.conv_tol_energy = 1e-7
    opts.initial_guess = vq.InitialGuess.SAD
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    t0 = time.perf_counter()
    try:
        r = run_rhf_periodic_gamma_gdf(system, basis, opts, progress=False)
        wall = time.perf_counter() - t0
        return dict(energy=r.energy, converged=r.converged,
                    n_iter=r.n_iter, wall=wall, note="ok")
    except Exception as e:
        wall = time.perf_counter() - t0
        return dict(energy=None, converged=False, n_iter=None, wall=wall,
                    note=f"{type(e).__name__}: {str(e)[:100]}")


def run_pyscf(label, pyscf_atoms, a_ang):
    atom_str = "; ".join(f"{el} {x:.6f} {y:.6f} {z:.6f}"
                         for (el, x, y, z) in pyscf_atoms)
    cell = pbc_gto.M(atom=atom_str,
                     a=[[a_ang, 0, 0], [0, a_ang, 0], [0, 0, a_ang]],
                     basis="sto-3g", unit="A", verbose=0,
                     mesh=[31, 31, 31], precision=1e-8)
    mf = pbc_scf.RHF(cell)
    mf.exxdiv = "ewald"
    mf.with_df = pbc_df.GDF(cell)
    mf.with_df.build()
    mf.max_cycle = 60
    mf.conv_tol = 1e-7
    t0 = time.perf_counter()
    e = mf.kernel()
    wall = time.perf_counter() - t0
    return dict(energy=float(e), converged=bool(mf.converged),
                n_iter=int(getattr(mf, "cycles", 0)),
                wall=wall, note="ok")


def main():
    print("=" * 80)
    print("  RHF-only parity (vibe-qc periodic_rhf_gdf vs PySCF.pbc.GDF)")
    print("=" * 80)
    cases = [mgo_rocksalt(), al2o3_cubic()]
    rows = []
    for label, sys_p, pyscf_atoms, a_ang in cases:
        print()
        print(f"  >>> {label}  (a = {a_ang:.3f} Å, "
              f"{len(sys_p.unit_cell)} atoms)")
        v = run_vibeqc(label, sys_p)
        flag = "✓" if v["converged"] else "✗"
        print(f"      vibeqc {flag}  iter={v['n_iter']}  E={v['energy']}  "
              f"wall={v['wall']:.1f}s  {v['note']}")
        p = run_pyscf(label, pyscf_atoms, a_ang)
        flag = "✓" if p["converged"] else "✗"
        print(f"      pyscf  {flag}  iter={p['n_iter']}  E={p['energy']}  "
              f"wall={p['wall']:.1f}s  {p['note']}")
        rows.append((label, v, p))

    print()
    print("=" * 92)
    print("  Summary")
    print("=" * 92)
    print(f"  {'system':>16s}  {'method':>9s}  {'vibeqc E':>14s}  "
          f"{'pyscf E':>14s}  {'ΔE (Ha)':>12s}  verdict")
    print("  " + "-" * 90)
    for label, v, p in rows:
        if v["energy"] is None or p["energy"] is None:
            verdict = "fail"
            de_str = " " * 12
        else:
            de = v["energy"] - p["energy"]
            de_str = f"{de:+12.4e}"
            if abs(de) < 1e-3:
                verdict = "✓ PARITY"
            elif abs(de) < 1.0:
                verdict = "~ off by mHa-Ha"
            else:
                verdict = "✗ BUG"
        print(f"  {label:>16s}  {'RHF':>9s}  {v['energy']:>14.6f}  "
              f"{p['energy']:>14.6f}  {de_str}  {verdict}")


if __name__ == "__main__":
    main()
