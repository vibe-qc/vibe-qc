"""Plain Roothaan SCF — no DIIS, no damping — on MgO and Al2O3.

We want to see the bare SCF behaviour on these ionic cells using the
new GDF driver. Expectation: deep-core ionic cells without
acceleration usually oscillate.

Prints every iteration to stdout.
"""
from __future__ import annotations

import time

import numpy as np

import vibeqc as vq
from vibeqc.periodic_rhf_gdf import run_rhf_periodic_gamma_gdf
from pyscf.pbc import gto as pbc_gto, scf as pbc_scf, df as pbc_df

ANG2BOHR = 1.0 / 0.529177210903


def mgo(a_ang=4.211):
    a = a_ang * ANG2BOHR
    mg_frac = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    o_frac = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
    cell = []
    for fx, fy, fz in mg_frac:
        cell.append(vq.Atom(12, [fx*a, fy*a, fz*a]))
    for fx, fy, fz in o_frac:
        cell.append(vq.Atom(8, [fx*a, fy*a, fz*a]))
    sys_p = vq.PeriodicSystem(3, np.diag([a, a, a]), cell)
    pyscf_atoms = []
    for fx, fy, fz in mg_frac:
        pyscf_atoms.append(("Mg", fx*a_ang, fy*a_ang, fz*a_ang))
    for fx, fy, fz in o_frac:
        pyscf_atoms.append(("O", fx*a_ang, fy*a_ang, fz*a_ang))
    return "MgO-rocksalt", sys_p, pyscf_atoms, a_ang


def al2o3(a_ang=5.05):
    a = a_ang * ANG2BOHR
    al_frac = [(0.25, 0.25, 0.25), (0.75, 0.75, 0.75)]
    o_frac = [(0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]
    cell = []
    for fx, fy, fz in al_frac:
        cell.append(vq.Atom(13, [fx*a, fy*a, fz*a]))
    for fx, fy, fz in o_frac:
        cell.append(vq.Atom(8, [fx*a, fy*a, fz*a]))
    sys_p = vq.PeriodicSystem(3, np.diag([a, a, a]), cell)
    pyscf_atoms = []
    for fx, fy, fz in al_frac:
        pyscf_atoms.append(("Al", fx*a_ang, fy*a_ang, fz*a_ang))
    for fx, fy, fz in o_frac:
        pyscf_atoms.append(("O", fx*a_ang, fy*a_ang, fz*a_ang))
    return "Al2O3-cubic", sys_p, pyscf_atoms, a_ang


def pyscf_ref(pyscf_atoms, a_ang):
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
    mf.max_cycle = 100
    mf.conv_tol = 1e-7
    return float(mf.kernel()), bool(mf.converged)


def run(label, sys_p, pyscf_atoms, a_ang, max_iter=80):
    print()
    print("=" * 80)
    print(f"  {label}  (plain Roothaan: no DIIS, no damping)")
    print("=" * 80)
    basis = vq.BasisSet(sys_p.unit_cell_molecule(), "sto-3g")

    opts = vq.PeriodicRHFOptions()
    opts.max_iter = max_iter
    opts.use_diis = False         # no DIIS
    opts.damping = 0.0            # no damping
    opts.conv_tol_energy = 1e-7
    opts.initial_guess = vq.InitialGuess.SAD

    t0 = time.perf_counter()
    r = run_rhf_periodic_gamma_gdf(sys_p, basis, opts, progress=True)
    wall = time.perf_counter() - t0

    e_pyscf, conv_pyscf = pyscf_ref(pyscf_atoms, a_ang)
    flag = "✓" if r.converged else "✗"
    print()
    print(f"  vibeqc {flag}  iter={r.n_iter}  E={r.energy:.6f}  wall={wall:.1f}s")
    print(f"  pyscf  {'✓' if conv_pyscf else '✗'}  E={e_pyscf:.6f}")
    if r.energy is not None:
        dE = r.energy - e_pyscf
        verdict = ("✓ PARITY" if abs(dE) < 1e-3
                   else "~ off by mHa-Ha" if abs(dE) < 1.0
                   else "✗ BUG")
        print(f"  ΔE = {dE:+.4e} Ha   {verdict}")


def main():
    run(*mgo(), max_iter=80)
    run(*al2o3(), max_iter=80)


if __name__ == "__main__":
    main()
