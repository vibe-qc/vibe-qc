"""Test the new periodic_rhf_gdf driver on MgO and compare to PySCF.

Targets:
  - Convergence: driver must converge in <= 60 iters
  - Parity: |E_vibeqc - E_pyscf| < 1 mHa, where E_pyscf = -1085.231546

This test bypasses the official ``mgo_al2o3_parity.py`` so we can
debug the new driver in isolation before swapping the dispatch.
"""
from __future__ import annotations

import time

import numpy as np

import vibeqc as vq
from vibeqc.periodic_rhf_gdf import run_rhf_periodic_gamma_gdf

ANG2BOHR = 1.0 / 0.529177210903
A_ANG = 4.211
PYSCF_E_REF = -1085.231546


def build_mgo():
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
    return sys_p, basis


def main():
    sys_p, basis = build_mgo()
    print(f"MgO sto-3g: nbasis = {basis.nbasis}, n_electrons = {sys_p.n_electrons()}")

    opts = vq.PeriodicRHFOptions()
    opts.max_iter = 60
    opts.use_diis = True
    opts.damping = 0.85
    opts.conv_tol_energy = 1e-7
    opts.initial_guess = vq.InitialGuess.SAD

    t0 = time.perf_counter()
    r = run_rhf_periodic_gamma_gdf(
        sys_p, basis, opts,
        progress=True,
    )
    wall = time.perf_counter() - t0

    print()
    print(f"Result: E = {r.energy:.6f}  converged={r.converged}  "
          f"iters={r.n_iter}  wall={wall:.1f}s")
    print(f"PySCF reference (GDF, exxdiv=ewald): E = {PYSCF_E_REF:.6f}")
    dE = r.energy - PYSCF_E_REF
    print(f"ΔE = {dE:+.4e} Ha")

    if abs(dE) < 1e-3:
        print("✓ PARITY (|ΔE| < 1 mHa)")
    elif abs(dE) < 1.0:
        print("~ off by mHa-Ha")
    else:
        print("✗ BUG (large mismatch)")


if __name__ == "__main__":
    main()
