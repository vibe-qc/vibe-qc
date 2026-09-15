"""Quick standalone Al2O3-cubic RHF parity test of the new GDF driver."""
from __future__ import annotations

import time

import numpy as np

import vibeqc as vq
from vibeqc.periodic_rhf_gdf import run_rhf_periodic_gamma_gdf
from pyscf.pbc import gto as pbc_gto, scf as pbc_scf, df as pbc_df

ANG2BOHR = 1.0 / 0.529177210903
A_ANG = 5.05


def build_al2o3():
    a = A_ANG * ANG2BOHR
    al_frac = [(0.25, 0.25, 0.25), (0.75, 0.75, 0.75)]
    o_frac = [(0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]
    cell_atoms = []
    for fx, fy, fz in al_frac:
        cell_atoms.append(vq.Atom(13, [fx * a, fy * a, fz * a]))
    for fx, fy, fz in o_frac:
        cell_atoms.append(vq.Atom(8, [fx * a, fy * a, fz * a]))
    sys_p = vq.PeriodicSystem(3, np.diag([a, a, a]), cell_atoms)
    basis = vq.BasisSet(sys_p.unit_cell_molecule(), "sto-3g")
    return sys_p, basis, al_frac, o_frac


def pyscf_ref(al_frac, o_frac):
    atom_lines = []
    for fx, fy, fz in al_frac:
        atom_lines.append(f"Al {fx*A_ANG:.6f} {fy*A_ANG:.6f} {fz*A_ANG:.6f}")
    for fx, fy, fz in o_frac:
        atom_lines.append(f"O  {fx*A_ANG:.6f} {fy*A_ANG:.6f} {fz*A_ANG:.6f}")
    cell = pbc_gto.M(atom="; ".join(atom_lines),
                     a=[[A_ANG, 0, 0], [0, A_ANG, 0], [0, 0, A_ANG]],
                     basis="sto-3g", unit="A", verbose=0,
                     mesh=[31, 31, 31], precision=1e-8)
    mf = pbc_scf.RHF(cell)
    mf.exxdiv = "ewald"
    mf.with_df = pbc_df.GDF(cell)
    mf.with_df.build()
    mf.max_cycle = 60
    mf.conv_tol = 1e-7
    return float(mf.kernel()), mf.converged


def main():
    sys_p, basis, al_frac, o_frac = build_al2o3()
    print(f"Al2O3-cubic sto-3g: nbasis = {basis.nbasis}, "
          f"n_electrons = {sys_p.n_electrons()}")

    opts = vq.PeriodicRHFOptions()
    opts.max_iter = 60
    opts.use_diis = True
    opts.damping = 0.85
    opts.conv_tol_energy = 1e-7
    opts.initial_guess = vq.InitialGuess.SAD

    t0 = time.perf_counter()
    r = run_rhf_periodic_gamma_gdf(sys_p, basis, opts, progress=True)
    wall = time.perf_counter() - t0

    e_pyscf, conv_pyscf = pyscf_ref(al_frac, o_frac)
    print()
    print(f"vibeqc:  E = {r.energy:.6f}  converged={r.converged}  iters={r.n_iter}  wall={wall:.1f}s")
    print(f"pyscf :  E = {e_pyscf:.6f}  converged={conv_pyscf}")
    dE = r.energy - e_pyscf
    print(f"ΔE = {dE:+.4e} Ha")
    if abs(dE) < 1e-3:
        print("✓ PARITY (|ΔE| < 1 mHa)")


if __name__ == "__main__":
    main()
