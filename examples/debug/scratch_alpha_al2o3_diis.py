"""Same α-Al2O3 corundum test as scratch_alpha_al2o3.py, but with DIIS
turned on (no damping). Compare iter count vs plain Roothaan.
"""
from __future__ import annotations

import time

import numpy as np

from ase.spacegroup import crystal as ase_crystal

import vibeqc as vq
from vibeqc.periodic_rhf_gdf import run_rhf_periodic_gamma_gdf
from pyscf.pbc import gto as pbc_gto, scf as pbc_scf, df as pbc_df

ANG2BOHR = 1.0 / 0.529177210903
A_ANG = 4.7589
C_ANG = 12.991
AL_Z = 0.35216
O_X = 0.30624


def build_corundum_ase():
    return ase_crystal(
        ["Al", "O"],
        basis=[(0, 0, AL_Z), (O_X, 0, 0.25)],
        spacegroup=167,
        cellpar=[A_ANG, A_ANG, C_ANG, 90, 90, 120],
    )


def build_vibeqc(atoms_ase):
    cell_bohr = atoms_ase.cell.array * ANG2BOHR
    cell_atoms = []
    Z_BY_SYMBOL = {"Al": 13, "O": 8}
    for sym, pos_ang in zip(atoms_ase.get_chemical_symbols(),
                            atoms_ase.get_positions()):
        cell_atoms.append(vq.Atom(Z_BY_SYMBOL[sym], list(pos_ang * ANG2BOHR)))
    sys_p = vq.PeriodicSystem(3, cell_bohr, cell_atoms)
    basis = vq.BasisSet(sys_p.unit_cell_molecule(), "sto-3g")
    return sys_p, basis


def pyscf_ref(atoms_ase):
    atom_str = "; ".join(
        f"{sym} {x:.6f} {y:.6f} {z:.6f}"
        for sym, (x, y, z) in zip(atoms_ase.get_chemical_symbols(),
                                  atoms_ase.get_positions())
    )
    cell = pbc_gto.M(atom=atom_str, a=atoms_ase.cell.array,
                     basis="sto-3g", unit="A", verbose=0, precision=1e-8)
    mf = pbc_scf.RHF(cell)
    mf.exxdiv = "ewald"
    mf.with_df = pbc_df.GDF(cell)
    mf.with_df.build()
    mf.max_cycle = 200
    mf.conv_tol = 1e-7
    return float(mf.kernel()), bool(mf.converged)


def main():
    atoms = build_corundum_ase()
    sys_p, basis = build_vibeqc(atoms)
    print(f"α-Al2O3 corundum: {len(atoms)} atoms, {basis.nbasis} AOs, "
          f"{sys_p.n_electrons()} e")

    opts = vq.PeriodicRHFOptions()
    opts.max_iter = 80
    opts.use_diis = True          # DIIS ON
    opts.damping = 0.0            # no damping
    opts.diis_start_iter = 2      # default
    opts.diis_subspace_size = 8   # default
    opts.conv_tol_energy = 1e-7
    opts.initial_guess = vq.InitialGuess.SAD

    print()
    print("=" * 80)
    print(f"  α-Al2O3 with DIIS (start iter {opts.diis_start_iter}, "
          f"subspace {opts.diis_subspace_size}), no damping")
    print("=" * 80)
    t0 = time.perf_counter()
    r = run_rhf_periodic_gamma_gdf(sys_p, basis, opts, progress=True)
    wall = time.perf_counter() - t0

    print()
    print("Computing PySCF reference...")
    e_pyscf, conv_pyscf = pyscf_ref(atoms)

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


if __name__ == "__main__":
    main()
