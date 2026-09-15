"""Test build_fock_2e_real_space (DIRECT method) vs PySCF on MgO RHF.

Plan:
  1. Build MgO sto-3g periodic system in vibe-qc.
  2. Get SAD density D (nbf × nbf, vibe-qc AO order).
  3. Wrap D as a degenerate LatticeMatrixSet (block 0 = D, others = 0).
  4. Call build_fock_2e_real_space(basis, system, opts, P_real_space,
     exchange_scale=1.0, omega=0.0) → LatticeMatrixSet of F_2e blocks.
  5. Bloch-sum at Γ to get Γ-point F_2e = J - ½ K.
  6. Compare to PySCF's mf.get_jk on the same density (after AO permutation).

If F_2e matches PySCF's J - ½ K (with Madelung correction) to ~µHa,
DIRECT is correct and just needs wiring into the dispatch.

Run with cutoff_bohr=18 to ensure lattice convergence on MgO sto-3g.
"""
from __future__ import annotations

import numpy as np

import vibeqc as vq
from vibeqc.periodic_rhf_gdf import (
    _build_permutation_matrix,
    _pyscf_cell_from_vibeqc,
)
from pyscf.pbc import scf as pbc_scf, df as pbc_df
from pyscf.pbc.tools import madelung as pyscf_madelung


ANG2BOHR = 1.0 / 0.529177210903
A_ANG = 4.211
CUTOFF = 18.0


def build_mgo():
    a = A_ANG * ANG2BOHR
    mg_frac = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    o_frac = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
    cell = []
    for fx, fy, fz in mg_frac:
        cell.append(vq.Atom(12, [fx*a, fy*a, fz*a]))
    for fx, fy, fz in o_frac:
        cell.append(vq.Atom(8, [fx*a, fy*a, fz*a]))
    sys_p = vq.PeriodicSystem(3, np.diag([a, a, a]), cell)
    basis = vq.BasisSet(sys_p.unit_cell_molecule(), "sto-3g")
    return sys_p, basis


def main():
    sys_p, basis = build_mgo()
    nbf = basis.nbasis
    print(f"MgO sto-3g: nbf = {nbf}, n_e = {sys_p.n_electrons()}")

    lat_opts = vq.LatticeSumOptions()
    lat_opts.coulomb_method = vq.CoulombMethod.DIRECT_TRUNCATED
    lat_opts.cutoff_bohr = CUTOFF
    lat_opts.nuclear_cutoff_bohr = max(25.0, 2.0 * CUTOFF)
    lat_opts.schwarz_threshold = 1e-12

    # SAD density (vibe-qc AO order)
    D_vq = np.asarray(vq.sad_density(sys_p.unit_cell_molecule(), basis))
    D_vq = 0.5 * (D_vq + D_vq.T)
    print(f"⟨D|S⟩(molecular) = {float(np.einsum('ij,ij->', D_vq, vq.compute_overlap(basis))):.4f}")

    # Wrap D as a degenerate LatticeMatrixSet (block 0 = D, others = 0).
    D_set = vq.compute_overlap_lattice(basis, sys_p, lat_opts)
    zero = np.zeros((nbf, nbf))
    for i in range(len(D_set)):
        D_set.set_block(i, D_vq if i == 0 else zero)

    # build_fock_2e_real_space — full Coulomb (omega=0), exchange_scale=1.
    # This returns F_2e (LatticeMatrixSet) = J - ½ K per cell.
    print("Calling build_fock_2e_real_space (this may take a while)...")
    import time
    t0 = time.perf_counter()
    F2_set = vq.build_fock_2e_real_space(
        basis, sys_p, lat_opts, D_set,
        1.0, 0.0,   # exchange_scale=1, omega=0
    )
    print(f"  done in {time.perf_counter()-t0:.1f}s")

    # Bloch-sum at Γ
    F2_gamma = np.real(vq.bloch_sum(F2_set, np.zeros(3)))
    F2_gamma = 0.5 * (F2_gamma + F2_gamma.T)

    # PySCF reference (cell built before P so we can pass it to perm)
    cell = _pyscf_cell_from_vibeqc(sys_p, "sto-3g", mesh=[31, 31, 31])
    mf = pbc_scf.RHF(cell)
    mf.exxdiv = "ewald"
    mf.with_df = pbc_df.GDF(cell)
    mf.with_df.build()

    # Permute D to PySCF AO order
    P = _build_permutation_matrix(sys_p, basis, cell)
    D_py = P @ D_vq @ P.T
    Jp, Kp = mf.get_jk(cell, D_py, hermi=1)
    F2_py = Jp - 0.5 * Kp     # PySCF's get_jk includes Madelung in K

    # Convert F2_py back to vibe-qc AO order for elementwise compare
    F2_py_vq = P.T @ F2_py @ P

    diff = F2_gamma - F2_py_vq
    print()
    print(f"max|F2_gamma - F2_py_vq|        = {np.max(np.abs(diff)):.3e}")
    print(f"||diff||/||F2_py||              = {np.linalg.norm(diff)/np.linalg.norm(F2_py_vq):.3e}")

    # Also: subtract the Madelung correction. PySCF's K has
    #   K_corrected = K_DF + madelung * S · D · S
    # so PySCF F_2e includes -0.5 * madelung * S D S that the DIRECT
    # path doesn't. Let's add it back to DIRECT for fair compare.
    madelung = float(pyscf_madelung(cell, np.zeros(3)))
    Sp = mf.get_ovlp()
    correction_py_basis = -0.5 * madelung * (Sp @ D_py @ Sp)
    correction_vq_basis = P.T @ correction_py_basis @ P
    F2_gamma_with_madelung = F2_gamma + correction_vq_basis
    diff2 = F2_gamma_with_madelung - F2_py_vq
    print(f"After Madelung correction added to DIRECT:")
    print(f"  max|diff| = {np.max(np.abs(diff2)):.3e}")
    print(f"  ||diff||/||F2_py|| = {np.linalg.norm(diff2)/np.linalg.norm(F2_py_vq):.3e}")


if __name__ == "__main__":
    main()
