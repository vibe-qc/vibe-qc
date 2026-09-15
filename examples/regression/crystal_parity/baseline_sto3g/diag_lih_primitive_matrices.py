"""Stage 2: vibe-qc LiH primitive-cell matrix dump for CRYSTAL parity.

CRYSTAL14 dumped F(G=0) and P(G=0) for the 2-atom LiH FCC primitive
cell, STO-3G, 6 BFs (Li 1s/2s/2p_x/2p_y/2p_z + H 1s).  See job
fce948157c88 in the parity table.

This script builds the SAME 2-atom primitive cell in vibe-qc, runs
gamma-only RHF to convergence, and dumps:
  * S(g=0)  — overlap (k-independent; should match CRYSTAL exactly)
  * T(g=0)  — kinetic (k-independent; should match CRYSTAL exactly)
  * V(g=0)  — nuclear attraction, Ewald gauge (k-independent)
  * F(g=0)  — converged Fock = T+V+J-0.5K (CRYSTAL has this)
  * P(g=0)  — converged density (CRYSTAL has this)
  * J(Γ)    — Coulomb matrix at Γ
  * K(Γ)    — exchange matrix at Γ

The output prints the 6x6 matrices in a CRYSTAL-style format so the
element-by-element comparison against CRYSTAL's F(G=0)/P(G=0) is
direct (modulo basis-ordering reconciliation — see note below).

Basis-ordering reconciliation:
  CRYSTAL14 LiH primitive AOs (verified from F(G=0) diagonal):
    1=Li_1s, 2=Li_2s, 3=Li_2p_x, 4=Li_2p_y, 5=Li_2p_z, 6=H_1s
  vibe-qc/libint STO-3G ordering needs to be confirmed at runtime.
  This script prints the diagonals of T+V (which separate the Li
  shells from H by ~order of magnitude) to make the mapping
  unambiguous.

Runs locally on the Mac — no compute-reference dependency.  6 AOs / 4 electrons
makes everything fast.

Compare to:
  /home/USER/.local/share/vq/jobs/fce948157c88/baseline_sto3g/lih-rhf-sto3g-matdump.out
"""
from __future__ import annotations

import sys

import numpy as np
import vibeqc as vq
from vibeqc._vibeqc_core import (
    LatticeSumOptions, bloch_sum,
    compute_overlap_lattice, compute_kinetic_lattice,
    build_jk_gamma_molecular_limit,
)
from vibeqc.periodic_v_ne import compute_nuclear_lattice_dispatch
from vibeqc.ewald_composed import build_j_ewald_3d
from vibeqc.ewald_j import auto_grid

ANG2BOHR = 1.0 / 0.529177210903


def lih_primitive():
    """LiH rocksalt FCC primitive cell (2 atoms).

    CRYSTAL14 input puts Li at (0,0,0) and H at (1/2,1/2,1/2) in the
    *conventional* cubic cell.  The primitive (FCC) lattice vectors
    in Cartesian are (0,a/2,a/2), (a/2,0,a/2), (a/2,a/2,0) with
    a = 4.084 Å.  Li at origin, H at the body-center of the
    conventional cell which in the primitive coords corresponds to
    Cartesian (a/2, a/2, a/2).
    """
    a_ang = 4.084
    a = a_ang * ANG2BOHR
    L = np.array([
        [0.0, a / 2, a / 2],
        [a / 2, 0.0, a / 2],
        [a / 2, a / 2, 0.0],
    ])
    atoms = [
        vq.Atom(3, [0.0, 0.0, 0.0]),                  # Li
        vq.Atom(1, [a / 2, a / 2, a / 2]),            # H at conv-cell body center
    ]
    system = vq.PeriodicSystem(3, L, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def fmt_matrix(M, label, fmt="{:>10.4f}"):
    """Print a square matrix in a readable form."""
    n = M.shape[0]
    print(f"\n{label}  ({n}x{n})")
    print("        " + "".join(f"{i+1:>10d}" for i in range(n)))
    for i in range(n):
        row = "  " + f"{i+1:>3d}  " + "".join(fmt.format(M[i, j]) for j in range(n))
        print(row)


def main():
    print(f"vibeqc {vq.__version__} — LiH FCC primitive matrix dump @ Γ", flush=True)
    print(f"compare to CRYSTAL14 job fce948157c88 (LiH primitive, STO-3G,"
          f" SHRINK 8 8, serial)\n", flush=True)

    system, basis = lih_primitive()
    print(f"system: {basis.nbasis} BFs, {system.n_electrons()} electrons, "
          f"dim={system.dim}", flush=True)
    print(f"lattice vectors (bohr):")
    print(system.lattice)
    print(f"atom positions (bohr):")
    for at in system.unit_cell_molecule().atoms:
        print(f"  {at}")

    # Run vibe-qc periodic RHF at Γ (same path GDF uses post-Step-3':
    # Ewald-J + real-space-K, bit-exact with EWALD_3D driver).
    opts = vq.PeriodicRHFOptions()
    opts.use_diis = True
    opts.damping = 0.0
    opts.max_iter = 30
    opts.conv_tol_energy = 1e-9

    print("\nrunning Γ-only RHF (native GDF driver, dim=3 -> Ewald-J + real-space-K)",
          flush=True)
    res = vq.run_rhf_periodic_gamma_gdf(system, basis, opts, progress=False)
    print(f"  converged={res.converged}  n_iter={res.n_iter}")
    print(f"  E_total    = {res.energy:.10f} Ha")
    print(f"  E_nuclear  = {res.e_nuclear:.10f}")
    print(f"  E_elec     = {res.e_electronic:.10f}")
    print(f"  E_J        = {res.e_coulomb:.10f}")
    print(f"  E_K        = {res.e_hf_exchange:.10f}")

    # Build the per-piece matrices at Γ = bloch_sum at k=0.
    lat = LatticeSumOptions()
    gl = LatticeSumOptions()
    gl.coulomb_method = vq.CoulombMethod.EWALD_3D
    kg = np.zeros(3)

    S = np.real(bloch_sum(compute_overlap_lattice(basis, system, lat), kg))
    T = np.real(bloch_sum(compute_kinetic_lattice(basis, system, lat), kg))
    V = np.real(bloch_sum(compute_nuclear_lattice_dispatch(basis, system, gl), kg))
    S = 0.5 * (S + S.T); T = 0.5 * (T + T.T); V = 0.5 * (V + V.T)
    Hcore = T + V

    # Final J and K at the converged density
    D = np.asarray(res.density)
    grid_shape = tuple(int(x) for x in auto_grid(np.asarray(system.lattice, float), 0.3))
    J = build_j_ewald_3d(basis, system, D, omega=0.5, lattice_opts=lat,
                        grid_shape=grid_shape, spacing_bohr=0.3)
    J = 0.5 * (J + J.T)
    jk = build_jk_gamma_molecular_limit(basis, system, lat, D, 0.0)
    K = np.asarray(jk.K); K = 0.5 * (K + K.T)

    F = Hcore + J - 0.5 * K
    F = 0.5 * (F + F.T)

    # Inferred basis ordering check: look at the diagonal of T+V (Hcore)
    # to distinguish Li shells from H. Li 1s should be deeply bound,
    # Li 2s less so, Li 2p the next, H 1s separate.
    print("\nbasis-ordering check (diagonal of Hcore = T+V):", flush=True)
    for i in range(basis.nbasis):
        print(f"  AO {i+1}:  Hcore[i,i] = {Hcore[i,i]:.4f}    "
              f"T[i,i] = {T[i,i]:.4f}    V[i,i] = {V[i,i]:.4f}")

    fmt_matrix(S, "S(Γ) overlap")
    fmt_matrix(T, "T(Γ) kinetic")
    fmt_matrix(V, "V(Γ) nuclear-attraction (Ewald gauge)")
    fmt_matrix(Hcore, "Hcore(Γ) = T + V")
    fmt_matrix(J, "J(Γ) Coulomb")
    fmt_matrix(K, "K(Γ) exchange")
    fmt_matrix(F, "F(Γ) = Hcore + J - 0.5 K (converged Fock)")
    fmt_matrix(D, "P(Γ) converged density")

    # Energy components from the matrices (sanity cross-check)
    E_kin = float(np.einsum("ij,ij->", D, T))
    E_ne = float(np.einsum("ij,ij->", D, V))
    E_J = 0.5 * float(np.einsum("ij,ij->", D, J))
    E_K = -0.25 * float(np.einsum("ij,ij->", D, K))
    print(f"\n=== energy components (from matrices, this Γ run) ===")
    print(f"  E_kin  = tr(D·T)/1  = {E_kin:.6f}")
    print(f"  E_ne   = tr(D·V)/1  = {E_ne:.6f}")
    print(f"  E_J    = tr(D·J)/2  = {E_J:.6f}")
    print(f"  E_K    = -tr(D·K)/4 = {E_K:.6f}  (note: convention check)")
    print(f"  E_ee   = E_J + E_K  = {E_J + E_K:.6f}")
    print(f"  E_nuc  =              {res.e_nuclear:.6f}")
    print(f"  sum    =              {E_kin + E_ne + E_J + E_K + res.e_nuclear:.6f}")
    print(f"  res.energy = {res.energy:.6f}  "
          f"(Δ = {(E_kin + E_ne + E_J + E_K + res.e_nuclear) - res.energy:+.2e})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
