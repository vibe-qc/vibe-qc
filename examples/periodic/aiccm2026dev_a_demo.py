"""Full AICCM stack on the ``aiccm2026dev-a`` Γ-CCM construction.

``aiccm2026dev-a`` is **Γ-CCM**, the union-and-weight/Wigner--Seitz
four-center construction within the variational finite-BvK-torus CCM family.
Its sibling ``aiccm2026dev-b`` is **χ-CCM**, the finite-translation-group
character construction. They are distinct approaches compared at a declared
common exchange-``q=0`` convention; equality is a result to establish, not a
naming premise.

The selector ``method="aiccm2026dev-a"`` and the ``run_ccm_scf`` aliases
``"aiccm2026dev-a"`` / ``"gamma-ccm"`` bind to the symmetric Wigner--Seitz
four-center route demonstrated here. The separate ``"neutral-bloch"`` / GDF
and ``"real-gamma"`` routes are neutral fitted-torus Bloch/real-Γ representation
controls. Their exact same-H Fourier relation is valid for the specified
block-circulant control Hamiltonian but is not Γ-CCM / χ-CCM construction
evidence.

Runs the complete Cyclic Cluster Model method stack on the symmetric Born–von
Kármán-torus four-center (``method="aiccm2026dev-a"``; AICCM_ALGORITHM.md §13) — HF,
open-shell HF, and the MP2 / UMP2 / CCSD(T) correlation hierarchy — and shows
the exact 8-fold ERI symmetry that the historical ``union12`` weight breaks in
≥2-D.

This is the development-line ("aiccm2026dev-a") demo used for the head-to-head against
an independent CCM derivation. Run:

    python examples/periodic/aiccm2026dev_a_demo.py

Expected output (STO-3G unless noted; energies in Ha):

    === 8-fold ERI symmetry (max permutation violation) ===
      1D H4 chain (4,1,1) : union12 1.63e-03   aiccm2026dev-a 0.00e+00
      2D hex      (2,2,1) : union12 2.12e-02   aiccm2026dev-a 3.10e-18
      2D oblique  (2,2,1) : union12 1.94e-02   aiccm2026dev-a 3.10e-18
    === Full stack on the 1D H4 chain (4,1,1), aiccm2026dev-a ===
      RHF     E/atom = -0.542676   (gold -0.542875)
      MP2     Ecorr  = -0.129514   /cell
      CCSD    Ecorr  = -0.190443   /cell
      CCSD(T) Ecorr  = -0.191100   /cell  [(T)=-6.57e-04]
    === 2D hexagonal lattice (2,2,1): aiccm2026dev-a vs union12 ===
      RHF E/atom : union12 -0.399044   aiccm2026dev-a -0.400456   (Δ=-1.413 mHa/atom —
                   eq-18 asymmetry bites in non-orthorhombic >=2D)
    === Madelung rank-1 diagnostic at a fixed reference ===
      neutral four-center g_eff: 8-fold sym 3.16e-17, cell Madelung ξ=0.0758
      (CᵀSC)_ia max |occ-virt| = 1.17e-15  → ξ·S⊗S has no occ-virt element
      ⇒ numerator vanishes at fixed orbitals; denominators still shift

Reference: Peintinger & Bredow, J. Comput. Chem. 35, 839 (2014),
doi:10.1002/jcc.23550; AICCM_ALGORITHM.md §13.
"""

from __future__ import annotations

import numpy as np

from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import (
    CCMSystem,
    ccm_eri_neutral,
    ccm_neutral_background_constant,
    ccm_overlap,
)
from vibeqc.periodic.ccm.ccsd import run_ccm_ccsd
from vibeqc.periodic.ccm.mp2 import run_ccm_mp2
from vibeqc.periodic.ccm.padded import build_padded_cluster, ccm_eri, ccm_eri_symmetric, eri_cells
from vibeqc.periodic.ccm.scf import run_ccm_rhf

BOHR = 1.0 / 0.529177210903


def max_asym(eff):
    nrm = np.linalg.norm(eff)
    return max(np.linalg.norm(eff - eff.transpose(1, 0, 2, 3)) / nrm,
               np.linalg.norm(eff - eff.transpose(0, 1, 3, 2)) / nrm,
               np.linalg.norm(eff - eff.transpose(2, 3, 0, 1)) / nrm)


def h4_1d():
    pos = [[x * BOHR, 0, 0] for x in (0.0, 0.8, 2.0, 2.8)]
    return PeriodicSystem(3, np.diag([4.0 * BOHR, 40.0, 40.0]),
                          [Atom(1, p) for p in pos], 0, 1)


def hex_2d(a=4.0):
    lat = np.array([[a, 0, 0], [a * 0.5, a * np.sqrt(3) / 2, 0], [0, 0, 40.0]])
    return PeriodicSystem(3, lat, [Atom(1, [0, 0, 0])], 0, 2)


def oblique_2d(a=4.0):
    lat = np.array([[a, 0, 0], [a * 0.4, a * 0.9, 0], [0, 0, 40.0]])
    return PeriodicSystem(3, lat, [Atom(1, [0, 0, 0])], 0, 2)


def main():
    print("=== 8-fold ERI symmetry (max permutation violation) ===")
    for name, unit, nrep in [("1D H4 chain (4,1,1)", h4_1d(), (4, 1, 1)),
                             ("2D hex      (2,2,1)", hex_2d(), (2, 2, 1)),
                             ("2D oblique  (2,2,1)", oblique_2d(), (2, 2, 1))]:
        ccm = CCMSystem(unit, nrep, "sto-3g")
        pad = build_padded_cluster(ccm, eri_cells(ccm))
        print(f"  {name} : union12 {max_asym(ccm_eri(ccm, pad)):.2e}   "
              f"aiccm2026dev-a {max_asym(ccm_eri_symmetric(ccm, pad)):.2e}")

    print("=== Full stack on the 1D H4 chain (4,1,1), aiccm2026dev-a ===")
    ccm = CCMSystem(h4_1d(), (4, 1, 1), "sto-3g")
    rhf = run_ccm_rhf(ccm, method="aiccm2026dev-a")
    mp2 = run_ccm_mp2(ccm, method="aiccm2026dev-a")
    cc = run_ccm_ccsd(ccm, method="aiccm2026dev-a", compute_triples=True)
    print(f"  RHF     E/atom = {rhf.energy_per_atom:.6f}   (gold -0.542875)")
    print(f"  MP2     Ecorr  = {mp2.e_correlation:.6f}   /cell")
    print(f"  CCSD    Ecorr  = {cc.e_ccsd_correlation:.6f}   /cell")
    print(f"  CCSD(T) Ecorr  = {cc.e_correlation:.6f}   /cell  [(T)={cc.e_t:.2e}]")

    print("=== 2D hexagonal lattice (2,2,1): aiccm2026dev-a vs union12 ===")
    ccm2 = CCMSystem(hex_2d(), (2, 2, 1), "sto-3g")
    e_u = run_ccm_rhf(ccm2, method="union12").energy_per_atom
    e_d = run_ccm_rhf(ccm2, method="aiccm2026dev-a").energy_per_atom
    print(f"  RHF E/atom : union12 {e_u:.6f}   aiccm2026dev-a {e_d:.6f}   "
          f"(Δ={(e_d - e_u) * 1e3:.3f} mHa/atom — eq-18 asymmetry bites in ≥2D)")

    print("=== Madelung rank-1 diagnostic at a fixed reference ===")
    # The neutral (Ewald) four-center g_eff = density-fit of v_E (§13.7/§13.9):
    # 8-fold symmetric, and = bare − ξ·S⊗S with ξ = the cell Madelung constant.
    ccm3 = CCMSystem(h4_1d(), (4, 1, 1), "sto-3g")
    xi = ccm_neutral_background_constant(ccm3)
    g_neu = ccm_eri_neutral(ccm3, ke_cutoff=120.0)
    print(f"  neutral four-center g_eff: 8-fold sym {max_asym(g_neu):.2e}, cell Madelung ξ={xi:.4f}")
    # ξ·S⊗S → ξ·(CᵀSC)⊗(CᵀSC) = ξ·I⊗I in the MO basis: its
    # occupied-virtual numerator block vanishes at this fixed reference. A
    # separately optimised reference changes the orbital gap, so neutral and
    # bare references are not interchangeable for correlation.
    rhf3 = run_ccm_rhf(ccm3, method="aiccm2026dev-a")
    S = ccm_overlap(ccm3)
    C = np.asarray(rhf3.mo_coeffs)
    n_occ = ccm3.supercell.n_electrons() // 2
    ov = float(np.max(np.abs((C.T @ S @ C)[:n_occ, n_occ:])))
    print(f"  (CᵀSC)_ia max |occ-virt| = {ov:.2e}  → ξ·S⊗S has no occ-virt element")
    print("  ⇒ numerator vanishes at fixed orbitals; denominators still shift")


if __name__ == "__main__":
    main()
