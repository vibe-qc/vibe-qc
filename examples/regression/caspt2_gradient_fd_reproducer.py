#!/usr/bin/env python3
"""CASPT2/NEVPT2 Z-vector gradient parity adjudicator.

Validates the analytic Z-vector gradient against full-energy finite
differences of the total (CASSCF + PT2) energy for H2/6-31G CAS(2,2).

Usage:
    python examples/regression/caspt2_gradient_fd_reproducer.py

Exit 0 if the Z-vector gradient equals full-energy FD within 5e-4 Ha/bohr
(1% of the full FD magnitude, or tighter if measured accuracy permits).
Exit 1 otherwise.

Target accuracies (H2/6-31G CAS(2,2), R=1.4 bohr):
  CASPT2 Z-vector: 99.99% (delta 6.77e-7 vs full FD)
  NEVPT2 Z-vector: 99.99% (delta 4.99e-7 vs full FD)
"""

from __future__ import annotations

import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")

from vibeqc import Atom, BasisSet, Molecule
from vibeqc.solvers import build_hamiltonian_mo, get_hf_orbital_provider
from vibeqc.solvers._casscf import casscf
from vibeqc.solvers._mrpt import (
    _add,
    _pt2_correction,
    _semicanonical_prep,
    apply_1body,
    apply_2body,
)
from vibeqc.solvers._rdm import make_rdm12


def main() -> int:
    mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
    basis = BasisSet(mol, "6-31g")
    C_hf = get_hf_orbital_provider(mol, basis)
    H = build_hamiltonian_mo(mol, basis, C_hf)
    sc = casscf(H.h1e, H.h2e, 2, 2, n_core=0, nuclear_repulsion=H.nuclear_repulsion)
    assert sc.converged, "CASSCF did not converge"
    C_conv = C_hf @ sc.mo_rotation
    rdm1, rdm2 = make_rdm12(sc.cas.ci_coeffs, sc.cas.determinants, 2)

    print(f"CASSCF energy = {sc.e_total:.8f} Ha")
    print()

    # ── CASPT2 ──────────────────────────────────────────────────────────
    from vibeqc.gradient._caspt2 import compute_caspt2_gradient

    g_caspt2_frozen = compute_caspt2_gradient(
        mol,
        basis,
        C_conv,
        sc.h1e_cas,
        sc.h2e_cas,
        n_core=0,
        n_active_orb=2,
        n_active_elec=2,
        rdm1=rdm1,
        rdm2=rdm2,
        use_zvector=False,
        fd_eps=0.001,
    )
    g_caspt2_zvec = compute_caspt2_gradient(
        mol,
        basis,
        C_conv,
        sc.h1e_cas,
        sc.h2e_cas,
        n_core=0,
        n_active_orb=2,
        n_active_elec=2,
        rdm1=rdm1,
        rdm2=rdm2,
        use_zvector=True,
        determinants=sc.cas.determinants,
        ci_coeffs=sc.cas.ci_coeffs,
        fd_eps=0.001,
    )

    # Full CASPT2 FD
    atoms_list = list(mol.atoms)
    xyz_ref = np.array([at.xyz for at in atoms_list], dtype=float)
    h_fd = 0.001

    def e_caspt2_full(xyz):
        m = Molecule([Atom(int(at.Z), list(xyz)) for at, xyz in zip(atoms_list, xyz)])
        b = BasisSet(m, "6-31g")
        Ch = get_hf_orbital_provider(m, b)
        Hd = build_hamiltonian_mo(m, b, Ch)
        sd = casscf(
            Hd.h1e, Hd.h2e, 2, 2, n_core=0, nuclear_repulsion=Hd.nuclear_repulsion
        )
        Pd = _semicanonical_prep(sd.h1e_cas, sd.h2e_cas, 0, 2, 2, 0)
        e2 = _pt2_correction(Pd, lambda s: apply_1body(s, Pd["F"], Pd["norb"]))
        return sd.e_total + e2

    xyz_p = xyz_ref.copy()
    xyz_p[0, 2] += h_fd
    xyz_m = xyz_ref.copy()
    xyz_m[0, 2] -= h_fd
    g_caspt2_full = (e_caspt2_full(xyz_p) - e_caspt2_full(xyz_m)) / (2.0 * h_fd)

    d_frozen = abs(g_caspt2_frozen[0, 2] - g_caspt2_full)
    d_zvec = abs(g_caspt2_zvec[0, 2] - g_caspt2_full)

    print(f"CASPT2 gradient (atom 0, z):")
    print(f"  full-energy FD  = {g_caspt2_full:+.8f} Ha/bohr")
    print(f"  frozen FD       = {g_caspt2_frozen[0, 2]:+.8f}  (delta {d_frozen:.2e})")
    print(f"  Z-vector        = {g_caspt2_zvec[0, 2]:+.8f}  (delta {d_zvec:.2e})")
    caspt2_ok = d_zvec < 1e-4
    print(f"  CASPT2 verdict  = {'PASS' if caspt2_ok else 'FAIL'}")

    # ── NEVPT2 ──────────────────────────────────────────────────────────
    from vibeqc.gradient._nevpt2 import compute_nevpt2_gradient

    g_nevpt2_frozen = compute_nevpt2_gradient(
        mol,
        basis,
        C_conv,
        sc.h1e_cas,
        sc.h2e_cas,
        n_core=0,
        n_active_orb=2,
        n_active_elec=2,
        rdm1=rdm1,
        rdm2=rdm2,
        use_zvector=False,
        fd_eps=0.001,
    )
    g_nevpt2_zvec = compute_nevpt2_gradient(
        mol,
        basis,
        C_conv,
        sc.h1e_cas,
        sc.h2e_cas,
        n_core=0,
        n_active_orb=2,
        n_active_elec=2,
        rdm1=rdm1,
        rdm2=rdm2,
        use_zvector=True,
        determinants=sc.cas.determinants,
        ci_coeffs=sc.cas.ci_coeffs,
        fd_eps=0.001,
    )

    def e_nevpt2_full(xyz):
        m = Molecule([Atom(int(at.Z), list(xyz)) for at, xyz in zip(atoms_list, xyz)])
        b = BasisSet(m, "6-31g")
        Ch = get_hf_orbital_provider(m, b)
        Hd = build_hamiltonian_mo(m, b, Ch)
        sd = casscf(
            Hd.h1e, Hd.h2e, 2, 2, n_core=0, nuclear_repulsion=Hd.nuclear_repulsion
        )
        Pd = _semicanonical_prep(sd.h1e_cas, sd.h2e_cas, 0, 2, 2, 0)
        norb = Pd["norb"]
        h1D = np.zeros((norb, norb))
        for p in range(norb):
            h1D[p, p] = Pd["eps"][p]
        h1D[Pd["act"], Pd["act"]] = Pd["h1a"]
        eriD = np.zeros_like(Pd["eri"])
        eriD[Pd["act"], Pd["act"], Pd["act"], Pd["act"]] = Pd["eri"][
            Pd["act"], Pd["act"], Pd["act"], Pd["act"]
        ]

        def apply_HD(s):
            return _add(
                apply_1body(s, h1D, norb), apply_2body(s, eriD, norb, idx=Pd["aidx"])
            )

        e2 = _pt2_correction(Pd, apply_HD)
        return sd.e_total + e2

    xyz_p = xyz_ref.copy()
    xyz_p[0, 2] += h_fd
    xyz_m = xyz_ref.copy()
    xyz_m[0, 2] -= h_fd
    g_nevpt2_full = (e_nevpt2_full(xyz_p) - e_nevpt2_full(xyz_m)) / (2.0 * h_fd)

    d_frozen2 = abs(g_nevpt2_frozen[0, 2] - g_nevpt2_full)
    d_zvec2 = abs(g_nevpt2_zvec[0, 2] - g_nevpt2_full)

    print(f"\nNEVPT2 gradient (atom 0, z):")
    print(f"  full-energy FD  = {g_nevpt2_full:+.8f} Ha/bohr")
    print(f"  frozen FD       = {g_nevpt2_frozen[0, 2]:+.8f}  (delta {d_frozen2:.2e})")
    print(f"  Z-vector        = {g_nevpt2_zvec[0, 2]:+.8f}  (delta {d_zvec2:.2e})")
    nevpt2_ok = d_zvec2 < 1e-4
    print(f"  NEVPT2 verdict  = {'PASS' if nevpt2_ok else 'FAIL'}")

    all_ok = caspt2_ok and nevpt2_ok
    print(f"\nFINAL VERDICT = {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
