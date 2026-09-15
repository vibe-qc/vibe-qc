"""Milestone 1: molecular RHF/CASCI/CASSCF/CASPT2 in a static external
point-charge field. Embedded-cluster CASSCF spike (see README +
handovers/HANDOVER_PERIODIC_MULTIREF.md, MR6-MR10).

Proves the enabler exists with NO new core code: V_ext via
compute_nuclear_with_charges injected into a custom Hcore +
run_rhf_scf_with_jk, then the integral-driven CAS solvers on the
V_ext-aware MO integrals.

Run:
    .venv/bin/python studies/embedded-cluster-cas/spike_external_field.py
"""
from __future__ import annotations

import dataclasses

import numpy as np

import vibeqc as vq
from vibeqc._vibeqc_core import (
    RHFOptions,
    compute_eri,
    compute_kinetic,
    compute_nuclear,
    compute_nuclear_with_charges,
    compute_overlap,
    make_direct_jk_builder,
    run_rhf_scf_with_jk,
)
from vibeqc.solvers import casci, caspt2, casscf
from vibeqc.solvers._hamiltonian import build_hamiltonian_ao, transform_hamiltonian

BOHR = 1.8897259886  # angstrom -> bohr


def _attr(obj, *names):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    raise AttributeError(f"none of {names} on {type(obj).__name__}")


def _energy(res):
    # RHFResult -> .energy ; CASCI/CASSCF/NEVPT2Result -> .e_total
    return float(_attr(res, "e_total", "energy"))


# ----- the embedded-reference construction (the whole enabler) -----

def qm_ext_repulsion(mol, ext_pos, ext_q):
    """sum_{QM nucleus A, external charge B} Z_A q_B / |R_A - R_B|."""
    e = 0.0
    for a in mol.atoms:
        Ra = np.asarray(a.xyz, dtype=float)
        for q, Rb in zip(ext_q, ext_pos):
            e += float(a.Z) * float(q) / np.linalg.norm(Ra - np.asarray(Rb, float))
    return e


def embedded_pieces(mol, basis, ext_pos, ext_q):
    S = np.asarray(compute_overlap(basis))
    T = np.asarray(compute_kinetic(basis))
    V_nuc = np.asarray(compute_nuclear(basis, mol))
    if ext_q:
        V_ext = np.asarray(compute_nuclear_with_charges(
            basis, [list(map(float, p)) for p in ext_pos], list(map(float, ext_q))))
    else:
        V_ext = np.zeros_like(T)
    Hcore = T + V_nuc + V_ext
    E_nuc = mol.nuclear_repulsion() + qm_ext_repulsion(mol, ext_pos, ext_q)
    jk = make_direct_jk_builder(basis)
    return S, Hcore, float(E_nuc), jk


def embedded_rhf(mol, basis, ext_pos, ext_q):
    S, Hcore, E_nuc, jk = embedded_pieces(mol, basis, ext_pos, ext_q)
    res = run_rhf_scf_with_jk(basis, mol.n_electrons(), S, Hcore, E_nuc, jk,
                              RHFOptions())
    return res, Hcore, E_nuc


def embedded_mo_hamiltonian(mol, basis, C, Hcore, E_nuc):
    """MO Hamiltonian whose h1e carries V_ext (Hcore already includes it)."""
    ham_ao = build_hamiltonian_ao(mol, basis)
    ham_ao = dataclasses.replace(ham_ao, h1e=np.asarray(Hcore),
                                 nuclear_repulsion=float(E_nuc))
    return transform_hamiltonian(ham_ao, np.asarray(C))


def embedded_cas(mol, basis, ext_pos, ext_q, nae, nao):
    """RHF -> CASCI -> CASSCF -> CASPT2(on CASCI), all in the field."""
    rhf, Hcore, E_nuc = embedded_rhf(mol, basis, ext_pos, ext_q)
    C = np.asarray(_attr(rhf, "mo_coeffs", "C", "orbitals"))
    H = embedded_mo_hamiltonian(mol, basis, C, Hcore, E_nuc)
    n_core = (mol.n_electrons() - nae) // 2
    n_virt = H.norb - n_core - nao
    ci = casci(H.h1e, H.h2e, nae, nao, n_core=n_core,
               nuclear_repulsion=H.nuclear_repulsion)
    sc = casscf(H.h1e, H.h2e, nae, nao, n_core=n_core,
                nuclear_repulsion=H.nuclear_repulsion)
    pt2 = caspt2(ci, H.h1e, H.h2e, n_core=n_core, n_virt=n_virt)
    return dict(rhf=_energy(rhf), casci=_energy(ci), casscf=_energy(sc),
                caspt2=_energy(pt2))


# ----------------------------- the spike -----------------------------

def main():
    # LiH / 6-31G, CAS(2,2): polar, well-behaved, trivial active space.
    mol = vq.Molecule([vq.Atom(3, [0.0, 0.0, 0.0]),
                       vq.Atom(1, [0.0, 0.0, 1.60 * BOHR])])
    basis = vq.BasisSet(mol, "6-31g")
    nae, nao = 2, 2

    # A crude "Madelung" cage: +/-q point charges flanking the bond axis.
    d = 4.0 * BOHR
    ext_pos = [[0.0, 0.0, -d], [0.0, 0.0, 1.60 * BOHR + d]]
    base_q = [+1.0, -1.0]

    print("=" * 68)
    print("Milestone 1: molecular CAS in a static external point-charge field")
    print("=" * 68)

    # (1) API cross-check: external-charge integral over the QM nuclei
    #     reproduces compute_nuclear (validates the primitive + sign).
    V_nuc = np.asarray(compute_nuclear(basis, mol))
    V_as_charges = np.asarray(compute_nuclear_with_charges(
        basis, [list(map(float, a.xyz)) for a in mol.atoms],
        [float(a.Z) for a in mol.atoms]))
    d1 = float(np.max(np.abs(V_nuc - V_as_charges)))
    print(f"\n(1) compute_nuclear_with_charges(QM) vs compute_nuclear:"
          f"  max|Δ| = {d1:.2e}   {'PASS' if d1 < 1e-10 else 'FAIL'}")

    # (2) V_ext = 0 limit: embedded RHF (custom Hcore, no charges) ==
    #     standard run_rhf.
    e_std = _energy(vq.run_rhf(mol, basis))
    rhf0, _, _ = embedded_rhf(mol, basis, [], [])
    d2 = abs(_energy(rhf0) - e_std)
    print(f"(2) embedded RHF (V_ext=0) vs run_rhf:"
          f"            |Δ| = {d2:.2e}   {'PASS' if d2 < 1e-9 else 'FAIL'}")

    # (3) Field effect: full CAS ladder, bare vs field, + a q-scan.
    bare = embedded_cas(mol, basis, [], [], nae, nao)
    field = embedded_cas(mol, basis, ext_pos, base_q, nae, nao)
    print("\n(3) CAS ladder in the field (Ha):")
    print(f"      {'method':8s} {'bare':>15s} {'field':>15s} {'Δ(field)':>13s}")
    for m in ("rhf", "casci", "casscf", "caspt2"):
        print(f"      {m:8s} {bare[m]:15.8f} {field[m]:15.8f} "
              f"{field[m] - bare[m]:13.2e}")

    print("\n    q-scan of CASSCF energy (smoothness + q=0 recovery):")
    scan = []
    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        e = embedded_cas(mol, basis, ext_pos, [s, -s], nae, nao)["casscf"]
        scan.append(e)
        print(f"      q = {s:.2f} :  E_casscf = {e:.8f}")
    q0_ok = abs(scan[0] - bare["casscf"]) < 1e-9
    # Bounded second differences => smooth/continuous response. The
    # q-dependence need NOT be monotone: the QM-nucleus<->charge term is
    # linear in q while the electronic response is ~quadratic, so the
    # total energy turns over (here near q~0.75) -- physical, not a bug.
    d2s = [scan[i + 1] - 2 * scan[i] + scan[i - 1] for i in range(1, len(scan) - 1)]
    smooth = max(abs(x) for x in d2s) < 5e-2
    nonzero = abs(field["casscf"] - bare["casscf"]) > 1e-6
    print(f"\n    q=0 recovers bare CASSCF: {'PASS' if q0_ok else 'FAIL'};"
          f"  smooth response: {'yes' if smooth else 'no'};"
          f"  field has effect: {'yes' if nonzero else 'no'}")

    ok = d1 < 1e-10 and d2 < 1e-9 and q0_ok and nonzero
    print("\n" + ("MILESTONE 1 ENABLER: PASS — molecular CAS runs in an "
                  "external point-charge field." if ok else "SOME CHECK FAILED"))


if __name__ == "__main__":
    main()
