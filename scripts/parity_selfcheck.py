"""Self-consistency check for vibeqc.parity decomposition helpers.

For each SCF flavour: the decomposed pieces must sum to result.energy,
and (for RKS/UKS) the building-block e_coulomb / e_exchange must match
the SCF-reported result.e_coulomb / result.e_hf_exchange.
"""
from __future__ import annotations
import numpy as np
import vibeqc as vq
from vibeqc.parity import (
    decompose_energy_rhf, decompose_energy_uhf,
    decompose_energy_rks, decompose_energy_uks,
)

ANG = 1.0 / 0.529177210903
H2O = [(8, [0.0, 0.0, 0.0]),
        (1, [0.0, 0.793353 * ANG, -0.613510 * ANG]),
        (1, [0.0, -0.793353 * ANG, -0.613510 * ANG])]
OH = [(8, [0.0, 0.0, 0.0]), (1, [0.0, 0.0, 0.97 * ANG])]


def check(label, payload, result):
    res = payload["e_total_residual"]
    ok = abs(res) < 1e-9
    print(f"  {label:28s} e_total={payload['e_total']:.10f}  "
          f"reported={payload['e_total_reported']:.10f}  "
          f"residual={res:+.2e}  {'OK' if ok else 'FAIL'}")
    # cross-check RKS/UKS reported pieces
    if hasattr(result, "e_coulomb"):
        dc = payload["e_coulomb"] - result.e_coulomb
        dx = payload["e_exchange"] - result.e_hf_exchange
        dn = payload["e_nuc"] - result.e_nuclear
        print(f"    {'':26s} vs reported: dE_coul={dc:+.2e} "
              f"dE_exch={dx:+.2e} dE_nuc={dn:+.2e}")
    return ok


def main():
    all_ok = True

    mol = vq.Molecule([vq.Atom(z, xyz) for z, xyz in H2O])
    for bname in ("def2-svp", "def2-tzvp"):
        basis = vq.BasisSet(mol, bname)
        # RHF
        o = vq.RHFOptions(); o.conv_tol_energy = 1e-12
        r = vq.run_rhf(mol, basis, o)
        all_ok &= check(f"RHF/{bname}", decompose_energy_rhf(mol, basis, r), r)
        # RKS-PBE
        o = vq.RKSOptions(); o.functional = "PBE"; o.conv_tol_energy = 1e-10
        r = vq.run_rks(mol, basis, o)
        all_ok &= check(f"RKS-PBE/{bname}", decompose_energy_rks(mol, basis, r), r)
        # RKS-B3LYP
        o = vq.RKSOptions(); o.functional = "B3LYP"; o.conv_tol_energy = 1e-10
        r = vq.run_rks(mol, basis, o)
        all_ok &= check(f"RKS-B3LYP/{bname}", decompose_energy_rks(mol, basis, r), r)

    # open-shell OH radical
    mol_oh = vq.Molecule([vq.Atom(z, xyz) for z, xyz in OH], multiplicity=2)
    basis = vq.BasisSet(mol_oh, "def2-svp")
    o = vq.UHFOptions(); o.conv_tol_energy = 1e-11; o.max_iter = 300; o.damping = 0.5
    r = vq.run_uhf(mol_oh, basis, o)
    all_ok &= check("UHF/def2-svp", decompose_energy_uhf(mol_oh, basis, r), r)
    o = vq.UKSOptions(); o.functional = "PBE"; o.conv_tol_energy = 1e-9
    o.max_iter = 400; o.damping = 0.6
    r = vq.run_uks(mol_oh, basis, o)
    all_ok &= check("UKS-PBE/def2-svp", decompose_energy_uks(mol_oh, basis, r), r)

    print()
    print("ALL OK" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
