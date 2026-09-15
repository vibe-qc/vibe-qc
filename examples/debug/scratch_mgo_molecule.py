"""Compare a single Mg + O molecule (no periodic) S matrix between
vibe-qc (libint) and PySCF (libcint). This isolates the basis-ordering
question from any periodic / lattice-cutoff confounds.
"""
from __future__ import annotations

import numpy as np

import vibeqc as vq

from pyscf import gto


def vibeqc_mol_S():
    # Mg at origin, O at (3, 0, 0) bohr.
    atoms = [vq.Atom(12, [0.0, 0.0, 0.0]),
             vq.Atom(8,  [3.0, 0.0, 0.0])]
    mol = vq.Molecule(atoms, charge=0, multiplicity=1)
    basis = vq.BasisSet(mol, "sto-3g")
    S = vq.compute_overlap(basis)
    return S, basis


def pyscf_mol_S():
    mol = gto.M(atom="Mg 0 0 0; O 1.587 0 0",  # 3 bohr ≈ 1.587 Å
                basis="sto-3g", unit="A", verbose=0)
    S = mol.intor("int1e_ovlp")
    return S, mol


def main():
    Sv, basis_v = vibeqc_mol_S()
    Sp, mol_p = pyscf_mol_S()

    print(f"vibeqc nbf = {basis_v.nbasis}, pyscf nao = {mol_p.nao}")
    print()

    print("vibeqc S diag (raw):")
    print(np.diag(Sv))
    print()
    print("pyscf S diag (raw):")
    print(np.diag(Sp))
    print()

    print("vibeqc shells (i, l, atom_index, nbf):")
    for i, sh in enumerate(basis_v.shells()):
        print(f"  {i:2d}  l={sh.l}  atom={getattr(sh,'atom_index','?')}")

    print()
    print("pyscf shells (i, l, atom_index, nctr*(2l+1)):")
    for ish in range(mol_p.nbas):
        l = mol_p.bas_angular(ish)
        atm = mol_p.bas_atom(ish)
        nctr = mol_p.bas_nctr(ish)
        nbf = nctr * (2*l + 1)
        print(f"  {ish:2d}  l={l}  atom={atm}  nbf={nbf}")

    print()
    print("AO labels (pyscf):")
    for i, lab in enumerate(mol_p.ao_labels()):
        print(f"  {i:2d}  {lab}")

    print()
    # Per-atom permutation: vibeqc index → pyscf index.
    # Within a p-shell, libint uses (m=-1,0,+1) = (py, pz, px) but
    # libcint (PySCF) uses (px, py, pz). So in addition to shell-order
    # remapping, we also cyclic-shift within each p-shell:
    #   vibeqc (py, pz, px) at relative indices (0, 1, 2)
    #     →  pyscf (px, py, pz) at relative indices (1, 2, 0)
    # Mg vibeqc layout: 1s(0) 2s(1) 2py(2) 2pz(3) 2px(4) 3s(5) 3py(6) 3pz(7) 3px(8)
    # Mg pyscf  layout: 1s(0) 2s(1) 3s(2)  2px(3) 2py(4) 2pz(5) 3px(6) 3py(7) 3pz(8)
    PERM_MG = [0, 1, 4, 5, 3, 2, 7, 8, 6]
    # O vibeqc layout: 1s(0) 2s(1) 2py(2) 2pz(3) 2px(4)
    # O pyscf  layout: 1s(0) 2s(1) 2px(2) 2py(3) 2pz(4)
    PERM_O = [0, 1, 3, 4, 2]
    n = 9 + 5
    P = np.zeros((n, n))
    for i, p in enumerate(PERM_MG):
        P[p, i] = 1.0
    for i, p in enumerate(PERM_O):
        P[9 + p, 9 + i] = 1.0
    Sv_perm = P @ Sv @ P.T
    diff = Sv_perm - Sp
    print(f"After permutation: max|elem diff| = {np.max(np.abs(diff)):.3e}")
    print(f"  ||Δ||/||S|| = {np.linalg.norm(diff)/np.linalg.norm(Sp):.3e}")
    if np.max(np.abs(diff)) > 1e-6:
        # Print a side-by-side of the worst rows
        bad_i, bad_j = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
        print(f"  worst at (row,col)=({bad_i},{bad_j}): "
              f"vq(perm)={Sv_perm[bad_i,bad_j]:+.6f} py={Sp[bad_i,bad_j]:+.6f}")
        print()
        print("Full Sv_perm (after permutation):")
        for i in range(n):
            print(" ".join(f"{x:+7.4f}" for x in Sv_perm[i]))
        print()
        print("Full Sp (pyscf):")
        for i in range(n):
            print(" ".join(f"{x:+7.4f}" for x in Sp[i]))


if __name__ == "__main__":
    main()
