"""Probe the residual in compute_3c_eri_gradient_weighted on H2CO/def2-tzvp.

Compares analytic 3c gradient (via compute_3c_eri_gradient_contribution
with fixed γ, D) against FD on the same contracted-with-fixed-γD
expression. Per HANDOVER_DF_GRADIENT_RESIDUAL.md the analytic kernel
is ~2.8 mHa off FD on heavy-atom z components.
"""

from __future__ import annotations

import numpy as np

from vibeqc import (
    Atom, Molecule, BasisSet, RHFOptions, run_rhf,
)
from vibeqc import _vibeqc_core as core

ANG = 1.8897261339213


def make_mol():
    coords = np.array([
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.205),
        (0.0, 0.943, -0.587),
        (0.0, -0.943, -0.587),
    ])
    Zs = [6, 8, 1, 1]
    return [(z, list(c * ANG)) for z, c in zip(Zs, coords)], coords, Zs


def main():
    atoms, coords, Zs = make_mol()
    mol = Molecule([Atom(z, xyz) for z, xyz in atoms])
    basis = BasisSet(mol, "def2-tzvp")
    aux = BasisSet(mol, "def2-tzvp-jk")
    opts = RHFOptions()
    opts.conv_tol_energy = 1e-12
    rhf = run_rhf(mol, basis, opts)
    D = np.asarray(rhf.density)

    # γ from the converged density
    P_munu = np.array(core.compute_3c_eri(basis, aux))  # (n_aux, n_orb, n_orb)
    rho = np.einsum("Pij,ij->P", P_munu, D)
    V_PQ = np.array(core.compute_2c_eri(aux))
    gamma = np.linalg.solve(V_PQ, rho)

    g_3c_analytic = np.array(
        core.compute_3c_eri_gradient_contribution(basis, aux, mol, D, gamma))

    # FD on the 3c piece with fixed γ, D
    def shift(atom_idx, comp, eps):
        c = coords.copy()
        c[atom_idx, comp] += eps
        m = Molecule(
            [Atom(z, list(cc * ANG)) for z, cc in zip(Zs, c)])
        b = BasisSet(m, "def2-tzvp")
        a = BasisSet(m, "def2-tzvp-jk")
        Pmn = np.array(core.compute_3c_eri(b, a))
        return np.einsum("Pij,P,ij->", Pmn, gamma, D)

    def fd_3c(atom_idx, comp, delta=1e-4):
        return (shift(atom_idx, comp, +delta) - shift(atom_idx, comp, -delta)) / (2 * delta * ANG)

    elem_labels = ["C", "O", "H", "H"]
    print(f"{'atom':<6} {'comp':<4} {'analytic':>14} {'fd':>14} {'diff':>14}")
    for atom_idx in range(4):
        for comp in range(3):
            a = g_3c_analytic[atom_idx, comp]
            f = fd_3c(atom_idx, comp)
            print(f"{atom_idx:<2} {elem_labels[atom_idx]:<4} {comp:<4} "
                  f"{a:>14.6e} {f:>14.6e} {a-f:>+14.3e}")


if __name__ == "__main__":
    main()
