"""Diagnostic v2: probe spheropole per-primitive-pair weights.

Run:
    .venv/bin/python examples/debug/probe_spheropole_v2.py
"""

from __future__ import annotations

import math

import numpy as np
import vibeqc as vq
from vibeqc._vibeqc_core import (
    CoulombMethod,
    InitialGuess,
    LatticeSumOptions,
    compute_multipole_moments_lattice,
    compute_overlap_lattice,
)
from vibeqc.bipole_ext_el_pole import (
    _split_basis_into_primitives,
)
from vibeqc.guess import initial_density_closed_shell

ANG2BOHR = 1.0 / 0.529177210903


def main():
    a = 4.21 * ANG2BOHR
    lattice = (a / 2.0) * np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    atoms = [vq.Atom(12, [0.0, 0.0, 0.0]), vq.Atom(8, [a / 2.0, a / 2.0, a / 2.0])]
    system = vq.PeriodicSystem(3, lattice, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    lat_opts = LatticeSumOptions()
    lat_opts.cutoff_bohr = 6.0
    lat_opts.coulomb_method = CoulombMethod.DIRECT_TRUNCATED

    # Primitive-split basis
    (prim_basis, prim_alpha, prim_origins, prim_N_lib, contraction_map) = (
        _split_basis_into_primitives(basis, system)
    )

    n_prim = int(prim_basis.nbasis)
    prim_ao_to_l = np.zeros(n_prim, dtype=int)
    ao_idx = 0
    for shell in prim_basis.shells():
        l = int(shell.l)
        for _ in range(2 * l + 1):
            prim_ao_to_l[ao_idx] = l
            ao_idx += 1

    # Get the original contracted basis coefficients for comparison
    print("Contracted shell info:")
    for i, shell in enumerate(basis.shells()):
        l = int(shell.l)
        exps = list(shell.exponents)
        coefs = list(shell.coefficients)
        origin = shell.origin
        print(
            f"  shell {i}: l={l}, origin=({origin[0]:.2f},{origin[1]:.2f},{origin[2]:.2f})"
        )
        for j, (exp, coef) in enumerate(zip(exps, coefs)):
            N = (2.0 * exp / math.pi) ** 0.75
            if l == 1:
                N *= (4.0 * exp) ** 0.5
            d_std = coef / N
            print(
                f"    prim {j}: alpha={exp:.4f}, c_lib={coef:.6f}, N={N:.4f}, d_std={d_std:.6f}"
            )

    # Build emultipole2 on primitive basis at g=0 only
    M_lat = compute_multipole_moments_lattice(
        prim_basis, system, lat_opts, 2, (0.0, 0.0, 0.0)
    )
    g0_idx = next(
        i
        for i, c in enumerate(M_lat.cells)
        if (np.asarray(c.index) == np.array([0, 0, 0])).all()
    )

    # Examine a few specific primitive pairs
    S = np.asarray(M_lat.blocks[g0_idx][0], dtype=float)
    trQ = (
        np.asarray(M_lat.blocks[g0_idx][4], dtype=float)
        + np.asarray(M_lat.blocks[g0_idx][7], dtype=float)
        + np.asarray(M_lat.blocks[g0_idx][9], dtype=float)
    )

    gamma_ab = prim_alpha[:, None] + prim_alpha[None, :]
    N_outer = prim_N_lib[:, None] * prim_N_lib[None, :]
    weight = gamma_ab / N_outer

    # Print some specific pairs
    pairs_to_check = []
    for a in range(n_prim):
        for b in range(n_prim):
            la, lb = prim_ao_to_l[a], prim_ao_to_l[b]
            # Pick representative pairs: s-s, p-s, p-p
            if la == 0 and lb == 0 and a < 3 and b < 3:
                pairs_to_check.append((a, b, "s-s"))
            elif la == 1 and lb == 0 and 6 <= a < 9 and b < 3:
                pairs_to_check.append((a, b, "p-s"))
            elif la == 1 and lb == 1 and 6 <= a < 9 and 6 <= b < 9:
                pairs_to_check.append((a, b, "p-p"))

    print(f"\nSample primitive-pair diagnostics (g=0):")
    for a, b, label in pairs_to_check[:6]:
        S_ab = abs(S[a, b])
        trQ_ab = trQ[a, b]
        w_ab = weight[a, b]
        print(
            f"  {label} ({a},{b}): S={S_ab:.6f}, trQ={trQ_ab:.6f}, "
            f"weight={w_ab:.6f}, weighted_trQ={w_ab * trQ_ab:.6f}"
        )

    # Now compute per-l-type averages
    for la in [0, 1]:
        for lb in [0, 1]:
            mask_a = prim_ao_to_l == la
            mask_b = prim_ao_to_l == lb
            sub_weight = weight[np.ix_(mask_a, mask_b)]
            sub_trQ = trQ[np.ix_(mask_a, mask_b)]
            sub_S = np.abs(S[np.ix_(mask_a, mask_b)])
            avg_w = float(np.mean(sub_weight))
            avg_trQ = float(np.mean(sub_trQ))
            avg_S = float(np.mean(sub_S))
            n_pairs = sub_weight.size
            print(f"\n  l_a={la}, l_b={lb}: {n_pairs} pairs")
            print(f"    avg |S| = {avg_S:.6f}, avg trQ = {avg_trQ:.6f}")
            print(f"    avg weight = {avg_w:.6f}")
            print(f"    avg weighted_trQ = {avg_w * avg_trQ:.6f}")


if __name__ == "__main__":
    main()
