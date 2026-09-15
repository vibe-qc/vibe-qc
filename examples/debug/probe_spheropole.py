"""Diagnostic: probe spheropole contributions by shell type.

Run:
    .venv/bin/python examples/debug/probe_spheropole.py
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
    compute_ext_el_spheropole,
)
from vibeqc.guess import initial_density_closed_shell

ANG2BOHR = 1.0 / 0.529177210903


def build_mgo():
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
    return system, basis


def build_he_cubic():
    lattice = np.eye(3) * 8.0
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(2, [0.0, 0.0, 0.0])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def main():
    system, basis = build_mgo()
    print(f"Basis: {basis.name}, nbf={basis.nbasis}, nshells={basis.nshells}")

    # Shell information
    print("\nShells:")
    for i, shell in enumerate(basis.shells()):
        print(
            f"  shell {i}: l={shell.l}, atom={shell.atom_index}, "
            f"n_prim={len(shell.exponents)}, origin=({shell.origin[0]:.3f}, {shell.origin[1]:.3f}, {shell.origin[2]:.3f})"
        )

    # Build density (SAD iter-1)
    lat_opts = LatticeSumOptions()
    lat_opts.cutoff_bohr = 6.0
    lat_opts.coulomb_method = CoulombMethod.DIRECT_TRUNCATED

    S_lat = compute_overlap_lattice(basis, system, lat_opts)
    n_occ = system.n_electrons() // 2
    D_sad = initial_density_closed_shell(
        system.unit_cell_molecule(),
        basis,
        n_occ,
        InitialGuess.SAD,
        is_periodic=True,
    )
    P_real = S_lat
    for g_idx in range(len(P_real.cells)):
        is_g0 = (np.asarray(P_real.cells[g_idx].index) == np.array([0, 0, 0])).all()
        P_real.set_block(g_idx, D_sad if is_g0 else np.zeros_like(D_sad))

    # Current spheropole value
    E_full = compute_ext_el_spheropole(P_real, basis, system, lat_opts)
    print(f"\nCurrent spheropole (all shells): E = {E_full:+.6f} Ha")
    print(f"CRYSTAL14 reference: +4.119 Ha")
    print(f"Ratio: {abs(E_full / 4.119) * 100:.1f}%")

    # Now probe: primitive-split basis and compute per-primitive-pair contributions
    (prim_basis, prim_alpha, prim_origins, prim_N_lib, contraction_map) = (
        _split_basis_into_primitives(basis, system)
    )

    # Map primitive AOs back to their shell type
    n_prim = int(prim_basis.nbasis)
    prim_ao_to_l = np.zeros(n_prim, dtype=int)
    ao_idx = 0
    for shell in prim_basis.shells():
        l = int(shell.l)
        n_ao = 2 * l + 1
        for _ in range(n_ao):
            prim_ao_to_l[ao_idx] = l
            ao_idx += 1

    # Count s-s vs other contributions
    n_ss = 0
    n_other = 0
    for a in range(n_prim):
        for b in range(n_prim):
            if prim_ao_to_l[a] == 0 and prim_ao_to_l[b] == 0:
                n_ss += 1
            else:
                n_other += 1
    print(
        f"\nPrimitive AO pairs: {n_ss} s-s, {n_other} other "
        f"({n_other / (n_ss + n_other) * 100:.1f}% other)"
    )

    # Compute spheropole with only s-s primitive pairs
    M_lat = compute_multipole_moments_lattice(
        prim_basis, system, lat_opts, 2, (0.0, 0.0, 0.0)
    )
    gamma_ab = prim_alpha[:, None] + prim_alpha[None, :]
    N_outer = prim_N_lib[:, None] * prim_N_lib[None, :]
    weight = gamma_ab / N_outer
    prim_origins_sq = (prim_origins**2).sum(axis=1)

    E_ss = 0.0
    E_other = 0.0
    V_cell = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    prefactor = (system.n_electrons() / 6.0) / (2.0 * V_cell * math.pi**1.5)

    for c, cell in enumerate(M_lat.cells):
        P_g = np.asarray(P_real.blocks[c], dtype=float)
        if P_g.size == 0:
            continue
        R_g = np.asarray(cell.r_cart, dtype=float)

        S_g = np.asarray(M_lat.blocks[c][0], dtype=float)
        Dx_g = np.asarray(M_lat.blocks[c][1], dtype=float)
        Dy_g = np.asarray(M_lat.blocks[c][2], dtype=float)
        Dz_g = np.asarray(M_lat.blocks[c][3], dtype=float)
        Qxx_g = np.asarray(M_lat.blocks[c][4], dtype=float)
        Qyy_g = np.asarray(M_lat.blocks[c][7], dtype=float)
        Qzz_g = np.asarray(M_lat.blocks[c][9], dtype=float)

        B_g = prim_origins + R_g[None, :]
        B_g_sq = (B_g**2).sum(axis=1)
        A_plus_B = prim_origins[:, None, :] + B_g[None, :, :]
        A_sq_plus_B_sq = prim_origins_sq[:, None] + B_g_sq[None, :]

        M_prim_g = 2.0 * (Qxx_g + Qyy_g + Qzz_g)
        M_prim_g -= 2.0 * (
            A_plus_B[..., 0] * Dx_g + A_plus_B[..., 1] * Dy_g + A_plus_B[..., 2] * Dz_g
        )
        M_prim_g += A_sq_plus_B_sq * S_g

        weighted = weight * M_prim_g

        # Split into s-s and other
        ss_mask = np.zeros((n_prim, n_prim), dtype=bool)
        for a in range(n_prim):
            for b in range(n_prim):
                if prim_ao_to_l[a] == 0 and prim_ao_to_l[b] == 0:
                    ss_mask[a, b] = True

        weighted_ss = weighted.copy()
        weighted_ss[~ss_mask] = 0.0
        weighted_other = weighted.copy()
        weighted_other[ss_mask] = 0.0

        SHIFT_ss = contraction_map @ weighted_ss @ contraction_map.T
        SHIFT_other = contraction_map @ weighted_other @ contraction_map.T

        E_ss += float(np.einsum("mn,mn->", P_g, SHIFT_ss))
        E_other += float(np.einsum("mn,mn->", P_g, SHIFT_other))

    print(f"\nPer-shell-type decomposition:")
    print(f"  E(s-s only)   = {prefactor * E_ss:+.6f} Ha")
    print(f"  E(other only) = {prefactor * E_other:+.6f} Ha")
    print(f"  E(total)      = {prefactor * (E_ss + E_other):+.6f} Ha")

    # Compare with contracted-basis direct computation
    M_contracted = compute_multipole_moments_lattice(
        basis, system, lat_opts, 2, (0.0, 0.0, 0.0)
    )
    # Contracted emultipole2: just tr(Q) = r² moment
    E_direct_r2 = 0.0
    for c in range(len(M_contracted.cells)):
        P_g = np.asarray(P_real.blocks[c], dtype=float)
        if P_g.size == 0:
            continue
        trQ = (
            np.asarray(M_contracted.blocks[c][4], dtype=float)
            + np.asarray(M_contracted.blocks[c][7], dtype=float)
            + np.asarray(M_contracted.blocks[c][9], dtype=float)
        )
        E_direct_r2 += float(np.sum(P_g * trQ))

    # The direct r² integral: ⟨r²⟩_rho = Σ P_μν · ⟨χ_μ|r²|χ_ν⟩
    print(f"\nDirect ⟨r²⟩ (contracted basis): {E_direct_r2:.6f} e·bohr²")
    n_elec = system.n_electrons()
    print(f"  ⟨r²⟩/N_elec = {E_direct_r2 / n_elec:.6f} bohr²")

    # Estimate spheropole from ⟨r²⟩ using the same prefactor as the bond-symmetrised formula
    # The CRYSTAL formula relates the bond-symmetrised integral to the spheropole.
    # For the contracted basis (no primitive splitting), the integral is different.
    # Let's try: E_sphero ≈ prefactor * (something) * E_direct_r2
    # From the s-s derivation: bond_sym = 2*r² - 2*(A+B)·D + (|A|²+|B|²)*S
    # The bond-symmetrised version is approximately 2*r² for well-separated atoms.
    # For MgO at g=0: A≈0, B≈2.1 bohr, D≈(B/2)*S, so:
    # bond_sym ≈ 2*r² - 2*(0+2.1)*(1.05)*S + (0+4.41)*S ≈ 2*r² - 4.41*S + 4.41*S = 2*r²
    print(
        f"  Rough estimate (assuming bond_sym ≈ 2*⟨r²⟩): "
        f"{prefactor * 2.0 * E_direct_r2:+.6f} Ha"
    )


if __name__ == "__main__":
    main()
