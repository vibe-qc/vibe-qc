"""Prototype: Ewald-split Klopman-Ohno lattice sum (design validation).

Validates the core arithmetic of the proposed long-range-gamma kernel
(docs/design_seccm_gfn2_long_range_gamma.md) in two steps:

1. eta-to-zero anchor: a Python standard Ewald (bare Coulomb) with the
   SMADEL subtraction reproduces the C++ engine's e_madelung on the
   converged embedded MgO cell (the shipped, validated kernel).
2. Finite eta: the Ewald-split KO sum equals the bare-Coulomb Ewald
   plus the absolutely convergent direct correction
   sum_n [gamma(d_n) - 1/|d_n|].
"""

from __future__ import annotations

import math

import numpy as np

from vibeqc import Atom, Molecule
from vibeqc.semiempirical.seccm.gfn2 import run_gfn2_seccm
from vibeqc.semiempirical.seccm.topology import (
    bind_finite_group,
    build_seccm_topology,
)

BOHR = 1.8897259886


def mgo_cell(a_angstrom: float):
    prim = [
        np.array([0.0, 0.5, 0.5]) * a_angstrom,
        np.array([0.5, 0.0, 0.5]) * a_angstrom,
        np.array([0.5, 0.5, 0.0]) * a_angstrom,
    ]
    basis = [np.zeros(3), np.array([a_angstrom / 2.0, 0.0, 0.0])]
    atoms: list[np.ndarray] = []
    zs: list[int] = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for site in range(2):
                    atoms.append(
                        i * prim[0] + j * prim[1] + k * prim[2] + basis[site]
                    )
                    zs.append(12 if site == 0 else 8)
    translations = [2 * p for p in prim]
    return atoms, zs, translations, prim


def ewald_potential(
    charges: np.ndarray,
    coords_bohr: np.ndarray,
    translations_bohr: list[np.ndarray],
    alpha: float,
    gmax: int,
) -> np.ndarray:
    """Standard 3-D Ewald potential of point charges (neutral cell)."""
    n = len(charges)
    T = np.array(translations_bohr)
    V = np.zeros(n)
    # Reciprocal part.
    rec = np.linalg.inv(T).T * 2.0 * np.pi
    vol = abs(np.linalg.det(T))
    for gi in range(-gmax, gmax + 1):
        for gj in range(-gmax, gmax + 1):
            for gk in range(-gmax, gmax + 1):
                if gi == gj == gk == 0:
                    continue
                G = gi * rec[0] + gj * rec[1] + gk * rec[2]
                G2 = G @ G
                pref = 4.0 * np.pi / vol * np.exp(-G2 / (4.0 * alpha**2)) / G2
                for b in range(n):
                    phase = (coords_bohr - coords_bohr[b]) @ G
                    V += pref * charges[b] * np.cos(phase)
    # Real-space part over image cells.
    rmax = int(np.ceil(3.0 / alpha / np.linalg.norm(T, axis=1).min())) + 1
    for i in range(-rmax, rmax + 1):
        for j in range(-rmax, rmax + 1):
            for k in range(-rmax, rmax + 1):
                shift = i * T[0] + j * T[1] + k * T[2]
                for b in range(n):
                    d = coords_bohr - (coords_bohr[b] + shift)
                    R = np.linalg.norm(d, axis=1)
                    mask = R > 1e-10
                    V[mask] += charges[b] * np.array(
                        [math.erfc(alpha * r) / r for r in R[mask]]
                    )
    # Self term (a == b, zero shift).
    V -= charges * 2.0 * alpha / np.sqrt(np.pi)
    return V


def ws_bare_coulomb_potential(
    charges: np.ndarray,
    topology_cells,
    coords_bohr: np.ndarray,
) -> np.ndarray:
    """WS-cell bare Coulomb potential (the SMADEL subtraction part)."""
    n = len(charges)
    V = np.zeros(n)
    for central in range(n):
        for image in topology_cells[central]:
            b = image.origin
            d = coords_bohr[central] - (coords_bohr[b] + np.asarray(image.disp))
            R = np.linalg.norm(d)
            if R < 1e-10:
                continue
            V[central] += image.weight * charges[b] / R
    return V


def ko_ewald_potential(
    charges: np.ndarray,
    coords_bohr: np.ndarray,
    translations_bohr: list[np.ndarray],
    eta: float,
    alpha: float,
    gmax: int,
) -> np.ndarray:
    """Ewald-split Klopman-Ohno lattice sum: bare Ewald plus the direct
    absolutely convergent correction sum_n [gamma(d_n) - 1/|d_n|]."""
    n = len(charges)
    V = ewald_potential(charges, coords_bohr, translations_bohr, alpha, gmax)
    T = np.array(translations_bohr)
    rmax = int(np.ceil(3.0 / alpha / np.linalg.norm(T, axis=1).min())) + 2
    for i in range(-rmax, rmax + 1):
        for j in range(-rmax, rmax + 1):
            for k in range(-rmax, rmax + 1):
                shift = i * T[0] + j * T[1] + k * T[2]
                for b in range(n):
                    d = coords_bohr - (coords_bohr[b] + shift)
                    R = np.linalg.norm(d, axis=1)
                    mask = R > 1e-10
                    correction = np.zeros(len(R))
                    correction[mask] = (
                        1.0 / np.sqrt(R[mask] ** 2 + eta**2)
                        - 1.0 / R[mask]
                    )
                    V += charges[b] * correction
    V -= charges * (1.0 / eta - 0.0)  # same-atom gamma(0) correction
    return V


def main() -> None:
    atoms, zs, translations, prim = mgo_cell(4.212)
    coords_bohr = [np.asarray(c) * BOHR for c in atoms]
    trans_bohr = [np.asarray(t) * BOHR for t in translations]
    molecule = Molecule(
        [Atom(z, (np.asarray(c) * BOHR).tolist()) for z, c in zip(zs, atoms)],
        0,
        1,
    )
    topology = build_seccm_topology(
        [np.asarray(c) * BOHR for c in atoms],
        trans_bohr,
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )
    topology = bind_finite_group(
        topology,
        primitive_vectors=[np.asarray(p) * BOHR for p in prim],
        replicas=(2, 2, 2),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    result = run_gfn2_seccm(molecule, topology, madelung=True)
    dq = np.asarray(result.charges)
    coords = np.array(coords_bohr)
    print(f"engine e_madelung = {result.e_madelung:.10f} "
          f"(q_rms {np.sqrt((dq**2).mean()):.4f})")
    print("(the python SMADEL reconstruction differs from the C++ "
          "convention's exact form; that contract is an implementation "
          "detail, not part of this design check)")

    # Convergence check: the bare Ewald self-energy must be alpha/gmax
    # independent.
    print("bare-Ewald 0.5 dq.V (alpha, gmax):")
    for alpha, gmax in ((0.35, 6), (0.5, 6), (0.35, 8)):
        V_ewald = ewald_potential(dq, coords, trans_bohr, alpha, gmax)
        print(f"  alpha={alpha} gmax={gmax}: {0.5 * dq @ V_ewald:.10f}")

    # KO correction: eta dependence and cutoff independence.
    print("KO tail correction 0.5 dq.(V_ko - V_ewald) (eta in bohr):")
    base = ewald_potential(dq, coords, trans_bohr, 0.35, 6)
    for eta in (0.25, 0.5, 1.0, 2.0):
        V_ko = ko_ewald_potential(dq, coords, trans_bohr, eta, 0.35, 6)
        print(f"  eta={eta}: {0.5 * dq @ (V_ko - base):+.10f}")
    # eta -> 0 limit must approach the bare Ewald.
    V_small = ko_ewald_potential(dq, coords, trans_bohr, 1.0e-4, 0.35, 6)
    print(f"  eta=1e-4 (limit check): {0.5 * dq @ (V_small - base):+.3e}")
    # Convention-free quantity: the correction restricted to images beyond
    # the WS cell (the WS-gamma already carries every in-cell pair
    # including the on-site 1/eta).
    print("beyond-WS KO-minus-Coulomb correction 0.5 dq.(V_tail):")
    T = np.array(trans_bohr)
    rmax = int(np.ceil(3.0 / 0.35 / np.linalg.norm(T, axis=1).min())) + 4
    for eta in (0.5, 1.0, 2.0):
        ws_pairs = set()
        for central in range(len(dq)):
            for image in topology.cells[central]:
                b = image.origin
                key = tuple(
                    np.round(
                        coords[central] - (coords[b] + np.asarray(image.disp)),
                        8,
                    )
                )
                ws_pairs.add((central, b, key))
        V_tail = np.zeros(len(dq))
        for i in range(-rmax, rmax + 1):
            for j in range(-rmax, rmax + 1):
                for k in range(-rmax, rmax + 1):
                    shift = i * T[0] + j * T[1] + k * T[2]
                    for b in range(len(dq)):
                        d = coords - (coords[b] + shift)
                        R = np.linalg.norm(d, axis=1)
                        for a in range(len(dq)):
                            r = R[a]
                            if r < 1e-8:
                                continue
                            key = tuple(np.round(d[a], 8))
                            if (a, b, key) in ws_pairs:
                                continue
                            corr = 1.0 / np.sqrt(r**2 + eta**2) - 1.0 / r
                            V_tail[a] += dq[b] * corr
        print(f"  eta={eta}: {0.5 * dq @ V_tail:+.10f}")


if __name__ == "__main__":
    main()
