"""Kernel element check: C++ ewald_shell_gamma_3d vs Python split-B reference.

Uses the shipped bindings (gfn2_enumerate_shells + SemiempiricalBasis) to
reconstruct the exact shell list the C++ adapter sees, rebuilds the
Ewald-split Klopman-Ohno kernel in Python (bare-Ewald + absolutely
convergent [gamma - 1/r] correction), and compares 1/2 dq . gamma . dq
against the C++ e_scc at the converged shell charges from a real run.

Also checks the eta->0 anchor against the bare-Coulomb Ewald kernel.
"""

from __future__ import annotations

import math

import numpy as np

from vibeqc import Atom, Molecule
from vibeqc._vibeqc_core.semiempirical import xtb as _xtb
from vibeqc._vibeqc_core.semiempirical import (
    gfn2_enumerate_shells,
    SemiempiricalBasis,
)
from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params
from vibeqc.semiempirical.seccm.gfn2 import run_gfn2_seccm
from vibeqc.semiempirical.seccm.topology import (
    bind_finite_group,
    build_seccm_topology,
)

BOHR = 1.8897259886
PI = math.pi
ALPHA = 0.35


def mgo_cell(a_angstrom: float):
    prim = [
        np.array([0.0, 0.5, 0.5]) * a_angstrom,
        np.array([0.5, 0.0, 0.5]) * a_angstrom,
        np.array([0.5, 0.5, 0.0]) * a_angstrom,
    ]
    basis = [np.zeros(3), np.array([a_angstrom / 2.0, 0.0, 0.0])]
    atoms, zs = [], []
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


def images(T: np.ndarray, rmax: float) -> list[np.ndarray]:
    norms = np.linalg.norm(T, axis=1)
    n1 = int(np.ceil(rmax / norms[0])) + 1
    n2 = int(np.ceil(rmax / norms[1])) + 1
    n3 = int(np.ceil(rmax / norms[2])) + 1
    out: list[np.ndarray] = []
    for i in range(-n1, n1 + 1):
        for j in range(-n2, n2 + 1):
            for k in range(-n3, n3 + 1):
                shift = i * T[0] + j * T[1] + k * T[2]
                if np.linalg.norm(shift) <= rmax:
                    out.append(shift)
    return out


def kernel_b(
    coords: np.ndarray,
    hardness: np.ndarray,
    T: np.ndarray,
    alpha: float,
    gmax: int,
    rmax: float,
    eta_override: float | None = None,
) -> np.ndarray:
    """Split-B kernel: bare Coulomb Ewald + sum_n [gamma - 1/r]."""
    n = len(coords)
    rec = np.linalg.inv(T).T * 2.0 * PI
    vol = abs(np.linalg.det(T))
    bg = PI / (alpha**2 * vol)
    self_const = 2.0 * alpha / math.sqrt(PI)

    gvecs: list[np.ndarray] = []
    gpref: list[float] = []
    gmax = int(gmax)
    for gi in range(-gmax, gmax + 1):
        for gj in range(-gmax, gmax + 1):
            for gk in range(-gmax, gmax + 1):
                if gi == gj == gk == 0:
                    continue
                g = gi * rec[0] + gj * rec[1] + gk * rec[2]
                g2 = g @ g
                gvecs.append(g)
                gpref.append(
                    4.0 * PI / vol * math.exp(-g2 / (4.0 * alpha**2)) / g2
                )

    imgs = images(T, rmax)
    G = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            d = coords[a] - coords[b]
            eta = (
                eta_override
                if eta_override is not None
                else 2.0 / (hardness[a] + hardness[b])
            )
            same_site = a == b
            value = bg
            for g, pref in zip(gvecs, gpref):
                value += pref * math.cos(g @ d)
            for shift in imgs:
                r = np.linalg.norm(d + shift)
                if r < 1e-10:
                    continue
                gamma = 1.0 / math.sqrt(r * r + eta * eta)
                value += math.erfc(alpha * r) / r + (gamma - 1.0 / r)
            if same_site:
                value += 1.0 / eta - self_const
            G[a, b] = value
    return G


def main() -> None:
    atoms, zs, translations, prim = mgo_cell(4.212)
    coords_ang = np.array(atoms)
    coords = coords_ang * BOHR
    T = np.array(translations) * BOHR
    molecule = Molecule(
        [Atom(z, (np.asarray(c) * BOHR).tolist()) for z, c in zip(zs, atoms)],
        0,
        1,
    )
    topology = build_seccm_topology(
        [np.asarray(c) * BOHR for c in atoms],
        [np.asarray(t) * BOHR for t in translations],
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
    result = run_gfn2_seccm(molecule, topology, ewald_gamma=True)
    dq = np.asarray(result.shell_charges)
    print(f"converged: iter={result.n_iter} q_rms="
          f"{np.sqrt((np.asarray(result.charges) ** 2).mean()):.4f}")

    params = load_gfn2_params()
    basis = SemiempiricalBasis.build(molecule, params, 0)
    shells = list(gfn2_enumerate_shells(basis, molecule, params))
    print(f"n_shells C++={len(dq)}, python enumeration={len(shells)}")
    hardness = np.array([s.hardness for s in shells])
    n = len(shells)
    coords_all = np.array(
        [coords[s.atom_idx] for s in shells]
    )

    # Convergence of the Python split-B kernel.
    for rmax in (25.0, 40.0, 60.0):
        g = kernel_b(coords_all, hardness, T, ALPHA, 8, rmax)
        e = 0.5 * dq @ g @ dq
        print(f"  rmax={rmax:5.1f}: 1/2 dq.G.dq = {e:+.8f}")

    # eta -> 0 anchor: off-diagonal blocks must equal the bare-Coulomb
    # Ewald kernel (with the same alpha/background convention).
    g_small = kernel_b(coords_all, hardness, T, ALPHA, 8, 60.0,
                       eta_override=1.0e-4)
    g_coul = kernel_b(coords_all, hardness, T, ALPHA, 8, 60.0,
                      eta_override=0.0)
    diff = g_small - g_coul
    print(f"eta=1e-4 vs coul: max|dG| = {np.abs(diff).max():.3e} "
          f"(on-diag excluded below)")
    mask = ~np.eye(n, dtype=bool)
    print(f"  off-diagonal max|dG| = {np.abs(diff[mask]).max():.3e}")

    # Diagonal must be 1/eta + image sum.
    g_finite = kernel_b(coords_all, hardness, T, ALPHA, 8, 60.0)
    diag_ref = 1.0 / (2.0 / (hardness + hardness))
    print(f"  diagonal vs 1/eta: {g_finite[0, 0] - diag_ref[0]:+.3e} "
          f"(shell 0, eta={2.0/hardness[0]:.4f})")

    print(f"  C++ e_scc (per prim cell, x8 -> cyclic) = "
          f"{result.cyclic_scc_energy:+.8f}")


if __name__ == "__main__":
    main()
