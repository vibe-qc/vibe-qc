"""Prototype: pure-reciprocal lattice sum of the Klopman-Ohno shell gamma.

The KO kernel gamma(R) = 1/sqrt(R^2 + eta^2) has no singularity at R=0,
so its periodic lattice sum is well-defined in reciprocal space:

    Gamma_ab = sum_{G != 0} (4 pi / V) * eta_ab * K1(G eta_ab) / G * cos(G.d)

with K1 the modified Bessel function (exponential decay at large G makes
the series absolutely convergent; the finite eta is the short-range
regulator, so no erfc split / background / self terms are needed).

Checks on the MgO 2x2x2 cell:
1. G-cut convergence of 1/2 dq.Gamma.dq at the converged shell charges;
2. the diagonal equals 1/eta_aa in the dense-grid (large supercell) limit;
3. the eta -> 0 energy limit equals the bare-Coulomb Ewald energy.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import k1 as _k1

from vibeqc import Atom, Molecule
from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params
from vibeqc.semiempirical.seccm.gfn2 import run_gfn2_seccm
from vibeqc.semiempirical.seccm.topology import (
    bind_finite_group,
    build_seccm_topology,
)
from vibeqc._vibeqc_core.semiempirical import (
    gfn2_enumerate_shells,
    SemiempiricalBasis,
)

BOHR = 1.8897259886
PI = math.pi


def mgo_cell(a_angstrom: float, reps=(2, 2, 2)):
    prim = [
        np.array([0.0, 0.5, 0.5]) * a_angstrom,
        np.array([0.5, 0.0, 0.5]) * a_angstrom,
        np.array([0.5, 0.5, 0.0]) * a_angstrom,
    ]
    basis = [np.zeros(3), np.array([a_angstrom / 2.0, 0.0, 0.0])]
    n1, n2, n3 = reps
    atoms, zs = [], []
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                for site in range(2):
                    atoms.append(
                        i * prim[0] + j * prim[1] + k * prim[2] + basis[site]
                    )
                    zs.append(12 if site == 0 else 8)
    translations = [n1 * prim[0], n2 * prim[1], n3 * prim[2]]
    return atoms, zs, translations, prim


def g_grid(T: np.ndarray, g_cut: float) -> np.ndarray:
    rec = np.linalg.inv(T).T * 2.0 * PI
    n1 = int(np.ceil(g_cut / np.linalg.norm(rec[0]))) + 1
    n2 = int(np.ceil(g_cut / np.linalg.norm(rec[1]))) + 1
    n3 = int(np.ceil(g_cut / np.linalg.norm(rec[2]))) + 1
    out = []
    for i in range(-n1, n1 + 1):
        for j in range(-n2, n2 + 1):
            for k in range(-n3, n3 + 1):
                if i == j == k == 0:
                    continue
                g = i * rec[0] + j * rec[1] + k * rec[2]
                if np.linalg.norm(g) <= g_cut:
                    out.append(g)
    return np.array(out)


def reciprocal_kernel(
    coords: np.ndarray,
    hardness: np.ndarray,
    T: np.ndarray,
    g_cut: float,
    eta_override: float | None = None,
) -> np.ndarray:
    n = len(coords)
    vol = abs(np.linalg.det(T))
    G = g_grid(T, g_cut)
    gn = np.linalg.norm(G, axis=1)
    gamma = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            d = coords[a] - coords[b]
            eta = (
                eta_override
                if eta_override is not None
                else 2.0 / (hardness[a] + hardness[b])
            )
            phase = G @ d
            terms = 4.0 * PI / vol * eta * _k1(gn * eta) / gn * np.cos(phase)
            gamma[a, b] = terms.sum()
    return gamma


def bare_coulomb_ewald(
    coords: np.ndarray, T: np.ndarray, alpha: float, gmax: int, rmax: float
) -> np.ndarray:
    """Validated bare-Coulomb Ewald kernel (real[erfc] + rec[erf] + bg + self)."""
    n = len(coords)
    rec = np.linalg.inv(T).T * 2.0 * PI
    vol = abs(np.linalg.det(T))
    bg = PI / (alpha**2 * vol)
    G = np.zeros((n, n))
    gvecs, gpref = [], []
    for gi in range(-gmax, gmax + 1):
        for gj in range(-gmax, gmax + 1):
            for gk in range(-gmax, gmax + 1):
                if gi == gj == gk == 0:
                    continue
                g = gi * rec[0] + gj * rec[1] + gk * rec[2]
                g2 = g @ g
                gvecs.append(g)
                gpref.append(4.0 * PI / vol * math.exp(-g2 / (4.0 * alpha**2)) / g2)
    norms = np.linalg.norm(T, axis=1)
    imgs = []
    for i in range(-int(rmax / norms[0]) - 1, int(rmax / norms[0]) + 2):
        for j in range(-int(rmax / norms[1]) - 1, int(rmax / norms[1]) + 2):
            for k in range(-int(rmax / norms[2]) - 1, int(rmax / norms[2]) + 2):
                shift = i * T[0] + j * T[1] + k * T[2]
                if np.linalg.norm(shift) <= rmax:
                    imgs.append(shift)
    for a in range(n):
        for b in range(n):
            d = coords[a] - coords[b]
            value = bg
            for g, pref in zip(gvecs, gpref):
                value += pref * math.cos(g @ d)
            for shift in imgs:
                r = np.linalg.norm(d + shift)
                if r < 1e-8:
                    continue
                value += math.erfc(alpha * r) / r
            if a == b:
                value -= 2.0 * alpha / math.sqrt(PI)
            G[a, b] = value
    return G


def main() -> None:
    atoms, zs, translations, prim = mgo_cell(4.212)
    coords = np.array(atoms) * BOHR
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
    params = load_gfn2_params()
    basis = SemiempiricalBasis.build(molecule, params, 0)
    shells = list(gfn2_enumerate_shells(basis, molecule, params))
    hardness = np.array([s.hardness for s in shells])
    coords_all = np.array([coords[s.atom_idx] for s in shells])
    print(f"hardness range: {hardness.min():.4f} .. {hardness.max():.4f}")
    eta_min = 2.0 / (hardness.max() + hardness.max())
    print(f"eta range: {2.0/(hardness.max()+hardness.max()):.4f} .. "
          f"{2.0/(hardness.min()+hardness.min()):.4f}, eta_min={eta_min:.4f}")

    print("G-cut convergence of 1/2 dq.Gamma.dq:")
    for g_cut in (6.0, 8.0, 10.0, 12.0, 15.0):
        g = reciprocal_kernel(coords_all, hardness, T, g_cut)
        e = 0.5 * dq @ g @ dq
        print(f"  g_cut={g_cut:5.1f}: {e:+.8f}")

    # Diagonal vs 1/eta in the dense-grid limit: build a big vacuum cell.
    print("diagonal dense-limit check (one Mg atom, 3x3x3 lattice of a=30 bohr):")
    Tbig = np.eye(3) * 30.0
    coords_big = np.zeros((1, 3))
    h_big = np.array([hardness[0]])
    g_dense = reciprocal_kernel(coords_big, h_big, Tbig, 6.0)
    print(f"  Gamma_00 = {g_dense[0, 0]:+.8f}, 1/eta = {2.0/hardness[0]:+.8f}")

    # eta -> 0 energy anchor vs bare-Coulomb Ewald.
    print("eta->0 anchor vs bare-Coulomb Ewald (fixed dq):")
    g0 = reciprocal_kernel(coords_all, hardness, T, 8.0, eta_override=1e-4)
    gc = bare_coulomb_ewald(coords_all, T, 0.35, 6, 40.0)
    print(f"  KO(eta=1e-4): {0.5 * dq @ g0 @ dq:+.8f}")
    print(f"  Coulomb Ewald: {0.5 * dq @ gc @ dq:+.8f}")


if __name__ == "__main__":
    main()
