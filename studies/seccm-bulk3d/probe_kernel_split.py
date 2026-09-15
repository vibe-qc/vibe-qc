"""Kernel-convention check: C++ split-A vs prototype split-B vs raw KO sum.

Settles the algebra of ewald_shell_gamma_3d before trusting any physics:
- A = real[gamma - erfc/r] + rec[erf prefactor e^{-G^2/4a^2}] + bg + on-site
- B = bare Ewald (real[erfc] + rec[erf] + bg + self) + corr sum [gamma - 1/r]
- raw = direct KO lattice sum over a big spherical cutoff (large-N reference)
All evaluated on the MgO 2x2x2 cell. If A == B == raw (to cutoff tolerance),
the shipped split is fine; otherwise fix before running SCC physics.
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
ALPHA = 0.35
PI = math.pi


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


def images(T: np.ndarray, rmax: float) -> list[np.ndarray]:
    """All lattice images with |n*T| <= rmax, spherical truncation."""
    norms = np.linalg.norm(T, axis=1)
    out: list[np.ndarray] = []
    n1 = int(np.ceil(rmax / norms[0])) + 1
    n2 = int(np.ceil(rmax / norms[1])) + 1
    n3 = int(np.ceil(rmax / norms[2])) + 1
    for i in range(-n1, n1 + 1):
        for j in range(-n2, n2 + 1):
            for k in range(-n3, n3 + 1):
                shift = i * T[0] + j * T[1] + k * T[2]
                if np.linalg.norm(shift) <= rmax:
                    out.append(shift)
    return out


def kernel_formula(
    coords: np.ndarray,
    T: np.ndarray,
    eta: float,
    alpha: float,
    gmax: int,
    rmax: float,
    formula: str,
) -> np.ndarray:
    """Pairwise kernel matrix between the coords for a given split formula.

    eta is scalar here (single hard pair) for the convention check; the
    real kernel generalises per pair.
    """
    n = len(coords)
    rec = np.linalg.inv(T).T * 2.0 * PI
    vol = abs(np.linalg.det(T))
    bg = PI / (alpha**2 * vol)
    G = np.zeros((n, n))

    gvecs: list[np.ndarray] = []
    gpref: list[float] = []
    for gi in range(-gmax, gmax + 1):
        for gj in range(-gmax, gmax + 1):
            for gk in range(-gmax, gmax + 1):
                if gi == gj == gk == 0:
                    continue
                g = gi * rec[0] + gj * rec[1] + gk * rec[2]
                g2 = g @ g
                gvecs.append(g)
                if formula in ("A", "B", "raw"):
                    # rec[erf] piece (the validated bare-Ewald reciprocal)
                    gpref.append(
                        4.0 * PI / vol * math.exp(-g2 / (4.0 * alpha**2)) / g2
                    )
                else:
                    raise ValueError(formula)

    imgs = images(T, rmax)
    for a in range(n):
        for b in range(n):
            d = coords[a] - coords[b]
            value = bg
            for g, pref in zip(gvecs, gpref):
                value += pref * math.cos(g @ d)
            for shift in imgs:
                r = np.linalg.norm(d + shift)
                if formula in ("A", "B", "raw"):
                    if r < 1e-8:
                        continue  # on-site bookkeeping below
                    if formula == "A":
                        value += (
                            1.0 / math.sqrt(r * r + eta * eta)
                            - math.erfc(alpha * r) / r
                        )
                    elif formula == "B":
                        value += (
                            math.erfc(alpha * r) / r
                            + 1.0 / math.sqrt(r * r + eta * eta)
                            - 1.0 / r
                        )
                    else:  # raw
                        value += 1.0 / math.sqrt(r * r + eta * eta)
                else:
                    raise ValueError(formula)
            # on-site bookkeeping for the zero-displacement pair (a == b
            # here since coords are distinct atoms; same-site handled):
            if a == b:
                if formula == "A":
                    value += 1.0 / eta - 2.0 * alpha / math.sqrt(PI)
                elif formula == "B":
                    value += 1.0 / eta - 2.0 * alpha / math.sqrt(PI)
                else:  # raw
                    value += 1.0 / eta
            G[a, b] = value
    return G


def main() -> None:
    atoms, zs, translations, prim = mgo_cell(4.212)
    coords = np.array(atoms) * BOHR
    T = np.array(translations) * BOHR
    eta = 0.6  # representative shell-pair hardness in bohr

    # Charge vector from a converged embedded run so the q-weighted
    # energy comparison is physically meaningful.
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
    result = run_gfn2_seccm(molecule, topology, madelung=True)
    dq = np.asarray(result.charges)
    print(f"q_rms = {np.sqrt((dq**2).mean()):.4f}")

    for rmax in (20.0, 30.0, 40.0, 60.0):
        ga = kernel_formula(coords, T, eta, ALPHA, 8, rmax, "A")
        gb = kernel_formula(coords, T, eta, ALPHA, 8, rmax, "B")
        gr = kernel_formula(coords, T, eta, ALPHA, 8, rmax, "raw")
        ea = 0.5 * dq @ ga @ dq
        eb = 0.5 * dq @ gb @ dq
        er = 0.5 * dq @ gr @ dq
        print(f"rmax={rmax:5.1f}  E_A={ea:+.8f}  E_B={eb:+.8f}  "
              f"E_raw={er:+.8f}  A-B={ea-eb:+.2e}  A-raw={ea-er:+.2e} "
              f"B-raw={eb-er:+.2e}")


if __name__ == "__main__":
    main()
