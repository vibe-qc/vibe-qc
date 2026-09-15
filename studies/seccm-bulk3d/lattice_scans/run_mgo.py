"""compute-cluster lattice scan: MgO rocksalt 2x2x2, madelung embedding.

Usage: run_mgo.py a1 a2 a3 ...  (lattice constants in Angstrom)
One line per point, flush=True; fits a parabola through the three lowest
points and reports the minimum. Exit 1 if any point fails to converge.
"""
from __future__ import annotations

import sys

import numpy as np

import vibeqc
from vibeqc import Atom, Molecule
from vibeqc.semiempirical import run_gfn2_seccm
from vibeqc.semiempirical.seccm.topology import (
    bind_finite_group,
    build_seccm_topology,
)

BOHR = 1.8897259886


def run_point(a_angstrom: float):
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
                    atoms.append(i * prim[0] + j * prim[1] + k * prim[2] + basis[site])
                    zs.append(12 if site == 0 else 8)
    translations = [2 * p for p in prim]
    mol = Molecule(
        [Atom(z, (np.asarray(c) * BOHR).tolist()) for z, c in zip(zs, atoms)],
        0,
        1,
    )
    topo = build_seccm_topology(
        [np.asarray(c) * BOHR for c in atoms],
        [np.asarray(t) * BOHR for t in translations],
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )
    topo = bind_finite_group(
        topo,
        primitive_vectors=[np.asarray(p) * BOHR for p in prim],
        replicas=(2, 2, 2),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    res = run_gfn2_seccm(mol, topo, max_iter=3000, madelung=True)
    q = np.asarray(res.charges)
    e_atom = res.energy * res.group_order / len(q)
    return e_atom, res.n_iter, float(np.sqrt((q**2).mean())), res


def main() -> None:
    print(f"vibeqc {vibeqc.__version__}  python {sys.version.split()[0]}",
          flush=True)
    a_list = [float(a) for a in sys.argv[1:]]
    if len(a_list) < 3:
        print("need at least 3 lattice constants", flush=True)
        raise SystemExit(2)
    print(f"{'a/A':>8} {'E/atom':>12} {'iter':>6} {'q_rms':>7}", flush=True)
    results: list[tuple[float, float]] = []
    failed = 0
    for a in a_list:
        try:
            e_atom, n_iter, q_rms, res = run_point(a)
        except RuntimeError as exc:
            print(f"{a:8.4f}  FAILED: {str(exc)[:60]}", flush=True)
            failed += 1
            continue
        print(f"{a:8.4f} {e_atom:12.6f} {n_iter:6d} {q_rms:7.3f}", flush=True)
        results.append((a, e_atom))
    if failed:
        print(f"{failed} point(s) failed", flush=True)
    if len(results) < 3:
        raise SystemExit(1)
    order = sorted(results, key=lambda item: item[1])
    a_fit = np.array([item[0] for item in order[:3]])
    e_fit = np.array([item[1] for item in order[:3]])
    coeff = np.polyfit(a_fit, e_fit, 2)
    a_min = -coeff[1] / (2.0 * coeff[0])
    print(f"parabolic minimum over 3 lowest points: a = {a_min:.4f} A "
          f"(E/atom = {np.polyval(coeff, a_min):.6f})", flush=True)
    if failed:
        raise SystemExit(1)
    print("SCAN OK", flush=True)


if __name__ == "__main__":
    main()
