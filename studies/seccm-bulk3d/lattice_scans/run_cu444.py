"""fcc Cu 4x4x4 lattice scan (even-atom metal cell, T = 0.005 Ha).

Usage: run_cu444.py a1 a2 a3 ...  One line per point, flush=True.
Slow: ~1 h per point on a loaded workstation.
"""
from __future__ import annotations

import sys
import time

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
    atoms = [
        i * prim[0] + j * prim[1] + k * prim[2]
        for i in range(4) for j in range(4) for k in range(4)
    ]
    translations = [4 * p for p in prim]
    mol = Molecule(
        [Atom(29, (np.asarray(c) * BOHR).tolist()) for c in atoms], 0, 1
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
        replicas=(4, 4, 4),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    res = run_gfn2_seccm(
        mol, topo, max_iter=3000, electronic_temperature=0.005
    )
    q = np.asarray(res.charges)
    e_atom = res.energy * res.group_order / len(q)
    return e_atom, res.n_iter, float(np.sqrt((q**2).mean()))


def main() -> None:
    print(f"vibeqc {vibeqc.__version__}  python {sys.version.split()[0]}",
          flush=True)
    a_list = [float(a) for a in sys.argv[1:]]
    if len(a_list) < 3:
        raise SystemExit(2)
    print(f"{'a/A':>8} {'E/atom':>12} {'iter':>6} {'q_rms':>7} {'t/s':>7}",
          flush=True)
    results: list[tuple[float, float]] = []
    failed = 0
    for a in a_list:
        t0 = time.time()
        try:
            e_atom, n_iter, q_rms = run_point(a)
        except RuntimeError as exc:
            print(f"{a:8.4f}  FAILED: {str(exc)[:60]}", flush=True)
            failed += 1
            continue
        print(f"{a:8.4f} {e_atom:12.6f} {n_iter:6d} {q_rms:7.3f} "
              f"{time.time()-t0:7.0f}", flush=True)
        results.append((a, e_atom))
    if len(results) < 3:
        raise SystemExit(1)
    order = sorted(results, key=lambda item: item[1])
    a_fit = np.array([item[0] for item in order[:3]])
    e_fit = np.array([item[1] for item in order[:3]])
    coeff = np.polyfit(a_fit, e_fit, 2)
    a_min = -coeff[1] / (2.0 * coeff[0])
    print(f"parabolic minimum: a = {a_min:.4f} A "
          f"(E/atom = {np.polyval(coeff, a_min):.6f})", flush=True)
    if failed:
        raise SystemExit(1)
    print("SCAN OK", flush=True)


if __name__ == "__main__":
    main()
