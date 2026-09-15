"""compute-cluster lattice scan: Al2O3 corundum primitive cell (30 atoms), T = 0.005.

Usage: run_corundum.py --a a1 a2 ... --c c1 c2 ...
At least one of --a / --c with >= 3 values. Scans a at fixed c0, then c at
the a-parabola minimum. One line per point, flush=True.
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
A0 = 4.7589
C0 = 12.991


def run_point(a_angstrom: float, c_angstrom: float):
    from ase.spacegroup import crystal as ase_crystal

    atoms_ase = ase_crystal(
        ["Al", "O"],
        basis=[(0, 0, 0.35216), (0.30624, 0, 0.25)],
        spacegroup=167,
        cellpar=[a_angstrom, a_angstrom, c_angstrom, 90, 90, 120],
    )
    cell_ang = atoms_ase.cell.array
    translations = [row for row in cell_ang]
    coords = [np.asarray(pos) for pos in atoms_ase.get_positions()]
    zs = [13 if s == "Al" else 8 for s in atoms_ase.get_chemical_symbols()]
    mol = Molecule(
        [Atom(z, (np.asarray(c) * BOHR).tolist()) for z, c in zip(zs, coords)],
        0,
        1,
    )
    topo = build_seccm_topology(
        [np.asarray(c) * BOHR for c in coords],
        [np.asarray(t) * BOHR for t in translations],
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )
    topo = bind_finite_group(
        topo,
        primitive_vectors=[np.asarray(p) * BOHR for p in translations],
        replicas=(1, 1, 1),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    res = run_gfn2_seccm(
        mol, topo, max_iter=3000, electronic_temperature=0.005
    )
    q = np.asarray(res.charges)
    e_atom = res.energy * res.group_order / len(q)
    return e_atom, res.n_iter, float(np.sqrt((q**2).mean()))


def parabola(xs, es):
    coeff = np.polyfit(np.array(xs), np.array(es), 2)
    return -coeff[1] / (2.0 * coeff[0]), np.polyval(coeff, -coeff[1] / (2.0 * coeff[0]))


def main() -> None:
    print(f"vibeqc {vibeqc.__version__}  python {sys.version.split()[0]}",
          flush=True)
    argv = sys.argv[1:]
    a_list: list[float] = []
    c_list: list[float] = []
    current: str | None = None
    for token in argv:
        if token in ("--a", "--c"):
            current = token
            continue
        if current == "--a":
            a_list.append(float(token))
        elif current == "--c":
            c_list.append(float(token))
        else:
            raise SystemExit(f"unexpected token: {token}")
    if len(a_list) < 3 and len(c_list) < 3:
        raise SystemExit("need at least 3 values under --a or --c")
    failed = 0
    results_a: list[tuple[float, float]] = []
    print(f"{'a/A':>7} {'c/A':>8} {'E/atom':>12} {'iter':>6} {'q_rms':>7}",
          flush=True)
    for a in a_list:
        try:
            e_atom, n_iter, q_rms = run_point(a, C0)
        except RuntimeError as exc:
            print(f"{a:7.4f} {C0:8.3f}  FAILED: {str(exc)[:50]}", flush=True)
            failed += 1
            continue
        print(f"{a:7.4f} {C0:8.3f} {e_atom:12.6f} {n_iter:6d} {q_rms:7.3f}",
              flush=True)
        results_a.append((a, e_atom))
    a_use = A0
    if len(results_a) >= 3:
        order = sorted(results_a, key=lambda item: item[1])[:3]
        a_use, _ = parabola([x[0] for x in order], [x[1] for x in order])
        print(f"a-parabola minimum: a = {a_use:.4f} A", flush=True)
    for c in c_list:
        try:
            e_atom, n_iter, q_rms = run_point(a_use, c)
        except RuntimeError as exc:
            print(f"{a_use:7.4f} {c:8.3f}  FAILED: {str(exc)[:50]}", flush=True)
            failed += 1
            continue
        print(f"{a_use:7.4f} {c:8.3f} {e_atom:12.6f} {n_iter:6d} {q_rms:7.3f}",
              flush=True)
    if failed:
        print(f"{failed} point(s) failed", flush=True)
        raise SystemExit(1)
    print("SCAN OK", flush=True)


if __name__ == "__main__":
    main()
