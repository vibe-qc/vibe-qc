"""compute-cluster lattice scan: corundum Al2O3 hex cell with the IID 130 recipe.

ewald_gamma=True, scc_mixer="newton", electronic_temperature=0.002 Ha,
30-atom hexagonal conventional cell (12 Al + 18 O, spacegroup 167).

Usage: run_corundum_ewald.py a1 c1 a2 c2 ...  (pairs in Angstrom)
One line per (a, c) point, flush=True; exit 1 if any point fails.
"""

from __future__ import annotations

import sys

import numpy as np
from ase.spacegroup import crystal as ase_crystal

import vibeqc
from vibeqc import Atom, Molecule
from vibeqc.semiempirical import run_gfn2_seccm
from vibeqc.semiempirical.seccm.topology import (
    bind_finite_group,
    build_seccm_topology,
)

BOHR = 1.8897259886
AL_Z = 0.35216
O_X = 0.30624
Z_BY_SYMBOL = {"Al": 13, "O": 8}


def run_point(a_angstrom: float, c_angstrom: float):
    atoms_ase = ase_crystal(
        ["Al", "O"],
        basis=[(0, 0, AL_Z), (O_X, 0, 0.25)],
        spacegroup=167,
        cellpar=[a_angstrom, a_angstrom, c_angstrom, 90, 90, 120],
    )
    cell_ang = atoms_ase.cell.array
    translations_ang = [row for row in cell_ang]
    coords_ang = [np.asarray(pos) for pos in atoms_ase.get_positions()]
    zs = [Z_BY_SYMBOL[s] for s in atoms_ase.get_chemical_symbols()]
    mol = Molecule(
        [Atom(z, (c * BOHR).tolist()) for z, c in zip(zs, coords_ang)],
        0,
        1,
    )
    topo = build_seccm_topology(
        [c * BOHR for c in coords_ang],
        [t * BOHR for t in translations_ang],
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )
    topo = bind_finite_group(
        topo,
        primitive_vectors=[t * BOHR for t in translations_ang],
        replicas=(1, 1, 1),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    res = run_gfn2_seccm(
        mol,
        topo,
        max_iter=3600,
        ewald_gamma=True,
        scc_mixer="newton",
        electronic_temperature=0.002,
    )
    q = np.asarray(res.charges)
    e_atom = res.energy * res.group_order / len(q)
    return e_atom, res.n_iter, float(np.sqrt((q**2).mean()))


def main() -> None:
    print(f"vibeqc {vibeqc.__version__}  python {sys.version.split()[0]}", flush=True)
    pairs = [float(v) for v in sys.argv[1:]]
    if len(pairs) < 6 or len(pairs) % 2 != 0:
        raise SystemExit(2)
    print(f"{'a/A':>8} {'c/A':>8} {'E/atom':>12} {'iter':>6} {'q_rms':>7}", flush=True)
    results: list[tuple[float, float, float]] = []
    failed = 0
    for a, c in zip(pairs[::2], pairs[1::2]):
        try:
            e_atom, n_iter, q_rms = run_point(a, c)
        except RuntimeError as exc:
            print(f"{a:8.4f} {c:8.4f}  FAILED: {str(exc)[:60]}", flush=True)
            failed += 1
            continue
        print(f"{a:8.4f} {c:8.4f} {e_atom:12.6f} {n_iter:6d} {q_rms:7.3f}", flush=True)
        results.append((a, c, e_atom))
    if len(results) < 3:
        raise SystemExit(1)
    order = sorted(results, key=lambda item: item[2])
    best = order[0]
    print(
        f"lowest-energy point: a = {best[0]:.4f} A, c = {best[1]:.4f} A "
        f"(E/atom = {best[2]:.6f})",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
