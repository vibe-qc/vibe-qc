"""MgO(100) slab ladder through the GFN2-xTB-SECCM boundary (2-D).

Rocksalt (100) is non-polar (Tasker type 1): every plane is a neutral
Mg/O checkerboard. The 2-D cyclic topology uses the primitive surface
vectors (a/2,a/2,0) and (-a/2,a/2,0), with two atoms per primitive plane.
The finite z stack carries no translation, so the vacuum direction is an
ordinary finite-cluster direction.

The opt-in Madelung/Ewald embedding uses the 2-D Parry/Heyes kernel.
The surface energy follows from E_slab(n) = n*E_layer + 2*A*gamma with
E_layer = 2 x E_bulk/atom from the embedded 3-D bulk cell at the same a.
"""

from __future__ import annotations

import argparse

import numpy as np

from vibeqc import Atom, Molecule
from vibeqc.semiempirical.seccm.gfn2 import run_gfn2_seccm
from vibeqc.semiempirical.seccm.topology import (
    bind_finite_group,
    build_seccm_topology,
)

BOHR = 1.8897259886


def mgo100_slab(
    a_angstrom: float, layers: int
) -> tuple[list[np.ndarray], list[int], list[np.ndarray], list[np.ndarray]]:
    """B1 rocksalt(100): neutral checkerboard planes, 2x2 surface cell."""
    if layers < 1:
        raise ValueError("layers must be positive")
    t1 = np.array([a_angstrom / 2.0, a_angstrom / 2.0, 0.0])
    t2 = np.array([-a_angstrom / 2.0, a_angstrom / 2.0, 0.0])
    unlike_offset = np.array([a_angstrom / 2.0, 0.0, 0.0])
    atoms: list[np.ndarray] = []
    zs: list[int] = []
    for k in range(layers):
        z_offset = np.array([0.0, 0.0, k * a_angstrom / 2.0])
        for i in range(2):
            for j in range(2):
                home = i * t1 + j * t2 + z_offset
                if k % 2 == 0:
                    atoms.extend((home, home + unlike_offset))
                    zs.extend((12, 8))
                else:
                    atoms.extend((home + unlike_offset, home))
                    zs.extend((12, 8))
    translations = [2.0 * t1, 2.0 * t2]
    primitives = [t1, t2]
    return atoms, zs, translations, primitives


def run_slab(a_angstrom: float, layers: int, temperature: float) -> dict:
    atoms_ang, zs, translations_ang, prim = mgo100_slab(a_angstrom, layers)
    molecule = Molecule(
        [Atom(z, (np.asarray(c) * BOHR).tolist()) for z, c in zip(zs, atoms_ang)],
        0,
        1,
    )
    topology = build_seccm_topology(
        [np.asarray(c) * BOHR for c in atoms_ang],
        [np.asarray(t) * BOHR for t in translations_ang],
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )
    topology = bind_finite_group(
        topology,
        primitive_vectors=[np.asarray(p) * BOHR for p in prim],
        replicas=(2, 2, 1),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    result = run_gfn2_seccm(
        molecule, topology, madelung=True,
        electronic_temperature=temperature,
    )
    return {
        "energy": result.energy,
        "gap": result.homo_lumo_gap,
        "n_iter": result.n_iter,
        "e_mad": result.e_madelung,
        "charges": np.asarray(result.charges),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=float, default=4.212)
    parser.add_argument(
        "--layers", nargs="+", type=int, default=[2, 4, 6]
    )
    parser.add_argument("--bulk-e-per-atom", type=float, default=-2.3734)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    print(f"{'layers':>7} {'E/prim':>13} {'E/atom':>12} {'gap/eV':>8} "
          f"{'iter':>6} {'e_mad':>10} {'gamma/J-m2':>11}")
    results = {}
    for n in args.layers:
        try:
            r = run_slab(args.a, n, args.temperature)
        except Exception as exc:  # noqa: BLE001
            print(f"{n:7d}  FAILED: {exc}")
            continue
        results[n] = r
        n_atoms = 2 * n
        e_layer = 2.0 * args.bulk_e_per_atom
        area_m2 = 0.5 * (args.a * 1.0e-10) ** 2
        gamma_ha = (r["energy"] - n * e_layer) / 2.0
        gamma_jm2 = gamma_ha * 4.359744722e-18 / area_m2
        print(f"{n:7d} {r['energy']:13.6f} {r['energy'] / n_atoms:12.6f} "
              f"{r['gap'] * 27.2114:8.3f} {r['n_iter']:6d} "
              f"{r['e_mad']:10.5f} {gamma_jm2:11.3f}")
        print(f"        q min/max/rms: "
              f"{r['charges'].min():+.3f}/{r['charges'].max():+.3f}/"
              f"{np.sqrt((r['charges'] ** 2).mean()):.3f}")


if __name__ == "__main__":
    main()
