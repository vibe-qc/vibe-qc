"""Rocksalt (MgO) lattice scans through the GFN2-xTB-SECCM boundary.

Bulk-3D validation ladder (user-directed, 2026-08-17). MgO rocksalt:
two atoms per primitive cell (cation at the origin, anion at the body
center), cubic lattice constant a. The 2x2x2 supercell holds 16 atoms.
"""

from __future__ import annotations

import argparse

import numpy as np

from vibeqc import Atom, Molecule, PeriodicSystem
from vibeqc._vibeqc_core.semiempirical import xtb as _xtb
from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params
from vibeqc.semiempirical.seccm.gfn2 import run_gfn2_seccm
from vibeqc.semiempirical.seccm.topology import (
    bind_finite_group,
    build_seccm_topology,
)

BOHR = 1.8897259886


def rocksalt_cell(
    a_angstrom: float,
    cation_z: int,
    anion_z: int,
    replicas: tuple[int, int, int],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[int]]:
    # Rhombohedral rocksalt primitive cell: fcc-like vectors with the
    # cation at the origin and the anion at (a/2, 0, 0).
    prim = [
        np.array([0.0, 0.5, 0.5]) * a_angstrom,
        np.array([0.5, 0.0, 0.5]) * a_angstrom,
        np.array([0.5, 0.5, 0.0]) * a_angstrom,
    ]
    basis = [np.zeros(3), np.array([a_angstrom / 2.0, 0.0, 0.0])]
    n1, n2, n3 = replicas
    atoms: list[np.ndarray] = []
    zs: list[int] = []
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                for site in range(2):
                    atoms.append(
                        i * prim[0] + j * prim[1] + k * prim[2] + basis[site]
                    )
                    zs.append(cation_z if site == 0 else anion_z)
    translations = [n1 * prim[0], n2 * prim[1], n3 * prim[2]]
    return atoms, translations, prim, zs


def run_seccm(
    a_angstrom: float,
    cation_z: int,
    anion_z: int,
    replicas: tuple[int, int, int],
    max_iter: int,
    temperature: float,
    madelung: bool,
) -> dict:
    atoms_ang, translations_ang, prim, zs = rocksalt_cell(
        a_angstrom, cation_z, anion_z, replicas
    )
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
        replicas=replicas,
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    result = run_gfn2_seccm(
        molecule, topology, max_iter=max_iter,
        electronic_temperature=temperature,
        madelung=madelung,
    )
    return {
        "energy": result.energy,
        "e_band0": result.e_band0,
        "e_scc": result.e_scc,
        "e_aes": result.e_aes,
        "e_3rd": result.e_3rd,
        "e_rep": result.e_repulsive,
        "e_mad": result.e_madelung,
        "gap": result.homo_lumo_gap,
        "n_iter": result.n_iter,
        "charges": np.asarray(result.charges),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cation", type=int, default=12)
    parser.add_argument("--anion", type=int, default=8)
    parser.add_argument("--replicas", nargs=3, type=int, default=[2, 2, 2])
    parser.add_argument(
        "--a-list", nargs="+", type=float,
        default=[4.3, 4.212, 4.1, 4.0, 3.9, 3.8, 3.7, 3.6],
    )
    parser.add_argument("--max-iter", type=int, default=3600)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--madelung", action="store_true")
    args = parser.parse_args()
    replicas = tuple(args.replicas)
    print(f"{'a/A':>8} {'E/cell':>12} {'band0':>9} {'scc':>8} {'aes':>8} "
          f"{'3rd':>8} {'rep':>7} {'mad':>9} {'gap/eV':>8} {'iter':>6} "
          f"{'q_rms':>6}")
    for a in args.a_list:
        try:
            r = run_seccm(
                a, args.cation, args.anion, replicas,
                args.max_iter, args.temperature, args.madelung,
            )
        except Exception as exc:  # noqa: BLE001 - scan resilience
            print(f"{a:8.4f}  FAILED: {exc}")
            continue
        print(f"{a:8.4f} {r['energy']:12.6f} {r['e_band0']:9.4f} "
              f"{r['e_scc']:8.4f} {r['e_aes']:8.4f} {r['e_3rd']:8.4f} "
              f"{r['e_rep']:7.4f} {r['e_mad']:9.4f} "
              f"{r['gap'] * 27.2114:8.3f} "
              f"{r['n_iter']:6d} "
              f"{np.sqrt((r['charges'] ** 2).mean()):6.3f}")


if __name__ == "__main__":
    main()
