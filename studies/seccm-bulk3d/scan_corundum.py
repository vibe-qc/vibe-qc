"""Corundum (Al2O3) a/c lattice scans through the GFN2-xTB-SECCM boundary.

Corundum R-3c (spacegroup 167): hexagonal conventional cell, 30 atoms
(12 Al + 18 O), a_exp = 4.759-4.761 A, c_exp = 12.99-13.00 A. The cell
is built with ase.spacegroup.crystal (the same builder the ab initio
corundum examples use) so the coordinates match the rest of the repo.
"""

from __future__ import annotations

import argparse

import numpy as np
from ase.spacegroup import crystal as ase_crystal

from vibeqc import Atom, Molecule
from vibeqc.semiempirical.seccm.gfn2 import run_gfn2_seccm
from vibeqc.semiempirical.seccm.topology import (
    bind_finite_group,
    build_seccm_topology,
)

BOHR = 1.8897259886
AL_Z = 0.35216
O_X = 0.30624
Z_BY_SYMBOL = {"Al": 13, "O": 8}


def corundum_cell(a_angstrom: float, c_angstrom: float) -> tuple[
    list[np.ndarray], list[np.ndarray], list[np.ndarray], list[int]
]:
    atoms_ase = ase_crystal(
        ["Al", "O"],
        basis=[(0, 0, AL_Z), (O_X, 0, 0.25)],
        spacegroup=167,
        cellpar=[a_angstrom, a_angstrom, c_angstrom, 90, 90, 120],
    )
    cell_ang = atoms_ase.cell.array
    translations_ang = [row for row in cell_ang]
    coords_ang = [
        np.asarray(pos) for pos in atoms_ase.get_positions()
    ]
    zs = [Z_BY_SYMBOL[s] for s in atoms_ase.get_chemical_symbols()]
    return coords_ang, translations_ang, translations_ang, zs


def run_seccm(
    a_angstrom: float, c_angstrom: float, max_iter: int, temperature: float,
    madelung: bool,
) -> dict:
    atoms_ang, translations_ang, prim, zs = corundum_cell(a_angstrom, c_angstrom)
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
        replicas=(1, 1, 1),
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
        "gap": result.homo_lumo_gap,
        "n_iter": result.n_iter,
        "charges": np.asarray(result.charges),
        "shell_charges": np.asarray(result.shell_charges),
        "e_scc": result.e_scc,
        "e_3rd": result.e_3rd,
        "e_mad": result.e_madelung,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a0", type=float, default=4.7589)
    parser.add_argument("--c0", type=float, default=12.991)
    parser.add_argument(
        "--a-list", nargs="+", type=float,
        default=[4.62, 4.68, 4.7589, 4.84, 4.90],
    )
    parser.add_argument(
        "--c-list", nargs="+", type=float,
        default=[12.5, 12.8, 12.991, 13.2, 13.5],
    )
    parser.add_argument("--scan", choices=["a", "c", "both"], default="both")
    parser.add_argument("--max-iter", type=int, default=3600)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--madelung", action="store_true")
    args = parser.parse_args()

    print(f"{'a/A':>7} {'c/A':>8} {'E/cell':>14} {'gap/eV':>8} {'iter':>6} "
          f"{'q_rms':>7} {'dqsh_rms':>9} {'e_scc':>9} {'e_3rd':>9} {'e_mad':>9}")
    if args.scan in ("a", "both"):
        for a in args.a_list:
            try:
                r = run_seccm(
                    a, args.c0, args.max_iter, args.temperature,
                    args.madelung,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"{a:7.4f} {args.c0:8.3f}  FAILED: {exc}")
                continue
            print(f"{a:7.4f} {args.c0:8.3f} {r['energy']:14.6f} "
                  f"{r['gap'] * 27.2114:8.3f} {r['n_iter']:6d} "
                  f"{np.sqrt((r['charges'] ** 2).mean()):7.3f} "
                  f"{np.sqrt((r['shell_charges'] ** 2).mean()):9.3f} "
                  f"{r['e_scc']:9.4f} {r['e_3rd']:9.4f} {r['e_mad']:9.4f}")
    if args.scan in ("c", "both"):
        for c in args.c_list:
            try:
                r = run_seccm(
                    args.a0, c, args.max_iter, args.temperature,
                    args.madelung,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"{args.a0:7.4f} {c:8.3f}  FAILED: {exc}")
                continue
            print(f"{args.a0:7.4f} {c:8.3f} {r['energy']:14.6f} "
                  f"{r['gap'] * 27.2114:8.3f} {r['n_iter']:6d} "
                  f"{np.sqrt((r['charges'] ** 2).mean()):7.3f} "
                  f"{np.sqrt((r['shell_charges'] ** 2).mean()):9.3f} "
                  f"{r['e_scc']:9.4f} {r['e_3rd']:9.4f} {r['e_mad']:9.4f}")


if __name__ == "__main__":
    main()
