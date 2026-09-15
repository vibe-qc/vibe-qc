"""fcc metal lattice scans through the GFN2-xTB-SECCM boundary.

Bulk-3D validation ladder (user-directed, 2026-08-17): diamond C, MgO,
corundum Al2O3, and a metal (fcc Cu / Pd / Ag). This script is the metal
arm: it scans the fcc lattice constant of Cu through the GFN2-xTB-SECCM
finite-cluster engine and, where requested, the Gamma-periodic driver on
the same supercell as a cross-check.

Per-cell energies are quoted per primitive cell (per atom for fcc).
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
Z = {"Cu": 29, "Pd": 46, "Ag": 47}


def fcc_primitive(a_angstrom: float) -> list[np.ndarray]:
    """Primitive fcc lattice vectors for conventional lattice constant a."""
    return [
        np.array([0.0, 0.5, 0.5]) * a_angstrom,
        np.array([0.5, 0.0, 0.5]) * a_angstrom,
        np.array([0.5, 0.5, 0.0]) * a_angstrom,
    ]


def supercell(
    a_angstrom: float,
    replicas: tuple[int, int, int],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Build supercell atoms (angstrom), cyclic translations, primitive vectors."""
    prim = fcc_primitive(a_angstrom)
    n1, n2, n3 = replicas
    atoms = [
        i * prim[0] + j * prim[1] + k * prim[2]
        for i in range(n1)
        for j in range(n2)
        for k in range(n3)
    ]
    translations = [n1 * prim[0], n2 * prim[1], n3 * prim[2]]
    return atoms, translations, prim


def run_seccm(
    element: str,
    a_angstrom: float,
    replicas: tuple[int, int, int],
    electronic_temperature: float,
    max_iter: int,
    madelung: bool,
) -> dict:
    atoms_ang, translations_ang, prim = supercell(a_angstrom, replicas)
    molecule = Molecule(
        [Atom(Z[element], (np.asarray(c) * BOHR).tolist()) for c in atoms_ang],
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
        molecule,
        topology,
        electronic_temperature=electronic_temperature,
        max_iter=max_iter,
        madelung=madelung,
    )
    return {
        "energy": result.energy,
        "e_band0": result.e_band0,
        "e_scc": result.e_scc,
        "e_aes": result.e_aes,
        "e_3rd": result.e_3rd,
        "e_rep": result.e_repulsive,
        "gap": result.homo_lumo_gap,
        "n_iter": result.n_iter,
        "n_records": result.n_records,
        "charges": np.asarray(result.charges),
        "shell_charges": np.asarray(result.shell_charges),
    }


def run_gamma(
    element: str,
    a_angstrom: float,
    replicas: tuple[int, int, int],
    electronic_temperature: float,
    cutoff_bohr: float = 15.0,
) -> dict:
    atoms_ang, translations_ang, _ = supercell(a_angstrom, replicas)
    lattice = np.column_stack(
        [np.asarray(t) * BOHR for t in translations_ang]
    )
    system = PeriodicSystem(
        3,
        lattice,
        [Atom(Z[element], (np.asarray(c) * BOHR).tolist()) for c in atoms_ang],
        0,
        1,
    )
    opts = _xtb.XTBSccOptions()
    opts.electronic_temperature = electronic_temperature
    result = _xtb.run_gfn2_xtb_gamma(
        system, load_gfn2_params(), opts, cutoff_bohr
    )
    return {
        "energy": float(result.energy),
        "n_iter": int(result.n_iter),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--element", default="Cu", choices=sorted(Z))
    parser.add_argument("--replicas", nargs=3, type=int, default=[2, 2, 2])
    parser.add_argument(
        "--a-list", nargs="+", type=float,
        default=[3.4, 3.434, 3.5, 3.615, 3.7, 3.8],
    )
    parser.add_argument("--temperature", type=float, default=0.005)
    parser.add_argument("--max-iter", type=int, default=3600)
    parser.add_argument("--madelung", action="store_true")
    parser.add_argument("--gamma", action="store_true", help="run Gamma cross-check")
    args = parser.parse_args()

    replicas = tuple(args.replicas)
    max_iter = args.max_iter
    header = (
        f"{'a/A':>8} {'E/atom':>12} {'band0':>10} {'scc':>10} {'aes':>10} "
        f"{'3rd':>10} {'rep':>10} {'gap/eV':>8} {'iter':>6} {'rec':>6}"
    )
    for a in args.a_list:
        try:
            r = run_seccm(
                args.element, a, replicas, args.temperature, max_iter,
                args.madelung,
            )
        except Exception as exc:  # noqa: BLE001 - scan resilience
            print(f"{a:8.4f}  SECCM FAILED: {exc}")
            continue
        print(header if a == args.a_list[0] else "", end="")
        print(
            f"{a:8.4f} {r['energy']:12.6f} {r['e_band0']:10.4f} "
            f"{r['e_scc']:10.4f} {r['e_aes']:10.4f} {r['e_3rd']:10.4f} "
            f"{r['e_rep']:10.4f} {r['gap'] * 27.2114:8.3f} "
            f"{r['n_iter']:6d} {r['n_records']:6d}"
        )
        print(f"        charges min/max/rms: "
              f"{r['charges'].min():+.4f}/{r['charges'].max():+.4f}/"
              f"{np.sqrt((r['charges'] ** 2).mean()):.4f}")
        if r["shell_charges"].size:
            dqs = r["shell_charges"]
            print(f"        shell dq min/max/rms: "
                  f"{dqs.min():+.4f}/{dqs.max():+.4f}/"
                  f"{np.sqrt((dqs ** 2).mean()):.4f}")
        if args.gamma:
            g = run_gamma(args.element, a, replicas, args.temperature)
            print(f"        gamma: E/atom={g['energy']:12.6f} "
                  f"iter={g['n_iter']}")


if __name__ == "__main__":
    main()
