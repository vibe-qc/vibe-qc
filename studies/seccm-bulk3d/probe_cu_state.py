"""Quick Cu 2x2x2 state check on the current tree (IID 130 follow-up)."""

from __future__ import annotations

import sys

import numpy as np

from vibeqc import Atom, Molecule
from vibeqc.semiempirical.seccm.gfn2 import run_gfn2_seccm
from vibeqc.semiempirical.seccm.topology import (
    bind_finite_group,
    build_seccm_topology,
)

BOHR = 1.8897259886
Z = 29  # Cu


def fcc_primitive(a_angstrom: float):
    return [
        np.array([0.0, 0.5, 0.5]) * a_angstrom,
        np.array([0.5, 0.0, 0.5]) * a_angstrom,
        np.array([0.5, 0.5, 0.0]) * a_angstrom,
    ]


def run(a_angstrom, replicas, ewald_gamma, madelung, temperature):
    prim = fcc_primitive(a_angstrom)
    n1, n2, n3 = replicas
    atoms = [
        i * prim[0] + j * prim[1] + k * prim[2]
        for i in range(n1)
        for j in range(n2)
        for k in range(n3)
    ]
    translations = [n1 * prim[0], n2 * prim[1], n3 * prim[2]]
    molecule = Molecule(
        [Atom(Z, (np.asarray(c) * BOHR).tolist()) for c in atoms], 0, 1
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
        replicas=replicas,
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    result = run_gfn2_seccm(
        molecule,
        topology,
        electronic_temperature=temperature,
        max_iter=3600,
        ewald_gamma=ewald_gamma,
        madelung=madelung,
    )
    q = np.asarray(result.charges)
    return result, q


def main():
    for mode in ("plain", "madelung", "ewald_gamma"):
        try:
            result, q = run(
                3.615,
                (2, 2, 2),
                ewald_gamma=(mode == "ewald_gamma"),
                madelung=(mode == "madelung"),
                temperature=0.005,
            )
            print(
                f"{mode:12s} conv={result.converged} phys={result.physical_basin} "
                f"E/atom={result.energy:+.6f} gap={result.homo_lumo_gap:.6f} "
                f"iter={result.n_iter} q_rms={np.sqrt((q**2).mean()):.4f} "
                f"q range [{q.min():+.4f},{q.max():+.4f}]"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{mode:12s} FAILED: {exc}")

    # Warm-start diagnostic: is the plain-kernel sane branch a fixed point
    # of the ewald_gamma map?
    print()
    try:
        plain, _ = run(
            3.615, (2, 2, 2), ewald_gamma=False, madelung=False, temperature=0.005
        )
        warm, q = run_warm(
            3.615,
            (2, 2, 2),
            plain.shell_charges,
            ewald_gamma=True,
            temperature=0.005,
        )
        print(
            f"{"warm-ewald":12s} conv={warm.converged} phys={warm.physical_basin} "
            f"E/atom={warm.energy:+.6f} gap={warm.homo_lumo_gap:.6f} "
            f"iter={warm.n_iter} q_rms={np.sqrt((q**2).mean()):.4f} "
            f"q range [{q.min():+.4f},{q.max():+.4f}]"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"{"warm-ewald":12s} FAILED: {exc}")


def run_warm(a_angstrom, replicas, shell_charges, ewald_gamma, temperature):
    prim = fcc_primitive(a_angstrom)
    n1, n2, n3 = replicas
    atoms = [
        i * prim[0] + j * prim[1] + k * prim[2]
        for i in range(n1)
        for j in range(n2)
        for k in range(n3)
    ]
    translations = [n1 * prim[0], n2 * prim[1], n3 * prim[2]]
    molecule = Molecule(
        [Atom(Z, (np.asarray(c) * BOHR).tolist()) for c in atoms], 0, 1
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
        replicas=replicas,
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    result = run_gfn2_seccm(
        molecule,
        topology,
        electronic_temperature=temperature,
        max_iter=3600,
        ewald_gamma=ewald_gamma,
        initial_shell_charges=np.asarray(shell_charges, dtype=float),
    )
    return result, np.asarray(result.charges)


if __name__ == "__main__":
    sys.exit(main())
