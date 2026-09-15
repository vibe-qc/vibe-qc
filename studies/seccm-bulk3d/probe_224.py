"""Probe the 2x2x4 non-cubic fcc Cu cell (IID 130's second case)."""

from __future__ import annotations

import numpy as np

from vibeqc import Atom, Molecule
from vibeqc.semiempirical.seccm.gfn2 import run_gfn2_seccm
from vibeqc.semiempirical.seccm.topology import (
    bind_finite_group,
    build_seccm_topology,
)

BOHR = 1.8897259886
T = 0.005


def build(replicas: tuple[int, int, int], a: float):
    prim = [
        np.array([0.0, 0.5, 0.5]) * a,
        np.array([0.5, 0.0, 0.5]) * a,
        np.array([0.5, 0.5, 0.0]) * a,
    ]
    n1, n2, n3 = replicas
    atoms = [
        i * prim[0] + j * prim[1] + k * prim[2]
        for i in range(n1) for j in range(n2) for k in range(n3)
    ]
    translations = [n1 * prim[0], n2 * prim[1], n3 * prim[2]]
    molecule = Molecule(
        [Atom(29, (np.asarray(c) * BOHR).tolist()) for c in atoms], 0, 1
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
    return molecule, topology


def main() -> None:
    for a in (3.615, 3.7958):
        for replicas, max_iter in [((2, 2, 4), 800), ((2, 2, 4), 3600)]:
            molecule, topology = build(replicas, a)
            try:
                result = run_gfn2_seccm(
                    molecule,
                    topology,
                    electronic_temperature=T,
                    max_iter=max_iter,
                )
                print(f"a={a} {replicas} max_iter={max_iter}: "
                      f"RETURNED E/atom={result.energy:12.6f} "
                      f"conv={result.converged} physical={result.physical_basin} "
                      f"iter={result.n_iter}")
            except Exception as exc:  # noqa: BLE001
                print(f"a={a} {replicas} max_iter={max_iter}: "
                      f"RAISED {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
