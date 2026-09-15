"""Probe the physical-basin gate on the compressed 2x2x2 Cu cell."""

from __future__ import annotations

import numpy as np

from vibeqc import Atom, Molecule
from vibeqc.semiempirical.seccm.gfn2 import run_gfn2_seccm
from vibeqc.semiempirical.seccm.topology import (
    bind_finite_group,
    build_seccm_topology,
)

BOHR = 1.8897259886
A = 3.434
T = 0.005


def main() -> None:
    prim = [
        np.array([0.0, 0.5, 0.5]) * A,
        np.array([0.5, 0.0, 0.5]) * A,
        np.array([0.5, 0.5, 0.0]) * A,
    ]
    atoms = [
        i * prim[0] + j * prim[1] + k * prim[2]
        for i in range(2) for j in range(2) for k in range(2)
    ]
    translations = [2 * p for p in prim]
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
        replicas=(2, 2, 2),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    try:
        result = run_gfn2_seccm(
            molecule, topology, electronic_temperature=T, max_iter=3600
        )
        print(f"RETURNED: converged={result.converged} "
              f"physical_basin={result.physical_basin} "
              f"energy={result.energy} n_iter={result.n_iter}")
    except Exception as exc:  # noqa: BLE001
        print(f"RAISED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
