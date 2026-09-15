"""Probe the 3-D SCC-DFTB-SECCM even/odd replica parity structure (IID 313).

Walks R = 2..7 on the H-Li cubic cell and reports per-primitive-cell
energies alongside the dense-k reference, to characterise the
even-replica branch divergence seen in the compute-cluster campaign lane.
"""

from __future__ import annotations

import sys
import time
from itertools import product

import numpy as np

from vibeqc import Atom, Molecule
from vibeqc.semiempirical import run_scc_dftb_seccm
from vibeqc.semiempirical.seccm import bind_finite_group, build_seccm_topology

CELL = 4.1
SITES = np.array([[0.17, 0.31, 0.0], [1.39, -0.22, 0.0]])
DENSE_K_LIMIT = -0.11893  # IID 313 reference


def topology_for(coords, primitives, replicas):
    topology = build_seccm_topology(
        coords,
        [replicas * np.asarray(p, dtype=float) for p in primitives],
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )
    return bind_finite_group(
        topology,
        primitive_vectors=[np.asarray(p, dtype=float) for p in primitives],
        replicas=(replicas, replicas, replicas),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )


def molecule_for(coords):
    return Molecule(
        [Atom(1 if k % 2 == 0 else 3, c.tolist()) for k, c in enumerate(coords)],
        0,
        1,
    )


def main():
    primitives = [
        np.array([CELL, 0.0, 0.0]),
        np.array([0.0, CELL, 0.0]),
        np.array([0.0, 0.0, CELL]),
    ]
    for r in range(2, 8):
        coords = np.array(
            [
                origin + site
                for idx in product(range(r), repeat=3)
                for origin in [
                    idx[0] * primitives[0]
                    + idx[1] * primitives[1]
                    + idx[2] * primitives[2]
                ]
                for site in SITES
            ]
        )
        t0 = time.time()
        result = run_scc_dftb_seccm(
            molecule_for(coords), topology_for(coords, primitives, r)
        )
        dt = time.time() - t0
        err = result.energy - DENSE_K_LIMIT
        print(
            f"R={r:2d}  atoms={len(coords):4d}  E/cell={result.energy:+.8f}  "
            f"err={err:+.8f}  conv={result.converged}  gap={result.homo_lumo_gap:.6f}  "
            f"{dt:.1f}s",
            flush=True,
        )
    print(f"dense-k reference: {DENSE_K_LIMIT:+.8f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
