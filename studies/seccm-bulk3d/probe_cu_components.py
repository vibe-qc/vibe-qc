"""Component decomposition of the fcc Cu SECCM energy vs lattice constant
(IID 130): isolate which channel carries the a-dependent offset vs xtb."""

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


def run(a_angstrom):
    prim = fcc_primitive(a_angstrom)
    atoms = [
        i * prim[0] + j * prim[1] + k * prim[2]
        for i in range(2)
        for j in range(2)
        for k in range(2)
    ]
    translations = [2 * prim[0], 2 * prim[1], 2 * prim[2]]
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
        replicas=(2, 2, 2),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    return run_gfn2_seccm(
        molecule,
        topology,
        max_iter=3600,
        electronic_temperature=0.002,
        ewald_gamma=True,
        scc_mixer="newton",
    )


def main():
    print(
        f"{'a':>6} {'E':>10} {'band0':>10} {'scc':>10} {'aes':>10} "
        f"{'3rd':>10} {'rep':>10} {'mad':>10}"
    )
    for a in (3.4, 3.615, 3.8, 4.0, 4.2, 4.4):
        result = run(a)
        print(
            f"{a:6.3f} {result.energy:10.6f} {result.e_band0:10.4f} "
            f"{result.e_scc:10.4f} {result.e_aes:10.4f} {result.e_3rd:10.4f} "
            f"{result.e_repulsive:10.4f} {result.e_madelung:10.4f}",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())
