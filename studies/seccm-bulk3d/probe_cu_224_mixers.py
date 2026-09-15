"""SECCM mixer sweep on the elongated 2x2x4 Cu cell (IID 130).

The molecular driver's polyalgorithm and SECCM simple/newton all fail
the 16-atom elongated cluster; xtb's Broyden converges it. This probe
tests SECCM's remaining mixers (diis, broyden) with the ewald_gamma
kernel and the plain WS kernel.
"""

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


def build(a_angstrom, replicas):
    prim = [
        np.array([0.0, 0.5, 0.5]) * a_angstrom,
        np.array([0.5, 0.0, 0.5]) * a_angstrom,
        np.array([0.5, 0.5, 0.0]) * a_angstrom,
    ]
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
    return molecule, topology


def try_case(label, mixer, ewald_gamma, temperature):
    molecule, topology = build(3.7958, (2, 2, 4))
    try:
        result = run_gfn2_seccm(
            molecule,
            topology,
            max_iter=3600,
            ewald_gamma=ewald_gamma,
            scc_mixer=mixer,
            electronic_temperature=temperature,
        )
        q = np.asarray(result.charges)
        print(
            f"{label:36s} conv={result.converged} phys={result.physical_basin} "
            f"E/atom={result.energy:+.6f} iter={result.n_iter} "
            f"q_rms={np.sqrt((q**2).mean()):.4f}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"{label:36s} FAILED: {exc}", flush=True)


def main():
    for mixer in ("diis", "broyden"):
        for ewald in (True, False):
            try_case(
                f"mixer={mixer} ewald={ewald} T=0.002", mixer, ewald, 0.002
            )


if __name__ == "__main__":
    sys.exit(main())
