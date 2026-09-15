"""Eyert-Broyden parameter sweep on the elongated 2x2x4 Cu cell (IID 130).

xtb's Broyden runs with bromix 0.4; the SECCM adapter passed its own
default charge_mixing (0.1) as alpha. Sweep alpha x temperature.
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


def try_case(alpha, temperature, ewald):
    molecule, topology = build(3.7958, (2, 2, 4))
    try:
        result = run_gfn2_seccm(
            molecule,
            topology,
            max_iter=3600,
            ewald_gamma=ewald,
            scc_mixer="broyden_eyert",
            charge_mixing=alpha,
            electronic_temperature=temperature,
        )
        q = np.asarray(result.charges)
        print(
            f"alpha={alpha} T={temperature} ewald={ewald} "
            f"conv={result.converged} phys={result.physical_basin} "
            f"E/atom={result.energy:+.6f} iter={result.n_iter} "
            f"q_rms={np.sqrt((q**2).mean()):.4f}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"alpha={alpha} T={temperature} ewald={ewald} FAILED: {exc}",
            flush=True,
        )


def main():
    for alpha in (0.2, 0.4, 0.6):
        for temperature in (0.002, 0.005):
            for ewald in (True, False):
                try_case(alpha, temperature, ewald)


if __name__ == "__main__":
    sys.exit(main())
