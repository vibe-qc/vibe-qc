"""Cu 2x2x2 kernel/mixer matrix probe (IID 130): which combination reaches
the sane branch with the image-summed gamma."""

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


def build(a_angstrom, replicas):
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
    return molecule, topology


def try_case(label, a, replicas, **kwargs):
    molecule, topology = build(a, replicas)
    try:
        result = run_gfn2_seccm(molecule, topology, max_iter=3600, **kwargs)
        q = np.asarray(result.charges)
        dq = np.asarray(result.shell_charges)
        print(
            f"{label:34s} conv={result.converged} phys={result.physical_basin} "
            f"E/atom={result.energy:+.6f} gap={result.homo_lumo_gap:.6f} "
            f"iter={result.n_iter} q_rms={np.sqrt((q**2).mean()):.4f} "
            f"shell_rms={np.sqrt((dq**2).mean()):.4f}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"{label:34s} FAILED: {exc}", flush=True)


def main():
    cases = [
        ("plain T=0.005", dict(electronic_temperature=0.005)),
        (
            "ewald T=0.005 newton",
            dict(
                electronic_temperature=0.005,
                ewald_gamma=True,
                scc_mixer="newton",
            ),
        ),
        (
            "ewald T=0.002 newton",
            dict(
                electronic_temperature=0.002,
                ewald_gamma=True,
                scc_mixer="newton",
            ),
        ),
        (
            "ewald T=0.005 newton onsite-mol",
            dict(
                electronic_temperature=0.005,
                ewald_gamma=True,
                ewald_gamma_molecular_onsite=True,
                scc_mixer="newton",
            ),
        ),
        (
            "ewald T=0.005 simple onsite-mol",
            dict(
                electronic_temperature=0.005,
                ewald_gamma=True,
                ewald_gamma_molecular_onsite=True,
            ),
        ),
    ]
    for label, kwargs in cases:
        try_case(label, 3.615, (2, 2, 2), **kwargs)


if __name__ == "__main__":
    sys.exit(main())
