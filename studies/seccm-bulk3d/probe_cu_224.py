"""fcc Cu 2x2x4 with the metal recipe (IID 130 closure arm 2)."""

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


def run(a_angstrom, replicas, temperature, mixer="newton", **extra):
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
        max_iter=3600,
        electronic_temperature=temperature,
        ewald_gamma=True,
        scc_mixer=mixer,
        **extra,
    )
    q = np.asarray(result.charges)
    dq = np.asarray(result.shell_charges)
    print(
        f"a={a_angstrom:7.4f} E/atom={result.energy:+.6f} "
        f"gap={result.homo_lumo_gap:.6f} iter={result.n_iter} "
        f"q_rms={np.sqrt((q**2).mean()):.4f} "
        f"shell_rms={np.sqrt((dq**2).mean()):.4f}",
        flush=True,
    )


def main():
    # The historical IID 130 blowup cell (2x2x4 at a=3.7958): which
    # mixer/temperature combination reaches the sane basin?
    cases = [
        ("newton T=0.002", dict(mixer="newton", temperature=0.002)),
        ("newton T=0.001", dict(mixer="newton", temperature=0.001)),
        ("newton T=0.0005", dict(mixer="newton", temperature=0.0005)),
        (
            "simple T=0.002 onsite-mol",
            dict(
                mixer="simple",
                temperature=0.002,
                ewald_gamma_molecular_onsite=True,
            ),
        ),
        (
            "newton T=0.002 onsite-mol",
            dict(
                mixer="newton",
                temperature=0.002,
                ewald_gamma_molecular_onsite=True,
            ),
        ),
    ]
    for label, kwargs in cases:
        print(f"{label:28s} ", end="", flush=True)
        try:
            run(3.7958, (2, 2, 4), kwargs.pop("temperature"), kwargs.pop("mixer"), **kwargs)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED: {exc}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
