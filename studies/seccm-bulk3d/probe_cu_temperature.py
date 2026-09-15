"""Cu 2x2x2 ewald_gamma probe: temperature sweep for the sane basin."""
from __future__ import annotations

import numpy as np

from vibeqc import Atom, Molecule
from vibeqc.semiempirical.seccm.gfn2 import run_gfn2_seccm
from vibeqc.semiempirical.seccm.topology import (
    bind_finite_group,
    build_seccm_topology,
)

BOHR = 1.8897259886


def fcc_cu(a_angstrom: float, replicas=(2, 2, 2)):
    prim = [
        np.array([0.0, 0.5, 0.5]) * a_angstrom,
        np.array([0.5, 0.0, 0.5]) * a_angstrom,
        np.array([0.5, 0.5, 0.0]) * a_angstrom,
    ]
    n1, n2, n3 = replicas
    atoms = [
        i * prim[0] + j * prim[1] + k * prim[2]
        for i in range(n1) for j in range(n2) for k in range(n3)
    ]
    translations = [n1 * prim[0], n2 * prim[1], n3 * prim[2]]
    return atoms, translations, prim, replicas


def main() -> None:
    for a in (3.615, 3.434):
        atoms, translations, prim, replicas = fcc_cu(a)
        molecule = Molecule(
            [Atom(29, (np.asarray(c) * BOHR).tolist()) for c in atoms],
            0,
            1,
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
        for T in (0.001, 0.005, 0.01, 0.02):
            try:
                r = run_gfn2_seccm(
                    molecule, topology, ewald_gamma=True,
                    electronic_temperature=T, max_iter=3600,
                )
                q = np.asarray(r.charges)
                sh = np.asarray(r.shell_charges)
                print(f"a={a} T={T}: E/atom={r.energy:12.6f} "
                      f"iter={r.n_iter:5d} q_rms={np.sqrt((q**2).mean()):6.3f} "
                      f"dqsh_rms={np.sqrt((sh**2).mean()):6.3f} "
                      f"gap={r.homo_lumo_gap*27.2114:6.3f}")
            except Exception as exc:  # noqa: BLE001
                print(f"a={a} T={T}: FAILED: {exc}")


if __name__ == "__main__":
    main()
