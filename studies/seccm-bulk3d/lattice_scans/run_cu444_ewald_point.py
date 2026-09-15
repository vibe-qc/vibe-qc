"""compute-cluster single-point driver: fcc Cu 4x4x4 with the IID 130 metal recipe.

One lattice constant per job (the 64-atom Newton map is serial and takes
hours per point). ewald_gamma=True, scc_mixer="newton",
electronic_temperature=0.002 Ha.

Usage: run_cu444_ewald_point.py A
Prints one result line; exit 1 if the point fails.
"""

from __future__ import annotations

import sys

import numpy as np

import vibeqc
from vibeqc import Atom, Molecule
from vibeqc.semiempirical import run_gfn2_seccm
from vibeqc.semiempirical.seccm.topology import (
    bind_finite_group,
    build_seccm_topology,
)

BOHR = 1.8897259886
Z = 29  # Cu


def run_point(a_angstrom: float):
    prim = [
        np.array([0.0, 0.5, 0.5]) * a_angstrom,
        np.array([0.5, 0.0, 0.5]) * a_angstrom,
        np.array([0.5, 0.5, 0.0]) * a_angstrom,
    ]
    atoms = [
        i * prim[0] + j * prim[1] + k * prim[2]
        for i in range(4)
        for j in range(4)
        for k in range(4)
    ]
    translations = [4 * p for p in prim]
    mol = Molecule(
        [Atom(Z, (np.asarray(c) * BOHR).tolist()) for c in atoms],
        0,
        1,
    )
    topo = build_seccm_topology(
        [np.asarray(c) * BOHR for c in atoms],
        [np.asarray(t) * BOHR for t in translations],
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )
    topo = bind_finite_group(
        topo,
        primitive_vectors=[np.asarray(p) * BOHR for p in prim],
        replicas=(4, 4, 4),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    res = run_gfn2_seccm(
        mol,
        topo,
        max_iter=3600,
        ewald_gamma=True,
        scc_mixer="newton",
        electronic_temperature=0.002,
    )
    q = np.asarray(res.charges)
    dq = np.asarray(res.shell_charges)
    e_atom = res.energy * res.group_order / len(q)
    print(
        f"a={a_angstrom:7.4f} E/atom={e_atom:+.8f} iter={res.n_iter} "
        f"q_rms={np.sqrt((q**2).mean()):.4f} "
        f"shell_rms={np.sqrt((dq**2).mean()):.4f}",
        flush=True,
    )


def main() -> None:
    print(f"vibeqc {vibeqc.__version__}  python {sys.version.split()[0]}", flush=True)
    if len(sys.argv) != 2:
        raise SystemExit(2)
    run_point(float(sys.argv[1]))


if __name__ == "__main__":
    sys.exit(main())
