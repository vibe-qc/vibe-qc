"""Corundum 2x2x2 hex cluster (240 atoms) probe (IID 130).

The 1x1x1 hex conventional cell is boundary-dominated (234/870 pairs are
WS ties), so its screened S suppresses the ionic branch. A 2x2x2 hex
cluster is interior-dominated; if the ionic branch returns (q_rms toward
xtb's 0.49) the corundum path is the size ladder, not a topology change.
Plain WS kernel first (cheapest); ewald_gamma at the end.
"""

from __future__ import annotations

import sys

import numpy as np

from vibeqc import Atom, Molecule
from vibeqc.semiempirical import run_gfn2_seccm
from vibeqc.semiempirical.seccm.topology import (
    bind_finite_group,
    build_seccm_topology,
)

BOHR = 1.8897259886
AL_Z = 0.35216
O_X = 0.30624
Z_BY_SYMBOL = {"Al": 13, "O": 8}


def build_cell(a_angstrom, c_angstrom, replicas):
    from ase.spacegroup import crystal as ase_crystal

    atoms_ase = ase_crystal(
        ["Al", "O"],
        basis=[(0, 0, AL_Z), (O_X, 0, 0.25)],
        spacegroup=167,
        cellpar=[a_angstrom, a_angstrom, c_angstrom, 90, 90, 120],
    )
    prim = [np.asarray(row) for row in atoms_ase.cell.array]
    sites = [np.asarray(p) for p in atoms_ase.get_positions()]
    zs = [Z_BY_SYMBOL[s] for s in atoms_ase.get_chemical_symbols()]
    n1, n2, n3 = replicas
    atoms = [
        i * prim[0] + j * prim[1] + k * prim[2] + site
        for i in range(n1)
        for j in range(n2)
        for k in range(n3)
        for site in sites
    ]
    zs = zs * (n1 * n2 * n3)
    translations = [n1 * prim[0], n2 * prim[1], n3 * prim[2]]
    mol = Molecule(
        [Atom(z, (c * BOHR).tolist()) for z, c in zip(zs, atoms)], 0, 1
    )
    topo = build_seccm_topology(
        [c * BOHR for c in atoms],
        [t * BOHR for t in translations],
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )
    topo = bind_finite_group(
        topo,
        primitive_vectors=[p * BOHR for p in prim],
        replicas=replicas,
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    return mol, topo


def run(a_angstrom, ewald_gamma):
    mol, topo = build_cell(a_angstrom, 12.99, (2, 2, 2))
    result = run_gfn2_seccm(
        mol,
        topo,
        max_iter=3600,
        ewald_gamma=ewald_gamma,
        scc_mixer="simple",
        electronic_temperature=0.002,
    )
    q = np.asarray(result.charges)
    e_atom = result.energy * result.group_order / len(q)
    S = np.asarray(result.overlap)
    ev = np.linalg.eigvalsh(S)
    print(
        f"a={a_angstrom:7.3f} ewald={ewald_gamma} E/atom={e_atom:+.6f} "
        f"iter={result.n_iter} q_rms={np.sqrt((q**2).mean()):.4f} "
        f"S_min={ev.min():+.6f} neg={(ev < 1e-6).sum()}",
        flush=True,
    )


def main():
    # compute-cluster-style: lattice constants from argv; one line per point.
    if len(sys.argv) > 1:
        for a in (float(v) for v in sys.argv[1:]):
            run(a, ewald_gamma=False)
        return
    for a in (4.55, 4.759, 4.95):
        run(a, ewald_gamma=False)


if __name__ == "__main__":
    sys.exit(main())
