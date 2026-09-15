"""Molecular-driver wall probe (IID 130 / IID 101 family).

Does the molecular GFN2 polyalgorithm (auto-stabilization ladder)
converge the same Cu clusters that GFN2-SECCM cannot? If yes, the
SECCM adapter's solver is the gap; if no, the wall is the GFN2 map.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from vibeqc import Atom, Molecule
from vibeqc._vibeqc_core.semiempirical import xtb as _xtb
from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

BOHR = 1.8897259886
Z = 29  # Cu


def fcc_primitive(a_angstrom: float):
    return [
        np.array([0.0, 0.5, 0.5]) * a_angstrom,
        np.array([0.5, 0.0, 0.5]) * a_angstrom,
        np.array([0.5, 0.5, 0.0]) * a_angstrom,
    ]


def cluster(a_angstrom, replicas):
    prim = fcc_primitive(a_angstrom)
    n1, n2, n3 = replicas
    atoms = [
        i * prim[0] + j * prim[1] + k * prim[2]
        for i in range(n1)
        for j in range(n2)
        for k in range(n3)
    ]
    return Molecule(
        [Atom(Z, (np.asarray(c) * BOHR).tolist()) for c in atoms], 0, 1
    )


def run(label, mol, params, temperature, mixer=None):
    opts = _xtb.XTBSccOptions()
    opts.electronic_temperature = temperature
    opts.max_iter = 3600
    if mixer is not None:
        opts.scc_mixer = mixer
    t0 = time.time()
    result = _xtb.run_gfn2_xtb(mol, params, opts)
    dt = time.time() - t0
    print(
        f"{label:36s} conv={result.converged} n_iter={result.n_iter} "
        f"E/atom={result.energy / len(mol.atoms):+.6f} "
        f"attempts={len(result.attempts)} {dt:.1f}s",
        flush=True,
    )


def main():
    params = load_gfn2_params()
    for replicas, a in (((2, 2, 4), 3.7958), ((4, 4, 4), 3.615)):
        mol = cluster(a, replicas)
        print(f"cluster {replicas} ({len(mol.atoms)} atoms):", flush=True)
        run(f"  default poly T=0.005", mol, params, 0.005)
        run(f"  default poly T=0.002", mol, params, 0.002)
        run(f"  default poly T=0.0", mol, params, 0.0)


if __name__ == "__main__":
    sys.exit(main())
