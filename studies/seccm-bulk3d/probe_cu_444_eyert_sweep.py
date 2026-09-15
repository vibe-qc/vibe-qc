"""Cu 4x4x4 Eyert T x alpha sweep (IID 130 metal lattice target).

The Eyert mixer reaches a fixed point on the 64-atom map fast; the
question was whether any (T, alpha) lands in the sane basin.

CONCLUDED BY MECHANISM (2026-08-26, without finishing the sweep): the
4x4x4 map has no sane fixed point for any (T, alpha) - its neutral-point
Jacobian has 63 eigenvalues below -1 (min -22) and every fixed-point
hunt diverges into the charge-density-wave basin. See
probe_wall_fixed_point_hunt.py and handovers/HANDOVER_SECCM_BULK3D.md.
Kept for the record; running it only reproduces fail-closed exits.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from vibeqc import Atom, Molecule
from vibeqc.semiempirical.seccm.gfn2 import run_gfn2_seccm
from vibeqc.semiempirical.seccm.topology import (
    bind_finite_group,
    build_seccm_topology,
)

BOHR = 1.8897259886
Z = 29  # Cu


def build(a_angstrom):
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
        [Atom(Z, (np.asarray(c) * BOHR).tolist()) for c in atoms], 0, 1
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
    return mol, topo


def try_case(temperature, alpha):
    mol, topo = build(3.615)
    t0 = time.time()
    try:
        r = run_gfn2_seccm(
            mol,
            topo,
            max_iter=500,
            ewald_gamma=True,
            scc_mixer="broyden_eyert",
            charge_mixing=alpha,
            electronic_temperature=temperature,
        )
        q = np.asarray(r.charges)
        print(
            f"T={temperature} alpha={alpha} conv={r.converged} "
            f"phys={r.physical_basin} E/atom={r.energy:+.6f} "
            f"iter={r.n_iter} q_rms={np.sqrt((q**2).mean()):.4f} "
            f"{time.time()-t0:.0f}s",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"T={temperature} alpha={alpha} FAILED {time.time()-t0:.0f}s: "
            f"{exc}",
            flush=True,
        )


def main():
    for temperature in (0.001, 0.0005, 0.002):
        for alpha in (0.2, 0.4):
            try_case(temperature, alpha)


if __name__ == "__main__":
    sys.exit(main())
