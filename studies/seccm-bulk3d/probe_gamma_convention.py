"""Probe: which PeriodicSystem lattice convention gives sane Gamma energies?

The previous session's Gamma numbers (-3.763 Ha/cell on the 8-atom 2x2x2
fcc Cu cell at a = 3.615 A, T = 0.005) must be reproducible; my
columns-as-vectors construction blew up. This script tests the candidate
constructions.
"""

from __future__ import annotations

import numpy as np

from vibeqc import Atom, PeriodicSystem
from vibeqc._vibeqc_core.semiempirical import xtb as _xtb
from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

BOHR = 1.8897259886
A = 3.615
T = 0.005


def fcc_primitive(a: float) -> list[np.ndarray]:
    return [
        np.array([0.0, 0.5, 0.5]) * a,
        np.array([0.5, 0.0, 0.5]) * a,
        np.array([0.5, 0.5, 0.0]) * a,
    ]


def atoms_8(a: float) -> list[np.ndarray]:
    prim = fcc_primitive(a)
    return [
        i * prim[0] + j * prim[1] + k * prim[2]
        for i in range(2) for j in range(2) for k in range(2)
    ]


def to_system(lattice_bohr: np.ndarray, coords_bohr: list[np.ndarray]) -> PeriodicSystem:
    return PeriodicSystem(
        3,
        lattice_bohr,
        [Atom(29, c.tolist()) for c in coords_bohr],
        0,
        1,
    )


def run(label: str, system: PeriodicSystem, cutoff: float) -> None:
    opts = _xtb.XTBSccOptions()
    opts.electronic_temperature = T
    try:
        result = _xtb.run_gfn2_xtb_gamma(system, load_gfn2_params(), opts, cutoff)
        print(f"{label}: E/atom = {float(result.energy) / 8:12.6f} "
              f"iter = {int(result.n_iter)}")
    except Exception as exc:  # noqa: BLE001
        print(f"{label}: FAILED {exc}")


def main() -> None:
    prim = fcc_primitive(A)
    coords = atoms_8(A)
    trans = [2 * p for p in prim]

    # (1) 8-atom cell, lattice columns = supercell translations (docstring
    #     convention).
    lattice_cols = np.column_stack([t * BOHR for t in trans])
    run("8at cols=trans", to_system(lattice_cols, [c * BOHR for c in coords]), 15.0)

    # (2) 8-atom cell, lattice rows = supercell translations.
    lattice_rows = np.vstack([t * BOHR for t in trans])
    run("8at rows=trans", to_system(lattice_rows, [c * BOHR for c in coords]), 15.0)

    # (3) 1-atom primitive cell, columns = primitive vectors.
    lattice_prim = np.column_stack([p * BOHR for p in prim])
    run("1at cols=prim", to_system(lattice_prim, [np.zeros(3) * BOHR]), 15.0)

    # (4) 8-atom cell with larger cutoff.
    run("8at cols=trans cutoff=25",
        to_system(lattice_cols, [c * BOHR for c in coords]), 25.0)


if __name__ == "__main__":
    main()
