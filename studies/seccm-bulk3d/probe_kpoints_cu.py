"""Confirm that the retired multi-k GFN2 Cu probe fails closed."""

from __future__ import annotations

import numpy as np
from vibeqc import Atom, PeriodicSystem
from vibeqc._vibeqc_core import monkhorst_pack
from vibeqc._vibeqc_core.semiempirical import xtb as _xtb
from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

BOHR = 1.8897259886


def build_cell() -> PeriodicSystem:
    a = 3.615
    prim = [
        np.array([0.0, 0.5, 0.5]) * a,
        np.array([0.5, 0.0, 0.5]) * a,
        np.array([0.5, 0.5, 0.0]) * a,
    ]
    lattice = np.column_stack([p * BOHR for p in prim])
    return PeriodicSystem(3, lattice, [Atom(29, [0.0, 0.0, 0.0])], 0, 1)


def main() -> None:
    system = build_cell()
    params = load_gfn2_params()
    mesh = monkhorst_pack(system, [2, 2, 2], [0, 0, 0], False)
    try:
        _xtb.run_gfn2_xtb_kpoints(system, params, mesh)
    except RuntimeError as exc:
        if "issue #351" not in str(exc):
            raise
        print(f"expected fail-closed result: {exc}")
        return
    raise RuntimeError("unsafe multi-k GFN2 route unexpectedly executed")


if __name__ == "__main__":
    main()
