"""Cutoff sweep of the Gamma driver on the 8-atom fcc Cu cell.

The previous session quoted -3.763 Ha/cell from run_gfn2_xtb_gamma on
this cell at T = 0.005, but the default cutoff=15 converges to an
absurd over-polarized state (-2.39e6 Ha/cell). Sweep the cutoff to find
the sane branch and its cutoff dependence.
"""

from __future__ import annotations

import numpy as np

from vibeqc import Atom, PeriodicSystem
from vibeqc._vibeqc_core.semiempirical import xtb as _xtb
from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

BOHR = 1.8897259886
A = 3.615
T = 0.005


def build_cell() -> PeriodicSystem:
    prim = [
        np.array([0.0, 0.5, 0.5]) * A,
        np.array([0.5, 0.0, 0.5]) * A,
        np.array([0.5, 0.5, 0.0]) * A,
    ]
    atoms = [
        i * prim[0] + j * prim[1] + k * prim[2]
        for i in range(2) for j in range(2) for k in range(2)
    ]
    lattice = np.column_stack(
        [(2 * prim[0]) * BOHR, (2 * prim[1]) * BOHR, (2 * prim[2]) * BOHR]
    )
    return PeriodicSystem(
        3, lattice, [Atom(29, (c * BOHR).tolist()) for c in atoms], 0, 1
    )


def main() -> None:
    system = build_cell()
    params = load_gfn2_params()
    print(f"{'cutoff/bohr':>12} {'E/atom':>14} {'Ef':>10} {'iter':>6} {'cells':>6}")
    for cutoff in [8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0]:
        opts = _xtb.XTBSccOptions()
        opts.electronic_temperature = T
        try:
            result = _xtb.run_gfn2_xtb_gamma(system, params, opts, cutoff)
            print(f"{cutoff:12.1f} {float(result.energy) / 8:14.6f} "
                  f"{float(result.fermi_level):10.4f} "
                  f"{int(result.n_iter):6d} {int(result.n_cells):6d}")
        except Exception as exc:  # noqa: BLE001
            print(f"{cutoff:12.1f}  FAILED: {exc}")


if __name__ == "__main__":
    main()
