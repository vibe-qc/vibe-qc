"""Decompose the Gamma driver's absurd Cu energy: which channel explodes?"""

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


def build_cell(replicas: tuple[int, int, int]) -> PeriodicSystem:
    prim = fcc_primitive(A)
    n1, n2, n3 = replicas
    atoms = [
        i * prim[0] + j * prim[1] + k * prim[2]
        for i in range(n1) for j in range(n2) for k in range(n3)
    ]
    lattice = np.column_stack(
        [(n1 * prim[0]) * BOHR, (n2 * prim[1]) * BOHR, (n3 * prim[2]) * BOHR]
    )
    return PeriodicSystem(
        3,
        lattice,
        [Atom(29, (c * BOHR).tolist()) for c in atoms],
        0,
        1,
    )


def run(label: str, system: PeriodicSystem, cutoff: float, temp: float) -> None:
    opts = _xtb.XTBSccOptions()
    opts.electronic_temperature = temp
    params = load_gfn2_params()
    result = _xtb.run_gfn2_xtb_gamma(system, params, opts, cutoff)
    occ = np.asarray(result.occupations)
    print(f"{label}: E={float(result.energy):14.4f} "
          f"Erep={float(result.e_repulsive):12.4f} "
          f"Ef={float(result.fermi_level):+10.4f} "
          f"iter={int(result.n_iter)} cells={int(result.n_cells)}")
    print(f"   occ min/max/mean: {occ.min():.4f}/{occ.max():.4f}/"
          f"{occ.mean():.4f}  n_occ={occ.size}")


def main() -> None:
    run("8at cutoff15 T=0.005", build_cell((2, 2, 2)), 15.0, 0.005)
    run("8at cutoff25 T=0.005", build_cell((2, 2, 2)), 25.0, 0.005)
    run("8at cutoff15 T=0     ", build_cell((2, 2, 2)), 15.0, 0.0)
    run("1at cutoff15 T=0.005", build_cell((1, 1, 1)), 15.0, 0.005)


if __name__ == "__main__":
    main()
