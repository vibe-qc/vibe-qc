"""Corundum hex-cell closest-image consistency diagnostic (IID 130).

Hypothesis: the 6 negative cyclic-overlap eigenvalues on the corundum
hex cell come from pairwise-closest image displacements that are not
globally consistent (the skew 120-degree cell wraps differently for
different pairs). A Hermitian-but-not-Gram S can be indefinite. This
probe checks the triangle condition r_ab + r_bc + r_ca = lattice vector
for every atom triple using the topology records.
"""

from __future__ import annotations

import sys
from itertools import combinations

import numpy as np

from vibeqc.semiempirical.seccm.topology import (
    bind_finite_group,
    build_seccm_topology,
)

BOHR = 1.8897259886

from xtb_corundum_reference import corundum_positions  # noqa: E402


def main():
    symbols, positions, lattice = corundum_positions(4.759, 12.99)
    coords = [np.asarray(p) * BOHR for p in positions]
    translations = [np.asarray(row) * BOHR for row in lattice]
    topo = build_seccm_topology(
        coords, translations, length_unit="bohr", geometry_quantum=1.0e-10
    )
    topo = bind_finite_group(
        topo,
        primitive_vectors=[np.asarray(row) * BOHR for row in lattice],
        replicas=(1, 1, 1),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    # Directed displacement from central a to origin b, closest image.
    disp = {}
    multi = {}
    for a, cell in enumerate(topo.cells):
        for image in cell:
            key = (a, image.origin)
            if key in disp:
                multi[key] = multi.get(key, 1) + 1
                continue
            disp[key] = np.asarray(image.disp)

    n_multi = sum(1 for k in multi if multi[k] > 1)
    print(f"n_atoms={len(coords)}  multi-image pairs={n_multi}")

    # Triangle consistency: r_ab + r_bc should equal r_ac modulo a
    # lattice translation, for every triple.
    violations = 0
    worst = 0.0
    for a, b, c in combinations(range(len(coords)), 3):
        for x, y, z in ((a, b, c), (a, c, b), (b, c, a)):
            r_xy = disp[(x, y)]
            r_yz = disp[(y, z)]
            r_xz = disp[(x, z)]
            path = r_xy + r_yz - r_xz
            # reduce modulo lattice: path + n1 T1 + n2 T2 + n3 T3
            residual = None
            for n1 in range(-2, 3):
                for n2 in range(-2, 3):
                    for n3 in range(-2, 3):
                        v = path + n1 * translations[0] + n2 * translations[1] + n3 * translations[2]
                        if residual is None or np.linalg.norm(v) < np.linalg.norm(residual):
                            residual = v
            mag = float(np.linalg.norm(residual))
            worst = max(worst, mag)
            if mag > 1.0e-6:
                violations += 1
    print(f"triangle violations: {violations}  worst residual {worst:.6f} bohr")
    # Hermiticity: r_ab == -r_ba modulo lattice.
    asym = 0
    for (a, b), r_ab in disp.items():
        r_ba = disp.get((b, a))
        if r_ba is None:
            continue
        delta = r_ab + r_ba
        mag = float(np.linalg.norm(delta))
        # delta must be a lattice vector
        residual = None
        for n1 in range(-2, 3):
            for n2 in range(-2, 3):
                for n3 in range(-2, 3):
                    v = delta + n1 * translations[0] + n2 * translations[1] + n3 * translations[2]
                    if residual is None or np.linalg.norm(v) < np.linalg.norm(residual):
                        residual = v
        if float(np.linalg.norm(residual)) > 1.0e-6:
            asym += 1
    print(f"non-opposite reverse pairs: {asym}")


if __name__ == "__main__":
    sys.exit(main())
