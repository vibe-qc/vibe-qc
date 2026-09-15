"""Dump SCC-DFTB-SECCM topology weights for R=2 vs R=3 (IID 313)."""

from __future__ import annotations

import sys
from itertools import product

import numpy as np

from vibeqc.semiempirical.seccm import bind_finite_group, build_seccm_topology

CELL = 4.1
SITES = np.array([[0.17, 0.31, 0.0], [1.39, -0.22, 0.0]])


def build(r):
    primitives = [
        np.array([CELL, 0.0, 0.0]),
        np.array([0.0, CELL, 0.0]),
        np.array([0.0, 0.0, CELL]),
    ]
    coords = np.array(
        [
            origin + site
            for idx in product(range(r), repeat=3)
            for origin in [
                idx[0] * primitives[0]
                + idx[1] * primitives[1]
                + idx[2] * primitives[2]
            ]
            for site in SITES
        ]
    )
    topology = build_seccm_topology(
        coords,
        [r * p for p in primitives],
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )
    return bind_finite_group(
        topology,
        primitive_vectors=primitives,
        replicas=(r, r, r),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )


def summarize(r):
    topo = build(r)
    natoms = len(topo.cells)
    weight_hist = {}
    boundary = 0
    total = 0.0
    for central, cell in enumerate(topo.cells):
        wt = topo.total_weight(central)
        total += wt
        weight_hist.setdefault(round(wt, 6), 0)
        weight_hist[round(wt, 6)] += 1
        if abs(wt - (natoms - 1)) > 1e-9:
            boundary += 1
    print(f"R={r}: natoms={natoms}")
    print(f"  total weight sum={total:.6f}  (expected {natoms*(natoms-1):.0f})")
    print(f"  per-central weight histogram: {dict(sorted(weight_hist.items()))}")
    print(f"  diagnostics: {[d.code.value for d in topo.diagnostics(natoms)]}")
    ties = [c for c in topo.candidate_classifications if c.reference_tie]
    print(f"  reference ties: {len(ties)}")
    # weight by origin for central=0
    for central in (0, 1, 2, natoms // 2, natoms - 1):
        cell = topo.cells[central]
        wt_by_origin = {}
        for image in cell:
            wt_by_origin[image.origin] = (
                wt_by_origin.get(image.origin, 0.0) + image.weight
            )
        interesting = {
            o: round(w, 4)
            for o, w in sorted(wt_by_origin.items())
            if w > 1e-9 and o not in (central,)
        }
        print(
            f"  central={central} site={central % 2} "
            f"total={round(topo.total_weight(central), 6)} n_images={len(cell)}"
        )


def main():
    for r in (2, 3, 4, 5):
        summarize(r)
        print()


if __name__ == "__main__":
    sys.exit(main())
