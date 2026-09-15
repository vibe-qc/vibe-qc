"""Probe: GFN2-SECCM AES invariance under torus-representative choice (issue #348)."""

from __future__ import annotations

import numpy as np

from vibeqc import Atom, Molecule
from vibeqc.semiempirical import run_gfn2_seccm
from vibeqc.semiempirical.seccm import bind_finite_group, build_seccm_topology

# H2O with H2 near the cell boundary; both typings run the multi-replica
# engine with identical bound WS records.
_A1 = np.array([4.5, 0.0, 0.0])
_COORDS = np.array(
    [[0.0, 0.0, 0.0], [1.8089, 0.0, 0.0], [4.5, 1.4316, 0.0]]
)


def _molecule(coords):
    return Molecule(
        [Atom(8, c.tolist()) for c in coords[:1]]
        + [Atom(1, c.tolist()) for c in coords[1:]],
        0,
        1,
    )


def _topology(coords):
    topology = build_seccm_topology(
        coords,
        [2.0 * _A1],
        length_unit="bohr",
        geometry_quantum=1.0e-10,
    )
    return bind_finite_group(
        topology,
        primitive_vectors=[_A1],
        replicas=(2, 1, 1),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )


def _records(topology):
    return sorted(
        (
            tuple(np.round(image.disp, 10)),
            image.origin,
            round(image.weight, 10),
        )
        for cell in topology.cells
        for image in cell
    )


def main():
    coords_typed = _COORDS.copy()
    coords_shifted = _COORDS.copy()
    coords_shifted[2] = coords_shifted[2] + 2.0 * _A1  # full torus over

    top_a = _topology(coords_typed)
    top_b = _topology(coords_shifted)
    print("records equal:", _records(top_a) == _records(top_b))

    result_a = run_gfn2_seccm(_molecule(coords_typed), top_a)
    result_b = run_gfn2_seccm(_molecule(coords_shifted), top_b)
    print("molecular_delegated:", bool(result_a.molecular_delegated),
          bool(result_b.molecular_delegated))
    print("converged:", bool(result_a.converged), bool(result_b.converged))
    for field in ("energy", "e_band0", "e_scc", "e_aes", "e_3rd",
                  "e_repulsive", "e_madelung"):
        va = float(getattr(result_a, field))
        vb = float(getattr(result_b, field))
        print(f"{field} delta: {va - vb:.6e}")
    print("charges A:", np.asarray(result_a.charges))
    print("charges B:", np.asarray(result_b.charges))
    print("shell_charges A:", np.asarray(result_a.shell_charges))
    print("shell_charges B:", np.asarray(result_b.shell_charges))

    control_a = run_gfn2_seccm(
        _molecule(coords_typed), top_a, include_aes=False
    )
    control_b = run_gfn2_seccm(
        _molecule(coords_shifted), top_b, include_aes=False
    )
    print("no-AES energy delta:", repr(
        float(control_a.energy) - float(control_b.energy)
    ))


if __name__ == "__main__":
    main()
