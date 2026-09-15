"""Probe: validate compute_ext_el_spheropole against CRYSTAL CYC 0.

For MgO/STO-3G at SAD (initial density), CRYSTAL prints
``::: EXT EL-SPHEROPOLE = +4.119189 Ha`` in the sealed ENECYCLE
reference. This probe runs vibe-qc's ``compute_ext_el_spheropole``
on the same SAD density and reports the difference.

**Status**: ``compute_ext_el_spheropole`` now computes the exact
bond-symmetrised second moment for all l (via libint ``emultipole2``)
and reproduces CRYSTAL's printed ``EXT EL-SPHEROPOLE`` to < 0.02 mHa
(see :func:`vibeqc.bipole_ext_el_pole.compute_ext_el_spheropole` and
``tests/test_bipole_ext_el_pole.py``). Earlier revisions of this probe
handled only the s-s term and under-reproduced CRYSTAL on p/d systems;
that gap is closed. The method is the published Gaussian-charge
electrostatic (spheropole) series of Saunders, Freyria-Fava, Dovesi,
Salasco & Roetti, Mol. Phys. 77, 629 (1992), validated here purely by
output parity against CRYSTAL.

Run from repo root:
    python examples/regression/crystal_parity/probe_spheropole_sad.py
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "examples/regression/crystal_parity")
from crystal_demos.builders import (
    build_mgo_sto3g,
    build_diamond_sto3g,
    build_sibulk_sto3g,
)

from vibeqc import (
    InitialGuess,
    LatticeSumOptions,
    compute_overlap_lattice,
)
from vibeqc.bipole_ext_el_pole import compute_ext_el_spheropole
from vibeqc.guess import initial_density_closed_shell


def _build_sad_density(system, basis, lat_opts):
    """Return a LatticeMatrixSet with SAD density at g=0, zeros elsewhere."""
    D_lat = compute_overlap_lattice(basis, system, lat_opts)
    D_sad = initial_density_closed_shell(
        system.unit_cell_molecule(), basis,
        system.n_electrons() // 2,
        InitialGuess.SAD,
        is_periodic=True,
    )
    for g_idx, cell in enumerate(D_lat.cells):
        if (np.asarray(cell.index) == np.array([0, 0, 0])).all():
            D_lat.set_block(g_idx, np.asarray(D_sad, dtype=float))
        else:
            D_lat.set_block(
                g_idx, np.zeros_like(np.asarray(D_sad), dtype=float)
            )
    return D_lat


def _probe(label, build_fn, crystal_cyc0_ha):
    print(f"\n=== {label} ===")
    system, basis = build_fn()
    lat_opts = LatticeSumOptions()
    lat_opts.cutoff_bohr = 12.0
    D_sad = _build_sad_density(system, basis, lat_opts)
    E_sphero = compute_ext_el_spheropole(
        D_sad, basis, system, lat_opts,
    )
    delta = E_sphero - crystal_cyc0_ha
    print(f"  vibe-qc EXT EL-SPHEROPOLE (SAD)   = {E_sphero:+.6f} Ha")
    print(f"  CRYSTAL EXT EL-SPHEROPOLE (CYC 0) = {crystal_cyc0_ha:+.6f} Ha")
    print(f"  Δ = {delta:+.6f} Ha ({delta * 1000.0:+.3f} mHa)")
    return delta


def main() -> int:
    # CRYSTAL CYC 0 EXT EL-SPHEROPOLE values from the sealed .out files:
    deltas = []
    deltas.append(_probe(
        "MgO STO-3G FCC",
        build_mgo_sto3g,
        +4.1191890135070,        # mgo_sto3g_enecycle.out:376
    ))
    deltas.append(_probe(
        "diamond STO-3G FCC",
        build_diamond_sto3g,
        +3.3151817973404,        # diamond_sto3g_enecycle.out:372
    ))

    print("\n=== summary ===")
    print(f"  max |Δ| = {max(abs(d) for d in deltas):+.6f} Ha")
    print("  (Partial implementation — see module docstring "
          "Limitations section.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
