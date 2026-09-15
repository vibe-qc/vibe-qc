"""BIPOLE bipolar far-field research diagnostic across Bravais lattices.

BIPOLE-EXACT-ZONE increment 5a. Exercises low-level quartet prototype
data structures for representative Bravais lattices plus 1D chains and
2D slabs. The periodic quartet expansion is from Pisani, Dovesi, and
Roetti (1988), Chapter II.4c, not Saunders's 1992 electrostatic-potential
paper.

For each lattice type, this script:
  1. Builds a minimal test system (H2 or He in the unit cell).
  2. Constructs the full far-field infrastructure.
  3. Compares the bipolar far-field Coulomb energy against the
     exact erfc-screened four-centre ERI J_SR.
  4. Reports the energy difference per lattice type.

This is not a certification script or a production SCF driver. In
particular, the exact scalar below covers the full J_SR domain while the
prototype scalar covers only its dispatched subset, so their difference is
diagnostic and is not an accuracy metric for a complete SCF replacement.

Usage
-----
    python examples/regression/bipole_parity/bipole_bravais_validation.py
"""

from __future__ import annotations

import sys
import time
import numpy as np

# All Bravais lattice types with their lattice matrices.
# Each entry: (name, dim, lattice_matrix, atom_positions, atom_Zs)
BRAVAIS_LATTICES = {
    # ---- 3D ----
    "cubic-P": {
        "dim": 3,
        "lattice": [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        "atoms": [([0.0, 0.0, 0.0], 2), ([0.7, 0.7, 0.7], 2)],
    },
    "cubic-I": {
        "dim": 3,
        "lattice": (4.0 / 2.0) * np.array(
            [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
        ).tolist(),
        "atoms": [([0.0, 0.0, 0.0], 2), ([0.5, 0.5, 0.5], 2)],
    },
    "cubic-F": {
        "dim": 3,
        "lattice": (4.0 / 2.0) * np.array(
            [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        ).tolist(),
        "atoms": [([0.0, 0.0, 0.0], 2), ([0.5, 0.5, 0.5], 2)],
    },
    "tetragonal-P": {
        "dim": 3,
        "lattice": [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 5.0]],
        "atoms": [([0.0, 0.0, 0.0], 2), ([0.5, 0.5, 0.5], 2)],
    },
    "hexagonal-P": {
        "dim": 3,
        "lattice": [
            [2.5, 0.0, 0.0],
            [-1.25, 2.5 * np.sqrt(3) / 2, 0.0],
            [0.0, 0.0, 4.0],
        ],
        "atoms": [([0.0, 0.0, 0.0], 2), ([0.5, 0.5, 0.5], 2)],
    },
    "orthorhombic-P": {
        "dim": 3,
        "lattice": [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]],
        "atoms": [([0.0, 0.0, 0.0], 2), ([0.5, 0.5, 0.5], 2)],
    },
    "monoclinic-P": {
        "dim": 3,
        "lattice": [
            [3.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [1.0, 0.0, 5.0],
        ],
        "atoms": [([0.0, 0.0, 0.0], 2), ([0.5, 0.5, 0.5], 2)],
    },
    "triclinic-P": {
        "dim": 3,
        "lattice": [
            [3.0, 0.2, 0.1],
            [0.3, 4.0, 0.2],
            [0.1, 0.3, 5.0],
        ],
        "atoms": [([0.0, 0.0, 0.0], 2), ([0.5, 0.5, 0.5], 2)],
    },
    # ---- 2D ----
    "square-slab": {
        "dim": 2,
        "lattice": [[6.0, 0.0, 0.0], [0.0, 6.0, 0.0]],
        "atoms": [([0.0, 0.0, 0.0], 2), ([0.5, 0.5, 0.0], 2)],
    },
    # ---- 1D ----
    "linear-chain": {
        "dim": 1,
        "lattice": [[4.0, 0.0, 0.0]],
        "atoms": [([0.0, 0.0, 0.0], 2), ([0.5, 0.0, 0.0], 2)],
    },
}


def _build_system(name, entry):
    """Build a PeriodicSystem from a lattice entry."""
    from vibeqc import Atom, PeriodicSystem

    dim = entry["dim"]
    lattice = np.asarray(entry["lattice"], dtype=float)
    atoms = [
        Atom(int(z), np.asarray(pos, dtype=float))
        for pos, z in entry["atoms"]
    ]
    return PeriodicSystem(dim, lattice, atoms)


def validate_one(name, entry, cutoff=6.0, omega=0.5, multipole_l_max=2):
    """Probe the low-level bipolar far field for one lattice type."""
    from vibeqc import BasisSet
    from vibeqc._vibeqc_core import (
        LatticeSumOptions,
        build_jk_2e_real_space,
        compute_multipole_moments_lattice,
        direct_lattice_cells,
        make_lattice_matrix_set,
    )
    from vibeqc.bipole_pair_moments import pair_center_moments
    from vibeqc.bipole_spherical_moment_buffer import (
        build_spherical_moment_buffer,
    )
    from vibeqc.bipole_dispatch import (
        build_penetration_dispatch_for_bipole_context,
    )
    from vibeqc.bipole_quartet_far_field import (
        build_bipolar_coulomb_far_field,
    )

    system = _build_system(name, entry)
    dim = entry["dim"]
    if dim < 3:
        print(f"  {name} (dim={dim}): SKIP — far-field dispatch requires 3D")
        return None

    basis = BasisSet(system.unit_cell_molecule(), "sto-3g")
    lo = LatticeSumOptions()
    lo.cutoff_bohr = cutoff
    cells = direct_lattice_cells(system, cutoff)

    # Trial density: identity at home cell.
    blocks = []
    for cell in cells:
        if np.allclose(cell.index, 0):
            blocks.append(np.eye(basis.nbasis, dtype=float))
        else:
            blocks.append(np.zeros((basis.nbasis, basis.nbasis), dtype=float))
    P_trial = make_lattice_matrix_set(basis.nbasis, cells, blocks)

    # ---- Infrastructure ----
    cart = compute_multipole_moments_lattice(
        basis, system, lo, 2, (0.0, 0.0, 0.0)
    )
    pair_mom = pair_center_moments(cart, basis)
    buf = build_spherical_moment_buffer(pair_mom, basis, L_max=multipole_l_max)
    dispatch = build_penetration_dispatch_for_bipole_context(
        basis, system, list(cart.cells),
        maximum_multipole_order=multipole_l_max,
    )

    # ---- Exact J_SR ----
    jk_exact = build_jk_2e_real_space(basis, system, lo, P_trial, omega)
    e_j_exact = 0.0
    for c, cell in enumerate(P_trial.cells):
        D = np.asarray(P_trial.blocks[c], dtype=float)
        J = np.asarray(jk_exact.J.blocks[c], dtype=float)
        e_j_exact += 0.5 * np.sum(D * J)

    # ---- Bipolar far-field ----
    density_dict = {
        (cell.index[0], cell.index[1], cell.index[2]): np.asarray(
            P_trial.blocks[c], dtype=float,
        )
        for c, cell in enumerate(P_trial.cells)
    }
    ff = build_bipolar_coulomb_far_field(
        buf, dispatch, density_dict,
        ewald_omega=omega, nbf=basis.nbasis,
    )

    diff = abs(e_j_exact - ff.e_coulomb_far)
    rel = diff / abs(e_j_exact) if abs(e_j_exact) > 1e-12 else 0.0

    return {
        "name": name,
        "dim": dim,
        "n_cells": len(cells),
        "n_quartets": ff.n_quartets,
        "e_j_exact": e_j_exact,
        "e_j_bipolar": ff.e_coulomb_far,
        "abs_diff": diff,
        "rel_diff": rel,
    }


def main():
    print("=" * 72)
    print("BIPOLE Bravais-Lattice Research Diagnostic")
    print("Pisani-Dovesi-Roetti 1988, doi:10.1007/978-3-642-93385-1")
    print("=" * 72)

    cutoff = 6.0
    omega = 0.5
    multipole_l_max = 2

    results = []
    for name, entry in BRAVAIS_LATTICES.items():
        print(f"\n--- {name} ---")
        t0 = time.perf_counter()
        try:
            r = validate_one(name, entry, cutoff, omega, multipole_l_max)
            dt = time.perf_counter() - t0
            if r is not None:
                r["time_s"] = dt
                results.append(r)
                print(
                    f"  {r['n_cells']} cells, {r['n_quartets']} quartets, "
                    f"{dt:.2f}s"
                )
                print(
                    f"  E_J exact:    {r['e_j_exact']:+.8f} Ha"
                )
                print(
                    f"  E_J bipolar:  {r['e_j_bipolar']:+.8f} Ha"
                )
                print(
                    f"  |ΔE|:         {r['abs_diff']:.2e} Ha "
                    f"({r['rel_diff']:.3e} rel)"
                )
            else:
                print(f"  SKIP (dim={entry['dim']})")
        except Exception as exc:
            dt = time.perf_counter() - t0
            print(f"  FAIL: {exc} ({dt:.1f}s)")

    # ---- Summary table ----
    if results:
        print("\n" + "=" * 72)
        print(f"{'Lattice':<20s} {'dim':>3s} {'cells':>6s} {'quartets':>9s} "
              f"{'|dE|/Ha':>12s} {'rel':>8s} {'t/s':>6s}")
        print("-" * 72)
        for r in results:
            print(
                f"{r['name']:<20s} {r['dim']:>3d} {r['n_cells']:>6d} "
                f"{r['n_quartets']:>9d} {r['abs_diff']:>12.2e} "
                f"{r['rel_diff']:>8.3e} {r['time_s']:>6.2f}"
            )
        print("=" * 72)


if __name__ == "__main__":
    main()
