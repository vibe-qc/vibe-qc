#!/usr/bin/env python3
"""Si diamond RHF/STO-3G parity — vibe-qc BIPOLE with FMIXING vs CRYSTAL23.

Runs the same Si diamond calculation (SG 227, a=5.43 Å, STO-3G,
SHRINK 2 2 ↔ 2×2×2 Monkhorst-Pack) with ``fmixing_percent=30``
(CRYSTAL's FMIXING 30 analogue) now wired through the BIPOLE driver.

The key fix: ``pbc_bipole.py`` now consumes ``opts.fock_mixing``
and applies Fock-matrix mixing in the SCF loop before diagonalisation,
matching CRYSTAL's FMIXING behaviour.

Usage:
    cd <your vibe-qc checkout>
    .venv/bin/python examples/regression/crystal_parity/crystal_demos/si_diamond_sto3g_2x2x2/run_vibeqc_si_diamond.py
"""

from __future__ import annotations

import sys
import time

import numpy as np
import vibeqc as vq
from vibeqc import Atom, BasisSet, PeriodicSystem, attach_symmetry, monkhorst_pack


def build_si_diamond_sto3g(a_ang: float = 5.43):
    """Si diamond, SG 227 (Fd-3m), 2 atoms/primitive cell, STO-3G."""
    ang2bohr = 1.0 / 0.529177210903
    a = a_ang * ang2bohr
    lattice = (
        a
        / 2.0
        * np.array(
            [
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        )
    )
    s = a / 8.0
    atoms = [
        Atom(14, [s, s, s]),
        Atom(14, [3.0 * s, 3.0 * s, 3.0 * s]),
    ]
    system = PeriodicSystem(3, lattice, atoms)
    attach_symmetry(system)
    basis = BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def main() -> int:
    system, basis = build_si_diamond_sto3g(a_ang=5.43)
    n_elec = system.n_electrons()
    print(f"Si diamond: {n_elec} electrons, {basis.nbasis} BFs, {basis.nshells} shells")

    # Symmetry-reduced 2×2×2 Monkhorst-Pack → 3 k-points (IBZ)
    kmesh = monkhorst_pack(system, [2, 2, 2], use_symmetry=True)
    n_k = len(list(kmesh.kpoints))
    print(f"k-mesh: 2×2×2 → {n_k} irreducible k-points (IBZ)")

    print()
    print("=== Run 1: default (no FMIXING, no damping) ===")
    t0 = time.time()
    result1 = vq.run_periodic_job(
        system,
        basis,
        method="RHF",
        jk_method="bipole",
        kpoints=kmesh,
        output="si_diamond_default",
        max_iter=80,
        conv_tol_energy=1e-7,
        output_qvf=False,
    )
    t1 = time.time() - t0
    print(f"  converged = {result1.converged}, n_iter = {result1.n_iter}")
    print(f"  E_total   = {result1.energy:.10f} Ha")
    print(f"  wall      = {t1:.1f}s")

    print()
    print("=== Run 2: FMIXING 30 (fock_mixing=0.30) ===")
    t0 = time.time()
    result2 = vq.run_periodic_job(
        system,
        basis,
        method="RHF",
        jk_method="bipole",
        kpoints=kmesh,
        output="si_diamond_fmix30",
        fmixing_percent=30,  # <-- Fock-matrix mixing, CRYSTAL FMIXING 30
        max_iter=80,
        conv_tol_energy=1e-7,
        output_qvf=False,
    )
    t2 = time.time() - t0
    print(f"  converged = {result2.converged}, n_iter = {result2.n_iter}")
    print(f"  E_total   = {result2.energy:.10f} Ha")
    print(f"  wall      = {t2:.1f}s")

    print()
    print("=== Run 3: FMIXING 30 + damping 0.3 ===")
    t0 = time.time()
    result3 = vq.run_periodic_job(
        system,
        basis,
        method="RHF",
        jk_method="bipole",
        kpoints=kmesh,
        output="si_diamond_fmix30_damp03",
        fmixing_percent=30,
        damping=0.3,
        max_iter=80,
        conv_tol_energy=1e-7,
        output_qvf=False,
    )
    t3 = time.time() - t0
    print(f"  converged = {result3.converged}, n_iter = {result3.n_iter}")
    print(f"  E_total   = {result3.energy:.10f} Ha")
    print(f"  wall      = {t3:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
