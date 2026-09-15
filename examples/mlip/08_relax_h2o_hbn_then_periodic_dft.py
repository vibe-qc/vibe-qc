#!/usr/bin/env python3
"""Relax H2O on h-BN with MACE, then optionally run periodic PBE.

The default command performs only the MACE geometry stage on a genuine
``dim=2`` sheet and writes an Extended XYZ structure:

    .venv-mace/bin/python \
        examples/mlip/08_relax_h2o_hbn_then_periodic_dft.py

Add ``--electronic`` to hand the relaxed ``PeriodicSystem`` directly to the
periodic electronic-structure driver. That second stage is a compact tutorial
probe, not a converged adsorption calculation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write
from ase.units import Bohr

import vibeqc as vq
from vibeqc.mlip import MLIPOptions
from vibeqc.mlip.mace import optimize_periodic_mace_positions


def hbn_2x2() -> vq.PeriodicSystem:
    """Return a flat, closed-shell 2 x 2 h-BN sheet with true 2D PBC."""
    lattice_constant = 2.50  # Angstrom; compact tutorial input, not a fit
    primitive_a1 = np.array([lattice_constant, 0.0, 0.0])
    primitive_a2 = np.array(
        [lattice_constant / 2.0, np.sqrt(3.0) * lattice_constant / 2.0, 0.0]
    )
    basis_n = (primitive_a1 + primitive_a2) / 3.0
    atoms = []
    for i in range(2):
        for j in range(2):
            origin = i * primitive_a1 + j * primitive_a2
            atoms.append(vq.Atom(5, (origin / Bohr).tolist()))
            atoms.append(vq.Atom(7, ((origin + basis_n) / Bohr).tolist()))
    return vq.slab_2d(
        (2.0 * primitive_a1 / Bohr).tolist(),
        (2.0 * primitive_a2 / Bohr).tolist(),
        atoms,
    )


def write_extended_xyz(system: vq.PeriodicSystem, path: Path) -> None:
    """Write the fixed-cell handoff geometry with its true PBC mask."""
    molecule = system.unit_cell_molecule()
    atoms = Atoms(
        numbers=[atom.Z for atom in molecule.atoms],
        positions=np.asarray([atom.xyz for atom in molecule.atoms]) * Bohr,
        cell=np.asarray(system.lattice).T * Bohr,
        pbc=tuple(axis < int(system.dim) for axis in range(3)),
    )
    write(path, atoms, format="extxyz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--electronic",
        action="store_true",
        help="also run the periodic PBE/def2-SVP single point",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("mace-handoff-output"),
        help="directory for the relaxed geometry and calculation siblings",
    )
    args = parser.parse_args()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    geometry = work_dir / "mace-h2o-hbn-relaxed.xyz"
    electronic_output = work_dir / "mace-h2o-hbn-periodic-pbe"

    sheet = hbn_2x2()
    a1 = np.asarray(sheet.lattice)[:, 0] * Bohr
    a2 = np.asarray(sheet.lattice)[:, 1] * Bohr
    hollow = 2.0 * (a1 + a2) / 3.0
    adsorbed = vq.place_adsorbate(
        sheet,
        "H2O",
        position=(float(hollow[0]), float(hollow[1])),
        height=2.90,
        orientation="flat",
        anchor="O",
    )
    fixed = list(range(len(sheet.unit_cell)))
    relaxed = optimize_periodic_mace_positions(
        adsorbed,
        MLIPOptions(model="medium-mpa-0", device="cpu", dtype="float64"),
        fmax=0.05,
        max_steps=200,
        fixed_indices=fixed,
    )
    write_extended_xyz(relaxed.system, geometry)

    print("model=medium-mpa-0 loader=mace_mp license=MIT")
    print(f"dim={relaxed.dim} pbc={relaxed.pbc} fixed_indices={fixed}")
    print(
        f"converged={relaxed.converged} steps={relaxed.n_steps} "
        f"max_force={relaxed.max_force_eva:.6f} eV/Angstrom"
    )
    print(f"geometry={geometry}")
    if not relaxed.converged:
        raise SystemExit(
            "MACE optimization hit max_steps; inspect the structure before "
            "starting an electronic single point."
        )

    if not args.electronic:
        print("Periodic electronic stage not requested; rerun with --electronic.")
        return

    basis = vq.BasisSet(relaxed.system.unit_cell_molecule(), "def2-svp")
    electronic = vq.run_periodic_job(
        relaxed.system,
        basis,
        method="RKS",
        functional="pbe",
        jk_method="auto",
        kpoints=(2, 2, 1),
        output=electronic_output,
    )
    print(f"periodic PBE energy={float(electronic.energy):.12f} Ha")
    print(
        "This electronic total is not comparable to the absolute MACE "
        "energy; the MACE result selected only the geometry."
    )


if __name__ == "__main__":
    main()
