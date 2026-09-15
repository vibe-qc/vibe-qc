#!/usr/bin/env python3
"""Compare MACE and PM6 H2O stretch curves without mixing energy scales.

Requires Python <= 3.13 and the optional MACE stack:

    pip install -e '.[mace]'
    python examples/mlip/05_mace_pm6_relative_curve.py

The same fixed-angle H2O geometry is evaluated over a short symmetric
O-H stretch. Each method is normalized to its own sampled minimum. The
printed relative curves can be compared; the raw absolute energies
cannot. Neither method is an external reference, so this example makes
no accuracy claim.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree

from vibeqc import Atom, Molecule
from vibeqc.mlip import MLIPOptions, resolve_model
from vibeqc.mlip.mace import mace_calculator
from vibeqc.semiempirical import PM6Model
from vibeqc.semiempirical.methods.pm6_params import load_pm6_params_auto

MODEL = "medium-mpa-0"
ANGLE_DEG = 104.5
DISTANCES_ANGSTROM = np.array([0.86, 0.91, 0.96, 1.01, 1.06, 1.11, 1.16])


def coordinates(r_oh: float) -> np.ndarray:
    """Fixed-angle water coordinates in Angstrom."""
    half_angle = np.deg2rad(ANGLE_DEG / 2.0)
    x = r_oh * np.sin(half_angle)
    y = r_oh * np.cos(half_angle)
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [x, y, 0.0],
            [-x, y, 0.0],
        ]
    )


def as_molecule(positions_angstrom: np.ndarray) -> Molecule:
    """Convert O,H,H Angstrom coordinates to a vibe-qc Molecule."""
    atoms = [
        Atom(z, list(xyz / Bohr))
        for z, xyz in zip((8, 1, 1), positions_angstrom, strict=True)
    ]
    return Molecule(atoms, charge=0, multiplicity=1)


def main() -> None:
    options = MLIPOptions(model=MODEL, device="cpu", dtype="float64")
    model_info = resolve_model(options.model)
    calculator = mace_calculator(options)
    pm6_params = load_pm6_params_auto([8, 1, 1])

    mace_ev = []
    pm6_ev = []
    for distance in DISTANCES_ANGSTROM:
        positions = coordinates(float(distance))

        ase_atoms = Atoms(numbers=[8, 1, 1], positions=positions)
        ase_atoms.calc = calculator
        mace_ev.append(float(ase_atoms.get_potential_energy()))

        pm6 = PM6Model(as_molecule(positions), params=pm6_params)
        pm6_ev.append(float(pm6.energy()) * Hartree)

    mace_relative = np.asarray(mace_ev) - min(mace_ev)
    pm6_relative = np.asarray(pm6_ev) - min(pm6_ev)

    print(f"# MACE model={model_info.key} loader={model_info.loader}")
    print(f"# MACE license={model_info.license} citation={model_info.citation}")
    print("# PM6 energy convention=vibe-qc PM6-like total")
    print("# Values below are independently normalized relative energies.")
    print("# No external reference is included; this is not an accuracy validation.")
    print("r_oh_angstrom,mace_delta_ev,pm6_delta_ev")
    for distance, mace_de, pm6_de in zip(
        DISTANCES_ANGSTROM,
        mace_relative,
        pm6_relative,
        strict=True,
    ):
        print(f"{distance:.4f},{mace_de:.8f},{pm6_de:.8f}")


if __name__ == "__main__":
    main()
