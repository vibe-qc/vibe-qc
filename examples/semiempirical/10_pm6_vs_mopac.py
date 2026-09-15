#!/usr/bin/env python3
"""PM6 energy comparison — Stewart 2007 vs MOPAC 2016 parameters.

Compares vibe-qc PM6 energies against reference MOPAC values for a
standard test set.  Demonstrates both parameter sets and the MOPAC
reference oracle.

Run:
    .venv/bin/python examples/semiempirical/10_pm6_vs_mopac.py
"""

from __future__ import annotations

import numpy as np
from vibeqc import Atom, Molecule
from vibeqc.semiempirical import PM6Model
from vibeqc.semiempirical.methods.mopac_params import load_mopac_pm6_params
from vibeqc.semiempirical.methods.pm6_params import load_pm6_params

# Reference energies from MOPAC PM6 (pinned in runner_mopac.py)
_MOPAC_REF = {
    "H2": -0.71,
    "H2O": -6.72,
    "CH4": -1.48,
    "NH3": -3.18,
    "CO2": -8.66,
    "C2H4": -4.84,
}


def _make_molecule(name: str) -> Molecule:
    """Build a test molecule (geometry in bohr)."""
    if name == "H2":
        return Molecule([Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])], 0, 1)
    if name == "H2O":
        t = np.deg2rad(104.5 / 2)
        r = 1.81
        return Molecule(
            [
                Atom(8, [0, 0, 0]),
                Atom(1, [r * np.sin(t), r * np.cos(t), 0]),
                Atom(1, [-r * np.sin(t), r * np.cos(t), 0]),
            ],
            0,
            1,
        )
    if name == "CH4":
        a = 2.0 / np.sqrt(3)
        return Molecule(
            [
                Atom(6, [0, 0, 0]),
                Atom(1, [a, a, a]),
                Atom(1, [-a, -a, a]),
                Atom(1, [-a, a, -a]),
                Atom(1, [a, -a, -a]),
            ],
            0,
            1,
        )
    if name == "NH3":
        b = 1.9
        d = 0.38
        return Molecule(
            [
                Atom(7, [0, 0, d]),
                Atom(1, [0, b, -0.5 * d]),
                Atom(1, [b * np.sqrt(3) / 2, -0.5 * b, -0.5 * d]),
                Atom(1, [-b * np.sqrt(3) / 2, -0.5 * b, -0.5 * d]),
            ],
            0,
            1,
        )
    if name == "CO2":
        return Molecule(
            [Atom(6, [0, 0, 0]), Atom(8, [2.2, 0, 0]), Atom(8, [-2.2, 0, 0])], 0, 1
        )
    if name == "C2H4":
        return Molecule(
            [
                Atom(6, [0, 0, 0]),
                Atom(6, [2.5, 0, 0]),
                Atom(1, [-0.55, 1.75, 0]),
                Atom(1, [-0.55, -1.75, 0]),
                Atom(1, [3.05, 1.75, 0]),
                Atom(1, [3.05, -1.75, 0]),
            ],
            0,
            1,
        )
    raise ValueError(f"Unknown molecule: {name}")


def main():
    params_s = load_pm6_params()
    params_m = load_mopac_pm6_params()

    header = f"{'Molecule':<8} {'Stewart 2007':<14} {'MOPAC 2016':<14} {'MOPAC ref':<14} {'Δ Stewart':<12} {'Δ MOPAC 2016':<14}"
    print(header)
    print("-" * len(header))

    for name in ["H2", "H2O", "CH4", "NH3", "CO2", "C2H4"]:
        mol = _make_molecule(name)

        # Stewart 2007
        m_s = PM6Model(mol, params=params_s)
        e_s = m_s.energy()

        # MOPAC 2016
        m_m = PM6Model(mol, params=params_m)
        e_m = m_m.energy()

        ref = _MOPAC_REF[name]
        print(
            f"{name:<8} {e_s:<+14.4f} {e_m:<+14.4f} {ref:<+14.4f} "
            f"{e_s - ref:<+12.4f} {e_m - ref:<+14.4f}"
        )

    print()
    print("Notes:")
    print("  - Stewart 2007: 5-element set with Ohno-Klopman gamma")
    print(
        "  - MOPAC 2016:   78-element set with STO-6G multipole gamma + diatomic pairs"
    )
    print("  - MOPAC ref:    pinned from MOPAC 2016 (PM6 Hamiltonian)")
    print("  - The ~0.5 Ha Stewart gap and ~3 Ha MOPAC 2016 gap come from")
    print("    different core-core repulsion formulas vs MOPAC's implementation.")


if __name__ == "__main__":
    main()
