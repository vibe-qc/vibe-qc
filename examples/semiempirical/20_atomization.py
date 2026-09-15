"""Semiempirical atomization energies and thermodynamics — PES validation.

Computes atomization energies for the standard test set using PM6
and compares against reference values.  Also reports the PES minimum
location from a bond-length scan.

Run:  .venv/bin/python examples/semiempirical/20_atomization.py

Priority: validates that energy differences (not just absolute energies)
are physical, which the FD gradient validation cannot catch.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
from vibeqc import Atom, Molecule
from vibeqc.semiempirical.dftb0 import DFTB0Model
from vibeqc.semiempirical.methods.gfn2 import GFN2Model
from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params
from vibeqc.semiempirical.methods.pm6 import PM6Model, UPM6Model

HERE = Path(__file__).resolve().parent

# ── Atomization energy reference (kcal/mol) ─────────────────────────────
# NIST CCCBDB experimental values
_ATOM_REF = {
    "H2": 109.5,
    "H2O": 232.2,
    "CH4": 419.3,
    "NH3": 297.6,
    "CO2": 389.1,
}

# Ground-state atomic multiplicities
_ATOM_MULT = {1: 2, 6: 3, 7: 4, 8: 3, 9: 2}


def atomic_energy(z: int) -> float:
    """Compute atomic energy using UPM6Model for open-shell ground states."""
    mol = Molecule([Atom(z, [0, 0, 0])], 0, _ATOM_MULT.get(z, 1))
    m = UPM6Model(mol)
    return m.energy()


def molecular_energy(atoms: list, method: str) -> float:
    """Compute molecular energy."""
    mol = Molecule(atoms, 0, 1)
    if method == "dftb0":
        return DFTB0Model(mol).energy()
    elif method == "gfn2":
        params = load_gfn2_params()
        return GFN2Model(mol, params, warn=False).energy()
    elif method == "pm6":
        return PM6Model(mol).energy()
    raise ValueError(f"Unknown method: {method}")


def atomization_energy(atoms: list, method: str) -> float:
    """Compute atomization energy: sum(atomic E) - molecular E."""
    e_mol = molecular_energy(atoms, method)
    e_atoms = sum(atomic_energy(at.Z) for at in atoms)
    return e_atoms - e_mol


def water_oh_scan() -> list[dict]:
    """Scan O-H distance for water with all methods."""
    rows = []
    for r in [1.50, 1.55, 1.60, 1.65, 1.70, 1.75, 1.81, 1.90, 2.00, 2.10, 2.20]:
        a = math.radians(52.25)
        z = r * math.cos(a)
        y = r * math.sin(a)
        atoms = [Atom(8, [0, 0, 0]), Atom(1, [0, y, z]), Atom(1, [0, -y, z])]
        for method in ["dftb0", "gfn2", "pm6"]:
            e = molecular_energy(atoms, method)
            rows.append({"r": r, "method": method, "energy": e})
    return rows


def find_min(rows: list[dict], method: str) -> float:
    mr = [r for r in rows if r["method"] == method]
    best = min(mr, key=lambda x: x["energy"])
    return best["r"]


def main():
    print("=" * 70)
    print(" Semiempirical atomization energies & PES validation")
    print("=" * 70)
    print()

    # ── Atomization energies ──────────────────────────────────────────
    print("Atomization energies (kcal/mol):")
    print(f"{'Molecule':<8} {'DFTB0':>10} {'GFN2':>10} {'PM6':>10} {'Ref':>10}")
    print("-" * 52)

    molecules = {
        "H2": [Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])],
        "H2O": water_atoms(1.81),
        "CH4": ch4_atoms(),
        "NH3": nh3_atoms(),
        "CO2": co2_atoms(),
    }

    for name, atoms in molecules.items():
        ref = _ATOM_REF.get(name, None)
        vals = {}
        for method in ["dftb0", "pm6"]:
            try:
                ae = atomization_energy(atoms, method)
                vals[method] = ae * 627.509  # Ha -> kcal/mol
            except Exception:
                vals[method] = None
        dftb0_str = f"{vals['dftb0']:10.1f}" if vals["dftb0"] else "       N/A"
        pm6_str = f"{vals['pm6']:10.1f}" if vals["pm6"] else "       N/A"
        ref_str = f"{ref:10.1f}" if ref else "       N/A"
        print(f"{name:<8} {dftb0_str} {'  N/A':>10} {pm6_str} {ref_str}")
    print("  Note: GFN2 closed-shell only — cannot compute open-shell atomic energies")
    print()

    # ── PES scan ──────────────────────────────────────────────────────
    print("Water O-H PES scan (bohr):")
    rows = water_oh_scan()
    for method in ["dftb0", "gfn2", "pm6"]:
        r_min = find_min(rows, method)
        sane = "✅" if 1.70 <= r_min <= 1.95 else "❌"
        print(
            f"  {method:8s}: minimum at r ≈ {r_min:.2f} bohr = {r_min * 0.529:.3f} Å {sane}"
        )
    print("  Reference: r_min ≈ 1.81 bohr = 0.957 Å (DFTB0, MSINDO, RHF/6-31G)")
    print()

    # ── Detailed PES table ────────────────────────────────────────────
    print("Detailed PES values:")
    print(f"{'r':>6s} {'DFTB0':>12s} {'GFN2':>12s} {'PM6':>12s}")
    for r in [1.50, 1.65, 1.81, 2.00, 2.20]:
        d = molecular_energy(water_atoms(r), "dftb0")
        g = molecular_energy(water_atoms(r), "gfn2")
        p = molecular_energy(water_atoms(r), "pm6")
        print(f"{r:6.2f} {d:12.6f} {g:12.6f} {p:12.6f}")


def water_atoms(r: float) -> list:
    a = math.radians(52.25)
    z = r * math.cos(a)
    y = r * math.sin(a)
    return [Atom(8, [0, 0, 0]), Atom(1, [0, y, z]), Atom(1, [0, -y, z])]


def ch4_atoms() -> list:
    d = 2.0 / math.sqrt(3)
    return [
        Atom(6, [0, 0, 0]),
        Atom(1, [d, d, d]),
        Atom(1, [-d, -d, d]),
        Atom(1, [-d, d, -d]),
        Atom(1, [d, -d, -d]),
    ]


def nh3_atoms() -> list:
    return [
        Atom(7, [0, 0, 0]),
        Atom(1, [0, 0, 1.91]),
        Atom(1, [1.77, 0, -0.72]),
        Atom(1, [-1.77, 0, -0.72]),
    ]


def co2_atoms() -> list:
    return [Atom(6, [0, 0, 0]), Atom(8, [2.2, 0, 0]), Atom(8, [-2.2, 0, 0])]


if __name__ == "__main__":
    main()
