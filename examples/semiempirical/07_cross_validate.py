#!/usr/bin/env python3
"""Smoke comparison of all semiempirical methods in vibe-qc.

Compares energies across methods for a standard test set and checks the
one robust ordering expectation used here: SCC-DFTB should not be higher
than DFTB0 on the same geometry. Do not read a universal ordering across
DFTB, GFN2-xTB, PM6, and OMx from this table; they are different models
with different parameter quality, and GFN2-xTB is currently experimental.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
from vibeqc import Atom, Molecule
from vibeqc.runner import run_job

# ── Test systems ──────────────────────────────────────────────────────────
MOLECULES = {
    "H2": Molecule(
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [1.4, 0.0, 0.0])], charge=0, multiplicity=1
    ),
    "H2O": Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [1.809, 0.0, 0.0]),
            Atom(1, [-0.453, 1.752, 0.0]),
        ],
        charge=0,
        multiplicity=1,
    ),
    "CH4": Molecule(
        [
            Atom(6, [0.0, 0.0, 0.0]),
            Atom(1, [1.087, 0.0, 0.0]),
            Atom(1, [-0.362, 1.025, 0.0]),
            Atom(1, [-0.362, -0.513, 0.887]),
            Atom(1, [-0.362, -0.513, -0.887]),
        ],
        charge=0,
        multiplicity=1,
    ),
    "NH3": Molecule(
        [
            Atom(7, [0.0, 0.0, 0.0]),
            Atom(1, [1.012, 0.0, 0.0]),
            Atom(1, [-0.506, 0.876, 0.0]),
            Atom(1, [-0.506, -0.876, 0.0]),
        ],
        charge=0,
        multiplicity=1,
    ),
    "CO2": Molecule(
        [
            Atom(6, [0.0, 0.0, 0.0]),
            Atom(8, [2.196, 0.0, 0.0]),
            Atom(8, [-2.196, 0.0, 0.0]),
        ],
        charge=0,
        multiplicity=1,
    ),
    "C2H4": Molecule(
        [
            Atom(6, [0.0, 0.0, 0.0]),
            Atom(6, [2.5, 0.0, 0.0]),
            Atom(1, [-0.55, 1.75, 0.0]),
            Atom(1, [-0.55, -1.75, 0.0]),
            Atom(1, [3.05, 1.75, 0.0]),
            Atom(1, [3.05, -1.75, 0.0]),
        ],
        charge=0,
        multiplicity=1,
    ),
}

# Methods to test — grouped by family
METHODS = [
    "dftb0",  # DFTB non-SCC
    "scc_dftb",  # DFTB SCC
    "pm6",  # NDDO PM6
    "gfn2_xtb",  # GFN2-xTB
    "om1",  # OM1
    "om2",  # OM2
    "om3",  # OM3
]


def main():
    print("=" * 80)
    print("Semiempirical smoke comparison — energy table")
    print("=" * 80)

    results = {}

    with tempfile.TemporaryDirectory() as tmp:
        for name, mol in MOLECULES.items():
            results[name] = {}
            for method in METHODS:
                out = os.path.join(tmp, f"{name}_{method}")
                try:
                    r = run_job(
                        mol,
                        method=method,
                        output=out,
                        write_molden_file=False,
                        write_xyz_file=False,
                        citations=False,
                    )
                    e = float(r.energy) if r else np.nan
                    conv = r.converged if r else False
                except Exception as exc:
                    e = np.nan
                    conv = False
                results[name][method] = {"E": e, "conv": conv}

    # ── Print main table ──
    header = f"{'Molecule':<8}" + "".join(f"{m:<14}" for m in METHODS)
    print(f"\n{header}")
    print("-" * len(header))

    for name in MOLECULES:
        row = f"{name:<8}"
        for method in METHODS:
            e = results[name][method]["E"]
            conv = results[name][method]["conv"]
            tag = "✓" if conv else "✗"
            if np.isfinite(e):
                row += f"{e:<+10.4f} {tag:<3}"
            else:
                row += f"{'FAIL':<14}"
        print(row)

    # ── DFTB-family ordering checks ──
    print("\n" + "=" * 80)
    print("Consistency checks")
    print("=" * 80)

    checks_passed = 0
    checks_total = 0

    for name in MOLECULES:
        e = results[name]
        if all(np.isfinite(e[m]["E"]) for m in METHODS):
            # SCC-DFTB should not sit above the DFTB0 energy.
            checks_total += 1
            if e["scc_dftb"]["E"] <= e["dftb0"]["E"] + 1e-8:
                checks_passed += 1
            else:
                print(
                    f"  ⚠ {name}: SCC ({e['scc_dftb']['E']:.4f}) > DFTB0 ({e['dftb0']['E']:.4f})"
                )
        else:
            print(f"  ⚠ {name}: some methods failed to converge")

    print(f"\n  SCC-vs-DFTB0 checks: {checks_passed}/{checks_total} passed")

    # ── Summary ──
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    for name in MOLECULES:
        es = results[name]
        print(f"\n{name}:")
        for method in METHODS:
            e = es[method]
            tag = "✓" if e["conv"] else "✗"
            if np.isfinite(e["E"]):
                print(f"  {method:<12}  E = {e['E']:>+12.6f} Ha  {tag}")
        print()


if __name__ == "__main__":
    main()
