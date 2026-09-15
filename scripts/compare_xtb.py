#!/usr/bin/env python3
"""Compare vibe-qc GFN2-xTB energies against the reference xtb program.

For each test molecule, runs ``xtb --gfn 2`` and computes
vibe-qc GFN2-xTB energy.  Exports per-molecule comparison JSON.

Important note on expected differences
--------------------------------------
The reference xtb program (Grimme group) includes the anisotropic
second-order electrostatic contribution from the H¹ atomic
multipole moment.  vibe-qc's GFN2-xTB implementation does not yet
include this term, so a systematic offset of up to ~0.5 kcal/mol
is expected for hydrogen-containing molecules.  This difference is
documented, not a bug — it will disappear once vibe-qc implements
the full anisotropic multipole electrostatics.

Usage:
    python scripts/compare_xtb.py
    python scripts/compare_xtb.py --output benchmark-exports/comparison_xtb.json
    python scripts/compare_xtb.py --xtb /path/to/xtb

Requires:
    - vibe-qc editable install
    - xtb binary on PATH (or --xtb)
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ── Add vibe-qc python to path if needed ────────────────────────
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_VIBEQC_PYTHON = _REPO / "python"
if str(_VIBEQC_PYTHON) not in sys.path:
    sys.path.insert(0, str(_VIBEQC_PYTHON))

# ── Constants ───────────────────────────────────────────────────
BOHR_TO_ANGSTROM = 0.529177210903
EV_PER_HA = 27.211407952353

# Molecules in bohr (from generate_semiempirical_benchmarks.py)
MOLECULES: dict[str, list[tuple[str, list[float]]]] = {
    "H2": [
        ("H", [0.0, 0.0, 0.0]),
        ("H", [1.4, 0.0, 0.0]),
    ],
    "H2O": [
        ("O", [0.0000, 0.0000, 0.1173]),
        ("H", [0.0000, 1.4315, -0.9386]),
        ("H", [0.0000, -1.4315, -0.9386]),
    ],
    "CH4": [
        ("C", [0.0000, 0.0000, 0.0000]),
        ("H", [1.1869, 1.1869, 1.1869]),
        ("H", [-1.1869, -1.1869, 1.1869]),
        ("H", [1.1869, -1.1869, -1.1869]),
        ("H", [-1.1869, 1.1869, -1.1869]),
    ],
    "NH3": [
        ("N", [0.0000, 0.0000, 0.1147]),
        ("H", [0.0000, 1.7670, -0.4810]),
        ("H", [1.5303, -0.8835, -0.4810]),
        ("H", [-1.5303, -0.8835, -0.4810]),
    ],
    "CO2": [
        ("C", [0.0000, 0.0000, 0.0000]),
        ("O", [0.0000, 0.0000, 2.1960]),
        ("O", [0.0000, 0.0000, -2.1960]),
    ],
    "H2CO": [
        ("C", [0.0000, 0.0000, 0.0000]),
        ("O", [0.0000, 0.0000, 2.2740]),
        ("H", [0.0000, 1.7700, -1.0820]),
        ("H", [0.0000, -1.7700, -1.0820]),
    ],
}

# Atomic number map
_Z_MAP = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
}


# ── Helpers ─────────────────────────────────────────────────────


def _bohr_to_ang(xyz: list[float]) -> list[float]:
    return [c * BOHR_TO_ANGSTROM for c in xyz]


def _find_xtb(binary_override: str | None = None) -> str | None:
    """Locate the xtb binary."""
    if binary_override:
        p = Path(binary_override)
        if p.is_file():
            return str(p.resolve())
        return binary_override
    found = shutil.which("xtb")
    if found:
        return found
    return None


def _write_xyz(atoms: list[tuple[str, list[float]]], path: Path) -> None:
    """Write an XYZ file (coordinates in Å)."""
    n = len(atoms)
    lines = [str(n), ""]
    for sym, xyz in atoms:
        x, y, z = _bohr_to_ang(xyz)
        lines.append(f"{sym:2s}  {x:14.8f}  {y:14.8f}  {z:14.8f}")
    path.write_text("\n".join(lines) + "\n")


def _parse_xtb_energy(output: str) -> float | None:
    """Extract total energy (Ha) from xtb stdout.

    xtb prints a SUMMARY block with:
        :: total energy         -5.768891678223 Eh        ::
    """
    m = re.search(r"total energy\s+([-\d.]+)\s+Eh", output, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # Fallback: "TOTAL ENERGY" line
    m = re.search(r"TOTAL ENERGY\s+([-\d.]+)\s+Eh", output, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def _run_xtb_molecule(
    name: str,
    atoms: list[tuple[str, list[float]]],
    xtb_bin: str,
) -> float | None:
    """Run xtb --gfn 2 on a molecule; return total energy in Ha."""
    try:
        with tempfile.TemporaryDirectory(prefix=f"xtb_{name}_") as tmp:
            tmp_path = Path(tmp)
            xyz_path = tmp_path / "coord.xyz"
            _write_xyz(atoms, xyz_path)

            proc = subprocess.run(
                [xtb_bin, str(xyz_path), "--gfn", "2"],
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=300,
            )
            combined = proc.stdout
            # xtb also writes to xtb.out
            out_file = tmp_path / "xtb.out"
            if out_file.exists():
                combined = out_file.read_text() + "\n" + combined
            e = _parse_xtb_energy(combined)
            if e is None and proc.returncode != 0:
                print(
                    f"  [warn] {name} xtb: rc={proc.returncode}\n"
                    f"    stderr: {proc.stderr[:200]}",
                    file=sys.stderr,
                )
            return e
    except subprocess.TimeoutExpired:
        print(f"  [warn] {name} xtb: timed out", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"  [warn] {name} xtb: {exc}", file=sys.stderr)
        return None


def _make_molecule(atoms: list[tuple[str, list[float]]]):
    """Build a vibeqc Molecule from (symbol, coords) list."""
    from vibeqc._vibeqc_core import Atom, Molecule

    return Molecule([Atom(_Z_MAP.get(sym, 0), xyz) for sym, xyz in atoms])


def _run_vibeqc_gfn2(name: str, mol) -> float | None:
    """Run vibe-qc GFN2-xTB; return total energy in Ha."""
    try:
        from vibeqc.semiempirical.methods.gfn2 import GFN2Model
        from vibeqc.semiempirical.methods.gfn2_params import (
            load_gfn2_params,
        )

        params = load_gfn2_params()
        model = GFN2Model(mol, params)
        return float(model.energy())
    except Exception as exc:
        print(
            f"  [warn] {name} vibe-qc GFN2-xTB: {exc}",
            file=sys.stderr,
        )
        return None


# ── Main ────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare vibe-qc GFN2-xTB vs reference xtb (--gfn 2)"
    )
    parser.add_argument(
        "--output",
        default="benchmark-exports/comparison_xtb.json",
        help="Output JSON path",
    )
    parser.add_argument("--xtb", default=None, help="Path to xtb binary")
    args = parser.parse_args()

    # Locate xtb
    xtb_bin = _find_xtb(args.xtb)
    have_xtb = xtb_bin is not None
    if not have_xtb:
        print(
            "xtb not found. Install xtb or use --xtb.",
            file=sys.stderr,
        )
        print(
            "Skipping xtb comparison — output will contain only vibe-qc data.",
        )
    else:
        print(f"xtb binary: {xtb_bin}")
    print()

    print(
        "NOTE: The reference xtb includes the H¹ anisotropic "
        "multipole contribution.\n"
        "      vibe-qc GFN2-xTB does not yet implement this term.\n"
        "      Expect ~0.1-0.5 kcal/mol systematic offset for "
        "H-containing molecules.\n"
    )

    results: dict = {}

    for mol_name, atoms in MOLECULES.items():
        print(f"--- {mol_name} ---")
        mol = _make_molecule(atoms)

        # vibe-qc
        v_gfn2 = _run_vibeqc_gfn2(mol_name, mol)
        print(f"  vibe-qc  GFN2-xTB: {v_gfn2}")

        # xtb
        if have_xtb:
            r_gfn2 = _run_xtb_molecule(mol_name, atoms, xtb_bin)
            print(f"  xtb      GFN2-xTB: {r_gfn2}")
        else:
            r_gfn2 = None
            print("  xtb      <not available>")

        delta = (
            (v_gfn2 - r_gfn2) if (v_gfn2 is not None and r_gfn2 is not None) else None
        )

        results[mol_name] = {
            "GFN2-xTB": {
                "vibeqc_ha": v_gfn2,
                "reference_ha": r_gfn2,
                "delta_ha": delta,
                "delta_ev": delta * EV_PER_HA if delta is not None else None,
                "_note": (
                    "xtb reference includes H¹ anisotropic "
                    "multipole term; vibe-qc does not.  Systematic "
                    "offset expected for molecules with H."
                )
                if r_gfn2 is not None
                else None,
            }
        }
        print()

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
