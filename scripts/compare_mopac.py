#!/usr/bin/env python3
"""Compare vibe-qc PM6 energies against the reference MOPAC program.

For each test molecule, runs MOPAC with the PM6 Hamiltonian and
computes vibe-qc PM6 energies with both the 5-element Stewart-2007
parameter set and the full 75-chemical-element MOPAC parameter set.  Exports
per-molecule comparison JSON.

Important note on MOPAC's energy convention
-------------------------------------------
MOPAC reports the **heat of formation** (ΔHf) by default, which is
not the same as the total electronic energy.  This script extracts
the **total energy** (ETOT) from MOPAC output.  If your MOPAC
installation outputs only ΔHf, set the environment variable
``MOPAC_USE_ETOT=0`` to fall back to ΔHf comparison (less reliable).

Usage:
    python scripts/compare_mopac.py
    python scripts/compare_mopac.py --output benchmark-exports/comparison_mopac.json
    python scripts/compare_mopac.py --mopac /path/to/mopac

Requires:
    - vibe-qc editable install
    - MOPAC binary on PATH (or --mopac)
"""

from __future__ import annotations

import json
import os
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


def _find_mopac(binary_override: str | None = None) -> str | None:
    """Locate the mopac binary."""
    if binary_override:
        p = Path(binary_override)
        if p.is_file():
            return str(p.resolve())
        return binary_override
    found = shutil.which("mopac")
    if found:
        return found
    return None


def _write_mopac_input(atoms: list[tuple[str, list[float]]], path: Path) -> None:
    """Write a MOPAC input file (.mop) with PM6, 1SCF, no GEO refinement.

    MOPAC input format (first 3 lines are comments):
        PM6 1SCF NOINTER XYZ CHARGE=0
        comment line 2
        comment line 3
        O   0.00000 0  0.00000 0  0.06207 0
        H   0.75732 0  0.00000 0 -0.49666 0
        H  -0.75732 0  0.00000 0 -0.49666 0
    """
    lines = [
        "PM6 1SCF NOINTER XYZ CHARGE=0",
        "vibe-qc PM6 comparison benchmark",
        "",
    ]
    for sym, xyz in atoms:
        x, y, z = _bohr_to_ang(xyz)
        lines.append(f"{sym:2s}  {x:10.6f} 1 {y:10.6f} 1 {z:10.6f} 1")
    path.write_text("\n".join(lines) + "\n")


def _parse_mopac_energy(text: str) -> float | None:
    """Extract total energy from MOPAC output.

    MOPAC writes lines like:
        TOTAL ENERGY            =        -13.46422 EV
    or (older versions):
        TOTAL ENERGY            =        -13.46422 EV
        ELECTRONIC ENERGY       =       -276.12345 EV
        CORE-CORE REPULSION     =        262.65923 EV

    We extract TOTAL ENERGY in eV and convert to Ha.
    """
    m = re.search(r"TOTAL ENERGY\s+=\s+([-\d.]+)\s+EV", text, re.IGNORECASE)
    if m:
        return float(m.group(1)) / EV_PER_HA
    return None


def _parse_mopac_heat_of_formation(text: str) -> float | None:
    """Extract heat of formation (kcal/mol) from MOPAC output.

    Used as a fallback if TOTAL ENERGY is unavailable.
    Converted to Ha via: 1 kcal/mol = 0.001593601 Ha
    """
    m = re.search(r"HEAT OF FORMATION\s+=\s+([-\d.]+)\s+KCAL", text, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 0.001593601
    return None


def _run_mopac_molecule(
    name: str,
    atoms: list[tuple[str, list[float]]],
    mopac_bin: str,
) -> float | None:
    """Run MOPAC PM6 on a molecule; return total energy in Ha."""
    try:
        with tempfile.TemporaryDirectory(prefix=f"mopac_{name}_") as tmp:
            tmp_path = Path(tmp)
            base = tmp_path / name
            mop_path = tmp_path / f"{name}.mop"
            _write_mopac_input(atoms, mop_path)

            proc = subprocess.run(
                [mopac_bin, str(mop_path)],
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=300,
            )

            # MOPAC writes {name}.out
            out_path = tmp_path / f"{name}.out"
            text = ""
            if out_path.exists():
                text = out_path.read_text()
            if not text:
                text = proc.stdout

            e_ha = _parse_mopac_energy(text)
            if e_ha is not None:
                return e_ha

            # Fallback: try ETOT (alternate naming)
            m = re.search(r"ETOT\s*=\s*([-\d.]+)\s*EV", text, re.IGNORECASE)
            if m:
                return float(m.group(1)) / EV_PER_HA

            # Fallback: heat of formation
            if os.environ.get("MOPAC_USE_ETOT", "1") != "0":
                e_hf = _parse_mopac_heat_of_formation(text)
                if e_hf is not None:
                    print(
                        f"  [warn] {name} MOPAC: using ΔHf fallback "
                        f"(set MOPAC_USE_ETOT=0 to disable)",
                        file=sys.stderr,
                    )
                    return e_hf

            if proc.returncode != 0:
                print(
                    f"  [warn] {name} MOPAC: rc={proc.returncode}",
                    file=sys.stderr,
                )
            return None
    except subprocess.TimeoutExpired:
        print(f"  [warn] {name} MOPAC: timed out", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"  [warn] {name} MOPAC: {exc}", file=sys.stderr)
        return None


def _make_molecule(atoms: list[tuple[str, list[float]]]):
    """Build a vibeqc Molecule from (symbol, coords) list."""
    from vibeqc._vibeqc_core import Atom, Molecule

    return Molecule([Atom(_Z_MAP.get(sym, 0), xyz) for sym, xyz in atoms])


def _run_vibeqc_pm6(name: str, mol) -> dict[str, float | None]:
    """Run vibe-qc PM6 (both 5-el and 82-el); return results dict."""
    from vibeqc._vibeqc_core.semiempirical.nddo import run_pm6
    from vibeqc.semiempirical.methods.pm6_params import (
        load_pm6_mopac_params,
        load_pm6_params,
    )

    results: dict[str, float | None] = {}

    for label, loader in [
        ("PM6 (5 el)", load_pm6_params),
        ("PM6 (82 el)", load_pm6_mopac_params),
    ]:
        try:
            params = loader()
            r = run_pm6(mol, params, max_iter=200)
            results[label] = float(r.energy)
        except Exception as exc:
            print(
                f"  [warn] {name} vibe-qc {label}: {exc}",
                file=sys.stderr,
            )
            results[label] = None

    return results


# ── Main ────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare vibe-qc PM6 vs reference MOPAC (PM6)"
    )
    parser.add_argument(
        "--output",
        default="benchmark-exports/comparison_mopac.json",
        help="Output JSON path",
    )
    parser.add_argument("--mopac", default=None, help="Path to mopac binary")
    args = parser.parse_args()

    # Locate MOPAC
    mopac_bin = _find_mopac(args.mopac)
    have_mopac = mopac_bin is not None
    if not have_mopac:
        print(
            "mopac not found. Install MOPAC or use --mopac.",
            file=sys.stderr,
        )
        print(
            "Skipping MOPAC comparison — output will contain only vibe-qc data.",
        )
    else:
        print(f"mopac binary: {mopac_bin}")
    print()

    results: dict = {}

    for mol_name, atoms in MOLECULES.items():
        print(f"--- {mol_name} ---")
        mol = _make_molecule(atoms)

        # vibe-qc
        v_pm6 = _run_vibeqc_pm6(mol_name, mol)
        for label in ("PM6 (5 el)", "PM6 (82 el)"):
            print(f"  vibe-qc  {label:14s}: {v_pm6.get(label)}")

        # MOPAC
        if have_mopac:
            r_pm6 = _run_mopac_molecule(mol_name, atoms, mopac_bin)
            print(f"  MOPAC    {'PM6':14s}: {r_pm6}")
        else:
            r_pm6 = None
            print("  MOPAC    <not available>")

        entry: dict = {}

        for label in ("PM6 (5 el)", "PM6 (82 el)"):
            v = v_pm6.get(label)
            delta = (v - r_pm6) if (v is not None and r_pm6 is not None) else None
            entry[label] = {
                "vibeqc_ha": v,
                "reference_ha": r_pm6,
                "delta_ha": delta,
                "delta_ev": delta * EV_PER_HA if delta is not None else None,
            }

        results[mol_name] = entry
        print()

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
