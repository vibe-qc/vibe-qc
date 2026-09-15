#!/usr/bin/env python3
"""Name all qc-input-library benchmark systems with the IUPAC naming engine.

Usage:
    python scripts/name_systems.py [--json] [--output NAME_SYSTEMS.csv]

Reads geometry definitions from ``scripts/_geometries.py``, extracts atom
coordinates, calls :func:`vibeqc.naming.name_from_atoms_detailed`, and
reports the IUPAC name, source, and confidence alongside the ad-hoc system
name.

Also handles periodic systems (named by composition without guessing a phase).

References
----------
- Handover: ``handovers/HANDOVER_IUPAC_NAMING.md`` work stream 5
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

# Resolve paths relative to qc-input-library repo root
_INPUT_LIB = Path(__file__).resolve().parent.parent
_GEOMETRIES = _INPUT_LIB / "scripts" / "_geometries.py"

try:
    from vibeqc.naming import (
        Confidence,
        NamedResult,
        NamingSource,
        name_from_atoms_detailed,
    )
except ImportError:
    print("ERROR: vibeqc must be installed in the Python path.")
    sys.exit(1)

# ── Mapping from element symbol → atomic number ──────────────────────────────

_SYMBOL_TO_Z = {
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
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "Pb": 82,
}


# ── Load geometry definitions ───────────────────────────────────────────────


def _load_geometries() -> list[tuple[str, dict]]:
    """Import all ``build_system`` functions from _geometries.py.

    Returns list of (function_name, kwargs) tuples.
    """
    geometries_mod_path = str(_GEOMETRIES)
    if not Path(geometries_mod_path).exists():
        print(f"WARNING: {_GEOMETRIES} not found — no geometries to name.")
        return []

    # Execute the module to access its functions
    import importlib.util

    spec = importlib.util.spec_from_file_location("_geometries", geometries_mod_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[possibly-unresolved-attribute]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    results: list[tuple[str, dict]] = []
    for name in sorted(dir(mod)):
        obj = getattr(mod, name)
        if callable(obj) and not name.startswith("_"):
            try:
                kw = obj()
                if isinstance(kw, dict) and "atoms" in kw:
                    results.append((name, kw))
            except Exception:  # noqa: BLE001 — some functions may fail independently
                pass
    return results


def _geometries_to_atoms(
    atoms_list: list[tuple[str, ...]],
) -> list[tuple[int, float, float, float]]:
    """Convert [{'symbol', x, y, z}] to (Z, x_ang, y_ang, z_ang) tuples."""
    result = []
    for entry in atoms_list:
        if isinstance(entry, tuple) and len(entry) >= 4:
            symbol, x, y, z = (
                entry[0],
                float(entry[1]),
                float(entry[2]),
                float(entry[3]),
            )
            z_num = _SYMBOL_TO_Z.get(symbol, -1)
            result.append((z_num, x, y, z))
        elif isinstance(entry, dict):
            symbol = entry.get("symbol", entry.get("element", ""))
            pos = entry.get("position", entry.get("pos", [0, 0, 0]))
            if hasattr(pos, "__iter__"):
                x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            else:
                x, y, z = 0.0, 0.0, 0.0
            z_num = _SYMBOL_TO_Z.get(symbol, -1)
            result.append((z_num, x, y, z))
    return result


def _formula_from_atoms(atoms_list: list[tuple[str, ...]]) -> str:
    """Derive Hill formula from atom entries."""
    from collections import Counter

    counts: dict[str, int] = {}
    for entry in atoms_list:
        symbol = (
            entry[0] if isinstance(entry, (tuple, list)) else entry.get("symbol", "?")
        )
        counts[symbol] = counts.get(symbol, 0) + 1

    parts: list[str] = []
    if "C" in counts:
        c = counts.pop("C")
        parts.append(f"C{c}" if c > 1 else "C")
    if "H" in counts:
        h = counts.pop("H")
        parts.append(f"H{h}" if h > 1 else "H")
    for sym in sorted(counts):
        cnt = counts[sym]
        parts.append(f"{sym}{cnt}" if cnt > 1 else sym)
    return "".join(parts)


# ── Periodic system naming ───────────────────────────────────────────────────


def _name_periodic(system_name: str, formula: str) -> NamedResult:
    """Name composition without assigning a mineral from formula alone."""
    from vibeqc_naming.solids import name_solid_from_formula

    result = name_solid_from_formula(formula)
    return NamedResult(
        name=result.name, source=NamingSource(result.source.value),
        confidence=Confidence(result.confidence.value), formula=result.formula,
        is_trivial=result.is_trivial,
    )


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    """Run naming on all systems and report results."""
    systems = _load_geometries()
    if not systems:
        print("No geometry definitions found in scripts/_geometries.py.")
        return

    rows: list[dict] = []
    molecule_count = 0
    periodic_count = 0

    for sys_name, kwargs in systems:
        atoms_raw = kwargs.get("atoms", [])
        formula = _formula_from_atoms(atoms_raw)
        atoms = _geometries_to_atoms(atoms_raw)

        if not atoms:
            continue

        # Check for periodic flag
        is_periodic = kwargs.get("periodic", kwargs.get("pbc", False))
        has_lattice = "lattice" in kwargs or "cell" in kwargs

        if is_periodic or has_lattice:
            result = _name_periodic(sys_name, formula)
            periodic_count += 1
        else:
            result = name_from_atoms_detailed(
                atoms,
                prefer_trivial=True,
                with_external=False,
            )
            molecule_count += 1

        rows.append(
            {
                "system": sys_name,
                "formula": formula,
                "type": "periodic" if (is_periodic or has_lattice) else "molecular",
                "name": result.name,
                "source": result.source.value,
                "confidence": result.confidence.value,
            }
        )

    # Print summary
    print(
        f"IUPAC naming report — {len(rows)} systems ({molecule_count} molecular, {periodic_count} periodic)"
    )
    print()
    print(
        f"{'system':<25s} {'formula':<10s} {'name':<40s} {'source':<28s} {'confidence'}"
    )
    print("-" * 120)
    for row in rows:
        print(
            f"{row['system']:<25s} "
            f"{row['formula']:<10s} "
            f"{row['name']:<40s} "
            f"{row['source']:<28s} "
            f"{row['confidence']}"
        )

    # JSON output option
    if "--json" in sys.argv:
        print()
        print("--- JSON ---")
        print(json.dumps(rows, indent=2))

    # CSV file output
    output_path = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg.startswith("--output="):
            output_path = arg.split("=", 1)[1]
        elif arg == "--output" and i < len(sys.argv) - 1:
            output_path = sys.argv[i + 1]

    if output_path:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV written to {output_path}")


if __name__ == "__main__":
    main()
