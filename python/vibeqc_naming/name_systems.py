#!/usr/bin/env python3
"""Name all qc-input-library benchmark systems with the IUPAC naming engine.

Usage:
    PYTHONPATH=python python3 python/vibeqc_naming/name_systems.py [--json] [--output NAME_SYSTEMS.csv]

Reads geometry definitions from the qc-input-library's ``scripts/_geometries.py``,
extracts atom coordinates, calls :func:`vibeqc_naming.name_from_atoms_detailed`,
and reports the IUPAC name, source, and confidence for each system.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
# Also try the standalone package path
sys.path.insert(0, "python")

try:
    from vibeqc_naming import (
        Confidence,
        NamedResult,
        NamingSource,
        name_from_atoms_detailed,
    )
    from vibeqc_naming.iupac_name import _compositional_name_from_formula
except ImportError:
    print("ERROR: vibeqc_naming must be in the Python path.")
    print("Run: PYTHONPATH=python python3 python/vibeqc_naming/name_systems.py")
    sys.exit(1)

# ── Symbol → atomic number ───────────────────────────────────────────────────

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

# Find qc-input-library
_THIS = Path(__file__).resolve().parent
_INPUT_LIB_CANDIDATES = [
    _THIS.parent.parent / "qc-input-library" / "scripts" / "_geometries.py",
]

_GEOMETRIES = None
for candidate in _INPUT_LIB_CANDIDATES:
    if candidate.exists():
        _GEOMETRIES = candidate
        break


def _load_geometries() -> list[tuple[str, dict]]:
    """Load all geometry building functions from _geometries.py."""
    if _GEOMETRIES is None:
        print(f"WARNING: Could not find _geometries.py in {_INPUT_LIB_CANDIDATES}")
        return []

    import importlib.util

    spec = importlib.util.spec_from_file_location("_geometries", str(_GEOMETRIES))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    results: list[tuple[str, dict]] = []
    for name in sorted(dir(mod)):
        obj = getattr(mod, name)
        if callable(obj) and not name.startswith("_"):
            try:
                kw = obj()
                if isinstance(kw, dict) and "atoms" in kw:
                    results.append((name, kw))
            except Exception:
                pass
    return results


def _to_atom_tuples(atoms_list) -> list[tuple[int, float, float, float]]:
    """Convert geometry entries to (Z, x, y, z) tuples."""
    result = []
    for entry in atoms_list:
        if isinstance(entry, tuple) and len(entry) >= 4:
            symbol, x, y, z = (
                entry[0],
                float(entry[1]),
                float(entry[2]),
                float(entry[3]),
            )
            result.append((_SYMBOL_TO_Z.get(symbol, -1), x, y, z))
        elif isinstance(entry, dict):
            symbol = entry.get("symbol", entry.get("element", ""))
            pos = entry.get("position", [0, 0, 0])
            z = _SYMBOL_TO_Z.get(symbol, -1)
            result.append((z, float(pos[0]), float(pos[1]), float(pos[2])))
    return result


def _formula(atoms_list) -> str:
    """Hill formula from atom entries."""
    from collections import Counter

    counts = Counter(
        e[0] if isinstance(e, tuple) else e.get("symbol", "?") for e in atoms_list
    )
    parts = []
    if "C" in counts:
        c = counts.pop("C")
        parts.append(f"C{c}" if c > 1 else "C")
    if "H" in counts:
        h = counts.pop("H")
        parts.append(f"H{h}" if h > 1 else "H")
    for sym, cnt in sorted(counts.items()):
        parts.append(f"{sym}{cnt}" if cnt > 1 else sym)
    return "".join(parts)


# ── Periodic naming ──────────────────────────────────────────────────────────


def _name_periodic(formula: str) -> NamedResult:
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
    systems = _load_geometries()
    if not systems:
        print("No geometry definitions found.")
        return

    rows = []
    mol_count = peri_count = 0

    for sys_name, kwargs in systems:
        atoms_raw = kwargs.get("atoms", [])
        if not atoms_raw:
            continue
        formula = _formula(atoms_raw)
        atoms = _to_atom_tuples(atoms_raw)
        if not atoms:
            continue

        is_periodic = (
            kwargs.get("periodic", False) or "lattice" in kwargs or "cell" in kwargs
        )

        if is_periodic:
            result = _name_periodic(formula)
            peri_count += 1
        else:
            result = name_from_atoms_detailed(
                atoms, prefer_trivial=True, with_external=False
            )
            mol_count += 1

        rows.append(
            {
                "system": sys_name,
                "formula": formula,
                "type": "periodic" if is_periodic else "molecular",
                "name": result.name,
                "source": result.source.value,
                "confidence": result.confidence.value,
            }
        )

    print(
        f"IUPAC naming report — {len(rows)} systems ({mol_count} molecular, {peri_count} periodic)"
    )
    print()
    print(f"{'system':<25s} {'formula':<10s} {'name':<40s} {'source':<28s} {'conf'}")
    print("-" * 120)
    for row in rows:
        print(
            f"{row['system']:<25s} "
            f"{row['formula']:<10s} "
            f"{row['name']:<40s} "
            f"{row['source']:<28s} "
            f"{row['confidence']}"
        )

    if "--json" in sys.argv:
        print("\n--- JSON ---")
        print(json.dumps(rows, indent=2))

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
