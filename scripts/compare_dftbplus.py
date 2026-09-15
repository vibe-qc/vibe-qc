#!/usr/bin/env python3
"""Compare vibe-qc DFTB0 / SCC-DFTB energies against DFTB+.

For each test molecule, runs DFTB+ (mio-1-1 SKF) in both non-SCC
and SCC modes, computes vibe-qc DFTB0 + SCC-DFTB energies, and
exports a per-molecule comparison JSON.

Usage:
    python scripts/compare_dftbplus.py
    python scripts/compare_dftbplus.py --output benchmark-exports/comparison_dftbplus.json
    python scripts/compare_dftbplus.py --dftbplus /path/to/dftb+

Requires:
    - vibe-qc editable install
    - DFTB+ binary on PATH (or --dftbplus)
    - mio-1-1 SKF files accessible (DFTB+ default search path,
      or set DFTBPLUS_PARAM_DIR)
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

# Max angular momentum map for DFTB+ SKF selection
_DFTB_LMAX = {
    "H": "s",
    "He": "s",
    "C": "p",
    "N": "p",
    "O": "p",
    "F": "p",
    "Si": "p",
    "P": "p",
    "S": "p",
    "Cl": "p",
}


# ── Helpers ─────────────────────────────────────────────────────


def _bohr_to_ang(xyz: list[float]) -> list[float]:
    return [c * BOHR_TO_ANGSTROM for c in xyz]


def _find_dftbplus(binary_override: str | None = None) -> str | None:
    """Locate the dftb+ binary."""
    if binary_override:
        p = Path(binary_override)
        if p.is_file():
            return str(p.resolve())
        return binary_override  # trust it's a name on PATH
    for name in ("dftb+", "dftbplus"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _write_gen_geometry(atoms: list[tuple[str, list[float]]], path: Path) -> None:
    """Write a DFTB+ GEN-format geometry file (coordinates in Å)."""
    symbols = [sym for sym, _ in atoms]
    unique = sorted(set(symbols))
    n_atoms = len(atoms)

    lines = [f"{n_atoms}  {' '.join(unique)}"]
    lines.append(" ".join(symbols))
    for i, (sym, xyz) in enumerate(atoms):
        x, y, z = _bohr_to_ang(xyz)
        lines.append(
            f" {i + 1:4d}  {_Z_MAP.get(sym, 0):2d}  {x:14.8f}  {y:14.8f}  {z:14.8f}"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_dftb_hsd(
    atoms: list[tuple[str, list[float]]], work_dir: Path, scc: bool
) -> None:
    """Write DFTB+ HSD input (dftb_in.hsd + geom.gen)."""
    _write_gen_geometry(atoms, work_dir / "geom.gen")

    symbols = sorted(set(sym for sym, _ in atoms))
    lmax_lines = "\n".join(f'    {s} = "{_DFTB_LMAX.get(s, "p")}"' for s in symbols)

    # Use DFTBPLUS_PARAM_DIR if set; otherwise assume mio-1-1/ in cwd
    param_dir = os.environ.get("DFTBPLUS_PARAM_DIR", "./mio-1-1/")

    if scc:
        scc_block = (
            "Scc = Yes\n"
            "    SCCTolerance = 1e-8\n"
            "    Mixer = Broyden {\n"
            "        MixingParameter = 0.2\n"
            "    }\n"
            "    MaxSCCIterations = 200"
        )
    else:
        scc_block = "Scc = No"

    hsd = f"""Geometry = GenFormat {{
    <<< "geom.gen"
}}

Driver = {{}}

Hamiltonian = DFTB {{
    {scc_block}
    SlaterKosterFiles = Type2FileNames {{
        Prefix = "{param_dir}"
        Separator = "-"
        Suffix = ".skf"
    }}
    MaxAngularMomentum = {{
{lmax_lines}
    }}
}}

Options = {{}}

Analysis = {{
    CalculateForces = No
}}
"""
    (work_dir / "dftb_in.hsd").write_text(hsd)


def _parse_dftbplus_energy(output: str) -> float | None:
    """Extract total energy (Ha) from DFTB+ detailed.out or stdout."""
    for pat in (
        r"Total energy:\s+([-\d.]+)\s+H",
        r"Total Energy:\s+([-\d.]+)\s+H",
    ):
        m = re.search(pat, output)
        if m:
            return float(m.group(1))
    return None


def _run_dftbplus_molecule(
    name: str,
    atoms: list[tuple[str, list[float]]],
    dftbplus_bin: str,
) -> tuple[float | None, float | None]:
    """Run DFTB+ non-SCC and SCC; return (dftb0_ha, scc_ha)."""
    results: dict[str, float | None] = {}

    for mode, use_scc in [("dftb0", False), ("scc", True)]:
        try:
            with tempfile.TemporaryDirectory(prefix=f"dftbplus_{name}_{mode}_") as tmp:
                tmp_path = Path(tmp)
                _write_dftb_hsd(atoms, tmp_path, scc=use_scc)

                proc = subprocess.run(
                    [dftbplus_bin],
                    cwd=tmp_path,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                combined = proc.stdout + "\n" + proc.stderr
                detail = tmp_path / "detailed.out"
                if detail.exists():
                    combined = detail.read_text() + "\n" + combined
                e = _parse_dftbplus_energy(combined)
                results[mode] = e
                if e is None and proc.returncode != 0:
                    print(
                        f"  [warn] {name} DFTB+ {mode}: rc={proc.returncode}",
                        file=sys.stderr,
                    )
        except subprocess.TimeoutExpired:
            print(
                f"  [warn] {name} DFTB+ {mode}: timed out",
                file=sys.stderr,
            )
            results[mode] = None
        except Exception as exc:
            print(
                f"  [warn] {name} DFTB+ {mode}: {exc}",
                file=sys.stderr,
            )
            results[mode] = None

    return results.get("dftb0"), results.get("scc")


def _make_molecule(atoms: list[tuple[str, list[float]]]):
    """Build a vibeqc Molecule from (symbol, coords) list."""
    from vibeqc._vibeqc_core import Atom, Molecule

    return Molecule([Atom(_Z_MAP.get(sym, 0), xyz) for sym, xyz in atoms])


def _run_vibeqc_dftb(name: str, mol):
    """Run vibe-qc DFTB0 and SCC-DFTB; return (dftb0_ha, scc_ha)."""
    from vibeqc.semiempirical import DFTB0Model, SCCDFTBModel
    from vibeqc.semiempirical.parameters import default_parameters

    p = default_parameters()
    results: dict[str, float | None] = {}

    for label, cls in [("dftb0", DFTB0Model), ("scc", SCCDFTBModel)]:
        try:
            model = cls(mol, p)
            results[label] = float(model.energy())
        except Exception as exc:
            print(
                f"  [warn] {name} vibe-qc {label}: {exc}",
                file=sys.stderr,
            )
            results[label] = None

    return results.get("dftb0"), results.get("scc")


# ── Main ────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare vibe-qc DFTB0/SCC-DFTB vs DFTB+ (mio-1-1)"
    )
    parser.add_argument(
        "--output",
        default="benchmark-exports/comparison_dftbplus.json",
        help="Output JSON path",
    )
    parser.add_argument("--dftbplus", default=None, help="Path to dftb+ binary")
    args = parser.parse_args()

    # Locate DFTB+
    dftbplus_bin = _find_dftbplus(args.dftbplus)
    have_dftbplus = dftbplus_bin is not None
    if not have_dftbplus:
        print(
            "DFTB+ not found. Install DFTB+ or use --dftbplus.",
            file=sys.stderr,
        )
        print(
            "Skipping DFTB+ comparison — output will contain only vibe-qc data.",
        )
    else:
        print(f"DFTB+ binary: {dftbplus_bin}")
    print()

    results: dict = {}

    for mol_name, atoms in MOLECULES.items():
        print(f"--- {mol_name} ---")
        mol = _make_molecule(atoms)

        # vibe-qc
        v_dftb0, v_scc = _run_vibeqc_dftb(mol_name, mol)
        print(f"  vibe-qc  DFTB0:    {v_dftb0}")
        print(f"  vibe-qc  SCC-DFTB: {v_scc}")

        # DFTB+
        if have_dftbplus:
            r_dftb0, r_scc = _run_dftbplus_molecule(mol_name, atoms, dftbplus_bin)
            print(f"  DFTB+    DFTB0:    {r_dftb0}")
            print(f"  DFTB+    SCC-DFTB: {r_scc}")
        else:
            r_dftb0, r_scc = None, None
            print("  DFTB+    <not available>")

        entry: dict = {}

        def _cmp(vibeqc_ha, ref_ha):
            delta = (
                (vibeqc_ha - ref_ha)
                if (vibeqc_ha is not None and ref_ha is not None)
                else None
            )
            return {
                "vibeqc_ha": vibeqc_ha,
                "reference_ha": ref_ha,
                "delta_ha": delta,
                "delta_ev": delta * EV_PER_HA if delta is not None else None,
            }

        entry["DFTB0"] = _cmp(v_dftb0, r_dftb0)
        entry["SCC-DFTB"] = _cmp(v_scc, r_scc)

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
