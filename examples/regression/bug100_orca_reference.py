"""ORCA 6 reference — acetic acid B3LYP/def2-TZVP CPCM(water) comparison.

Part of BUG 100 investigation.  Runs the identical molecular system in
ORCA 6.1.1 and reports wall time, energy, and cavity details for
cross-code performance comparison.

Prerequisites
------------
* ORCA 6.1.1 installed and ``orca`` on PATH.
* The ORCA input is written to a temporary directory and executed via
  subprocess; output is captured and parsed.

Usage
-----
    .venv/bin/python examples/regression/bug100_orca_reference.py

Output
------
Prints a comparison table with wall times, energies, and cavity metrics.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

ORCA_INPUT = """! B3LYP def2-TZVP CPCM(Water) TightSCF NoPOP
%cpcm
  epsilon 78.39
end
%maxcore  8000
* xyz 0 1
C          1.8835100000       -0.1108300000        0.0000000000
H          1.4172800000       -1.0972200000       -0.0000000000
H          1.4170500000        0.4559200000        0.8847900000
H          1.4170500000        0.4559200000       -0.8847900000
C          3.4035100000       -0.1108300000        0.0000000000
O          3.9980700000       -1.1745400000       -0.0000000000
O          3.9980700000        1.0762700000        0.0000000000
H          4.9597100000        0.9390400000       -0.0000000000
*
"""


def _run_orca(input_text: str, timeout_s: int = 1800) -> tuple[float, str]:
    """Run ORCA, return (wall_seconds, stdout)."""
    work_dir = Path(tempfile.mkdtemp(prefix="orca_bug100_"))
    inp = work_dir / "acetic.inp"
    inp.write_text(input_text)

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            ["orca", str(inp)],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "ORCA not found on PATH. Install ORCA 6.1.1 or later: "
            "https://www.faccts.de/orca/"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ORCA timed out after {timeout_s}s")

    dt = time.perf_counter() - t0
    if result.returncode != 0:
        out_path = work_dir / "acetic.out"
        tail = ""
        if out_path.exists():
            lines = out_path.read_text().splitlines()
            tail = "\n".join(lines[-20:])
        raise RuntimeError(
            f"ORCA exited with code {result.returncode}\n"
            f"Last 20 lines of output:\n{tail}"
        )
    out_path = work_dir / "acetic.out"
    return dt, out_path.read_text() if out_path.exists() else ""


def _parse_orca_output(text: str) -> dict:
    """Extract key values from ORCA output."""
    info: dict = {}

    # Final single point energy
    m = re.search(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", text)
    if m:
        info["e_total"] = float(m.group(1))

    # CPCM solvation energy
    m = re.search(r"G_solv.*=\s+(-?\d+\.\d+)\s+Eh", text)
    if m:
        info["e_solv_eh"] = float(m.group(1))

    # SCF iterations
    m = re.search(r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES", text)
    if m:
        info["n_scf_iters"] = int(m.group(1))

    # Number of surface tesserae (CPCM cavity)
    m = re.search(r"Number of surface charges\s+\.+\s+(\d+)", text)
    if m:
        info["n_tesserae"] = int(m.group(1))

    # Cavity surface area
    m = re.search(r"Cavity surface area\s+\.+\s+(\d+\.\d+)\s+ang", text)
    if m:
        info["surface_area_ang2"] = float(m.group(1))

    return info


def _run_vibeqc() -> tuple[float, dict]:
    """Run the same calculation in vibe-qc."""
    import numpy as np
    import vibeqc as vq

    atoms = [
        vq.Atom(6, [1.8835100000, -0.1108300000, 0.0000000000]),
        vq.Atom(1, [1.4172800000, -1.0972200000, -0.0000000000]),
        vq.Atom(1, [1.4170500000, 0.4559200000, 0.8847900000]),
        vq.Atom(1, [1.4170500000, 0.4559200000, -0.8847900000]),
        vq.Atom(6, [3.4035100000, -0.1108300000, 0.0000000000]),
        vq.Atom(8, [3.9980700000, -1.1745400000, -0.0000000000]),
        vq.Atom(8, [3.9980700000, 1.0762700000, 0.0000000000]),
        vq.Atom(1, [4.9597100000, 0.9390400000, -0.0000000000]),
    ]
    mol = vq.Molecule(atoms, 0, 1)
    basis = vq.BasisSet(mol, "def2-tzvp")

    sm = vq.SolventModel(epsilon=78.39, name="water",
                         n_points_per_sphere=302)

    t0 = time.perf_counter()
    sol = vq.run_cpcm_scf(mol, basis, method="rks",
                          solvent=sm,
                          options=vq.RKSOptions())
    dt = time.perf_counter() - t0

    A2 = 1.8897261339213  # bohr per Å
    surface_area_ang2 = float(sol.cavity.total_surface_area_bohr2) / (A2 * A2)

    return dt, {
        "e_total": sol.energy,
        "e_solv_eh": sol.e_solv,
        "n_scf_iters": sol.scf.n_iter,
        "n_macro_iter": sol.n_macro_iter,
        "n_tesserae": sol.cavity.n_points,
        "surface_area_ang2": surface_area_ang2,
    }


def main() -> None:
    print("BUG 100 — ORCA 6.1.1 vs vibe-qc CPCM comparison")
    print("Acetic acid, B3LYP/def2-TZVP, CPCM water (ε=78.39)")
    print()

    # vibe-qc
    print("Running vibe-qc ...")
    vq_wall, vq_info = _run_vibeqc()
    print(f"  vibe-qc: {vq_wall:.1f}s")

    # ORCA
    print("Running ORCA ...")
    try:
        orca_wall, orca_text = _run_orca(ORCA_INPUT)
        orca_info = _parse_orca_output(orca_text)
        print(f"  ORCA:    {orca_wall:.1f}s")
    except RuntimeError as exc:
        print(f"  ORCA:    SKIPPED ({exc})")
        orca_wall = None
        orca_info = {}

    print()
    print(f"{'':30s}  {'vibe-qc':>12s}  {'ORCA':>12s}")
    print("-" * 58)

    rows = [
        ("Wall time (s)", f"{vq_wall:.1f}", f"{orca_wall:.1f}" if orca_wall else "N/A"),
        ("E_total (Ha)", f"{vq_info.get('e_total', 0):.8f}",
         f"{orca_info.get('e_total', 0):.8f}" if orca_info else "N/A"),
        ("E_solv (kcal/mol)",
         f"{vq_info.get('e_solv_eh', 0) * 627.509:.2f}",
         f"{orca_info.get('e_solv_eh', 0) * 627.509:.2f}" if orca_info else "N/A"),
        ("SCF iterations", str(vq_info.get("n_scf_iters", "?")),
         str(orca_info.get("n_scf_iters", "?"))),
        ("CPCM macro-iters", str(vq_info.get("n_macro_iter", "?")), "N/A"),
        ("Cavity points", str(vq_info.get("n_tesserae", "?")),
         str(orca_info.get("n_tesserae", "?"))),
        ("Surface area (Å²)", f"{vq_info.get('surface_area_ang2', 0):.1f}",
         f"{orca_info.get('surface_area_ang2', 0):.1f}" if orca_info else "N/A"),
    ]
    for label, vq_val, orca_val in rows:
        print(f"  {label:<28s}  {vq_val:>12s}  {orca_val:>12s}")


if __name__ == "__main__":
    main()
