#!/usr/bin/env python
"""CRYSTAL23 limit of the real-Gamma ladder cells (RG-13, GitLab #519).

At a finite mesh CRYSTAL's exact exchange is not the torus exchange of the
real-Gamma producer: CRYSTAL sums the exchange series in real space over the
whole lattice with a density Fourier-interpolated from the k-mesh, while the
real-Gamma producer, and PySCF with ``exxdiv='ewald'``, evaluate the
Born-von Karman torus with the Madelung-corrected ``q = 0`` term. The two
agree only in the thermodynamic limit. This script therefore runs CRYSTAL at
converged meshes, and ``ladder_pyscf.py`` supplies the rungs whose limit it is
compared with; the same-mesh comparison is the PySCF one.

The basis is vibe-qc's own: its shells are written into the input with
coefficients for normalised primitives, as ``ladder_pyscf.py`` transports
them, and CRYSTAL's atomic guess fills them by aufbau. The integral tolerances
are tighter than CRYSTAL's defaults (``TOLINTEG 8 8 8 8 16``).

CRYSTAL runs out of process; nothing is imported from it.

Usage
-----
    python ladder_crystal.py [--cases c-diamond lih-rocksalt mgo]
                             [--shrink 8 12] [--crystal Pcrystal --mpi 6]
                             [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ladder_pyscf import CASES, _runtime_identity, _transported_bases  # noqa: E402

BOHR_TO_ANGSTROM = 0.529177210903
TOLINTEG = "8 8 8 8 16"
TOLDEE = 10
LATVEC = 50000

# Space-group input of each cell; it must describe the same primitive cell as
# the case's builder, which the volume check below enforces.
GEOMETRY = {
    "c-diamond": ("227", "3.5670", [(6, (0.125, 0.125, 0.125))]),
    "lih-rocksalt": ("225", "4.0840", [(3, (0.0, 0.0, 0.0)), (1, (0.5, 0.5, 0.5))]),
    "mgo": ("225", "4.22389871", [(12, (0.0, 0.0, 0.0)), (8, (0.5, 0.5, 0.5))]),
    "be-fcc": ("225", "3.20", [(4, (0.0, 0.0, 0.0))]),
}

_SCF_ENDED = re.compile(
    r"SCF ENDED - CONVERGENCE ON ENERGY\s+E\(AU\)\s+(\S+)\s+CYCLES\s+(\d+)")
_VOLUME = re.compile(r"PRIMITIVE CELL.*VOLUME=\s*([0-9.]+)")


def _crystal_basis(z: int, shells: list) -> list:
    """One atom's basis block, the shells filled by aufbau for the guess."""
    lat = {0: 0, 1: 2, 2: 3, 3: 4}
    left = float(z)
    lines = [f"{z} {len(shells)}"]
    for shell in shells:
        l, prims = shell[0], shell[1:]
        che = min(left, 2.0 * (2 * l + 1))
        left -= che
        lines.append(f"0 {lat[l]} {len(prims)} {che:.1f} 1.0")
        lines.extend(f"{a:.10e} {c:.10e}" for a, c in prims)
    if left > 0:
        raise ValueError(f"Z={z}: the basis holds fewer than Z electrons")
    return lines


def crystal_input(name: str, shrink: int) -> str:
    sg, a, atoms = GEOMETRY[name]
    case = CASES[name]
    ao_by_z, _, _ = _transported_bases(case["build"](), case["basis"])
    lines = [f"{name} RHF/{case['basis']} (vibe-qc basis transported)", "CRYSTAL",
             "0 0 0", sg, a, str(len(atoms))]
    lines += [f"{z} {x:.6f} {y:.6f} {w:.6f}" for z, (x, y, w) in atoms]
    # The tight tolerances reach further than the default lattice-vector
    # table on cells with diffuse shells (Li 2sp, exponent 0.048).
    lines += ["LATVEC", str(LATVEC), "END"]
    for z in sorted(ao_by_z):
        lines += _crystal_basis(z, ao_by_z[z])
    lines += ["99 0", "END", "SHRINK", f"{shrink} {shrink}", "TOLINTEG", TOLINTEG,
              "TOLDEE", str(TOLDEE), "MAXCYCLE", "200", "END"]
    return "\n".join(lines) + "\n"


def run_crystal(name: str, shrink: int, crystal: str, mpi: int = 0) -> dict:
    text = crystal_input(name, shrink)
    command = [crystal] if mpi <= 1 else ["mpirun", "-n", str(mpi), crystal]
    with tempfile.TemporaryDirectory() as tmp:
        # The serial binary reads the deck on stdin, Pcrystal reads INPUT.
        deck = Path(tmp) / "INPUT"
        deck.write_text(text)
        t0 = time.perf_counter()
        with deck.open() as stdin:
            proc = subprocess.run(command, stdin=stdin, capture_output=True, text=True,
                                  cwd=tmp, check=False)
        wall = time.perf_counter() - t0
    # Under mpirun the deck's report can arrive on either stream.
    out = proc.stdout + "\n" + proc.stderr
    ended = _SCF_ENDED.search(out)
    if ended is None:
        raise RuntimeError(f"CRYSTAL did not converge on {name} SHRINK {shrink}:\n"
                           f"{out[-3000:]}")
    found = _VOLUME.search(out)
    if found is None:
        raise RuntimeError(f"{name}: no primitive-cell volume in the CRYSTAL output")
    volume = float(found.group(1))
    cell = CASES[name]["build"]()
    ours = abs(np.linalg.det(np.asarray(cell.lattice, dtype=float))) * BOHR_TO_ANGSTROM ** 3
    if abs(volume - ours) > 1e-3 * ours:
        raise RuntimeError(f"{name}: CRYSTAL cell volume {volume} A^3 is not the "
                           f"case's {ours:.4f} A^3")
    return {"shrink": shrink, "energy": float(ended.group(1)),
            "cycles": int(ended.group(2)), "volume_A3": volume,
            "conducting_warning": "POSSIBLY CONDUCTING STATE" in out,
            "wall_s": wall}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", nargs="+", default=["c-diamond", "lih-rocksalt", "mgo"],
                    choices=list(GEOMETRY))
    ap.add_argument("--shrink", nargs="+", type=int, default=[8, 12])
    ap.add_argument("--crystal", default=shutil.which("crystal") or "crystal",
                    help="CRYSTAL23 executable; it reads the input on stdin")
    ap.add_argument("--mpi", type=int, default=0,
                    help="run the executable (Pcrystal) under mpirun with this many ranks")
    ap.add_argument("--out", type=Path, default=HERE / "results_ladder_crystal.json")
    args = ap.parse_args()
    results = {"runtime": _runtime_identity(), "tolinteg": TOLINTEG,
               "toldee": TOLDEE, "latvec": LATVEC, "systems": {}}
    for name in args.cases:
        rows = []
        for shrink in args.shrink:
            rows.append(run_crystal(name, shrink, args.crystal, args.mpi))
            print(f"{name} SHRINK {shrink}: {rows[-1]['energy']:.10f} Ha/cell "
                  f"({rows[-1]['cycles']} cycles, {rows[-1]['wall_s']:.0f} s)", flush=True)
        results["systems"][name] = {"basis": CASES[name]["basis"], "crystal": rows}
        args.out.write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
