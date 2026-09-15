"""Out-of-process xtb GFN2 reference for the fcc Cu 2x2x2 ladder (IID 130).

Writes coord files, runs the xtb binary per lattice constant, and prints
the per-atom GFN2 total energy. xtb only via subprocess (rule 2).


CORRECTION (2026-08-26): the "$lattice:" xyz comment line is
IGNORED by xtb (verified: bit-identical energy without it), and
xtb 6.7.0 aborts on genuine periodic GFN2 input ("Multipoles not
available with PBC"). Energies produced here are FREE MOLECULAR
CLUSTER references, not periodic ones. See
handovers/HANDOVER_SECCM_BULK3D.md, evidence-correction section.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def _find_xtb() -> str:
    resolved = shutil.which("xtb")
    if resolved:
        return resolved
    fallback = Path.home() / "bin" / "xtb"
    if fallback.is_file():
        return str(fallback)
    raise RuntimeError("xtb binary not found on PATH or in ~/bin/xtb")


XT_BIN = _find_xtb()

A_LIST = [3.4, 3.434, 3.5, 3.615, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.4]


def fcc_primitive(a_angstrom: float):
    return [
        np.array([0.0, 0.5, 0.5]) * a_angstrom,
        np.array([0.5, 0.0, 0.5]) * a_angstrom,
        np.array([0.5, 0.5, 0.0]) * a_angstrom,
    ]


def coord_file(a_angstrom: float, path: Path) -> None:
    prim = fcc_primitive(a_angstrom)
    lattice = np.column_stack([2 * p for p in prim])
    atoms = [
        i * prim[0] + j * prim[1] + k * prim[2]
        for i in range(2)
        for j in range(2)
        for k in range(2)
    ]
    n = len(atoms)
    lines = [
        str(n),
        "$lattice: " + " ".join(f"{v:.10f}" for row in lattice for v in row),
    ]
    for atom in atoms:
        lines.append(f"Cu {atom[0]:14.8f} {atom[1]:14.8f} {atom[2]:14.8f}")
    path.write_text("\n".join(lines) + "\n")


def run_xtb(coord: Path, workdir: Path) -> float:
    completed = subprocess.run(
        [XT_BIN, str(coord), "--gfn", "2", "--acc", "1.0", "--parallel", "1"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    stdout = completed.stdout
    for line in stdout.splitlines():
        if "TOTAL ENERGY" in line:
            for token in line.replace("|", " ").split():
                try:
                    return float(token)
                except ValueError:
                    continue
    print(stdout[-2000:], file=sys.stderr)
    raise RuntimeError("xtb total energy not found in output")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="xtb_cu_") as tmp:
        tmpdir = Path(tmp)
        for a in A_LIST:
            coord = tmpdir / f"cu_{a}.xyz"
            coord_file(a, coord)
            workdir = tmpdir / f"run_{a}"
            workdir.mkdir()
            total = run_xtb(coord, workdir)
            print(f"a={a:7.4f} E/atom={total / 8:+.8f} (total {total:+.8f})", flush=True)


if __name__ == "__main__":
    sys.exit(main())
