"""xtb reference on the elongated 2x2x4 Cu cell (IID 130).

Does the xtb binary converge the same elongated cell that every vibe-qc
mixer fails? If xtb also fails, the wall is the GFN2 map for this cell,
not the mixer.


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


def fcc_primitive(a_angstrom: float):
    return [
        np.array([0.0, 0.5, 0.5]) * a_angstrom,
        np.array([0.5, 0.0, 0.5]) * a_angstrom,
        np.array([0.5, 0.5, 0.0]) * a_angstrom,
    ]


def main():
    a = 3.7958
    replicas = (2, 2, 4)
    prim = fcc_primitive(a)
    atoms = [
        i * prim[0] + j * prim[1] + k * prim[2]
        for i in range(replicas[0])
        for j in range(replicas[1])
        for k in range(replicas[2])
    ]
    lattice = np.column_stack([r * p for r, p in zip(replicas, prim)])
    n = len(atoms)
    lines = [
        str(n),
        "$lattice: " + " ".join(f"{v:.10f}" for row in lattice for v in row),
    ]
    for atom in atoms:
        lines.append(f"Cu {atom[0]:14.8f} {atom[1]:14.8f} {atom[2]:14.8f}")
    xyz = "\n".join(lines) + "\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "input.xyz"
        path.write_text(xyz, encoding="utf-8")
        proc = subprocess.run(
            [_find_xtb(), "input.xyz", "--gfn", "2", "--acc", "1.0",
             "--parallel", "1"],
            cwd=tmp, capture_output=True, text=True, timeout=3600,
        )
        for line in proc.stdout.splitlines():
            if "TOTAL ENERGY" in line or "SCC iter" in line or "convergence" in line.lower():
                print(line.strip())
        tail = proc.stdout.strip().splitlines()
        print("last lines:")
        for line in tail[-6:]:
            print(" ", line.strip())


if __name__ == "__main__":
    sys.exit(main())
