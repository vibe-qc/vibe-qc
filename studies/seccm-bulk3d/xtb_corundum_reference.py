"""xtb Gamma-only periodic GFN2 reference for corundum (Al2O3).

Rule-2-sanctioned: xtb runs as a separate process (examples/regression
pattern); nothing imports it. Emits an xtb periodic XYZ for the 30-atom
corundum primitive cell (R-3c, spacegroup 167) and parses the total
energy and Mulliken charges.

Purpose: the GFN2-SECCM corundum SCC surface is branch-discontinuous
(razor basin boundary near a = 4.759 A; see the SECCM audit handover).
This reference identifies which charge state is physical so the branch
question can be settled. Out-of-process results are the only valid
comparison targets.


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
import tempfile
from pathlib import Path

import numpy as np

A0 = 4.7589
C0 = 12.991


def _find_xtb() -> str:
    resolved = shutil.which("xtb")
    if resolved:
        return resolved
    fallback = Path.home() / "bin" / "xtb"
    if fallback.is_file():
        return str(fallback)
    raise RuntimeError("xtb binary not found on PATH or in ~/bin/xtb")


def corundum_positions(a_angstrom: float = A0, c_angstrom: float = C0):
    """30-atom corundum primitive cell (hexagonal), same builder as
    studies/seccm-bulk3d/scan_corundum.py (AL_Z 0.35216, O_X 0.30624)."""
    from ase.spacegroup import crystal as ase_crystal

    atoms_ase = ase_crystal(
        ["Al", "O"],
        basis=[(0, 0, 0.35216), (0.30624, 0, 0.25)],
        spacegroup=167,
        cellpar=[a_angstrom, a_angstrom, c_angstrom, 90, 90, 120],
    )
    cell_ang = atoms_ase.cell.array
    symbols = atoms_ase.get_chemical_symbols()
    positions = [p.tolist() for p in atoms_ase.get_positions()]
    return symbols, positions, [row.tolist() for row in cell_ang]


def run_xtb_reference(a_angstrom: float = A0, c_angstrom: float = C0) -> None:
    symbols, positions, lattice = corundum_positions(a_angstrom, c_angstrom)
    n = len(symbols)
    lines = [str(n), "$lattice: " + " ".join(
        f"{v:.10f}" for row in lattice for v in row)]
    for symbol, pos in zip(symbols, positions):
        lines.append(f"{symbol:2s} {pos[0]:14.8f} {pos[1]:14.8f} {pos[2]:14.8f}")
    xyz = "\n".join(lines) + "\n"

    cmd = [_find_xtb(), "input.xyz", "--gfn", "2", "--acc", "1.0", "--parallel", "1"]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "input.xyz"
        path.write_text(xyz, encoding="utf-8")
        proc = subprocess.run(
            cmd, cwd=tmp, capture_output=True, text=True, timeout=1800
        )
        out = proc.stdout
        energy = None
        gap = None
        for line in out.splitlines():
            s = line.strip()
            if "TOTAL ENERGY" in s:
                for token in s.replace("|", " ").split():
                    try:
                        energy = float(token)
                        break
                    except ValueError:
                        continue
            if "HL-Gap" in s:
                parts = s.split()
                for i, word in enumerate(parts):
                    if word == "eV" and i > 0:
                        try:
                            gap = float(parts[i - 1])
                        except ValueError:
                            pass
        charges: list[float] = []
        in_table = False
        for line in out.splitlines():
            if "covCN" in line:
                in_table = True
                continue
            if not in_table:
                continue
            tokens = line.split()
            if len(tokens) >= 5 and tokens[2] in ("Al", "O"):
                try:
                    charges.append(float(tokens[4]))
                except ValueError:
                    pass
            if len(charges) >= n:
                break
        q = np.array(charges)
        print(f"xtb corundum a={a_angstrom} c={c_angstrom}: total energy "
              f"{energy}  (per atom {energy / n:.6f})")
        print(f"xtb corundum: HL gap {gap} eV")
        print(f"xtb corundum: q rms "
              f"{float(np.sqrt((q ** 2).mean())):.4f}  "
              f"range=[{q.min():.3f}, {q.max():.3f}]")


if __name__ == "__main__":
    run_xtb_reference()
