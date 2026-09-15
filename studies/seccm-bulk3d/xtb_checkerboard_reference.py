"""xtb Gamma-only periodic GFN2 reference for B1 rocksalt(100).

Rule-2-sanctioned: xtb runs as a separate process (examples/regression
pattern); nothing imports it. Emits an xtb periodic XYZ with a vacuum box
in z and parses the total energy, the gap, and the Mulliken charges.

The former reference used an x-striped four-coordinate lattice rather
than rocksalt and its numerical anchors are withdrawn. This generator
uses the primitive B1(100) surface vectors and a neutral slab centred in
a 40 A vacuum box. The plane count is a command-line argument so the
4-plane defect-ladder cell (IID 215) and the 2-plane smoke share one
builder; out-of-process results are the only valid comparison targets.


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

A = 4.212


def _find_xtb() -> str:
    resolved = shutil.which("xtb")
    if resolved:
        return resolved
    fallback = Path.home() / "bin" / "xtb"
    if fallback.is_file():
        return str(fallback)
    raise RuntimeError("xtb binary not found on PATH or in ~/bin/xtb")


def checkerboard_positions(
    n_planes: int = 2, vacancy: bool = False
) -> tuple[
    list[str], list[list[float]], list[list[float]]
]:
    """Neutral B1 rocksalt(100) slab, centred in a 40 A vacuum cell.
    `vacancy` removes the surface O at (a/2, 0, 0) (the F0 centre of the
    defect ladder)."""
    t1 = np.array([A / 2.0, A / 2.0, 0.0])
    t2 = np.array([-A / 2.0, A / 2.0, 0.0])
    unlike_offset = np.array([A / 2.0, 0.0, 0.0])
    atoms: list[np.ndarray] = []
    zs: list[str] = []
    for k in range(n_planes):
        z_offset = np.array([0.0, 0.0, k * A / 2.0])
        for i in range(2):
            for j in range(2):
                home = i * t1 + j * t2 + z_offset
                if k % 2 == 0:
                    atoms.extend((home, home + unlike_offset))
                    zs.extend(("Mg", "O"))
                else:
                    atoms.extend((home + unlike_offset, home))
                    zs.extend(("Mg", "O"))
    if vacancy:
        target = np.array([A / 2.0, 0.0, 0.0])
        for idx, (coord, symbol) in enumerate(zip(atoms, zs)):
            if symbol == "O" and np.linalg.norm(coord - target) < 1.0e-6:
                del atoms[idx]
                del zs[idx]
                break
        else:
            raise RuntimeError("vacancy site not found")
    z_center = 20.0 - (A * (n_planes - 1) / 4.0)
    positions = [np.array([x, y, z + z_center]) for (x, y, z) in atoms]
    lattice = [(2.0 * t1).tolist(), (2.0 * t2).tolist(), [0.0, 0.0, 40.0]]
    return zs, [p.tolist() for p in positions], lattice


def run_xtb_reference(n_planes: int = 2, vacancy: bool = False) -> None:
    symbols, positions, lattice = checkerboard_positions(n_planes, vacancy)
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
                    if word == "Eh" and i + 1 < len(parts):
                        try:
                            gap = float(parts[i + 1])
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
            if len(tokens) >= 5 and tokens[2] in ("Mg", "O"):
                try:
                    charges.append(float(tokens[4]))
                except ValueError:
                    pass
            if len(charges) >= n:
                break
        q = np.array(charges)
        label = f"xtb {n_planes}-plane" + (" vacancy" if vacancy else "")
        print(f"{label}: total energy {energy}  (per atom {energy / n:.6f})")
        print(f"{label}: HL gap {gap} eV")
        print(f"{label}: q rms "
              f"{float(np.sqrt((q ** 2).mean())):.4f}  "
              f"range=[{q.min():.3f}, {q.max():.3f}]")


if __name__ == "__main__":
    planes = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    vacancy = "--vacancy" in sys.argv[2:]
    run_xtb_reference(planes, vacancy)
