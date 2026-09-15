"""Out-of-process xtb parity references for the bulk-3D SECCM ladder.

Rule-2-sanctioned: xtb runs as a separate process (examples/regression
pattern); nothing imports it. Emits xtb periodic XYZ ($lattice header)
and parses the total energy / gap.


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
from ase.spacegroup import crystal as ase_crystal


def _find_xtb() -> str:
    resolved = shutil.which("xtb")
    if resolved:
        return resolved
    fallback = Path.home() / "bin" / "xtb"
    if fallback.is_file():
        return str(fallback)
    raise RuntimeError("xtb binary not found on PATH or in ~/bin/xtb")

AL_Z = 0.35216
O_X = 0.30624


def fcc_cu_8(a: float) -> tuple[list[str], list[list[float]], list[list[float]]]:
    prim = [
        np.array([0.0, 0.5, 0.5]) * a,
        np.array([0.5, 0.0, 0.5]) * a,
        np.array([0.5, 0.5, 0.0]) * a,
    ]
    atoms = [
        (i * prim[0] + j * prim[1] + k * prim[2]).tolist()
        for i in range(2) for j in range(2) for k in range(2)
    ]
    symbols = ["Cu"] * 8
    return symbols, atoms, [2 * p for p in prim]


def rocksalt_16(a: float) -> tuple[list[str], list[list[float]], list[list[float]]]:
    prim = [
        np.array([0.0, 0.5, 0.5]) * a,
        np.array([0.5, 0.0, 0.5]) * a,
        np.array([0.5, 0.5, 0.0]) * a,
    ]
    basis = [np.zeros(3), np.array([a / 2.0, 0.0, 0.0])]
    atoms: list[list[float]] = []
    symbols: list[str] = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for site in range(2):
                    atoms.append(
                        (i * prim[0] + j * prim[1] + k * prim[2] + basis[site]).tolist()
                    )
                    symbols.append("Mg" if site == 0 else "O")
    return symbols, atoms, [2 * p for p in prim]


def corundum_30(a: float, c: float) -> tuple[list[str], list[list[float]], list[list[float]]]:
    atoms_ase = ase_crystal(
        ["Al", "O"],
        basis=[(0, 0, AL_Z), (O_X, 0, 0.25)],
        spacegroup=167,
        cellpar=[a, a, c, 90, 90, 120],
    )
    return (
        list(atoms_ase.get_chemical_symbols()),
        [p.tolist() for p in atoms_ase.get_positions()],
        [row.tolist() for row in atoms_ase.cell.array],
    )


def run_xtb(
    label: str,
    symbols: list[str],
    atoms: list[list[float]],
    lattice: list[list[float]],
    etemp: float | None = None,
) -> None:
    n = len(symbols)
    lines = [str(n), "$lattice: " + " ".join(f"{v:.10f}" for row in lattice for v in row)]
    for symbol, pos in zip(symbols, atoms):
        lines.append(f"{symbol:2s} {pos[0]:14.8f} {pos[1]:14.8f} {pos[2]:14.8f}")
    xyz = "\n".join(lines) + "\n"

    cmd = [_find_xtb(), "input.xyz", "--gfn", "2", "--acc", "1.0", "--parallel", "1"]
    if etemp is not None:
        cmd += ["--etemp", str(etemp)]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "input.xyz"
        path.write_text(xyz, encoding="utf-8")
        try:
            proc = subprocess.run(
                cmd, cwd=tmp, capture_output=True, text=True, timeout=600
            )
        except subprocess.TimeoutExpired:
            print(f"{label}: TIMEOUT")
            return
        if "MgO" in label:
            dump = Path(__file__).resolve().parent / "logs" / "xtb_mgo_stdout.txt"
            dump.write_text(proc.stdout, encoding="utf-8")
        energy = None
        gap = None
        charges: list[float] = []
        in_charges = False
        for line in proc.stdout.splitlines():
            s = line.strip()
            if "total energy" in s.lower():
                for part in s.split(":"):
                    for token in part.strip().split():
                        try:
                            energy = float(token)
                            break
                        except ValueError:
                            continue
                    if energy is not None:
                        break
            if "HOMO-LUMO gap" in s or "HOMO-LUMO GAP" in s:
                parts = s.split()
                for i, w in enumerate(parts):
                    if w.lower() == "gap" and i + 1 < len(parts):
                        try:
                            gap = float(parts[i + 1])
                        except ValueError:
                            pass
            if "covCN" in s or "#" in s and "Z" in s:
                in_charges = False
            if in_charges:
                tokens = s.split()
                if len(tokens) == 3:
                    try:
                        charges.append(float(tokens[2]))
                    except ValueError:
                        pass
            if s.startswith("#") and "Z" in s:
                in_charges = True
        if energy is None:
            tail = "\n".join(proc.stdout.splitlines()[-15:])
            print(f"{label}: FAILED rc={proc.returncode}\n{tail}")
            return
        per_atom = energy / n
        q_rms = 0.0
        if charges:
            q_rms = float(np.sqrt(np.mean(np.array(charges) ** 2)))
        print(f"{label}: E_tot={energy:14.6f} E/atom={per_atom:12.6f} "
              f"gap={gap if gap is not None else float('nan'):6.2f} eV "
              f"q_rms={q_rms:6.3f} q_min={min(charges) if charges else float('nan'):+.3f} "
              f"q_max={max(charges) if charges else float('nan'):+.3f}")


def main() -> None:
    run_xtb("Cu 2x2x2 a=3.615", *fcc_cu_8(3.615), etemp=300.0)
    run_xtb("MgO 2x2x2 a=4.212", *rocksalt_16(4.212))
    run_xtb("Al2O3 hex a=4.7589 c=12.991", *corundum_30(4.7589, 12.991))


if __name__ == "__main__":
    main()
