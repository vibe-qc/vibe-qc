"""Re-run only the .py inputs whose <stem>.out is missing or doesn't
contain a parseable final energy. Skip the ones that already converged.

Useful after a partial orchestrator run (disk full, OOM, kill, etc.).
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

ENERGY_PATTERNS = [
    r"E_total\s*\(Ha\)\s*=\s*([-+]?\d+\.\d+)",
    r"converged SCF energy\s*=\s*([-+]?\d+\.\d+)",
    r"Final:\s*E\s*=\s*([-+]?\d+\.\d+)\s*Ha",
]


def has_energy(out_path: Path) -> bool:
    if not out_path.is_file():
        return False
    txt = out_path.read_text()
    for pat in ENERGY_PATTERNS:
        if re.search(pat, txt):
            return True
    return False


def discover_missing(parent: Path) -> list[Path]:
    if not parent.is_dir():
        return []
    found: list[Path] = []
    for sub in sorted(parent.iterdir()):
        if not sub.is_dir():
            continue
        for f in sorted(sub.glob("*.py")):
            if f.name.startswith("_"):
                continue
            if not has_energy(f.with_suffix(".out")):
                found.append(f)
    return found


def main():
    targets = (
        discover_missing(ROOT / "periodic") +
        discover_missing(ROOT / "periodic_pyscf")
    )
    if not targets:
        print("Nothing missing — every .out has a parseable energy.")
        return
    print(f"Re-running {len(targets)} missing input(s):")
    for p in targets:
        print(f"  - {p.parents[1].name}/{p.parent.name}/{p.name}")
    print()
    for p in targets:
        label = f"{p.parents[1].name}/{p.parent.name}/{p.stem}"
        print()
        print("=" * 76)
        print(f"  {label}")
        print("=" * 76)
        t0 = time.perf_counter()
        proc = subprocess.run(
            [PY, "-u", p.name], cwd=str(p.parent),
            capture_output=True, text=True,
        )
        wall = time.perf_counter() - t0
        if proc.stdout: print(proc.stdout, end="")
        if proc.stderr: print("STDERR:", proc.stderr, end="")
        rc_flag = "✓" if proc.returncode == 0 else "✗"
        print(f"  {rc_flag}  done in {wall:.1f}s  rc={proc.returncode}")


if __name__ == "__main__":
    main()
