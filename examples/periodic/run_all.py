"""Run every periodic example sequentially.

Each example lives in ``examples/periodic/<stem>/<stem>.py`` (the
script's name = its parent directory's name = the shared filename
stem; outputs are written as ``<stem>.{out,system,molden,xsf}``
alongside the script). The naming convention is
``Formula-CrystalLattice-Method-Basis[-options]``, e.g.
``MgO-rocksalt-RHF-sto3g``.

This script discovers every such pair under ``examples/periodic/``
and ``examples/periodic_pyscf/``, runs each in its own directory,
parses the final energy out of the ``.out``, and prints a one-line
summary plus a vibe-qc vs PySCF parity table.

Run::
    .venv/bin/python examples/periodic/run_all.py
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]    # examples/
PY = sys.executable


def discover(parent: Path) -> list[Path]:
    """List every .py file under each subdirectory of `parent`,
    excluding helpers prefixed with ``_`` (e.g., ``_systems.py``,
    ``_gen.py``)."""
    if not parent.is_dir():
        return []
    found: list[Path] = []
    for sub in sorted(parent.iterdir()):
        if not sub.is_dir():
            continue
        for f in sorted(sub.glob("*.py")):
            if f.name.startswith("_"):
                continue
            found.append(f)
    return found


def run_one(input_path: Path) -> dict:
    """Run a single <stem>.py with cwd = its directory."""
    stem = input_path.stem
    parent_label = input_path.parents[1].name   # 'periodic' or 'periodic_pyscf'
    print()
    print("=" * 74)
    print(f"  {parent_label}/{stem}")
    print("=" * 74)
    t0 = time.perf_counter()
    proc = subprocess.run(
        [PY, "-u", input_path.name],
        cwd=str(input_path.parent),
        capture_output=True,
        text=True,
    )
    wall = time.perf_counter() - t0
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print("STDERR:", proc.stderr, end="")
    energy = _extract_energy(input_path.with_suffix(".out"))
    return {
        "label": f"{parent_label}/{stem}",
        "stem": stem,
        "kind": parent_label,
        "energy": energy,
        "wall": wall,
        "rc": proc.returncode,
    }


def _extract_energy(out_path: Path) -> float | None:
    """Parse <stem>.out for the final total energy. Recognizes both
    vibe-qc periodic_runner and PySCF .out conventions."""
    if not out_path.is_file():
        return None
    text = out_path.read_text()
    # vibe-qc: "    E_total (Ha)  =       -1085.2315464583"
    m = re.search(r"E_total\s*\(Ha\)\s*=\s*([-+]?\d+\.\d+)", text)
    if m:
        return float(m.group(1))
    # PySCF: "    converged SCF energy = -1085.23154645828"
    m = re.search(r"converged SCF energy\s*=\s*([-+]?\d+\.\d+)", text)
    if m:
        return float(m.group(1))
    # PySCF stdout fallback: "Final: E = -1085.2315464583 Ha"
    m = re.search(r"Final:\s*E\s*=\s*([-+]?\d+\.\d+)\s*Ha", text)
    if m:
        return float(m.group(1))
    return None


def main():
    targets = (
        discover(ROOT / "periodic") +
        discover(ROOT / "periodic_pyscf")
    )
    if not targets:
        print("No examples found under examples/periodic or examples/periodic_pyscf")
        return
    rows = [run_one(p) for p in targets]

    print()
    print("=" * 74)
    print("  Summary (energies in Ha; wall in s)")
    print("=" * 74)
    for r in rows:
        e = (f"{r['energy']:14.6f}" if r["energy"] is not None
             else "    n/a       ")
        flag = "✓" if r["rc"] == 0 else "✗"
        print(f"  {flag}  {r['label']:<54s}  E = {e}  wall = {r['wall']:7.1f}")

    # Pair vibe-qc vs pyscf rows by stem and print ΔE.
    by_stem: dict[str, dict[str, float | None]] = {}
    for r in rows:
        by_stem.setdefault(r["stem"], {})[r["kind"]] = r["energy"]
    print()
    print("  Parity (vibe-qc − PySCF, |ΔE| < 1 mHa = ✓):")
    print(f"    {'system / method / basis':<48s}  {'ΔE (Ha)':>12s}  verdict")
    for stem, kinds in sorted(by_stem.items()):
        ev = kinds.get("periodic")
        ep = kinds.get("periodic_pyscf")
        if ev is None or ep is None:
            continue
        de = ev - ep
        verdict = ("✓ PARITY" if abs(de) < 1e-3
                   else "~ off by mHa-Ha" if abs(de) < 1.0
                   else "✗ BUG")
        print(f"    {stem:<48s}  {de:>+12.4e}  {verdict}")


if __name__ == "__main__":
    main()
