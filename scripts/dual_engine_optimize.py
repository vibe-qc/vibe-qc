#!/usr/bin/env python3
"""Dual-engine basis-set optimization: CRYSTAL14 + vibe-qc BIPOLE.

Runs pob-TZVP / HF single-point energies on CaO, LiF, MgO (rocksalt)
through both SCF engines, then reports side-by-side results. This is
the first step toward replacing CRYSTAL14 with vibe-qc BIPOLE as the
SCF engine inside the basis-set optimizer.

Usage (on compute-host-a):
    PYTHONUNBUFFERED=1 ~/gitlab/vibeqc-dev/.venv/bin/python dual_engine_optimize.py

Requirements:
    - vibe-qc installed (with BIPOLE driver)
    - CRYSTAL14 binary on PATH (crystal)
    - run-crystal.sh in ~/bin/
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---- Structures (rocksalt, SG 225, 2-atom primitive) ----
# All geometries from the PT2013 T4 test set.

ANG2BOHR = 1.0 / 0.529177210903

SYSTEMS = {
    "MgO": {
        "a_ang": 4.189,  # from the user's MgO_seg_PW1PW.d12
        "z1": 12,
        "sym1": "Mg",
        "z2": 8,
        "sym2": "O",
    },
}


def fcc_primitive(a_bohr: float) -> np.ndarray:
    return (a_bohr / 2.0) * np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )


# =========================================================================
# CRYSTAL14 engine
# =========================================================================


def run_crystal14(name: str, workdir: Path) -> Optional[float]:
    """Run CRYSTAL14 on a .d12 deck in *workdir*.

    Expects ``{name}.d12`` in workdir. Writes output to
    ``{name}.out``. Returns total energy in Hartree or None.
    """
    d12 = workdir / f"{name}.d12"
    if not d12.exists():
        print(f"  CRYSTAL14: missing {d12}")
        return None

    out = workdir / f"{name}.out"
    # vq jobs don't have ~/bin on PATH; use full paths.
    crystal_bin = Path.home() / "bin" / "crystal"
    d12_abs = d12.resolve()
    out_abs = out.resolve()
    cmd = ["bash", "-c", f"'{crystal_bin}' < '{d12_abs}' > '{out_abs}'"]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=3600,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(f"  CRYSTAL14: TIMEOUT on {name}")
        return None
    dt = time.perf_counter() - t0

    if not out.exists():
        print(f"  CRYSTAL14: no output for {name}")
        print(f"  CRYSTAL14: stderr: {proc.stderr[:500]}")
        return None

    text = out.read_text()
    energy = _parse_crystal_energy(text)
    if energy is None:
        print(f"  CRYSTAL14 {name}: FAILED to parse energy  ({dt:.0f}s)")
        # Save output snippet for debugging.
        (workdir / f"{name}_debug.txt").write_text(
            f"=== STDOUT (last 2000 chars) ===\n{text[-2000:]}\n"
            f"=== STDERR ===\n{proc.stderr[-500:]}"
        )
    else:
        print(f"  CRYSTAL14 {name}: E = {energy:.8f} Ha  ({dt:.0f}s)")
    return energy


def _parse_crystal_energy(text: str) -> Optional[float]:
    """Extract final total energy from CRYSTAL14 output."""
    # Look for "TOTAL ENERGY" line, last occurrence.
    best: Optional[float] = None
    for line in text.splitlines():
        if "TOTAL ENERGY" in line.upper():
            parts = line.split()
            for p in parts:
                try:
                    best = float(p)
                except ValueError:
                    continue
    return best


def emit_crystal_d12(
    name: str,
    a_ang: float,
    Z1: int,
    Z2: int,
    basis_text: str,
    method: str = "RHF",
) -> str:
    """Emit a CRYSTAL14 .d12 deck for a rocksalt primitive cell.

    Uses the pob-TZVP basis in CRYSTAL inline format (passed as
    *basis_text*). SHRINK 8 8, TOLINTEG 9 9 9 18 54.

    CRYSTAL14 .d12 uses fractional coordinates and the conventional
    cubic lattice constant (in Å), with space group 225 for rocksalt.
    """
    # In .d12, lattice parameter is the conventional cubic edge in Å.
    # Positions are fractional: Mg at (0,0,0), O at (0.5, 0.5, 0.5).
    #
    # CRYSTAL14 public 1.0.2 rejects "99 0" in inline basis within
    # a .d12 deck; it's a .basis-file-only marker. Strip it.
    basis_clean = basis_text.replace(" 99 0\n", "").replace(" 99 0", "").rstrip()
    deck = f"""{name} rocksalt — pob-TZVP {method}
CRYSTAL
   0 0 0
 225
   {a_ang:.6f}
   2
   {Z1}  0.0 0.0 0.0
   {Z2}  0.5 0.5 0.5
OPTGEOM
ENDOPT
ENDGEOM
{basis_clean}
ENDBS
{method}
TOLINTEG
 9 9 9 18 54
SHRINK
 8 8
END
"""
    return deck


# =========================================================================
# vibe-qc BIPOLE engine
# =========================================================================


def run_bipole(name: str, a_ang: float, Z1: int, Z2: int) -> BipoleOutcome:
    """Run vibe-qc BIPOLE RHF / pob-TZVP on a rocksalt primitive cell.

    Returns the outcome (energy, SCF verdict, iteration count). The
    energy is the final SCF energy even when the run did not converge,
    for diagnosis; callers must consult ``converged`` before treating
    it as comparable.
    """
    import vibeqc as vq
    from vibeqc.pbc_bipole import run_pbc_bipole_rhf

    a = a_ang * ANG2BOHR
    lattice = fcc_primitive(a)
    pos2 = a / 2.0
    atoms = [
        vq.Atom(Z1, [0.0, 0.0, 0.0]),
        vq.Atom(Z2, [pos2, pos2, pos2]),
    ]
    sysp = vq.PeriodicSystem(3, lattice, atoms)

    basis = vq.BasisSet(sysp.unit_cell_molecule(), "pob-tzvp")
    print(f"  BIPOLE {name}: {basis.nbasis} BFs/cell")

    kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])
    print(f"  BIPOLE {name}: {basis.nbasis} BFs/cell, Gamma-only")

    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 10.0
    opts.lattice_opts.nuclear_cutoff_bohr = 10.0
    opts.max_iter = 30
    opts.use_diis = True
    opts.damping = 0.0
    opts.conv_tol_energy = 1e-7
    opts.initial_guess = vq.InitialGuess.SAD

    t0 = time.perf_counter()
    try:
        result = run_pbc_bipole_rhf(
            sysp,
            basis,
            kmesh,
            opts,
            use_ewald_j_split=True,
            ewald_precision=1e-8,
        )
    except Exception as exc:
        print(f"  BIPOLE {name}: EXCEPTION: {exc}")
        return BipoleOutcome(energy=None, converged=False, n_iter=0)
    dt = time.perf_counter() - t0

    if not result.converged:
        print(f"  BIPOLE {name}: NOT CONVERGED after {result.n_iter} iters")
        # Return last energy anyway for diagnosis.
    print(
        f"  BIPOLE {name}: E = {result.energy:.8f} Ha  "
        f"({result.n_iter} iters, {dt:.0f}s)  converged={result.converged}"
    )
    return BipoleOutcome(
        energy=float(result.energy),
        converged=bool(result.converged),
        n_iter=int(result.n_iter),
    )


# =========================================================================
# pob-TZVP basis in CRYSTAL inline format (for CRYSTAL14)
# =========================================================================


def _find_pob_source(Z: int, basis: str = "pob-TZVP") -> Optional[Path]:
    """Locate a pob per-element CRYSTAL source file."""
    import vibeqc as vq

    pkg_root = Path(vq.__file__).parent
    src_dir = pkg_root / "basis_library" / "sources" / basis
    if not src_dir.exists():
        return None
    candidates = sorted(src_dir.glob(f"{Z:02d}_*"))
    return candidates[0] if candidates else None


def get_pob_tzvp_inline_crystal() -> str:
    """Return pob-TZVP as CRYSTAL inline basis text.

    Reads from the bundled per-element source files and emits in the
    format CRYSTAL14 expects (0-based LAT, per-element blocks with 99 0
    only at the very end).
    """
    from vibeqc.basis_crystal import emit_crystal, parse_crystal_atom_basis_file

    elements = {
        1: "H",
        3: "Li",
        8: "O",
        9: "F",
        11: "Na",
        12: "Mg",
        17: "Cl",
        19: "K",
        20: "Ca",
    }
    atoms = []
    for Z, _ in sorted(elements.items()):
        src = _find_pob_source(Z, "pob-TZVP")
        if src is not None:
            atoms.append(parse_crystal_atom_basis_file(src))
    return emit_crystal(atoms)


# =========================================================================
# Main
# =========================================================================


@dataclass
class BipoleOutcome:
    """Outcome of one vibe-qc BIPOLE run.

    Carries the SCF verdict alongside the energy so callers can
    serialize both instead of only the energy.
    """

    energy: Optional[float]
    converged: bool
    n_iter: int


@dataclass
class SystemResult:
    name: str
    crystal_energy: Optional[float] = None
    crystal_wall_s: float = 0.0
    bipole_energy: Optional[float] = None
    bipole_wall_s: float = 0.0
    bipole_converged: bool = False
    bipole_n_iter: int = 0
    delta_mha: Optional[float] = None
    notes: list[str] = field(default_factory=list)


def main():
    # Unbuffered output for vq job monitoring.
    import sys

    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    print("=" * 70)
    print("Dual-engine basis-set optimization — pob-TZVP / HF")
    print("Compounds: CaO, LiF, MgO (rocksalt, SG 225)")
    print("=" * 70)

    # Generate CRYSTAL inline basis text once.
    print("\n[0] Generating pob-TZVP CRYSTAL inline basis...")
    crystal_basis = get_pob_tzvp_inline_crystal()
    print(f"     {len(crystal_basis)} characters")

    results: dict[str, SystemResult] = {}

    for name, cfg in SYSTEMS.items():
        print(f"\n{'─' * 70}")
        print(f"[{name}] a = {cfg['a_ang']} Å  ({cfg['sym1']} + {cfg['sym2']})")
        print(f"{'─' * 70}")

        res = SystemResult(name=name)

        # --- CRYSTAL14 ---
        print(f"\n  >> CRYSTAL14 pob-TZVP / RHF")
        wd = Path(f"crystal_{name}")
        wd.mkdir(parents=True, exist_ok=True)
        deck = emit_crystal_d12(
            name,
            cfg["a_ang"],
            cfg["z1"],
            cfg["z2"],
            crystal_basis,
            method="RHF",
        )
        (wd / f"{name}.d12").write_text(deck)
        t0 = time.perf_counter()
        res.crystal_energy = run_crystal14(name, wd)
        res.crystal_wall_s = time.perf_counter() - t0

        # --- vibe-qc BIPOLE ---
        print(f"\n  >> vibe-qc BIPOLE pob-TZVP / RHF")
        t0 = time.perf_counter()
        outcome = run_bipole(
            name,
            cfg["a_ang"],
            cfg["z1"],
            cfg["z2"],
        )
        res.bipole_energy = outcome.energy
        res.bipole_converged = outcome.converged
        res.bipole_n_iter = outcome.n_iter
        res.bipole_wall_s = time.perf_counter() - t0

        # delta_mha compares two engines and is only meaningful when
        # BIPOLE actually converged: a non-converged cycle energy would
        # pollute the comparison, so it is withheld and noted instead.
        if res.crystal_energy is not None and res.bipole_energy is not None:
            if res.bipole_converged:
                res.delta_mha = (res.bipole_energy - res.crystal_energy) * 1000.0
            else:
                res.notes.append("BIPOLE did not converge; delta_mha withheld")

        results[name] = res

    # ---- Summary ----
    print(f"\n\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(
        f"{'System':<6} {'CRYSTAL14 (Ha)':>18} {'BIPOLE (Ha)':>18} "
        f"{'Δ (mHa)':>10} {'BIPOLE τ':>8}"
    )
    print(f"{'─' * 6} {'─' * 18} {'─' * 18} {'─' * 10} {'─' * 8}")
    for name, r in results.items():
        cry = f"{r.crystal_energy:.8f}" if r.crystal_energy else "FAILED"
        bip = f"{r.bipole_energy:.8f}" if r.bipole_energy else "FAILED"
        delta = f"{r.delta_mha:+.3f}" if r.delta_mha is not None else "—"
        wall = f"{r.bipole_wall_s:.0f}s" if r.bipole_wall_s else "—"
        print(f"{name:<6} {cry:>18} {bip:>18} {delta:>10} {wall:>8}")

    # Write JSON for later reference.
    out_path = Path("dual_engine_results.json")
    out_data = {}
    for name, r in results.items():
        out_data[name] = {
            "crystal_energy_ha": r.crystal_energy,
            "crystal_wall_s": r.crystal_wall_s,
            "bipole_energy_ha": r.bipole_energy,
            "bipole_wall_s": r.bipole_wall_s,
            "bipole_converged": r.bipole_converged,
            "bipole_n_iter": r.bipole_n_iter,
            "delta_mha": r.delta_mha,
            "notes": r.notes,
        }
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"\nResults written to {out_path.resolve()}")

    return results


if __name__ == "__main__":
    main()
