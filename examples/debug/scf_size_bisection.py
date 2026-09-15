"""Periodic-RKS size bisection — find the smallest system that exhibits a bug.

Runs the same RKS-LDA / sto-3g / EWALD_3D / Γ-only SCF setup at
five increasing system sizes:

    1. He cubic            — 1 atom,  ~1 bf,  ~30 s
    2. Mg cubic            — 1 atom,  ~5 bf,  ~30 s
    3. NaCl primitive FCC  — 2 atoms, 9 bf,  XFAILS (non-orthorhombic
                              FCC primitive — EWALD_3D requires
                              orthorhombic; included here as a
                              cross-check that the dispatcher's
                              error path fires correctly)
    4. NaCl conventional   — 8 atoms, 72 bf, ~1-3 min
    5. MgO conventional    — 8 atoms, 148 bf, ~3-10 min

For each system, reports:
    converged?  |  n_iter  |  E (Ha)  |  wall (s)  |  notes

Use case: "this bug shows up on NaCl-conv; does it also show on
He / Mg?" Smallest-reproducing system → fastest iteration cycle
on the actual bug.

Wall: ~5-15 min total on a laptop with v0.5.5 Schwarz screening.
Run with FAST_DEBUG=1 to coarsen all knobs and finish in ~1-3 min:

    VIBEQC_FAST_DEBUG=1 \\
        .venv/bin/python examples/debug/scf_size_bisection.py
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

import vibeqc as vq

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output" / "size-bisection"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = OUT_DIR / "summary.csv"

_FAST_DEBUG = os.environ.get(
    "VIBEQC_FAST_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")

if _FAST_DEBUG:
    SPACING, CUTOFF, NUC_CUT, OMEGA, CONV, MAX_IT = 0.8, 8.0, 15.0, 0.7, 1e-5, 12
else:
    SPACING, CUTOFF, NUC_CUT, OMEGA, CONV, MAX_IT = 0.5, 10.0, 18.0, 0.5, 1e-6, 25


def _mk_opts():
    o = vq.PeriodicKSOptions()
    o.functional = "LDA"
    o.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    o.lattice_opts.cutoff_bohr = CUTOFF
    o.lattice_opts.nuclear_cutoff_bohr = NUC_CUT
    o.conv_tol_energy = CONV
    o.max_iter = MAX_IT
    return o


def he_cubic() -> tuple[vq.PeriodicSystem, str]:
    a = 3.0 / 0.529177210903
    sys_ = vq.PeriodicSystem(3, a * np.eye(3), [vq.Atom(2, [0.0, 0.0, 0.0])])
    return sys_, "sto-3g"


def mg_cubic() -> tuple[vq.PeriodicSystem, str]:
    a = 3.5 / 0.529177210903
    sys_ = vq.PeriodicSystem(3, a * np.eye(3), [vq.Atom(12, [0.0, 0.0, 0.0])])
    return sys_, "sto-3g"


def nacl_primitive() -> tuple[vq.PeriodicSystem, str]:
    a = 5.640 / 0.529177210903
    half = a / 2
    lat = np.array([[0, half, half], [half, 0, half], [half, half, 0]])
    return (vq.PeriodicSystem(3, lat, [
        vq.Atom(11, [0.0, 0.0, 0.0]),
        vq.Atom(17, [half, half, half]),
    ]), "sto-3g")


def nacl_conventional() -> tuple[vq.PeriodicSystem, str]:
    a = 5.640 / 0.529177210903
    NA = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    CL = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
    cell = ([vq.Atom(11, [f * a for f in p]) for p in NA] +
            [vq.Atom(17, [f * a for f in p]) for p in CL])
    return vq.PeriodicSystem(3, a * np.eye(3), cell), "sto-3g"


def mgo_conventional() -> tuple[vq.PeriodicSystem, str]:
    a = 4.213 / 0.529177210903
    MG = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    O = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
    cell = ([vq.Atom(12, [f * a for f in p]) for p in MG] +
            [vq.Atom(8,  [f * a for f in p]) for p in O])
    return vq.PeriodicSystem(3, a * np.eye(3), cell), "pob-tzvp"


@dataclass
class SizeResult:
    label: str
    n_atoms: int
    n_bf: int
    converged: Optional[bool]
    n_iter: Optional[int]
    energy_ha: Optional[float]
    wall_s: float
    note: str


CASES: list[tuple[str, Callable[[], tuple[vq.PeriodicSystem, str]]]] = [
    ("01_he_cubic_1atom",        he_cubic),
    ("02_mg_cubic_1atom",        mg_cubic),
    ("03_nacl_primitive_2atom",  nacl_primitive),  # XFAIL on EWALD_3D
    ("04_nacl_conventional_8atom", nacl_conventional),
    ("05_mgo_conventional_8atom",  mgo_conventional),
]


def run_one(label: str, build_fn) -> SizeResult:
    print(f"\n[{label}]", flush=True)
    system, basis_name = build_fn()
    basis = vq.BasisSet(system.unit_cell_molecule(), basis_name)
    print(f"  {len(system.unit_cell)} atoms, {basis.nbasis} bf  (basis: {basis_name})")

    out_stem = OUT_DIR / label
    rec = SizeResult(
        label=label, n_atoms=len(system.unit_cell), n_bf=basis.nbasis,
        converged=None, n_iter=None, energy_ha=None,
        wall_s=0.0, note="",
    )
    t0 = time.perf_counter()
    try:
        with vq.crash_dump_context(out_stem):
            with vq.perf_log(out_stem.with_suffix(".perf")):
                result = vq.run_rks_periodic_scf(
                    system, basis, vq.KPoints.gamma(system),
                    _mk_opts(), progress=False,
                    spacing_bohr=SPACING, omega=OMEGA,
                )
        rec.wall_s = time.perf_counter() - t0
        rec.converged = bool(result.converged)
        rec.n_iter = int(result.n_iter)
        rec.energy_ha = float(result.energy)
        rec.note = "ok" if rec.converged else "max_iter, no convergence"
    except Exception as exc:
        rec.wall_s = time.perf_counter() - t0
        rec.converged = False
        rec.note = f"{type(exc).__name__}: {str(exc)[:60]}"

    flag = "✓" if rec.converged else "✗"
    print(f"  {flag} iter={rec.n_iter}  E={rec.energy_ha}  "
          f"wall={rec.wall_s:.1f}s  {rec.note}")
    return rec


def main() -> None:
    profile = "FAST_DEBUG" if _FAST_DEBUG else "default"
    print("=" * 72)
    print(f" Size bisection — {profile} profile")
    print("=" * 72)
    print(f"  spacing={SPACING}  cutoff={CUTOFF}  omega={OMEGA}  "
          f"max_iter={MAX_IT}")

    results = [run_one(lbl, fn) for lbl, fn in CASES]

    with open(SUMMARY_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "n_atoms", "n_bf", "converged", "n_iter",
                    "energy_ha", "wall_s", "note"])
        for r in results:
            w.writerow([r.label, r.n_atoms, r.n_bf, r.converged,
                        r.n_iter, r.energy_ha, r.wall_s, r.note])

    print()
    print("=" * 72)
    print(" Summary")
    print("=" * 72)
    print(f"  {'label':35s}  {'atom':4s}  {'bf':4s}  {'flag':4s}  "
          f"{'iter':4s}  {'wall':>6s}  note")
    for r in results:
        flag = "OK" if r.converged else "FAIL"
        i_str = f"{r.n_iter:4d}" if r.n_iter is not None else "  - "
        print(f"  {r.label:35s}  {r.n_atoms:4d}  {r.n_bf:4d}  {flag:4s}  "
              f"{i_str}  {r.wall_s:5.1f}s  {r.note[:50]}")
    print()
    print(f"  CSV: {SUMMARY_CSV.relative_to(HERE.parent.parent)}")
    print(f"  Failures: per-label .dump files under "
          f"{OUT_DIR.relative_to(HERE.parent.parent)}/")


if __name__ == "__main__":
    main()
