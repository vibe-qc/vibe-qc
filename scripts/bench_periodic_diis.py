#!/usr/bin/env python3
"""Track C — micro-benchmark: per-k Pulay DIIS vs Γ-folded DIIS.

The periodic SCF stack ships two distinct DIIS designs:

  (A) **C++ Γ-folded DIIS** (``cpp/src/periodic_scf.cpp``). The
      multi-k Fock list is folded to its Γ representative
      ``F_Γ = Σ_k w_k F(k)`` and a single nbf × nbf history is
      Pulay-extrapolated. Cheap (one nbf² block per history slot),
      strictly correct only in the molecular limit. Entry point:
      ``vibeqc.run_rhf_periodic``.

  (B) **Python per-k Pulay DIIS** (``_MultiKPulayDIIS`` in
      ``python/vibeqc/periodic_rhf_multi_k_ewald.py``). Per-k
      Fock + error history; B matrix uses a k-weighted Frobenius
      inner product ``B_ij = Σ_k w_k Re tr(e_i(k)^† e_j(k))``.
      Proper multi-k formulation; n_k × nbf² per history slot.
      Entry point: ``vibeqc.run_rhf_periodic_multi_k_ewald3d``.

Both paths exist on purpose — (A) is the C++ accelerator wired
into the new uniform scf_accelerator surface (DIIS / EDIIS /
EDIIS+DIIS), and (B) is the older Python multi-k Ewald path that
the regression / smearing / multi-k tests still drive.

What this script measures
-------------------------

For a small set of representative systems (molecular-limit + a
modest 1D H chain) we run both paths, collect:

  * iter count to ``conv_tol_grad = 1e-7``
  * wall time
  * converged energy

and print a markdown table. Energies must agree to ~1e-9 Ha
(otherwise the comparison is meaningless and we flag it).

Usage
-----

    python scripts/bench_periodic_diis.py
    python scripts/bench_periodic_diis.py --only h2-3d-k222

Caveats
-------

This is a *DIIS-design* benchmark, not an absolute-performance
benchmark — both drivers do their full SCF, including JK builds.
So the wall-time difference is dominated by the Python-vs-C++
Fock-build cost, not by the DIIS arithmetic. The interesting
column is **iter count**: it isolates the DIIS algorithm quality.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable, List

import numpy as np

import vibeqc as vq


# ---------------------------------------------------------------------------
# Test systems — kept tiny so the benchmark runs in seconds.
# ---------------------------------------------------------------------------

H2 = [vq.Atom(1, [0.0, 0.0, 0.0]),
      vq.Atom(1, [0.0, 0.0, 1.4])]

H2O = [vq.Atom(8, [0.0,  0.0,  0.0]),
       vq.Atom(1, [0.0,  1.43, -0.98]),
       vq.Atom(1, [0.0, -1.43, -0.98])]


def _big_box_system(atoms, dim, box=50.0, vacuum=30.0):
    if dim == 1:   lat = np.diag([box, vacuum, vacuum])
    elif dim == 2: lat = np.diag([box, box, vacuum])
    else:          lat = np.diag([box, box, box])
    return vq.PeriodicSystem(dim, lat, atoms)


@dataclass
class Case:
    name: str
    factory: Callable[[], tuple]   # returns (sysp, basis, mesh)


CASES = [
    Case("h2-3d-gamma",
         lambda: (_big_box_system(H2, 3),
                  vq.BasisSet(_big_box_system(H2, 3).unit_cell_molecule(),
                              "sto-3g"),
                  [1, 1, 1])),
    Case("h2-3d-k222",
         lambda: (_big_box_system(H2, 3),
                  vq.BasisSet(_big_box_system(H2, 3).unit_cell_molecule(),
                              "sto-3g"),
                  [2, 2, 2])),
    Case("h2o-3d-k222",
         lambda: (_big_box_system(H2O, 3),
                  vq.BasisSet(_big_box_system(H2O, 3).unit_cell_molecule(),
                              "sto-3g"),
                  [2, 2, 2])),
]


# ---------------------------------------------------------------------------
# Per-driver runners.
# ---------------------------------------------------------------------------

@dataclass
class Run:
    iters: int
    energy: float
    wall_s: float
    converged: bool


def _cxx_gamma_folded_diis(sysp, basis, mesh) -> Run:
    """C++ multi-k driver — Γ-folded DIIS in periodic_scf.cpp."""
    o = vq.PeriodicSCFOptions()
    o.lattice_opts.cutoff_bohr = 15.0
    o.lattice_opts.nuclear_cutoff_bohr = 15.0
    o.conv_tol_energy = 1e-10
    o.conv_tol_grad = 1e-7
    o.max_iter = 100
    o.scf_accelerator = vq.SCFAccelerator.DIIS
    km = vq.monkhorst_pack(sysp, mesh)
    t0 = time.perf_counter()
    r = vq.run_rhf_periodic(sysp, basis, km, o)
    dt = time.perf_counter() - t0
    n_iter = len(r.scf_trace) if getattr(r, "scf_trace", None) else r.n_iter
    return Run(iters=n_iter, energy=r.energy, wall_s=dt, converged=r.converged)


def _python_perk_pulay_diis(sysp, basis, mesh) -> Run:
    """Python multi-k driver — proper per-k Pulay DIIS."""
    o = vq.PeriodicRHFOptions()
    o.lattice_opts.cutoff_bohr = 15.0
    o.lattice_opts.nuclear_cutoff_bohr = 15.0
    o.conv_tol_energy = 1e-10
    o.conv_tol_grad = 1e-7
    o.max_iter = 100
    o.use_diis = True
    o.diis_start_iter = 2
    o.diis_subspace_size = 8
    km = vq.monkhorst_pack(sysp, mesh)
    t0 = time.perf_counter()
    r = vq.run_rhf_periodic_multi_k_ewald3d(sysp, basis, km, o)
    dt = time.perf_counter() - t0
    n_iter = getattr(r, "n_iter", 0)
    return Run(iters=n_iter, energy=r.energy, wall_s=dt,
               converged=getattr(r, "converged", True))


# ---------------------------------------------------------------------------
# Main loop.
# ---------------------------------------------------------------------------

def _fmt_row(name: str, run_a: Run, run_b: Run) -> str:
    de = abs(run_a.energy - run_b.energy)
    agree = "✓" if de < 1e-9 else f"⚠ Δ={de:.2e}"
    return (f"| {name:<14} "
            f"| {run_a.iters:>4} / {run_b.iters:>4} "
            f"| {run_a.wall_s*1000:>7.1f} / {run_b.wall_s*1000:>7.1f} "
            f"| {run_a.energy:>14.8f} "
            f"| {agree:<14} |")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="Run only the named case (h2-3d-gamma, "
                         "h2-3d-k222, h2o-3d-k222)")
    args = ap.parse_args()

    cases = CASES if args.only is None else [
        c for c in CASES if c.name == args.only]
    if not cases:
        print(f"No matching case for '{args.only}'.")
        return 2

    print("Track C — Γ-folded DIIS (C++) vs per-k Pulay DIIS (Python)")
    print()
    print("| system         | iters (C++ / Py) | wall ms (C++ / Py) "
          "| energy (Ha)    | E-agree        |")
    print("|----------------|-------------------|--------------------"
          "|----------------|----------------|")

    failures: List[str] = []
    for case in cases:
        sysp, basis, mesh = case.factory()
        try:
            run_cxx = _cxx_gamma_folded_diis(sysp, basis, mesh)
        except Exception as e:
            failures.append(f"{case.name} C++ path raised: {e}")
            continue
        try:
            run_py = _python_perk_pulay_diis(sysp, basis, mesh)
        except Exception as e:
            failures.append(f"{case.name} Python path raised: {e}")
            continue
        if not run_cxx.converged:
            failures.append(f"{case.name}: C++ path did not converge")
        if not run_py.converged:
            failures.append(f"{case.name}: Python path did not converge")
        print(_fmt_row(case.name, run_cxx, run_py))

    print()
    if failures:
        print("Failures:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("Notes:")
    print("  * 'iters' isolates the DIIS algorithm quality — interpret it,")
    print("    not the wall-times (those mix in Python-vs-C++ Fock-build).")
    print("  * The C++ Γ-folded path is rigorous only in the molecular limit;")
    print("    for genuine multi-k physics, prefer the per-k Pulay path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
