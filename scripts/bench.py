#!/usr/bin/env python3
"""vibe-qc benchmark harness — OpenMP scaling and absolute wall-times.

Usage
-----

    python scripts/bench.py                      # default thread sweep
    python scripts/bench.py --threads 1,4,8,16   # custom
    python scripts/bench.py --only molecular     # restrict benchmarks

Controls ``OMP_NUM_THREADS`` per invocation by re-execing Python with the
environment variable set; prints a markdown table with wall-time and
relative speedup against the 1-thread baseline.

Benchmarks
----------

The set targets the hot paths that Phase P1 parallelised:

* **molecular RHF** — H₂O / cc-pVDZ. Dominates in ``build_fock_g`` + ``compute_eri``.
* **molecular RKS/PBE** — H₂O / cc-pVDZ. Adds the XC grid loop.
* **periodic RHF** — linear H-chain, unit cell = 6 H, cutoff=12 bohr.
  Dominates in ``build_fock_2e_real_space`` (triple lattice loop).
* **Ewald Madelung** — NaCl, large real-space cutoff so both the real-
  space and reciprocal-space Ewald loops have work to do.

Each benchmark is run twice (warm-up + measurement) — the second run is
reported.
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _child(argv: list[str], threads: int) -> str:
    """Spawn a child Python with OMP_NUM_THREADS set; return its stdout."""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    env["VECLIB_MAXIMUM_THREADS"] = str(threads)
    res = subprocess.run(
        [sys.executable] + argv, env=env, cwd=REPO,
        capture_output=True, text=True, check=True,
    )
    return res.stdout.strip()


def _run_once(script: str, threads: int) -> float:
    """Run an inline bench script in a child interpreter and parse the
    "TIME:<seconds>" line it emits. Re-execing isolates the OpenMP
    environment per thread-count cleanly."""
    out = _child(["-c", script], threads)
    for line in out.splitlines():
        if line.startswith("TIME:"):
            return float(line[5:])
    raise RuntimeError(f"bench script emitted no TIME line:\n{out}")


# -----------------------------------------------------------------------
# Inline bench scripts. Kept small so subprocess startup stays a small
# fraction of the measured wall-time.
# -----------------------------------------------------------------------

_MOLECULAR_RHF = """
import time
import vibeqc as vq
mol = vq.Molecule([
    vq.Atom(8, [0,0,0]),
    vq.Atom(1, [0, 1.43, -0.98]),
    vq.Atom(1, [0, -1.43, -0.98]),
])
basis = vq.BasisSet(mol, 'cc-pvdz')
opts = vq.RHFOptions(); opts.conv_tol_energy = 1e-10
# warm-up + measurement
vq.run_rhf(mol, basis, opts)
t = time.perf_counter()
for _ in range(3):
    vq.run_rhf(mol, basis, opts)
print(f'TIME:{(time.perf_counter() - t) / 3:.6f}')
"""

_MOLECULAR_RKS = """
import time
import vibeqc as vq
mol = vq.Molecule([
    vq.Atom(8, [0,0,0]),
    vq.Atom(1, [0, 1.43, -0.98]),
    vq.Atom(1, [0, -1.43, -0.98]),
])
basis = vq.BasisSet(mol, 'cc-pvdz')
opts = vq.RKSOptions(); opts.functional = 'PBE'; opts.conv_tol_energy = 1e-10
vq.run_rks(mol, basis, opts)
t = time.perf_counter()
for _ in range(3):
    vq.run_rks(mol, basis, opts)
print(f'TIME:{(time.perf_counter() - t) / 3:.6f}')
"""

_PERIODIC_RHF = """
import time
import numpy as np
import vibeqc as vq
unit = []
for i in range(6):
    unit.append(vq.Atom(1, [0, 0, 1.4 * i]))
a = 1.4 * 6
sysp = vq.PeriodicSystem(1, np.diag([a, 30.0, 30.0]), unit,
                          charge=0, multiplicity=1)
basis = vq.BasisSet(sysp.unit_cell_molecule(), 'sto-3g')
km = vq.monkhorst_pack(sysp, [2, 1, 1])
opts = vq.PeriodicSCFOptions()
opts.lattice_opts.cutoff_bohr = 12.0
opts.lattice_opts.nuclear_cutoff_bohr = 20.0
opts.conv_tol_energy = 1e-8
opts.max_iter = 20
vq.run_rhf_periodic(sysp, basis, km, opts)
t = time.perf_counter()
vq.run_rhf_periodic(sysp, basis, km, opts)
print(f'TIME:{time.perf_counter() - t:.6f}')
"""

_EWALD = """
import time
import numpy as np
import vibeqc as vq
a = 5.6 / 0.529177210903
lat = 0.5 * a * np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float).T
positions = np.column_stack([[0,0,0], [0.5*a, 0, 0]])
charges = np.array([11.0, 17.0])
opts = vq.EwaldOptions(); opts.real_cutoff_bohr = 40.0
vq.ewald_point_charge_energy(lat, positions, charges, opts)
t = time.perf_counter()
for _ in range(10):
    vq.ewald_point_charge_energy(lat, positions, charges, opts)
print(f'TIME:{(time.perf_counter() - t) / 10:.6f}')
"""

BENCHES = {
    "molecular-rhf":  ("H2O / cc-pVDZ  RHF",                   _MOLECULAR_RHF),
    "molecular-rks":  ("H2O / cc-pVDZ  RKS (PBE)",             _MOLECULAR_RKS),
    "periodic-rhf":   ("H6 chain (1D)  periodic RHF, 2 k-pts", _PERIODIC_RHF),
    "ewald":          ("NaCl Madelung  Ewald",                 _EWALD),
}


def run_suite(threads_list: list[int], only: list[str] | None) -> None:
    print("\n# vibe-qc P1 benchmark results\n")
    print(f"threads tested: {threads_list}\n")
    names = list(BENCHES) if only is None else only

    for key in names:
        if key not in BENCHES:
            raise KeyError(f"unknown benchmark {key!r}; known: {list(BENCHES)}")
        label, script = BENCHES[key]
        print(f"## {label}")
        print()
        print("| threads | wall (s) | speedup |")
        print("|--------:|---------:|--------:|")
        baseline = None
        for nt in threads_list:
            t = _run_once(script, nt)
            if baseline is None:
                baseline = t
                sp = 1.00
            else:
                sp = baseline / t
            print(f"| {nt:7d} | {t:8.4f} | {sp:6.2f}× |")
        print()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--threads", default="1,2,4,8",
        help="comma-separated list of OMP_NUM_THREADS values"
             " (default: 1,2,4,8)",
    )
    p.add_argument(
        "--only", default=None,
        help=f"restrict to a subset. Known: {','.join(BENCHES)}",
    )
    args = p.parse_args()
    threads_list = [int(x) for x in args.threads.split(",") if x.strip()]
    only = args.only.split(",") if args.only else None
    run_suite(threads_list, only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
