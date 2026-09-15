"""BUG87-A reproducer: RIJCOSX SCF throughput on glycine, honest walls.

Fleet evidence (rp68perfcol08v1, glycine/def2-TZVP, 1 core, same node):
vibe-qc RHF RIJCOSX wall 2277.6 s vs ORCA 6.1.1 RIJCOSX 69.2 s (32.9x;
~53x after crediting ORCA's deck for EnGrad work vibe-qc never did).
The shipped .perf iteration table printed 0.000 s per iteration while
scf.rhf totalled 2258 s — the wall column was a renderer default, not a
measurement (SCFIteration carried no wall field).

This script measures the same route with honest wall clocks:

  * total run_rhf wall + n_iter + s/iter (the SCF black box),
  * one unbatched ``compute_cosx_k`` call at the converged density
    (the per-iteration exchange kernel, free-function path),
  * the COSX setup pieces (grid build, Q, Schwarz),
  * the DF J machinery setup + one J build,

so the SCF wall can be apportioned between per-iteration kernel cost
and setup cost without trusting any internal instrumentation.

Usage:
  python cosx_throughput_repro.py            # glycine/def2-svp (fast)
  python cosx_throughput_repro.py --tzvp     # the fleet defect case
  python cosx_throughput_repro.py --scf-only # skip component probes

Run single-threaded to match the fleet pair:
  OMP_NUM_THREADS=1 python cosx_throughput_repro.py

Baseline (this machine, OMP_NUM_THREADS=1, commit of record — see
EVIDENCE.md for the measured table).
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

import vibeqc as vq
from vibeqc import _vibeqc_core as core

# Geometry from the fleet artifact input.py (rp68perfcol08v1; coordinates
# in bohr as vq.Atom takes them).
GLYCINE = vq.Molecule(
    [
        vq.Atom(7, [-2.789235760000, 0.462982900000, 0.124721920000]),
        vq.Atom(6, [-0.423298650000, -0.880612370000, -0.175744530000]),
        vq.Atom(6, [1.838703520000, 0.850376760000, 0.160626720000]),
        vq.Atom(8, [1.702643240000, 3.085922760000, 0.623609620000]),
        vq.Atom(8, [4.072359800000, -0.319363720000, -0.081258220000]),
        vq.Atom(1, [-2.861045350000, 1.923741190000, -1.124387040000]),
        vq.Atom(1, [-4.270781040000, -0.697308940000, -0.211649330000]),
        vq.Atom(1, [-0.340150700000, -2.426408340000, 1.205645270000]),
        vq.Atom(1, [-0.323143170000, -1.725319950000, -2.067360380000]),
        vq.Atom(1, [5.444300970000, 0.861715110000, 0.136060280000]),
    ],
    charge=0,
    multiplicity=1,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tzvp", action="store_true",
                    help="def2-tzvp (the fleet defect case; slow pre-fix)")
    ap.add_argument("--scf-only", action="store_true")
    ap.add_argument("--max-iter", type=int, default=None,
                    help="bound the SCF for profiling")
    args = ap.parse_args()

    basis_name = "def2-tzvp" if args.tzvp else "def2-svp"
    print(f"glycine / {basis_name} / RIJCOSX RHF  "
          f"(OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')})")

    opts = vq.RHFOptions()
    opts.density_fit = True
    opts.aux_basis = "def2-tzvp-jk"
    opts.cosx = True
    if args.max_iter is not None:
        opts.max_iter = args.max_iter

    basis = vq.BasisSet(GLYCINE, basis_name)
    print(f"n_bf = {basis.nbasis}")

    t0 = time.perf_counter()
    result = vq.run_rhf(GLYCINE, basis, opts)
    t_scf = time.perf_counter() - t0
    n_iter = int(result.n_iter)
    print(f"SCF: E = {result.energy:.10f} Ha, converged={result.converged}, "
          f"n_iter={n_iter}")
    print(f"SCF wall: {t_scf:.2f} s  ({t_scf / max(n_iter, 1):.2f} s/iter)")

    if args.scf_only:
        return

    # ---- component probes at the converged density ----
    D = np.asarray(result.density)

    # The SCF (AUTO grid level) steps through the GridX stage
    # progression; probe the exchange kernel on the first (iteration
    # workhorse) and last (final/converged) stage grids.
    card = core.cosx_basis_cardinality_from_name(basis_name)
    level = core.resolve_cosx_grid_level(-1, card)
    stages = core.cosx_grid_stages_for_level(level)
    for label, gopts in (("first-stage", stages[0]), ("final-stage",
                                                      stages[-1])):
        t0 = time.perf_counter()
        grid = core.build_grid(GLYCINE, gopts)
        t_grid = time.perf_counter() - t0
        n_pts = grid.points.shape[0]
        print(f"[probe/{label}] grid build: {t_grid:.3f} s ({n_pts} points)")

        t0 = time.perf_counter()
        q = core.build_cosx_q(basis, grid)
        t_q = time.perf_counter() - t0
        print(f"[probe/{label}] build_cosx_q: {t_q:.3f} s")

        t0 = time.perf_counter()
        schwarz = core.build_cosx_schwarz(basis)
        t_schwarz = time.perf_counter() - t0
        print(f"[probe/{label}] build_cosx_schwarz: {t_schwarz:.3f} s")

        t0 = time.perf_counter()
        k = core.compute_cosx_k(basis, D, grid, q, schwarz)
        t_k = time.perf_counter() - t0
        e_k = -0.25 * float(np.tensordot(D, k))
        print(f"[probe/{label}] compute_cosx_k (unbatched): {t_k:.2f} s "
              f"(E_K = {e_k:.6f} Ha)")
        print(f"[probe/{label}] {n_iter} iters x K = {n_iter * t_k:.1f} s "
              f"vs SCF wall {t_scf:.1f} s")


if __name__ == "__main__":
    main()
