"""MgO conventional rocksalt — RHF / STO-3G / Γ-only GDF.

Standard ionic-wide-gap-insulator benchmark. 4 Mg + 4 O in the
cubic conventional cell, lattice constant a = 4.211 Å (experimental).
The vibe-qc half of the RHF parity test against PySCF.pbc (Γ-only
GDF with ``exxdiv='ewald'``); see ``examples/debug/mgo_al2o3_parity.py``
for the side-by-side comparison.

Runs native Gaussian density fitting through ``run_periodic_job`` at
the single Γ point. One k point is a crude Brillouin-zone sample for
this cell: the point here is a like-for-like parity target, not a
converged MgO energy. Pass a k mesh, e.g. ``kpoints=(4, 4, 4)``, for
that.

Run:
    .venv/bin/python examples/periodic/input-mgo-rocksalt-rhf.py

Expect 20 to 30 minutes with 3 or 4 OpenMP threads (vibe-qc 0.17.15).

Outputs (next to this script):
    input-mgo-rocksalt-rhf.out      — banner, SCF trace, energies
    input-mgo-rocksalt-rhf.system   — host / build / runtime manifest
    input-mgo-rocksalt-rhf.perf     — per-stage wall-clock
    plus the runner's structure, orbital, citation and QVF files
"""

from pathlib import Path

import numpy as np

import vibeqc as vq
from vibeqc import perf_log

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem  # → "input-mgo-rocksalt-rhf"
OUT  = HERE / STEM

ANG2BOHR = 1.0 / 0.529177210903

# MgO conventional rocksalt — 4 Mg + 4 O per cubic cell
A_ANG = 4.211                          # experimental lattice constant
A     = A_ANG * ANG2BOHR
mg_frac = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
o_frac  = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]

unit_cell = (
    [vq.Atom(12, [fx*A, fy*A, fz*A]) for fx, fy, fz in mg_frac]
    + [vq.Atom( 8, [fx*A, fy*A, fz*A]) for fx, fy, fz in o_frac]
)
sysp  = vq.PeriodicSystem(3, np.diag([A, A, A]), unit_cell)
basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

with perf_log(OUT.with_suffix(".perf")):
    result = vq.run_periodic_job(
        sysp, basis,
        method="RHF",
        jk_method="gdf",
        kpoints=(1, 1, 1),
        max_iter=60,
        conv_tol_energy=1e-7,
        output=str(OUT),
    )

print(f"  E = {result.energy:.8f} Ha   converged={result.converged}   "
      f"n_iter={result.n_iter}")
