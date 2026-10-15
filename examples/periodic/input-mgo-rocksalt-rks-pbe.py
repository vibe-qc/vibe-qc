"""MgO conventional rocksalt — RKS-PBE / STO-3G / Γ-only GDF.

Success criterion: the SCF converges within the iteration budget
(``max_iter = 60``) and energy tolerance (``conv_tol_energy = 1e-7``) set
below, and the |dE| against PySCF.pbc printed by
``examples/debug/mgo_al2o3_parity.py`` is at the mHa level. This
conventional 8-atom cell is a long run on the Gamma-only Ewald-3D route
(tens of minutes on a loaded machine), not a seconds-scale example.

DFT counterpart of ``input-mgo-rocksalt-rhf.py``. Same geometry,
same basis, same Γ-only GDF route; switches the SCF from restricted
Hartree-Fock to restricted Kohn-Sham with the PBE GGA functional.

The vibe-qc half of the PBE parity test against PySCF.pbc; see
``examples/debug/mgo_al2o3_parity.py`` for the side-by-side
comparison and the |ΔE| verdict. As for RHF, one Γ point is a parity
target, not a converged MgO energy; pass a k mesh for that.

Run:
    .venv/bin/python examples/periodic/input-mgo-rocksalt-rks-pbe.py

Expect 20 to 30 minutes with 3 or 4 OpenMP threads (vibe-qc 0.17.15).

Outputs (next to this script):
    input-mgo-rocksalt-rks-pbe.out      — banner, SCF trace, energies
    input-mgo-rocksalt-rks-pbe.system   — host / build / runtime manifest
    input-mgo-rocksalt-rks-pbe.perf     — per-stage wall-clock
    plus the runner's structure, orbital, citation and QVF files
"""

from pathlib import Path

import numpy as np

import vibeqc as vq
from vibeqc import perf_log

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem  # → "input-mgo-rocksalt-rks-pbe"
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
        method="RKS",
        functional="pbe",
        jk_method="gdf",
        kpoints=(1, 1, 1),
        max_iter=60,
        conv_tol_energy=1e-7,
        output=str(OUT),
    )

print(f"  E = {result.energy:.8f} Ha   converged={result.converged}   "
      f"n_iter={result.n_iter}")
