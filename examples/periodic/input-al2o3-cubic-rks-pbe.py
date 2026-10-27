"""Al2O3 cubic stuffed-fluorite model — RKS-PBE / STO-3G / Γ-only GDF.

DFT counterpart of ``input-al2o3-cubic-rhf.py``. Same geometry and
basis; switches to RKS with the PBE GGA functional.

The vibe-qc half of the PBE parity test against PySCF.pbc on an
Al2O3-stoichiometry crystal. See ``examples/debug/mgo_al2o3_parity.py``
for the side-by-side comparison and the |ΔE| verdict.

This model has no PBE gap at the Γ point: with integer occupations the
SCF does not converge (|dE| = 0.30 Ha after 60 cycles on vibe-qc
0.17.15), and with 0.01 Ha of Fermi-Dirac smearing it converges in 16
cycles with an electronic entropy of 5.2 k_B, i.e. with many partly
filled levels. The example therefore smears. Compare the free energy
(E - TS, printed in the ``.out``) against a reference run with the same
smearing.

Why a cubic *model* and not actual α-Al2O3 (corundum)
-----------------------------------------------------
It is a small, cheap parity target, not a real Al2O3 polymorph (see
``input-al2o3-cubic-rhf.py``). vibe-qc's periodic routes are
lattice-general; real hexagonal corundum belongs in the GDF parity
suite.

Run:
    .venv/bin/python examples/periodic/input-al2o3-cubic-rks-pbe.py

Outputs (next to this script):
    input-al2o3-cubic-rks-pbe.out      — banner, SCF trace, energies
    input-al2o3-cubic-rks-pbe.system   — host / build / runtime manifest
    input-al2o3-cubic-rks-pbe.perf     — per-stage wall-clock
    plus the runner's structure, orbital, citation and QVF files
"""

from pathlib import Path

import numpy as np

import vibeqc as vq
from vibeqc import perf_log

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem  # → "input-al2o3-cubic-rks-pbe"
OUT  = HERE / STEM

ANG2BOHR = 1.0 / 0.529177210903

# Cubic Al2O3 stuffed-fluorite model — 2 Al + 3 O per cell
A_ANG = 5.05
A     = A_ANG * ANG2BOHR
al_frac = [(0.25, 0.25, 0.25), (0.75, 0.75, 0.75)]
o_frac  = [(0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]

unit_cell = (
    [vq.Atom(13, [fx*A, fy*A, fz*A]) for fx, fy, fz in al_frac]
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
        smearing_temperature=0.01,   # Ha; no PBE gap at Gamma (see above)
        max_iter=60,
        conv_tol_energy=1e-7,
        output=str(OUT),
    )

print(f"  E = {result.energy:.8f} Ha   converged={result.converged}   "
      f"n_iter={result.n_iter}")
