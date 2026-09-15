"""MgO conventional rocksalt — RKS-PBE / STO-3G / Γ-only Ewald-3D.

DFT counterpart of ``input-mgo-rocksalt-rhf.py``. Same geometry,
same basis, same Ewald-3D Γ-only path; switches the SCF from
restricted Hartree-Fock to restricted Kohn-Sham with the PBE GGA
functional.

The vibe-qc half of the v0.7 PBE parity test against PySCF.pbc; see
``examples/debug/mgo_al2o3_parity.py`` for the side-by-side
comparison and the |ΔE| verdict.

Run:
    .venv/bin/python examples/periodic/input-mgo-rocksalt-rks-pbe.py

Outputs (next to this script):
    input-mgo-rocksalt-rks-pbe.out      — banner, SCF trace, energies
    input-mgo-rocksalt-rks-pbe.perf     — per-iteration wall-clock
    input-mgo-rocksalt-rks-pbe.system   — host / build / runtime manifest
"""

from pathlib import Path

import numpy as np

import vibeqc as vq
from vibeqc import perf_log, write_system_manifest

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

opts = vq.PeriodicKSOptions()
opts.functional                        = "pbe"
opts.lattice_opts.coulomb_method       = vq.CoulombMethod.EWALD_3D
opts.lattice_opts.cutoff_bohr          = 12.0
opts.lattice_opts.nuclear_cutoff_bohr  = 25.0
opts.max_iter        = 60
opts.use_diis        = True
opts.damping         = 0.85
opts.conv_tol_energy = 1e-7
opts.initial_guess   = vq.InitialGuess.SAD

with perf_log(OUT.with_suffix(".perf")):
    result = vq.run_rks_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.3, spacing_bohr=0.5, progress=False,
    )

with open(OUT.with_suffix(".out"), "w") as fh:
    fh.write("MgO conventional rocksalt — RKS-PBE / STO-3G / Γ-only Ewald-3D\n")
    fh.write(f"  lattice constant      = {A_ANG:.3f} Å ({A:.4f} bohr)\n")
    fh.write(f"  atoms per cell        = 4 Mg + 4 O = 8\n")
    fh.write(f"  basis                 = STO-3G ({basis.nbasis} fns / cell)\n")
    fh.write(f"  functional            = PBE\n")
    fh.write("\n")
    fh.write(f"  E_total               = {result.energy:.8f} Ha\n")
    fh.write(f"  converged             = {result.converged}\n")
    fh.write(f"  SCF iterations        = {result.n_iter}\n")

write_system_manifest(OUT.with_suffix(".system"))

print(f"  E = {result.energy:.8f} Ha   converged={result.converged}   "
      f"n_iter={result.n_iter}")
