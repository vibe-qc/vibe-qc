"""Al2O3 cubic stuffed-fluorite model — RHF / STO-3G / Γ-only Ewald-3D.

The vibe-qc half of the v0.7 RHF parity test against PySCF.pbc on
an Al2O3-stoichiometry crystal. See ``examples/debug/mgo_al2o3_parity.py``
for the side-by-side comparison and the |ΔE| verdict.

Why a cubic *model* and not actual α-Al2O3 (corundum)
-----------------------------------------------------
α-Al2O3 (corundum) is rhombohedral / hexagonal — incompatible with
vibe-qc's current FFT-Poisson solver, which is orthorhombic-only. We
use a 5-atom cubic stuffed-fluorite Al2O3 model instead: 2 Al at
body-diagonal positions, 3 O at face-centred positions, lattice
constant chosen to give a reasonable Al-O distance ~2 Å. This is
NOT a real Al2O3 polymorph — but both vibe-qc and PySCF.pbc solve
it with the same Hamiltonian, so the *parity* between the two
codes is the test, not the physical-realism of the crystal.

Run:
    .venv/bin/python examples/periodic/input-al2o3-cubic-rhf.py

Outputs (next to this script):
    input-al2o3-cubic-rhf.out      — banner, SCF trace, energies
    input-al2o3-cubic-rhf.perf     — per-iteration wall-clock
    input-al2o3-cubic-rhf.system   — host / build / runtime manifest
"""

from pathlib import Path

import numpy as np

import vibeqc as vq
from vibeqc import perf_log, write_system_manifest

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem  # → "input-al2o3-cubic-rhf"
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

opts = (vq.PeriodicRHFOptions() if hasattr(vq, "PeriodicRHFOptions")
        else vq.PeriodicKSOptions())
opts.lattice_opts.coulomb_method       = vq.CoulombMethod.EWALD_3D
opts.lattice_opts.cutoff_bohr          = 12.0
opts.lattice_opts.nuclear_cutoff_bohr  = 25.0
opts.max_iter        = 60
opts.use_diis        = True
opts.damping         = 0.85
opts.conv_tol_energy = 1e-7
opts.initial_guess   = vq.InitialGuess.SAD

with perf_log(OUT.with_suffix(".perf")):
    result = vq.run_rhf_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.3, spacing_bohr=0.5, progress=False,
    )

with open(OUT.with_suffix(".out"), "w") as fh:
    fh.write("Al2O3 cubic stuffed-fluorite model — RHF / STO-3G / Γ-only Ewald-3D\n")
    fh.write(f"  lattice constant      = {A_ANG:.3f} Å ({A:.4f} bohr)\n")
    fh.write(f"  atoms per cell        = 2 Al + 3 O = 5\n")
    fh.write(f"  basis                 = STO-3G ({basis.nbasis} fns / cell)\n")
    fh.write("\n")
    fh.write(f"  E_total               = {result.energy:.8f} Ha\n")
    fh.write(f"  converged             = {result.converged}\n")
    fh.write(f"  SCF iterations        = {result.n_iter}\n")

write_system_manifest(OUT.with_suffix(".system"))

print(f"  E = {result.energy:.8f} Ha   converged={result.converged}   "
      f"n_iter={result.n_iter}")
