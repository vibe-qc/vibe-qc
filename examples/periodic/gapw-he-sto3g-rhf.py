"""He in a 12-bohr cube — GAPW RHF / STO-3G via the all-electron route.

Demonstrates the M3c all-electron GAPW path: J via FFT-Poisson with
per-atom augmentation correction, V_ne + E_nn via the periodic Ewald
gauge. He is the simplest closed-shell atom and is a routine sanity
check for the GAPW build.

Run:
    .venv/bin/python examples/periodic/gapw-he-sto3g-rhf.py
"""

import warnings
from pathlib import Path

import numpy as np
import vibeqc as vq
from vibeqc import GAPWExperimentalWarning

warnings.simplefilter("ignore", category=GAPWExperimentalWarning)

HERE = Path(__file__).resolve().parent
OUT = HERE / Path(__file__).stem

L = 12.0  # bohr
lattice = L * np.eye(3)
atoms = [vq.Atom(2, [0.0, 0.0, 0.0])]
system = vq.PeriodicSystem(3, lattice, atoms)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

print("=" * 60)
print("GAPW RHF He STO-3G (all-electron augmentation)")
print("=" * 60)

result = vq.run_periodic_job(
    system,
    basis,
    method="RHF",
    jk_method="gapw",
    gapw_molecular_limit=True,
    output=str(OUT),
    max_iter=30,
    conv_tol_energy=1e-7,
    write_density=False,
    citations=False,
)

print(f"  E_total       = {result.energy:.8f} Ha")
print(f"  converged     = {result.converged}")
print(f"  n_iter        = {result.n_iter}")
print(
    f"  E = {result.energy:.8f} Ha   converged={result.converged}   "
    f"n_iter={result.n_iter}"
)
print()

# Compare with GPW
print("GPW reference (smooth-grid, no augmentation):")
gpw_result = vq.run_periodic_job(
    system,
    basis,
    method="RHF",
    jk_method="gpw",
    output=str(OUT) + "_gpw",
    max_iter=30,
    conv_tol_energy=1e-7,
    write_density=False,
    citations=False,
)
print(f"  GPW E_total   = {gpw_result.energy:.8f} Ha")
print(f"  GAPW E_total  = {result.energy:.8f} Ha")
print(f"  ΔE (GAPW-GPW) = {result.energy - gpw_result.energy:.8f} Ha")
