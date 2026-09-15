"""Li doublet in a 16-bohr cube, ROHF / STO-3G via the GPW route.

This is the smallest periodic restricted-open-shell example with both a
doubly occupied orbital and a singly occupied orbital.  The maintained-preview
public route is 3D, Gamma-only, and uses integer 2/1/0 occupations.

Run:
    .venv/bin/python examples/periodic/gpw-li-sto3g-rohf.py
"""

import os
from pathlib import Path

import numpy as np
import vibeqc as vq


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("VIBEQC_EXAMPLE_OUTPUT_DIR", HERE))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUTPUT_DIR / Path(__file__).stem

length_bohr = 16.0
lattice = length_bohr * np.eye(3)
atoms = [vq.Atom(3, [length_bohr / 2] * 3)]
system = vq.PeriodicSystem(
    3,
    lattice,
    atoms,
    charge=0,
    multiplicity=2,
)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

result = vq.run_periodic_job(
    system,
    basis,
    method="ROHF",
    jk_method="gpw",
    cutoff_ha=50.0,
    output=str(OUT),
    max_iter=80,
    conv_tol_energy=1e-8,
    write_density=False,
)

print(f"E_total    = {result.energy:.10f} Ha")
print(f"converged  = {result.converged}")
print(f"iterations = {result.n_iter}")
print(f"S^2        = {result.s_squared:.8f}")
print(f"occupations = {result.occupations.tolist()}")
