"""He in a 16-bohr cube — RKS / LDA / STO-3G via the GPW route (M3d).

Demonstrates the M3d closed-shell RKS extension of the GPW path: J via
FFT-Poisson, V_xc via libxc on the same smooth grid, V_ne + E_nn via
the periodic Ewald gauge. He is the smallest closed-shell atom and is
a routine sanity check for the LDA-GPW build.

Run:
    .venv/bin/python examples/periodic/gpw-he-sto3g-lda.py
"""

import warnings
from pathlib import Path

import numpy as np
import vibeqc as vq
from vibeqc import GAPWExperimentalWarning

warnings.simplefilter("ignore", category=GAPWExperimentalWarning)

HERE = Path(__file__).resolve().parent
OUT = HERE / Path(__file__).stem

L = 16.0  # bohr
lattice = L * np.eye(3)
atoms = [vq.Atom(2, [0.0, 0.0, 0.0])]
system = vq.PeriodicSystem(3, lattice, atoms)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

result = vq.run_periodic_job(
    system,
    basis,
    method="RKS",
    functional="lda",
    jk_method="gpw",
    output=str(OUT),
    max_iter=50,
    conv_tol_energy=1e-7,
    write_density=False,
    citations=False,
)

# Energy decomposition from the GPW runner result.
print(f"  E_total          = {result.energy:.8f} Ha")
print(f"  E_electronic     = {result.e_electronic:+.8f} Ha")
print(f"  E_nuclear (E_nn) = {result.e_nuclear:+.8f} Ha")
print(f"  E_coulomb (J)    = {result.e_coulomb:+.8f} Ha")
print(f"  E_xc (LDA)       = {result.e_xc:+.8f} Ha")
print(f"  E_hf_exchange    = {result.e_hf_exchange:+.8f} Ha")
print(f"  converged        = {result.converged}")
print(f"  E = {result.energy:.8f} Ha   converged={result.converged}   n_iter={result.n_iter}")
