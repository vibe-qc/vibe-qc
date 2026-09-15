"""H2 in a 16-bohr cube — RHF / STO-3G via the GPW route (M3a/M3b).

Drives the periodic SCF through :func:`vibeqc.run_periodic_job` with
``jk_method='gpw'``: the closed-shell M3a/M3b GPW path builds the
Hartree-J via FFT-Poisson on a smooth real-space grid while V_ne and
E_nn live in the matching periodic Ewald gauge. This is the simplest
end-to-end GPW example — a molecular H2 in vacuum padding.

Run:
    .venv/bin/python examples/periodic/gpw-h2-sto3g-rhf.py
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
atoms = [
    vq.Atom(1, [-0.7, 0.0, 0.0]),
    vq.Atom(1, [+0.7, 0.0, 0.0]),
]
system = vq.PeriodicSystem(3, lattice, atoms)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

result = vq.run_periodic_job(
    system,
    basis,
    method="RHF",
    jk_method="gpw",
    output=str(OUT),
    max_iter=50,
    conv_tol_energy=1e-7,
    write_density=False,
    citations=False,
)

print(f"  E_total       = {result.energy:.8f} Ha")
print(f"  converged     = {result.converged}")
print(f"  n_iter        = {result.n_iter}")
print(f"  E = {result.energy:.8f} Ha   converged={result.converged}   n_iter={result.n_iter}")
