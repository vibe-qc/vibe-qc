"""MgO primitive — run_periodic_job with jk_method='bipole'.

Demonstrates the high-level periodic_runner API with BIPOLE J/K
method. Same system as input-bipole-mgo-rhf.py but dispatched through
the standard run_periodic_job entry point.

Run:
    .venv/bin/python examples/periodic/input-bipole-runner-mgo-rhf.py
"""

from pathlib import Path

import numpy as np
from vibeqc.periodic_runner import run_periodic_job

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem
OUT = HERE / STEM

ANG2BOHR = 1.0 / 0.529177210903

import vibeqc as vq

a = 4.21 * ANG2BOHR
lattice = (a / 2.0) * np.array(
    [
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ]
)
atoms = [
    vq.Atom(12, [0.0, 0.0, 0.0]),
    vq.Atom(8, [a / 2.0, a / 2.0, a / 2.0]),
]
sysp = vq.PeriodicSystem(3, lattice, atoms)
basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

result = run_periodic_job(
    sysp,
    basis,
    method="RHF",
    jk_method="bipole",
    output=OUT,
    max_iter=50,
    conv_tol_energy=1e-7,
    initial_guess="SAD",
    progress=True,
    # The supported BIPOLE route uses exact erfc-screened ERIs plus
    # reciprocal J_LR. The quartet multipole prototype is unavailable:
    # explicit use_multipole_far_field=True raises before SCF setup.
    # multipole_l_max is inactive on this exact route.
)

print(
    f"  E = {result.energy:.8f} Ha   converged={result.converged}   "
    f"n_iter={result.n_iter}"
)
