"""H2 in a 16-bohr cube — RKS / PBE / STO-3G via the GPW route with full output (M3d).

Drives the closed-shell GPW RKS path through
:func:`vibeqc.run_periodic_job` with ``jk_method='gpw'`` and
``write_density=True`` so a primitive-cell ``.xsf`` lands alongside
the ``.out`` / ``.molden`` siblings. Useful as a template for any
GPW production-style job: select method, functional, and let the
runner emit the standard sidecars.

Run:
    .venv/bin/python examples/periodic/gpw-h2-sto3g-rks-pbe-with-output.py
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
    method="RKS",
    functional="pbe",
    jk_method="gpw",
    output=str(OUT),
    max_iter=50,
    conv_tol_energy=1e-7,
    write_density=True,
    density_spacing_bohr=0.3,
    citations=False,
)

print(f"  E_total   = {result.energy:.8f} Ha")
print(f"  E_xc      = {result.e_xc:+.8f} Ha")
print(f"  E_coulomb = {result.e_coulomb:+.8f} Ha")
print(f"  Output stem  = {OUT}")
print(f"  XSF density  = {OUT.with_suffix('.xsf')}  exists={OUT.with_suffix('.xsf').exists()}")
print(f"  E = {result.energy:.8f} Ha   converged={result.converged}   n_iter={result.n_iter}")
