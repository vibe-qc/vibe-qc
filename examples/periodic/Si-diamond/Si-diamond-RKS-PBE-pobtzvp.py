"""Si-diamond — RKS-PBE / pob-tzvp / Γ via the native GDF driver.

Standalone input: geometry is embedded below, so this file can be copied
to a scratch calculation directory and run without sibling helper modules.

Periodic SCF runs through ``vibeqc.run_periodic_job`` -> the native
``run_rhf_periodic_gamma_gdf`` driver (Lpq, S, T, V_ne, V_xc, SCF loop,
J/K einsums, and DIIS are vibe-qc-owned; PySCF is only an external
reference). ``fmixing_percent=30`` mirrors CRYSTAL's default FMIXING
Fock/KS matrix mixing.

Run::
    .venv/bin/python examples/periodic/Si-diamond/Si-diamond-RKS-PBE-pobtzvp.py

Produces (sibling files, sharing the script's stem):
    {stem}.out, {stem}.system, {stem}.molden
"""
from pathlib import Path

import numpy as np

from vibeqc import (
    Atom, BasisSet, PeriodicSystem,
    run_periodic_job,
)

ANG2BOHR = 1.0 / 0.529177210903
OUTPUT_STEM = Path(__file__).resolve().with_suffix("")

CELL_ANG = [[5.431, 0.0, 0.0], [0.0, 5.431, 0.0], [0.0, 0.0, 5.431]]
ATOM_DATA = [('Si', [0.0, 0.0, 0.0]),
 ('Si', [4.07325, 1.35775, 2.7155]),
 ('Si', [1.35775, 2.7155, 4.07325]),
 ('Si', [2.7155, 4.07325, 1.35775]),
 ('Si', [0.0, 2.7155, 2.7155]),
 ('Si', [4.07325, 4.07325, 0.0]),
 ('Si', [1.35775, 0.0, 1.35775]),
 ('Si', [2.7155, 1.35775, 4.07325]),
 ('Si', [2.7155, 0.0, 2.7155]),
 ('Si', [1.35775, 1.35775, 0.0]),
 ('Si', [4.07325, 2.7155, 1.35775]),
 ('Si', [0.0, 4.07325, 4.07325]),
 ('Si', [2.7155, 2.7155, 0.0]),
 ('Si', [1.35775, 4.07325, 2.7155]),
 ('Si', [4.07325, 0.0, 4.07325]),
 ('Si', [0.0, 1.35775, 1.35775])]

cell_bohr = np.array(CELL_ANG, dtype=float) * ANG2BOHR
Z_BY_SYM = {
    "H": 1, "Li": 3, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17,
    "Ti": 22, "Zn": 30,
}
atoms = [
    Atom(Z_BY_SYM[sym], [float(x) * ANG2BOHR for x in pos_ang])
    for sym, pos_ang in ATOM_DATA
]
system = PeriodicSystem(3, cell_bohr, atoms)
basis = BasisSet(system.unit_cell_molecule(), "pob-tzvp")

run_periodic_job(
    system, basis,
    method="RKS",
    functional="pbe",
    output=OUTPUT_STEM,
    use_diis=True,
    damping=0.0,
    fmixing_percent=30.0,
    max_iter=80,
    conv_tol_energy=1e-7,
    write_density=False,   # keep the generated parity grid lightweight.
)
