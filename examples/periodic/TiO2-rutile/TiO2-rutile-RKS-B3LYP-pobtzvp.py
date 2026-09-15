"""TiO2-rutile — RKS-B3LYP / pob-tzvp / Γ via the native GDF driver.

Standalone input: geometry is embedded below, so this file can be copied
to a scratch calculation directory and run without sibling helper modules.

Periodic SCF runs through ``vibeqc.run_periodic_job`` -> the native
``run_rhf_periodic_gamma_gdf`` driver (Lpq, S, T, V_ne, V_xc, SCF loop,
J/K einsums, and DIIS are vibe-qc-owned; PySCF is only an external
reference). ``fmixing_percent=30`` mirrors CRYSTAL's default FMIXING
Fock/KS matrix mixing.

Run::
    .venv/bin/python examples/periodic/TiO2-rutile/TiO2-rutile-RKS-B3LYP-pobtzvp.py

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

CELL_ANG = [[4.5937, 0.0, 0.0], [0.0, 4.5937, 0.0], [0.0, 0.0, 2.9587]]
ATOM_DATA = [('Ti', [0.0, 0.0, 0.0]),
 ('Ti', [2.29685, 2.29685, 1.47935]),
 ('O', [1.4024566100000002, 1.4024566100000002, 0.0]),
 ('O', [3.19124339, 3.19124339, 0.0]),
 ('O', [0.89439339, 3.6993066100000003, 1.47935]),
 ('O', [3.6993066100000003, 0.89439339, 1.47935])]

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
    functional="b3lyp",
    output=OUTPUT_STEM,
    use_diis=True,
    damping=0.0,
    fmixing_percent=30.0,
    max_iter=80,
    conv_tol_energy=1e-7,
    write_density=False,   # keep the generated parity grid lightweight.
)
