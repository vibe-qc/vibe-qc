"""ZnO-wurtzite — RHF / pob-tzvp / Γ via the native GDF driver.

Standalone input: geometry is embedded below, so this file can be copied
to a scratch calculation directory and run without sibling helper modules.

Periodic SCF runs through ``vibeqc.run_periodic_job`` -> the native
``run_rhf_periodic_gamma_gdf`` driver (Lpq, S, T, V_ne, V_xc, SCF loop,
J/K einsums, and DIIS are vibe-qc-owned; PySCF is only an external
reference). ``fmixing_percent=30`` mirrors CRYSTAL's default FMIXING
Fock/KS matrix mixing.

Run::
    .venv/bin/python examples/periodic/ZnO-wurtzite/ZnO-wurtzite-RHF-pobtzvp.py

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

CELL_ANG = [[3.2495, 0.0, 0.0],
 [-1.6247499999999995, 2.8141495495975333, 0.0],
 [0.0, 0.0, 5.2069]]
ATOM_DATA = [('Zn', [3.561873018753658e-16, 1.8760996997316888, 0.0]),
 ('Zn', [1.6247500000000004, 0.9380498498658445, 2.60345]),
 ('O', [3.561873018753658e-16, 1.8760996997316888, 1.9890358000000001]),
 ('O', [1.6247500000000004, 0.9380498498658445, 4.5924858])]

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
    method="RHF",
    output=OUTPUT_STEM,
    use_diis=True,
    damping=0.0,
    fmixing_percent=30.0,
    max_iter=80,
    conv_tol_energy=1e-7,
    write_density=False,   # keep the generated parity grid lightweight.
)
