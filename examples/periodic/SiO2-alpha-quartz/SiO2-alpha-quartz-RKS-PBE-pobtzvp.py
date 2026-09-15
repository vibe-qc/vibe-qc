"""SiO2-alpha-quartz — RKS-PBE / pob-tzvp / Γ via the native GDF driver.

Standalone input: geometry is embedded below, so this file can be copied
to a scratch calculation directory and run without sibling helper modules.

Periodic SCF runs through ``vibeqc.run_periodic_job`` -> the native
``run_rhf_periodic_gamma_gdf`` driver (Lpq, S, T, V_ne, V_xc, SCF loop,
J/K einsums, and DIIS are vibe-qc-owned; PySCF is only an external
reference). ``fmixing_percent=30`` mirrors CRYSTAL's default FMIXING
Fock/KS matrix mixing.

Run::
    .venv/bin/python examples/periodic/SiO2-alpha-quartz/SiO2-alpha-quartz-RKS-PBE-pobtzvp.py

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

CELL_ANG = [[4.9134, 0.0, 0.0],
 [-2.456699999999999, 4.255129218954461, 0.0],
 [0.0, 0.0, 5.4052]]
ATOM_DATA = [('Si', [2.30782398, 0.0, 1.8017333333333332]),
 ('Si', [-1.1539119899999997, 1.9986341941429104, 0.0]),
 ('Si', [1.3027880100000004, 2.2564950248115507, 3.6034666666666664]),
 ('Si', [-1.1539119899999997, 1.9986341941429104, 3.603466666666667]),
 ('Si', [2.30782398, 0.0, 0.0]),
 ('Si', [1.3027880100000004, 2.2564950248115507, 1.8017333333333332]),
 ('O', [1.3759976700000003, 1.1356939885389457, 0.6437593199999999]),
 ('O', [3.2418613200000004, 0.6238019434987238, 4.247225986666666]),
 ('O', [2.752241010000001, 2.4956332869167914, 2.445492653333333]),
 ('O', [0.2955410100000006, 1.7594959320376695, 4.76144068]),
 ('O', [-1.0807023299999996, 3.1194352304155153, 1.1579740133333334]),
 ('O', [0.7851613200000007, 3.631327275455737, 2.9597073466666663])]

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
