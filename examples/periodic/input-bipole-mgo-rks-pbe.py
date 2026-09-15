"""MgO primitive — RKS PBE / STO-3G / Γ-only BIPOLE.

DFT counterpart of input-bipole-mgo-rhf.py. Uses the BIPOLE RKS
driver with PBE functional.

Run:
    .venv/bin/python examples/periodic/input-bipole-mgo-rks-pbe.py
"""

from pathlib import Path

import numpy as np
import vibeqc as vq

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem
OUT = HERE / STEM

ANG2BOHR = 1.0 / 0.529177210903

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

from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks

kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])

opts = vq.PeriodicKSOptions()
opts.functional = "pbe"
opts.lattice_opts.cutoff_bohr = 10.0
opts.lattice_opts.nuclear_cutoff_bohr = 10.0
opts.max_iter = 50
opts.use_diis = True
opts.damping = 0.0
opts.conv_tol_energy = 1e-7
opts.initial_guess = vq.InitialGuess.SAD

result = run_pbc_bipole_rks(
    sysp,
    basis,
    kmesh,
    opts,
    functional="pbe",
    use_ewald_j_split=True,
    ewald_precision=1e-6,
    progress=True,
)

with open(OUT.with_suffix(".out"), "w") as fh:
    fh.write("MgO FCC primitive — RKS PBE / STO-3G / Gamma-only BIPOLE\n")
    fh.write(f"  lattice constant = 4.21 A\n")
    fh.write(f"  atoms per cell   = 1 Mg + 1 O = 2\n")
    fh.write(f"  functional       = PBE\n")
    fh.write("\n")
    fh.write(f"  E_total          = {result.energy:.8f} Ha\n")
    fh.write(f"  E_xc             = {result.e_xc:+.8f} Ha\n")
    fh.write(f"  E_coulomb        = {result.e_coulomb:+.8f} Ha\n")
    fh.write(f"  converged        = {result.converged}\n")
    fh.write(f"  SCF iterations   = {result.n_iter}\n")

print(
    f"  E = {result.energy:.8f} Ha   converged={result.converged}   "
    f"n_iter={result.n_iter}"
)
