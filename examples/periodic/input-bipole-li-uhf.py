"""Isolated Li atom — UHF / STO-3G / BIPOLE (molecular limit).

Open-shell UHF via the BIPOLE path. Single Li atom in a 10-bohr
cubic box, multiplicity=2 (doublet). The large vacuum pad ensures
the molecular-limit regime where the BIPOLE Ewald gauge reduces
to the standard molecular UHF energy.

Run:
    .venv/bin/python examples/periodic/input-bipole-li-uhf.py
"""

from pathlib import Path

import numpy as np
import vibeqc as vq

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem
OUT = HERE / STEM

# Li atom in a 10-bohr cubic box, multiplicity = 2
L = 10.0
lattice = np.diag([L, L, L])
atoms = [vq.Atom(3, [0.0, 0.0, 0.0])]
sysp = vq.PeriodicSystem(3, lattice, atoms, multiplicity=2)
basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

from vibeqc.pbc_bipole_uhf import run_pbc_bipole_uhf

kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])

opts = vq.PeriodicRHFOptions()
opts.lattice_opts.cutoff_bohr = 8.0
opts.lattice_opts.nuclear_cutoff_bohr = 8.0
opts.max_iter = 50
opts.use_diis = True
opts.damping = 0.0
opts.conv_tol_energy = 1e-7
opts.initial_guess = vq.InitialGuess.SAD

result = run_pbc_bipole_uhf(
    sysp,
    basis,
    kmesh,
    opts,
    use_ewald_j_split=True,
    ewald_precision=1e-6,
    progress=True,
)

with open(OUT.with_suffix(".out"), "w") as fh:
    fh.write("Li atom — UHF / STO-3G / BIPOLE (10 bohr box)\n")
    fh.write(f"  box size          = {L:.1f} bohr\n")
    fh.write(f"  multiplicity      = 2 (doublet)\n")
    fh.write("\n")
    fh.write(f"  E_total           = {result.energy:.8f} Ha\n")
    fh.write(f"  <S^2>             = {result.s_squared:.4f}\n")
    fh.write(f"  <S^2>_ideal       = {result.s_squared_ideal:.4f}\n")
    fh.write(f"  converged         = {result.converged}\n")
    fh.write(f"  SCF iterations    = {result.n_iter}\n")
    fh.write(f"  n_alpha / n_beta  = 2 / 1\n")

print(
    f"  E = {result.energy:.8f} Ha   S^2 = {result.s_squared:.4f}   "
    f"converged={result.converged}   n_iter={result.n_iter}"
)
