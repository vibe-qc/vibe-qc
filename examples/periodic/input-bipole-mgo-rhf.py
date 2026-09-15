"""MgO primitive — RHF / STO-3G / Γ-only BIPOLE.

CRYSTAL-gauge Ewald J-split via the BIPOLE path. Single Mg + O in
the FCC primitive cell, lattice constant a = 4.211 A (experimental).
The BIPOLE driver uses a shared Ewald alpha across V_ne/E_nn/J^LR.

Run:
    .venv/bin/python examples/periodic/input-bipole-mgo-rhf.py

Compare against:
    - CRYSTAL14  SHRINK 8 8:  E_total ≈ -271.218 Ha (reference)
    - GDF (EWALD_3D):        see input-mgo-rocksalt-rhf.py
    - BIPOLE RKS:            input-bipole-mgo-rks-pbe.py
"""

from pathlib import Path

import numpy as np
import vibeqc as vq

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem
OUT = HERE / STEM

ANG2BOHR = 1.0 / 0.529177210903

# MgO FCC primitive — 1 Mg + 1 O
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

# Use the BIPOLE driver directly
from vibeqc.pbc_bipole import run_pbc_bipole_rhf

kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])

opts = vq.PeriodicRHFOptions()
opts.lattice_opts.cutoff_bohr = 10.0
opts.lattice_opts.nuclear_cutoff_bohr = 10.0
opts.max_iter = 50
opts.use_diis = True
opts.damping = 0.0
opts.conv_tol_energy = 1e-7
opts.initial_guess = vq.InitialGuess.SAD

result = run_pbc_bipole_rhf(
    sysp,
    basis,
    kmesh,
    opts,
    use_ewald_j_split=True,
    ewald_precision=1e-6,
    progress=True,
    # The supported route is exact and defaults to
    # use_multipole_far_field=False. Explicit True raises before setup;
    # multipole_l_max is inactive on the exact route.
)

with open(OUT.with_suffix(".out"), "w") as fh:
    fh.write("MgO FCC primitive — RHF / STO-3G / Gamma-only BIPOLE\n")
    fh.write(f"  lattice constant = 4.21 A\n")
    fh.write(f"  atoms per cell   = 1 Mg + 1 O = 2\n")
    fh.write(f"  basis            = STO-3G ({basis.nbasis} BFs / cell)\n")
    fh.write(f"  J/K method       = BIPOLE (CRYSTAL-gauge Ewald J-split)\n")
    fh.write("\n")
    fh.write(f"  E_total          = {result.energy:.8f} Ha\n")
    fh.write(f"  converged        = {result.converged}\n")
    fh.write(f"  SCF iterations   = {result.n_iter}\n")
    if result.ewald_alpha_bohr_inv is not None:
        fh.write(f"  Ewald alpha      = {result.ewald_alpha_bohr_inv:.4f} bohr^-1\n")
    fh.write("\n")
    fh.write("  Energy components (final iter):\n")
    comp = result.energy_components[-1]
    fh.write(f"    E_kinetic          = {comp.e_kinetic:+.8f} Ha\n")
    fh.write(f"    E_nuclear_attract   = {comp.e_nuclear_attraction:+.8f} Ha\n")
    fh.write(f"    E_two_electron      = {comp.e_two_electron:+.8f} Ha\n")
    fh.write(f"    E_nuclear_repulsion = {comp.e_nuclear_repulsion:+.8f} Ha\n")
    if comp.e_j_short_range is not None:
        fh.write(f"    E_J_short_range     = {comp.e_j_short_range:+.8f} Ha\n")
    if comp.e_j_long_range is not None:
        fh.write(f"    E_J_long_range      = {comp.e_j_long_range:+.8f} Ha\n")
    if comp.e_exchange is not None:
        fh.write(f"    E_exchange          = {comp.e_exchange:+.8f} Ha\n")

print(
    f"  E = {result.energy:.8f} Ha   converged={result.converged}   "
    f"n_iter={result.n_iter}"
)
