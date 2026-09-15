"""MgO rocksalt — RHF / sto-3g / Γ via the new GDF driver.

Conventional cubic cell, a = 4.211 Å, 4 Mg + 4 O atoms, 80 electrons.
Plain Roothaan converges in 14 iters; DIIS in 9. We keep DIIS on for
the canonical example.

Run::
    .venv/bin/python examples/periodic/MgO-rocksalt-RHF-sto3g/MgO-rocksalt-RHF-sto3g.py

Produces (sibling files, sharing the script's stem):
    {stem}.out      — text log (banner, system, basis, SCF trace, energies)
    {stem}.system   — runtime manifest (CPU/OS/libs, wall-time)
    {stem}.molden   — Γ-point MOs (libint AO ordering)
    {stem}.xsf      — SCF density on a primitive-cell grid
"""
from pathlib import Path

import numpy as np

from vibeqc import (
    Atom, BasisSet, PeriodicSystem,
    run_periodic_job,
)

ANG2BOHR = 1.0 / 0.529177210903
A_ANG = 4.211

# Output stem = this script's path with the .py extension stripped.
# All generated files (.out, .system, .molden, .xsf) share that stem.
OUTPUT_STEM = Path(__file__).resolve().with_suffix("")

a = A_ANG * ANG2BOHR
mg_frac = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
o_frac = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
atoms = []
for fx, fy, fz in mg_frac:
    atoms.append(Atom(12, [fx * a, fy * a, fz * a]))
for fx, fy, fz in o_frac:
    atoms.append(Atom(8, [fx * a, fy * a, fz * a]))

system = PeriodicSystem(3, np.diag([a, a, a]), atoms)
basis = BasisSet(system.unit_cell_molecule(), "sto-3g")

run_periodic_job(
    system, basis,
    method="RHF",
    output=OUTPUT_STEM,
    use_diis=True,
    damping=0.0,
    max_iter=80,
    conv_tol_energy=1e-7,
    write_density=True,
    density_spacing_bohr=0.2,
)
