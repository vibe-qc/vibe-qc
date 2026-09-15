"""MgO primitive - BIPOLE workflow plus analytic-gradient diagnostic.

Demonstrates the supported fixed-cell BIPOLE workflow:
1. Single-point SCF with a research-preview analytic-gradient diagnostic
2. Atomic position relaxation
3. Energy and gradient diagnostics

Run:
    .venv/bin/python examples/periodic/input-bipole-workflow-mgo.py
"""

from pathlib import Path

import numpy as np
import vibeqc as vq

HERE = Path(__file__).resolve().parent
ANG2BOHR = 1.0 / 0.529177210903

# ---- Build MgO primitive cell -------------------------------------------
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
kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])
basis_name = "sto-3g"

print("=" * 60)
print("MgO FCC primitive — BIPOLE workflow")
print("=" * 60)

# ---- 1. Single-point SCF + gradient -------------------------------------
print("\n[1] Single-point BIPOLE RHF + research-preview analytic gradient")
from vibeqc.bipole_gradient import compute_bipole_gradient_rhf
from vibeqc.pbc_bipole import run_pbc_bipole_rhf

opts = vq.PeriodicRHFOptions()
opts.lattice_opts.cutoff_bohr = 8.0
opts.lattice_opts.nuclear_cutoff_bohr = 8.0
opts.max_iter = 30
opts.use_diis = True
opts.conv_tol_energy = 1e-7

basis = vq.BasisSet(sysp.unit_cell_molecule(), basis_name)
result = run_pbc_bipole_rhf(
    sysp,
    basis,
    kmesh,
    opts,
    use_ewald_j_split=True,
    ewald_precision=1e-8,
    progress=True,
)

grad = compute_bipole_gradient_rhf(
    sysp,
    basis,
    result,
    lattice_opts=opts.lattice_opts,
)

print(f"\n  E_total  = {result.energy:+.8f} Ha")
print(f"  n_iter   = {result.n_iter}")
print(f"  max|grad| = {np.max(np.abs(grad)):.4e} Ha/bohr")
print(f"  rms grad  = {np.sqrt(np.mean(grad**2)):.4e} Ha/bohr")
print(f"  Gradient (Ha/bohr):")
for i, atom in enumerate(sysp.unit_cell):
    print(
        f"    atom {i} (Z={atom.Z}): "
        f"Fx={grad[i, 0]:+.6f} Fy={grad[i, 1]:+.6f} Fz={grad[i, 2]:+.6f}"
    )

# ---- 2. Atomic relaxation -----------------------------------------------
print("\n[2] Atomic position relaxation (fixed lattice)")
from vibeqc.bipole_optimize import relax_atoms

atom_result = relax_atoms(
    sysp,
    basis_name,
    kmesh,
    method="RHF",
    max_iter=15,
    conv_tol_grad=1e-4,
    cutoff_bohr=8.0,
    ewald_precision=1e-8,
)
print(f"  E_final = {atom_result.energy:+.8f} Ha")

# Variable-cell BIPOLE optimization intentionally fails closed. Its former
# strain convention and coupled atom/cell convergence were not certified on
# one terminal geometry. Use a converged manual equation-of-state scan or a
# route with a validated production stress when the lattice must change.
