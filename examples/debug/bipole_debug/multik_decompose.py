"""Decompose the multi-k RHF gradient error by energy term.

The 0.034 Ha/bohr error on multi-k H₂ could be from:
1. J^LR cross-term bug: per-k ρ̂ vs total ρ̂
2. Exchange: K from D_real vs per-k
3. Something else

We isolate by comparing analytic vs FD for each energy component.
"""

from __future__ import annotations

import os
import warnings

import numpy as np

os.environ.setdefault("VIBEQC_AOPAIR_FT_BACKEND", "python")

import vibeqc as vq
from vibeqc.bipole_gradient import (
    compute_bipole_gradient_fd,
    compute_bipole_gradient_rhf,
)
from vibeqc.pbc_bipole import run_pbc_bipole_rhf

lattice = np.eye(3) * 5.0
atoms = [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.5])]
sysp = vq.PeriodicSystem(3, lattice, atoms)
basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
kmesh = vq.monkhorst_pack(sysp, [2, 1, 1])

opts = vq.PeriodicSCFOptions()
opts.lattice_opts.cutoff_bohr = 6.0
opts.lattice_opts.nuclear_cutoff_bohr = 6.0
opts.max_iter = 80
opts.use_diis = True
opts.conv_tol_energy = 1e-8

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result = run_pbc_bipole_rhf(
        sysp,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        ewald_precision=1e-8,
        progress=False,
    )
print(f"Multi-k: converged={result.converged}, E={result.energy:.8f}")

# Analytic gradient
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    grad_ana = compute_bipole_gradient_rhf(
        sysp, basis, result, lattice_opts=opts.lattice_opts, kmesh=kmesh
    )

# Now also run Γ-only for comparison
kmesh_gamma = vq.monkhorst_pack(sysp, [1, 1, 1])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result_gamma = run_pbc_bipole_rhf(
        sysp,
        basis,
        kmesh_gamma,
        opts,
        use_ewald_j_split=True,
        ewald_precision=1e-8,
        progress=False,
    )
print(f"Γ-only: converged={result_gamma.converged}, E={result_gamma.energy:.8f}")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    grad_ana_gamma = compute_bipole_gradient_rhf(
        sysp, basis, result_gamma, lattice_opts=opts.lattice_opts, kmesh=kmesh_gamma
    )

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    grad_fd_gamma = compute_bipole_gradient_fd(
        sysp,
        "sto-3g",
        kmesh_gamma,
        opts,
        method="RHF",
        step_bohr=1e-3,
        use_ewald_j_split=True,
        ewald_precision=1e-8,
    )

# FD for multi-k
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    grad_fd = compute_bipole_gradient_fd(
        sysp,
        "sto-3g",
        kmesh,
        opts,
        method="RHF",
        step_bohr=1e-3,
        use_ewald_j_split=True,
        ewald_precision=1e-8,
    )

print("\n=== Comparison ===")
print(f"Γ-only:  max|Δ| = {np.max(np.abs(grad_ana_gamma - grad_fd_gamma)):.6e}")
print(f"Multi-k: max|Δ| = {np.max(np.abs(grad_ana - grad_fd)):.6e}")

print("\nMulti-k forces (z-axis only):")
for a in range(2):
    print(
        f"  atom {a}: an={grad_ana[a, 2]:+.6f}  fd={grad_fd[a, 2]:+.6f}  Δ={grad_ana[a, 2] - grad_fd[a, 2]:+.6e}"
    )

print("\nΓ-only forces (z-axis only):")
for a in range(2):
    print(
        f"  atom {a}: an={grad_ana_gamma[a, 2]:+.6f}  fd={grad_fd_gamma[a, 2]:+.6f}  Δ={grad_ana_gamma[a, 2] - grad_fd_gamma[a, 2]:+.6e}"
    )

# The key insight: multi-k J^LR needs per-k densities
print("\n=== J^LR gradient check ===")
from vibeqc.bipole_gradient import _j_long_range_ewald_gradient, _matching_ewald_options

ew_alpha = getattr(result, "ewald_alpha_bohr_inv", None)
ew_opts = _matching_ewald_options(sysp, opts.lattice_opts, ew_alpha)

# J^LR gradient with gamma_local=False (multi-k convention)
jlr_multik = _j_long_range_ewald_gradient(
    sysp, basis, result.density, float(ew_alpha), gamma_local=False
)
print(f"J^LR multi-k (gamma_local=False):")
for a in range(2):
    print(f"  atom {a}: jlr_z={jlr_multik[a, 2]:+.6f}")

# J^LR gradient with gamma_local=True (Γ mixed convention)
jlr_gamma = _j_long_range_ewald_gradient(
    sysp, basis, result.density, float(ew_alpha), gamma_local=True
)
print(f"J^LR Γ (gamma_local=True):")
for a in range(2):
    print(f"  atom {a}: jlr_z={jlr_gamma[a, 2]:+.6f}")
