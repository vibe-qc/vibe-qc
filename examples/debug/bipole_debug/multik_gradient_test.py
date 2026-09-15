"""Fast multi-k RHF gradient test — simpler system (H₂ box, 2×1×1 mesh)."""

from __future__ import annotations

import os
import time
import warnings

import numpy as np

os.environ.setdefault("VIBEQC_AOPAIR_FT_BACKEND", "python")

import vibeqc as vq
from vibeqc.bipole_gradient import (
    compute_bipole_gradient_fd,
    compute_bipole_gradient_rhf,
)
from vibeqc.pbc_bipole import run_pbc_bipole_rhf

# H₂ box — small, fast
lattice = np.eye(3) * 5.0
atoms = [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.5])]
sysp = vq.PeriodicSystem(3, lattice, atoms)
basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

# Small multi-k mesh: 2×1×1 (2 k-points)
kmesh = vq.monkhorst_pack(sysp, [2, 1, 1])
print(f"Multi-k: {len(list(kmesh.kpoints))} k-points, weights={list(kmesh.weights)}")

opts = vq.PeriodicSCFOptions()
opts.lattice_opts.cutoff_bohr = 6.0
opts.lattice_opts.nuclear_cutoff_bohr = 6.0
opts.max_iter = 80
opts.use_diis = True
opts.conv_tol_energy = 1e-8

print("SCF...")
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
print(f"  converged={result.converged}, n_iter={result.n_iter}, E={result.energy:.8f}")

print("Analytic gradient...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    grad_ana = compute_bipole_gradient_rhf(
        sysp, basis, result, lattice_opts=opts.lattice_opts, kmesh=kmesh
    )

print("FD gradient (2 atoms → ~13 SCFs)...")
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

diff = grad_ana - grad_fd
max_abs = float(np.max(np.abs(diff)))
print(f"\nMax |Δ| = {max_abs:.6e} Ha/bohr (target: <1e-3)")
for a in range(len(sysp.unit_cell)):
    for d in range(3):
        print(
            f"  atom {a} axis {d}: an={grad_ana[a, d]:+.6f}  fd={grad_fd[a, d]:+.6f}  Δ={diff[a, d]:+.6e}"
        )
if max_abs < 1e-3:
    print("✅ M4 MULTI-K RHF GRADIENT PASSES!")
else:
    print(f"❌ FAIL — {max_abs:.4f} Ha/bohr")
