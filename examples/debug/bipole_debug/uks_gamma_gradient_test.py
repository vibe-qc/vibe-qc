"""Quick test of UKS Γ gradient on symmetric cell (M3).

A triplet H₂ in a box, SVWN LDA. The UKS gradient should match FD
on a symmetric cell where no orbital relaxation is needed.

Usage:
    VIBEQC_AOPAIR_FT_BACKEND=python python \
        examples/debug/bipole_debug/uks_gamma_gradient_test.py
"""

from __future__ import annotations

import os
import time
import warnings

import numpy as np

os.environ.setdefault("VIBEQC_AOPAIR_FT_BACKEND", "python")

import vibeqc as vq
from vibeqc.bipole_gradient import (
    compute_bipole_gradient_fd,
    compute_bipole_gradient_uks,
)
from vibeqc.pbc_bipole_uks import PeriodicKSOptions, run_pbc_bipole_uks

lattice = np.eye(3) * 5.0
atoms = [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.5])]
sysp = vq.PeriodicSystem(3, lattice, atoms)
sysp.multiplicity = 3  # triplet: α=2, β=0
basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])

opts = PeriodicKSOptions()
opts.lattice_opts.cutoff_bohr = 6.0
opts.lattice_opts.nuclear_cutoff_bohr = 6.0
opts.max_iter = 100
opts.use_diis = True
opts.conv_tol_energy = 1e-8
opts.functional = "svwn"
opts.initial_guess = vq.InitialGuess.SAD

print("UKS Γ gradient test — triplet H₂ SVWN")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result = run_pbc_bipole_uks(
        sysp,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        ewald_precision=1e-8,
        progress=False,
    )
print(
    f"SCF: converged={result.converged}, n_iter={result.n_iter}, E={result.energy:.8f}"
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    grad_ana = compute_bipole_gradient_uks(
        sysp, basis, result, lattice_opts=opts.lattice_opts
    )

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    grad_fd = compute_bipole_gradient_fd(
        sysp,
        "sto-3g",
        kmesh,
        opts,
        method="UKS",
        functional="svwn",
        step_bohr=1e-3,
        use_ewald_j_split=True,
        ewald_precision=1e-8,
    )

diff = grad_ana - grad_fd
max_abs = float(np.max(np.abs(diff)))
print(f"Max |Δ| = {max_abs:.6e} Ha/bohr (target: <1e-3)")
for a in range(len(sysp.unit_cell)):
    for d in range(3):
        print(
            f"  atom {a} axis {d}: an={grad_ana[a, d]:+.6f}  fd={grad_fd[a, d]:+.6f}  Δ={diff[a, d]:+.6e}"
        )
if max_abs < 1e-3:
    print("✅ PASS")
else:
    print("❌ FAIL")
