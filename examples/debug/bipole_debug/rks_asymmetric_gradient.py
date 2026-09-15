"""RKS Γ gradient on asymmetric cell — quantifies the remaining error.

The handover says the KS Bloch-CPHF relaxation is missing, giving
~0.5-1 Ha/bohr error on asymmetric crystals. Let's measure.
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
    compute_bipole_gradient_rks,
)
from vibeqc.pbc_bipole_rks import PeriodicKSOptions, run_pbc_bipole_rks


def build_beh2_asymmetric(a_bohr=6.0):
    """BeH₂ asymmetric cell: Be at 0,0, H at 0,0,2.5 and 0,0,4.2
    (different bond lengths)."""
    lattice = np.eye(3) * a_bohr
    atoms = [
        vq.Atom(4, [0, 0, 0]),
        vq.Atom(1, [0, 0, 2.5]),
        vq.Atom(1, [0, 0, 4.2]),
    ]
    return vq.PeriodicSystem(3, lattice, atoms)


print("RKS Γ gradient — asymmetric BeH₂")
sysp = build_beh2_asymmetric(a_bohr=6.0)
basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])

opts = PeriodicKSOptions()
opts.lattice_opts.cutoff_bohr = 7.0
opts.lattice_opts.nuclear_cutoff_bohr = 7.0
opts.max_iter = 100
opts.use_diis = True
opts.conv_tol_energy = 1e-8
opts.conv_tol_grad = 1e-5
opts.functional = "svwn"
opts.initial_guess = vq.InitialGuess.SAD

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result = run_pbc_bipole_rks(
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
    grad_analytic = compute_bipole_gradient_rks(
        sysp, basis, result, lattice_opts=opts.lattice_opts
    )

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    grad_fd = compute_bipole_gradient_fd(
        sysp,
        "sto-3g",
        kmesh,
        opts,
        method="RKS",
        functional="svwn",
        step_bohr=1e-3,
        use_ewald_j_split=True,
        ewald_precision=1e-8,
    )

diff = grad_analytic - grad_fd
max_abs = float(np.max(np.abs(diff)))
rms = float(np.sqrt(np.mean(diff**2)))
print(f"\nMax |Δ| = {max_abs:.6e} Ha/bohr (target: <1e-3)")
print(f"RMS    = {rms:.6e} Ha/bohr")
print(f"\nComponent breakdown:")
for a in range(len(sysp.unit_cell)):
    for d in range(3):
        print(
            f"  atom {a} axis {d}: an={grad_analytic[a, d]:+.6f}  fd={grad_fd[a, d]:+.6f}  Δ={diff[a, d]:+.6e}"
        )
