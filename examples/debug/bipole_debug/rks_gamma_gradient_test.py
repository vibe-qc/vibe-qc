"""Check RKS Γ analytic gradient vs FD — the M2 gap.

Quantifies the current error of the RKS analytic gradient at Γ
on a simple symmetric cell (H₂ box). Uses SVWN (LDA) first, then
optionally PBE (GGA).

Usage:
    VIBEQC_AOPAIR_FT_BACKEND=python python \
        examples/debug/bipole_debug/rks_gamma_gradient_test.py
"""

from __future__ import annotations

import os
import sys
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


def build_h2_box(a_bohr=5.0):
    """H₂ in a cubic box, symmetric."""
    lattice = np.eye(3) * a_bohr
    atoms = [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.5])]
    return vq.PeriodicSystem(3, lattice, atoms)


def run_and_check(func_name, system, basis_name="sto-3g", max_iter=100, verbose=True):
    print(f"\n{'=' * 60}")
    print(f"  RKS Γ gradient test — {func_name}")
    print(f"{'=' * 60}")
    basis = vq.BasisSet(system.unit_cell_molecule(), basis_name)
    kmesh = vq.monkhorst_pack(system, [1, 1, 1])

    opts = PeriodicKSOptions()
    opts.lattice_opts.cutoff_bohr = 6.0
    opts.lattice_opts.nuclear_cutoff_bohr = 6.0
    opts.max_iter = max_iter
    opts.use_diis = True
    opts.conv_tol_energy = 1e-8
    opts.conv_tol_grad = 1e-5
    opts.functional = func_name
    opts.initial_guess = vq.InitialGuess.SAD

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_pbc_bipole_rks(
            system,
            basis,
            kmesh,
            opts,
            use_ewald_j_split=True,
            ewald_precision=1e-8,
            progress=False,
        )
    t_scf = time.time() - t0
    print(
        f"  SCF: converged={result.converged}, "
        f"n_iter={result.n_iter}, E={result.energy:.8f}, "
        f"time={t_scf:.1f}s"
    )

    if not result.converged:
        print(f"  SKIP — SCF not converged")
        return

    # RKS analytic gradient
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grad_analytic = compute_bipole_gradient_rks(
            system, basis, result, lattice_opts=opts.lattice_opts
        )
    t_ana = time.time() - t0
    print(f"  Analytic gradient: time={t_ana:.1f}s")

    # FD reference
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grad_fd = compute_bipole_gradient_fd(
            system,
            basis_name,
            kmesh,
            opts,
            method="RKS",
            functional=func_name,
            step_bohr=1e-3,
            use_ewald_j_split=True,
            ewald_precision=1e-8,
        )
    t_fd = time.time() - t0
    print(f"  FD gradient: time={t_fd:.1f}s")

    diff = grad_analytic - grad_fd
    max_abs = float(np.max(np.abs(diff)))
    rms = float(np.sqrt(np.mean(diff**2)))
    print(f"\n  Max |Δ| = {max_abs:.6e} Ha/bohr")
    print(f"  RMS    = {rms:.6e} Ha/bohr")
    print(f"  Translation ΣF = {np.sum(grad_analytic, axis=0)}")
    print(f"  FD ΣF = {np.sum(grad_fd, axis=0)}")

    if max_abs < 1e-3:
        print(f"  ✅ PASS — RKS Γ gradient matches FD to <1e-3 Ha/bohr")
    else:
        print(f"  ❌ FAIL — RKS Γ gradient error {max_abs:.4f} > 1e-3")

    # Component-wise
    print(f"\n  Component breakdown:")
    for a in range(len(system.unit_cell)):
        for d in range(3):
            print(
                f"    atom {a} axis {d}: an={grad_analytic[a, d]:+.6f}  "
                f"fd={grad_fd[a, d]:+.6f}  Δ={diff[a, d]:+.6e}"
            )


if __name__ == "__main__":
    print("RKS Γ analytic gradient test (M2)")
    sysp = build_h2_box(a_bohr=5.0)
    print(f"System: H₂ box a=5.0 bohr")
    # LDA first (XC kernel exact for LDA)
    run_and_check("svwn", sysp)
    # Then GGA if desired
    # run_and_check("pbe", sysp)
