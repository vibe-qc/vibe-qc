"""Selected CI past the determinant wall: CASCI, CASSCF, and PT2 at scale.

The dense determinant CASCI engine caps out near CAS(8,8) (a few thousand
determinants); the direct C++ engine reaches ~2 million.  The *selected*
CI engine (roadmap 25i) grows only the determinants that matter, by the
multi-root CIPSI criterion, so active spaces with billions of
determinants become routine.  This example walks the three public
surfaces:

1. ``vibeqc.solvers.selected_casci``: a ``casci``-compatible solver for
   an active space far past the dense/direct walls.
2. ``run_job(method="casscf", ...)`` with
   ``CASSCFOptions(ci_solver="selected_ci")``: orbital-optimized CASSCF
   driven by the selected-CI engine, reachable from the public API.
3. ``CASSCFOptions(pt2=SelectedCIPT2Options(...))``: the Epstein-Nesbet
   PT2 stage (Sharma et al, JCTC 13, 1595 (2017)) on the converged
   selected wavefunction, deterministic or semistochastic.

Run with:

    python examples/wavefunction/02_selected_ci_at_scale.py

Sample output (machine-dependent timings elided):

    1. selected_casci: N2 / cc-pVDZ CAS(10,16)
       full determinant count : 19,079,424   (dense cap ~2,000,000)
       selected (5,000 dets)  : E = -109.166147 Ha
    2. run_job CASSCF(selected_ci): N2 / STO-3G CAS(10,8)
       E(CASSCF) = -107.652101 Ha   (converged, selected-CI backend)
    3. + Epstein-Nesbet PT2 on the selected wavefunction
       E(PT2)      = -0.000508 Ha   (2,929 perturbers)
       E(var+PT2)  = -107.652609 Ha
"""

from __future__ import annotations

from math import comb

from vibeqc import Atom, BasisSet, Molecule
from vibeqc.runner import run_job
from vibeqc.solvers import (
    CASSCFOptions,
    SelectedCIOptions,
    SelectedCIPT2Options,
    build_hamiltonian_mo,
    get_hf_orbital_provider,
    selected_casci,
)

# N2 near equilibrium (Bohr).
n2 = Molecule([Atom(7, [0.0, 0.0, 0.0]), Atom(7, [0.0, 0.0, 2.074])])

print("=" * 64)
print("Selected CI at scale: N2")
print("=" * 64)

# ── 1. selected_casci: a CASCI past the determinant wall ──────────────
# CAS(10,16) over the cc-pVDZ orbitals: ~19 million determinants, well
# past the ~2 million direct-CI cap.  The selected engine diagonalizes
# only a grown subset; the energy is variational (an upper bound to the
# exact CASCI), tightening as target_size / pt2_threshold tighten.
print("\n1. selected_casci: N2 / cc-pVDZ CAS(10,16)")
basis = BasisSet(n2, "cc-pvdz")
C = get_hf_orbital_provider(n2, basis)
H = build_hamiltonian_mo(n2, basis, C)

n_act = 16
full_dets = comb(n_act, 5) ** 2  # 10 electrons, M_s = 0
print(f"   full determinant count : {full_dets:,}   (dense cap ~2,000,000)")

sel_opts = SelectedCIOptions(
    target_size=5000,
    max_iter=20,
    conv_tol_energy=1e-9,
    pt2_threshold=1e-9,
    max_det_per_iter=5000,
    significant_coeff=0.005,
    # Heat-bath prefilter: for large actives, drop sub-threshold
    # candidate contributions (the C++ kernel walks presorted |H|
    # lists).  0.0 keeps exact CIPSI scoring; a small positive value
    # trades a few µHa for speed.
    select_eps=0.0,
)
cas = selected_casci(
    H.h1e, H.h2e, n_active_elec=10, n_active_orb=n_act, n_core=2,
    nuclear_repulsion=H.nuclear_repulsion, ms2=0, options=sel_opts,
)
print(f"   selected ({cas.n_det:,} dets)  : E = {cas.e_total:.6f} Ha")

# ── 2. run_job CASSCF with the selected-CI backend ────────────────────
# Orbital-optimized CASSCF, driven by the selected-CI engine inside each
# macro-iteration, reachable from the public API.  STO-3G keeps the
# orbital optimization quick for the example; the backend is the same
# one that scales to 30-40 active orbitals.  orbital_step="nr" is the
# robust choice for large selected actives (see the user guide).
print("\n2. run_job CASSCF(selected_ci): N2 / STO-3G CAS(10,8)")
# A deliberately tight variational space (200 of the 3,136 CAS(8,10)
# determinants) so the PT2 stage below has real correlation to recover;
# tighten target_size for a more converged variational energy.
job_sel_opts = SelectedCIOptions(
    target_size=200, max_iter=40, conv_tol_energy=1e-12,
    pt2_threshold=1e-12, max_det_per_iter=100, significant_coeff=1e-7,
)
res = run_job(
    n2,
    basis="sto-3g",
    method="casscf",
    active_space=(8, 10),  # (n_active_orb, n_active_elec)
    output="output-n2-selected-casscf",
    casscf_options=CASSCFOptions(
        orbital_step="nr",
        ci_solver="selected_ci",
        selected_ci_options=job_sel_opts,
        # ── 3. Epstein-Nesbet PT2 on the converged selected wavefunction
        # The headline energy stays variational; the PT2 estimate
        # surfaces on SolverResult.selected_pt2 and in the .out file.
        # n_samples=0 is the deterministic mode; set n_samples>=2 (with
        # eps2/eps2_loose) for the semistochastic estimator with an
        # error bar.
        pt2=SelectedCIPT2Options(eps2=0.0),
    ),
)
print(f"   E(CASSCF) = {res.energy:.6f} Ha   "
      f"(converged={res.converged}, {res.method})")

pt2 = res.selected_pt2[0]
print("\n3. + Epstein-Nesbet PT2 on the selected wavefunction")
print(f"   E(PT2)     = {pt2['e_pt2']:.6f} Ha   "
      f"({pt2['n_perturbers']:,} perturbers)")
print(f"   E(var+PT2) = {pt2['e_total']:.6f} Ha")
print("\n   (full PT2 block, leading configurations, and natural")
print("    occupations are written to output-n2-selected-casscf.out)")

# Semistochastic variant: deterministic to a loose threshold, then a
# sampled unbiased correction down to a tight one (memory-light for the
# 30-40-orbital regime).  Returns an energy with a statistical error.
print("\n   semistochastic PT2 (eps2=1e-9, loose=1e-6, 40 samples):")
res_semi = run_job(
    n2,
    basis="sto-3g",
    method="casscf",
    active_space=(8, 10),
    output="output-n2-selected-casscf-semi",
    casscf_options=CASSCFOptions(
        orbital_step="nr",
        ci_solver="selected_ci",
        selected_ci_options=job_sel_opts,
        pt2=SelectedCIPT2Options(
            eps2=1e-9, eps2_loose=1e-6, n_samples=40, sample_size=200,
            seed=1,
        ),
    ),
)
ps = res_semi.selected_pt2[0]
print(f"   E(PT2)     = {ps['e_pt2']:.6f} +/- {ps['stderr']:.6f} Ha")

print("\n" + "=" * 64)
print("Selected CI grows only the determinants that matter, so the")
print("active space is bounded by physics (correlated orbitals), not")
print("by the factorial determinant count.")
print("=" * 64)
