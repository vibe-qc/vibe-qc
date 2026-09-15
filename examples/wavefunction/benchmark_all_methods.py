"""Compare all vibe-qc methods on H2O/STO-3G.

Prints a side-by-side energy + timing table for every method that
the `run_job` dispatcher supports: HF, KS-DFT, FCI, Selected-CI,
DMRG, v2RDM, and Transcorrelated CI.

Run:
    .venv/bin/python examples/wavefunction/benchmark_all_methods.py
"""

from __future__ import annotations

import time

import numpy as np
from vibeqc import Atom, BasisSet, Molecule
from vibeqc.runner import run_job

# H2O Pitzer/Pulay geometry
R_oh, theta = 1.8089, np.radians(104.52 / 2)
mol = Molecule(
    [
        Atom(8, [0, 0, 0]),
        Atom(1, [0, R_oh * np.sin(theta), -R_oh * np.cos(theta)]),
        Atom(1, [0, -R_oh * np.sin(theta), -R_oh * np.cos(theta)]),
    ]
)

methods = [
    ("rhf", {}, "Hartree-Fock"),
    ("rks", {"functional": "PBE"}, "RKS / PBE"),
    ("uks", {"functional": "PBE"}, "UKS / PBE"),
    ("fci", {}, "Full CI (441 dets)"),
    ("selected_ci", {}, "Selected-CI + PT2"),
    ("dmrg", {}, "DMRG (MPS)"),
    ("v2rdm", {}, "Variational 2-RDM"),
    ("transcorrelated_ci", {}, "Transcorrelated + CI"),
]

print(f"{'Method':<24s} {'E(Ha)':>16s} {'Time':>8s} {'Note':>20s}")
print("-" * 72)
for method, opts, desc in methods:
    t0 = time.perf_counter()
    result = run_job(
        mol,
        basis="sto-3g",
        method=method,
        output="/tmp/h2o_all",
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
        citations=False,
        **opts,
    )
    dt = time.perf_counter() - t0
    note = ""
    if hasattr(result, "ci_labels") and result.ci_labels:
        note = f"ndet={len(result.ci_labels)}"
    print(f"{desc:<24s} {result.energy:>16.10f} {dt:>7.2f}s {note:>20s}")
