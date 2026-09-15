"""H2O — CASSCF(4,4) reference + internally-contracted CASPT2.

The standard multireference workflow: optimize the orbitals for the
active space (CASSCF), then recover dynamic correlation perturbatively
on that reference (CASPT2).  Passing ``casscf_options=`` to
``run_job(method="caspt2")`` switches the PT2 reference from the
default CASCI-on-HF-orbitals to the orbital-optimized CASSCF — the
method label gains a ``_casscf`` suffix.

CASSCF is non-convex: the optimizer converges to a stationary point in
the basin of the starting (HF) orbitals.  H2O CAS(4,4) is famously
basin-rich — different programs can land on different stationary
points (see ``tests/test_solvers_mrpt_parity.py``
``TestCASSCFReferencedCASPT2`` for the recorded OpenMolcas / PySCF /
vibe-qc comparison); on unique-minimum systems (H2, N2) vibe-qc matches
OpenMolcas ``&RASSCF`` + ``&CASPT2`` to ≤5e-6 Ha.

Run:
    .venv/bin/python examples/molecular/input-h2o-casscf-caspt2.py

Outputs (next to this script):
    input-h2o-casscf-caspt2.out      — banner + solver trace
    input-h2o-casscf-caspt2.system   — provenance manifest
"""

from pathlib import Path

import vibeqc as vq
from vibeqc.solvers import CASSCFOptions

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem  # → "input-h2o-casscf-caspt2"

# H2O (bohr) — the geometry used across the multireference test suite.
mol = vq.Molecule(
    [
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 1.43, -0.93]),
        vq.Atom(1, [0.0, -1.43, -0.93]),
    ]
)

# 1. CASSCF(4,4): 4 electrons in 4 orbitals, 3 doubly-occupied core
#    orbitals below — orbital-optimized reference.
casscf = vq.run_job(
    mol,
    basis="sto-3g",
    method="casscf",
    active_space=(4, 4),
    output=HERE / (STEM + "-ref"),
)

# 2. CASPT2 on the CASSCF reference (casscf_options= switches the
#    reference; an empty CASSCFOptions() uses single-state defaults).
caspt2 = vq.run_job(
    mol,
    basis="sto-3g",
    method="caspt2",
    active_space=(4, 4),
    casscf_options=CASSCFOptions(),
    output=HERE / STEM,
)

e_pt2 = caspt2.energy - casscf.energy

print(f"  E(CASSCF(4,4)/STO-3G)   = {casscf.energy:16.10f} Ha   [{casscf.method}]")
print(f"  E(CASPT2 on CASSCF)     = {caspt2.energy:16.10f} Ha   [{caspt2.method}]")
print(f"  E2 (dynamic corr.)      = {e_pt2:16.10f} Ha")

# Expected output (2026-06-10, vibe-qc 0.12.0.dev0):
#   E(CASSCF(4,4)/STO-3G)   =   -74.9793183480 Ha   [casscf(4e,4o)]
#   E(CASPT2 on CASSCF)     =   -74.9822007943 Ha   [caspt2(4e,4o)_casscf]
#   E2 (dynamic corr.)      =    -0.0028824464 Ha
