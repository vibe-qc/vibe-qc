"""RHF with explicit direct (integral-driven) Fock build on H2O / 6-31G*.

The DIRECT mode replaces the in-core 4-index ERI tensor with on-the-fly
Schwarz-screened libint quartet evaluation. On this 18-basis-function
H2O/6-31G* example the dense tensor is only 0.801 MiB, so the in-core path is
usually faster. The SCFMode.DIRECT selection is the point, not the speed.

Run:
    .venv/bin/python input-h2o-rhf-direct.py

Contrast with:
    input-h2o-rhf-conventional.py  — CONVENTIONAL (in-core ERI tensor)
    input-h2o-rhf.py               — AUTO (picks CONVENTIONAL here)
"""

from __future__ import annotations

from pathlib import Path

from vibeqc import Molecule, RHFOptions, SCFMode, run_rhf

HERE = Path(__file__).parent

mol = Molecule.from_xyz(HERE.parent / "h2o.xyz")

opts = RHFOptions()
opts.scf_mode = SCFMode.DIRECT  # explicit: always integral-driven
opts.schwarz_threshold = 1e-10  # per-quartet skip bound (ORCA convention)
opts.incremental_fock = True  # ΔP caching — ~2× faster on larger systems
opts.incremental_fock_reset_freq = 8  # full rebuild every 8 iters to dam float drift
opts.conv_tol_energy = 1e-8

result = run_rhf(mol, "6-31g*", opts)

print(f"Energy: {result.energy:.10f} Ha")
print(f"Converged in {result.n_iter} iterations")
