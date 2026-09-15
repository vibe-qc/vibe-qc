"""H2 in a 12-bohr cube — Fermi-Dirac smearing on the multi-k GPW route.

Demonstrates :func:`vibeqc.run_periodic_rks_gpw_multi_k` with
``smearing_temperature=`` (in Hartree): occupations switch from sharp
Aufbau to Fermi-Dirac, and the breakdown gains the variationally-correct
electronic free energy ``F = E − T·S`` plus the smearing entropy. On a
*gapped* molecular reference (vacuum-padded H2) the entropy stays ~0 and
``F ≈ E`` — the diagnostic that you're in the molecular limit.

Reference (internal cross-validation, CLAUDE.md §10): vacuum-padded H2 is
gapped, so smearing leaves the energy essentially unchanged; no external
program is invoked.

NOTE: the multi-k entry takes a **mesh object** from
``vq.monkhorst_pack(system, [..])`` — NOT a bare ``kmesh=(2, 2, 2)`` tuple
(the historical GPW-AUDIT-004 docs issue; see ``docs/user_guide/gapw.md``).

Run:
    .venv/bin/python examples/periodic/gpw-h2-sto3g-smearing.py
"""

import warnings

import numpy as np
import vibeqc as vq
from vibeqc import GAPWExperimentalWarning

warnings.simplefilter("ignore", category=GAPWExperimentalWarning)

L = 12.0  # bohr — vacuum-padded
lattice = L * np.eye(3)
atoms = [vq.Atom(1, [-0.7, 0.0, 0.0]), vq.Atom(1, [0.7, 0.0, 0.0])]
system = vq.PeriodicSystem(3, lattice, atoms)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
kmesh = vq.monkhorst_pack(system, [2, 2, 2])

result = vq.run_periodic_rks_gpw_multi_k(
    system, basis, kmesh,
    functional="lda",
    cutoff_ha=300.0,
    smearing_temperature=0.005,  # ~1580 K
    max_iter=80,
    conv_tol_energy=1e-7,
    quiet=True,
)

print(f"  E (internal)        = {result.energy:.8f} Ha   converged={result.converged}")
F = getattr(result, "free_energy", None)
S = getattr(result, "smearing_entropy", None)
T = getattr(result, "smearing_temperature", None)
if F is not None:
    print(f"  F = E - T*S         = {F:.8f} Ha")
if T is not None:
    print(f"  smearing T          = {T:.4f} Ha")
if S is not None:
    print(f"  smearing entropy S  = {S:.3e}  (gapped H2 → ~0)")
assert result.converged, "smeared multi-k SCF did not converge"
