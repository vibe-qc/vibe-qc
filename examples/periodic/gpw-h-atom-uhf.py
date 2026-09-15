"""Single H atom in a 12-bohr cube — open-shell UHF via the GPW route.

Demonstrates :func:`vibeqc.run_periodic_uhf_gpw`: a Γ-only spin-polarised
GPW SCF taking explicit per-spin occupations (``n_alpha`` / ``n_beta``)
and returning a ``GpwUhfScfResult`` with per-spin densities, MO
coefficients, and MO energies. It reduces bit-for-bit to the closed-shell
``run_periodic_rhf_gpw`` when ``n_alpha == n_beta``.

Reference (internal cross-validation, CLAUDE.md §10): a single H atom is
the spin-polarised 1s reference — `(n_alpha, n_beta) = (1, 0)`. On a
vacuum-padded cell the periodic answer is the atomic UHF limit, the same
number vibe-qc's molecular UHF gives; no external program is invoked.

Run:
    .venv/bin/python examples/periodic/gpw-h-atom-uhf.py
"""

import warnings

import numpy as np
import vibeqc as vq
from vibeqc import GAPWExperimentalWarning

warnings.simplefilter("ignore", category=GAPWExperimentalWarning)

L = 12.0  # bohr — vacuum-padded cubic cell
mid = L / 2
atom = vq.Atom(1, [mid, mid, mid])
system = vq.PeriodicSystem(3, L * np.eye(3), [atom])
# An odd electron count needs multiplicity 2 — `Molecule(..., 0, 1)` (as
# older tutorial drafts showed) and `system.unit_cell_molecule()` both raise
# "n_electrons and multiplicity are inconsistent".
basis = vq.BasisSet(vq.Molecule([atom], 0, 2), "sto-3g")

result = vq.run_periodic_uhf_gpw(
    system, basis,
    n_alpha=1, n_beta=0,
    cutoff_ha=200.0,
    max_iter=60,
    conv_tol_energy=1e-8,
)

print(f"  E (UHF/STO-3G) = {result.energy:.8f} Ha")
print(f"  n_alpha        = {result.n_alpha}")
print(f"  n_beta         = {result.n_beta}")
print(f"  converged      = {result.converged}  ({result.n_iter} iter)")
s2 = getattr(result, "spin_squared", None)
if s2 is not None:
    print(f"  <S^2>          = {s2:.4f}  (exact doublet = 0.75)")
assert result.converged, "UHF SCF did not converge"
