"""H2 in a 16-bohr cube — density of states + HOMO/LUMO from a GPW SCF.

Demonstrates the v0.12 post-SCF analysis helpers in
:mod:`vibeqc.periodic_gapw_postscf` (re-exported on the top-level
``vibeqc.*`` namespace) on top of a closed-shell GPW RKS-LDA result:

* :func:`vibeqc.compute_homo_lumo` — ``(e_homo, e_lumo, gap)`` from the
  converged spectrum.
* :func:`vibeqc.compute_dos_from_result` — a Gaussian-broadened density
  of states over an energy grid.

Reference (internal cross-validation, CLAUDE.md §10): on a vacuum-padded
*neutral* cell the GPW total and spectrum reproduce vibe-qc's own
*molecular* RKS-LDA/STO-3G result to the finite-cell / finite-grid floor
(see ``docs/user_guide/gapw.md`` § "The Madelung-shift convention"); no
external program is invoked. H2/STO-3G is closed-shell with a single
occupied σ_g orbital, so the HOMO/LUMO gap is large and positive.

Run:
    .venv/bin/python examples/periodic/gpw-h2-sto3g-dos.py
"""

import warnings

import numpy as np
import vibeqc as vq
from vibeqc import GAPWExperimentalWarning

warnings.simplefilter("ignore", category=GAPWExperimentalWarning)

L = 16.0  # bohr — vacuum padding so the periodic answer matches molecular
lattice = L * np.eye(3)
atoms = [
    vq.Atom(1, [-0.7, 0.0, 0.0]),
    vq.Atom(1, [+0.7, 0.0, 0.0]),
]
system = vq.PeriodicSystem(3, lattice, atoms)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

# Standalone GPW RKS-LDA SCF — returns a GpwScfResult carrying the MO
# energies the post-SCF helpers read. NOTE: the Γ-only RKS GPW entry is
# ``run_periodic_rhf_gpw`` with a ``functional=`` argument (there is no
# separate ``run_periodic_rks_gpw``; see ``docs/user_guide/gapw.md``).
result = vq.run_periodic_rhf_gpw(
    system,
    basis,
    functional="lda",
    cutoff_ha=300.0,
    max_iter=50,
    conv_tol_energy=1e-8,
)

# NOTE: compute_homo_lumo / compute_dos_from_result need ``system=`` (to
# infer the electron count) and compute_dos_from_result returns an
# ``(e_grid, dos)`` tuple and takes an ``e_range=(lo, hi)`` window, not an
# ``e_grid=`` array.
e_homo, e_lumo, gap = vq.compute_homo_lumo(result, system=system)

e_grid, dos = vq.compute_dos_from_result(
    result, system=system, sigma=0.01, e_range=(e_homo - 0.5, e_lumo + 0.5),
)

print(f"  E_total   = {result.energy:.8f} Ha   converged={result.converged}")
print(f"  HOMO      = {e_homo:+.4f} Ha")
print(f"  LUMO      = {e_lumo:+.4f} Ha")
print(f"  gap       = {gap:.4f} Ha")
print(f"  DOS grid  = {dos.shape[0]} points; max = {dos.max():.2f} states/Ha")
print(f"  ∫DOS dE   = {np.trapezoid(dos, e_grid):.2f}  (electron-weighted, closed-shell)")
