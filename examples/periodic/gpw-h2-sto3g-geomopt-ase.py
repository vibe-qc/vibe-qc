"""H2 in a 12-bohr cube — geometry optimisation via ASE BFGS + GPW forces.

Demonstrates the :class:`vibeqc.VibeqcGPW` ASE Calculator driving a
geometry relaxation. ``VibeqcGPW`` exposes ``energy`` and ``forces``
through the standard ``ase.Atoms`` interface, so any optimiser in
``ase.optimize`` (BFGS / LBFGS / FIRE) can relax the structure. On the
Γ-only route the forces use the analytic-where-possible GPW gradient
(`vibeqc.periodic_gapw_gradient.compute_gradient_gpw`); pass
``use_numerical_forces=True`` to fall back to the 6N-SCF numerical path.

Reference (internal cross-validation, CLAUDE.md §10): on a vacuum-padded
neutral cell the GPW forces relax H2 toward the **molecular RHF/STO-3G
equilibrium bond length (~1.35 bohr)** — the same minimum vibe-qc's own
molecular optimiser finds; no external program is invoked. Starting from a
stretched 1.6-bohr bond, BFGS should contract it toward that minimum.

Requires the ``[ase]`` extra (``pip install -e '.[ase]'``).

Run:
    .venv/bin/python examples/periodic/gpw-h2-sto3g-geomopt-ase.py
"""

import warnings

import numpy as np
from ase import Atoms
from ase.optimize import BFGS
from ase.units import Bohr
import vibeqc as vq
from vibeqc import GAPWExperimentalWarning

warnings.simplefilter("ignore", category=GAPWExperimentalWarning)

L_bohr = 12.0  # vacuum-padded cubic cell
L_ang = L_bohr * Bohr
r0_bohr = 1.6  # stretched start (equilibrium is ~1.35 bohr)
mid = L_bohr / 2
p1 = np.array([mid - r0_bohr / 2, mid, mid]) * Bohr
p2 = np.array([mid + r0_bohr / 2, mid, mid]) * Bohr

atoms = Atoms("H2", positions=[p1, p2], pbc=True)
atoms.set_cell(np.eye(3) * L_ang)
atoms.calc = vq.VibeqcGPW(basis="sto-3g", cutoff_ha=200.0)

r_initial = atoms.get_distance(0, 1) / Bohr
BFGS(atoms, logfile=None).run(fmax=0.05, steps=12)
r_final = atoms.get_distance(0, 1) / Bohr

print(f"  initial bond = {r_initial:.3f} bohr")
print(f"  final   bond = {r_final:.3f} bohr   (RHF/STO-3G equilibrium ~1.35 bohr)")
print(f"  final energy = {atoms.get_potential_energy():.6f} eV")
assert r_final < r_initial, "bond should contract toward equilibrium"
print("  bond contracted toward the equilibrium ✓")
