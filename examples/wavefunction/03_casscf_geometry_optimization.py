"""CASSCF full-energy FD geometry optimization: H2/6-31G CAS(2,2).

vibe-qc relaxes a molecular geometry on the CASSCF surface using the
central finite difference of the fully optimized CASSCF energy.
Starting from a stretched H2 at R = 1.5 bohr, the optimizer walks the
CAS(2,2) potential toward its minimum near R = 1.4 bohr.

The public analytic CASSCF gradient is currently incomplete and is not used
to steer this optimization. The same FD safety path is used for state-
specific and state-averaged runs.

For the single-point analytic gradient and its accuracy notes, see
02_casscf_gradient.py.

Run with:
    python examples/wavefunction/03_casscf_geometry_optimization.py
"""

from __future__ import annotations

import numpy as np
from vibeqc import Atom, Molecule
from vibeqc.molecular_optimize import optimize_molecule
from vibeqc.solvers import CASSCFOptions


def _bond_length(mol: Molecule) -> float:
    a = np.asarray(mol.atoms[0].xyz, dtype=float)
    b = np.asarray(mol.atoms[1].xyz, dtype=float)
    return float(np.linalg.norm(a - b))


# Stretched H2 (Bohr): start well past the equilibrium bond length.
h2 = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.5])])

print("=" * 64)
print("CASSCF(2,2)/6-31G full-energy FD geometry optimization of H2")
print("=" * 64)
print(f"start:  R = {_bond_length(h2):.4f} bohr")

result = optimize_molecule(
    h2,
    "6-31g",
    method="casscf",
    active_space=(2, 2),
    casscf_options=CASSCFOptions(),  # solver surface used at every FD point
    max_iter=20,
    conv_tol_grad=1e-2,
    progress=False,
)

final_geom = result.trajectory_frames[-1] if result.trajectory_frames else h2
print(f"final:  R = {_bond_length(final_geom):.4f} bohr")
print(f"\nfinal energy: {result.energy:.8f} Ha")
print(f"converged:    {result.converged}  (steps: {result.n_iter})")
gmax = float(np.max(np.abs(np.asarray(result.gradient, dtype=float))))
print(f"max |dE/dR|:  {gmax:.2e} Ha/bohr")

print(
    "\nFull-energy central finite differences drive this relaxation."
    "\nrun_job(method='casscf', optimize=True) uses the same safety path."
)
