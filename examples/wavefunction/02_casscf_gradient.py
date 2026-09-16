"""CASSCF analytic-gradient diagnostic: H2O/STO-3G CAS(4,4).

The exposed analytic value is the complete derivative of the variational
CASSCF energy and is the production force. This example prints its
components and the translational-invariance residual for inspection.

Run with:
    python examples/wavefunction/02_casscf_gradient.py
"""

from __future__ import annotations

import numpy as np
from vibeqc import Atom, Molecule
from vibeqc.runner import run_job

# Water molecule (Bohr)
h2o = Molecule(
    [
        Atom(8, [0.0, 0.0, 0.117]),
        Atom(1, [0.0, 0.757, -0.469]),
        Atom(1, [0.0, -0.757, -0.469]),
    ]
)

print("=" * 60)
print("CASSCF Analytic-Gradient Preview - H2O/STO-3G CAS(4,4)")
print("=" * 60)

# Run CASSCF + compute analytic gradient
result = run_job(
    h2o,
    basis="sto-3g",
    method="casscf",
    active_space=(4, 4),
    write_xyz_file=False,
    write_molden_file=False,
    write_population_file=False,
    citations=False,
)

print(f"\nCASSCF energy: {result.energy:.10f} Ha")
print(f"Converged:     {result.converged}")
print(f"Iterations:    {result.n_iter}")

grad = result.gradient
if grad is None:
    print("\nERROR: gradient is None! (SA-CASSCF or non-converged?)")
    exit(1)

print(f"\nAnalytic gradient (dE/dR, Hartree/bohr):")
print(f"  Shape: {grad.shape}")
for i, row in enumerate(grad):
    print(f"  atom {i}: {row[0]:12.8f} {row[1]:12.8f} {row[2]:12.8f}")

# Translational invariance check
net = np.sum(grad, axis=0)
print(
    f"\nTranslational invariance: Σ dE/dR = ({net[0]:.2e}, {net[1]:.2e}, {net[2]:.2e})"
)
print(f"  (should be ~0 for a free molecule)")

# Forces: F = -dE/dR
forces = -grad
print(f"\nForces (Ha/bohr), norm = {np.linalg.norm(forces):.4f}:")
for i, row in enumerate(forces):
    print(f"  atom {i}: {row[0]:12.8f} {row[1]:12.8f} {row[2]:12.8f}")

# Validation note
print(f"\nCorrectness note:")
print("  The analytic CASSCF gradient is the complete derivative of the")
print("  variational CASSCF energy (full-energy FD to ~2e-7 Ha/bohr).")
print("  compute_wz=True is a no-op alias of it (GitLab #516).")
print(f"  See examples/regression/casscf_gradient_fd_reproducer.py.")
print(f"\nFor full-energy finite-difference geometry optimization, see")
print(f"  03_casscf_geometry_optimization.py.")
