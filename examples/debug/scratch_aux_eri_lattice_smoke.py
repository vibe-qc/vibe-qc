"""Unit smoke for the new periodic 2c/3c ERI kernels (slice 1 of the
native-GDF feature work).

Two sanity checks that don't yet require PySCF:

  1. At ``cutoff_bohr = 0.5`` (only the zero cell survives the cutoff —
     the next image of a 20 bohr cubic lattice is at 20 bohr), the
     periodic 2c/3c kernels MUST equal the molecular 2c/3c kernels
     elementwise.
  2. Both periodic tensors must be symmetric (M = M^T; T(P, μ, ν) =
     T(P, ν, μ)) — this is an analytic property at Γ.

Run::

    .venv/bin/python examples/debug/scratch_aux_eri_lattice_smoke.py

If either check fails, slice 1 is structurally wrong; do not proceed
to PySCF parity.
"""
from __future__ import annotations

import numpy as np

import vibeqc as vq

ANG2BOHR = 1.0 / 0.529177210903

# H2 molecule, ~0.74 Å bond, in a 20 Å cubic box. Box is large enough
# that no lattice image other than (0, 0, 0) lies inside cutoff_bohr=1.0.
A = 20.0 * ANG2BOHR
atoms = [
    vq.Atom(1, [0.0, 0.0, 0.0]),
    vq.Atom(1, [0.74 * ANG2BOHR, 0.0, 0.0]),
]
system = vq.PeriodicSystem(3, np.diag([A, A, A]), atoms)
mol = system.unit_cell_molecule()

# Use cc-pVDZ as the orbital basis (small but non-trivial: 5 BFs per H,
# includes a p-shell, so we exercise within-shell ordering) and
# cc-pVDZ-rifit as the aux basis.
basis = vq.BasisSet(mol, "cc-pvdz")
aux = vq.BasisSet(mol, "cc-pvdz-rifit")

opts = vq.LatticeSumOptions()
opts.cutoff_bohr = 1.0   # << A: only g = 0 survives.

print(f"H2 / cc-pvdz: nbasis = {basis.nbasis}, naux = {aux.nbasis}")
print(f"box: a = {A:.3f} bohr; cutoff = {opts.cutoff_bohr} bohr "
      f"(only g=0 cell)")

# --- 2c ---
M_mol = np.asarray(vq.compute_2c_eri(aux))
M_lat = np.asarray(vq.compute_2c_eri_lattice(aux, system, opts))

dM = float(np.max(np.abs(M_mol - M_lat)))
asymM = float(np.max(np.abs(M_lat - M_lat.T)))
print(f"\n2c metric (n_aux x n_aux = {M_mol.shape}):")
print(f"  max|M_mol - M_lat|   = {dM:.3e}")
print(f"  max|M_lat - M_lat.T| = {asymM:.3e}")
ok_2c = (dM < 1e-12) and (asymM < 1e-12)
print(f"  verdict: {'PASS' if ok_2c else 'FAIL'}")

# --- 3c ---
T_mol = np.asarray(vq.compute_3c_eri(basis, aux))
T_lat = np.asarray(vq.compute_3c_eri_lattice(basis, aux, system, opts))

dT = float(np.max(np.abs(T_mol - T_lat)))
asymT = float(np.max(np.abs(T_lat - T_lat.transpose(0, 2, 1))))
print(f"\n3c tensor (n_aux x n_orb x n_orb = {T_mol.shape}):")
print(f"  max|T_mol - T_lat|              = {dT:.3e}")
print(f"  max|T_lat - T_lat.swap(mu,nu)|  = {asymT:.3e}")
ok_3c = (dT < 1e-12) and (asymT < 1e-12)
print(f"  verdict: {'PASS' if ok_3c else 'FAIL'}")

if ok_2c and ok_3c:
    print("\n✓ slice-1 structural check passed — proceed to PySCF parity")
else:
    print("\n✗ slice-1 structural check FAILED — fix before parity test")
    raise SystemExit(1)
