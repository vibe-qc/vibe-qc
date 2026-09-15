"""Empirical diagnostic for slice 2c of the native-GDF feature work.

Question: does the bare-image-sum periodic 2c metric M_PQ converge
fast enough on the v0.7.1 sweep systems to give sub-µHa SCF parity
without a modrho compensation step?

For each (orbital_basis, aux_basis) combination, this script:

  1. Builds vibe-qc PeriodicSystem + BasisSet for MgO rocksalt.
  2. Computes M_PQ at increasing image-sum cutoffs.
  3. Reports the eigenvalue spectrum of M (smallest = SPD margin),
     the matrix Frobenius norm vs cutoff (convergence proxy), and
     whether Cholesky succeeds.
  4. If Cholesky succeeds, also reports a few (large) elements of
     the resulting Lpq tensor.

If a configuration shows monotone-converging Frobenius norm AND a
positive-definite metric at the largest tested cutoff, the bare
lattice sum is good enough — modrho compensation can stay deferred
for that aux. If the Frobenius norm grows without bound OR Cholesky
fails, modrho is required to make that aux usable.

Run::

    .venv/bin/python examples/debug/scratch_lpq_native_smoke.py

The output guides the slice-3 decision: which aux bases need modrho
(work) vs which are usable as-is (ship).
"""
from __future__ import annotations

import numpy as np

import vibeqc as vq
from vibeqc.aux_basis import build_lpq_native, make_aux_basis_set

ANG2BOHR = 1.0 / 0.529177210903

# MgO rocksalt: same setup as
# examples/periodic/MgO-rocksalt/MgO-rocksalt-RHF-sto3g.py
A_ANG = 4.211
A = A_ANG * ANG2BOHR
mg_frac = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
o_frac = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
atoms = []
for fx, fy, fz in mg_frac:
    atoms.append(vq.Atom(12, [fx * A, fy * A, fz * A]))
for fx, fy, fz in o_frac:
    atoms.append(vq.Atom(8, [fx * A, fy * A, fz * A]))

system = vq.PeriodicSystem(3, np.diag([A, A, A]), atoms)
mol = system.unit_cell_molecule()
basis = vq.BasisSet(mol, "sto-3g")

print(f"MgO rocksalt / sto-3g")
print(f"  a = {A_ANG} Å = {A:.3f} bohr")
print(f"  n_atoms = {len(atoms)}, n_orb = {basis.nbasis}")

# Two aux candidates:
#   def2-svp-jkfit          - tight, smaller (no L=4 g, no extreme diffuse)
#   def2-universal-jkfit    - PySCF's auto-pick; diffuse, includes L=4 g
aux_candidates = ["def2-svp-jkfit", "def2-universal-jkfit"]

cutoffs = [4.0, 8.0, 12.0, 16.0, 20.0]   # bohr; must be strictly < min(image distance)

for aux_name in aux_candidates:
    print()
    print("=" * 74)
    print(f"  aux: {aux_name}")
    print("=" * 74)

    try:
        aux = make_aux_basis_set(mol, aux_name=aux_name)
    except Exception as exc:
        print(f"  could not load aux basis: {exc}")
        continue
    print(f"  n_aux = {aux.nbasis}")

    prev_M_norm = None
    for cut in cutoffs:
        opts = vq.LatticeSumOptions()
        opts.cutoff_bohr = cut

        M = np.asarray(vq.compute_2c_eri_lattice(aux, system, opts))
        # Force symmetry (kernel already symmetrizes; defensive)
        M = 0.5 * (M + M.T)

        eigvals = np.linalg.eigvalsh(M)
        Mnorm = float(np.linalg.norm(M))
        dM = (Mnorm - prev_M_norm) if prev_M_norm is not None else 0.0

        # SPD margin
        spd = eigvals[0] > 1e-10

        # Try Cholesky
        try:
            np.linalg.cholesky(M)
            chol_ok = True
        except np.linalg.LinAlgError:
            chol_ok = False

        print(f"  cutoff = {cut:5.1f} bohr  ||M|| = {Mnorm:11.4e}  "
              f"Δ||M|| = {dM:+.3e}  "
              f"min(eig) = {eigvals[0]:+.3e}  max(eig) = {eigvals[-1]:.3e}  "
              f"chol={'✓' if chol_ok else '✗'}")
        prev_M_norm = Mnorm

    # Use the largest cutoff for the Lpq build attempt.
    opts.cutoff_bohr = cutoffs[-1]
    print(f"\n  building Lpq at cutoff = {opts.cutoff_bohr} bohr...")
    try:
        Lpq = build_lpq_native(system, basis, aux, lat_opts=opts)
        print(f"  Lpq shape = {Lpq.shape}, "
              f"||Lpq||_max = {float(np.max(np.abs(Lpq))):.3e}, "
              f"||Lpq||_F = {float(np.linalg.norm(Lpq)):.3e}")
        # Sanity: J = Σ_L Lpq[L] · ρ_L should be symmetric for a
        # symmetric input D. Quick check on identity D.
        D_id = np.eye(basis.nbasis)
        rho = np.einsum("Lab,ab->L", Lpq, D_id)
        J = np.einsum("L,Lij->ij", rho, Lpq)
        Jasym = float(np.max(np.abs(J - J.T)))
        print(f"  J(D=I) symmetry: max|J - J.T| = {Jasym:.3e}")
    except Exception as exc:
        print(f"  Lpq build FAILED: {type(exc).__name__}: {exc}")
