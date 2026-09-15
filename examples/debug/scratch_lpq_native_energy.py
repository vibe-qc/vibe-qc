"""End-to-end energy test for the native Lpq pipeline.

Builds the same MgO/sto-3g system as the v0.7.1 baseline runs through
the existing GDF driver, then OVERRIDES the Lpq tensor with the
native one (built via build_lpq_native) and reruns the SCF. The
energy delta tells us whether the bare-image-sum + SVD-truncation
approach is good enough for sub-µHa parity, or whether modrho
compensation is required.

Three configurations:

  (A) PySCF Lpq (baseline)         — what the current driver does.
  (B) Native Lpq, cutoff=4 bohr    — cutoff where M is SPD; Cholesky.
  (C) Native Lpq, cutoff=20 bohr   — falls to SVD pseudoinverse.

Run::

    .venv/bin/python examples/debug/scratch_lpq_native_energy.py

If (B) or (C) is within sub-mHa of (A), the native pipeline is good
enough for slice-3 to be deferred. If both are off by > mHa, modrho
must land before the SCF parity sweep can pass.
"""
from __future__ import annotations

import logging

import numpy as np

import vibeqc as vq
from vibeqc.aux_basis import build_lpq_native, make_aux_basis_set
from vibeqc.periodic_rhf_gdf import (
    _build_permutation_matrix,
    _lpq_tensor_from_pyscf,
    _pyscf_cell_from_vibeqc,
    run_rhf_periodic_gamma_gdf,
)

logging.basicConfig(level=logging.WARNING, format="%(message)s")

ANG2BOHR = 1.0 / 0.529177210903
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

# Aux: def2-universal-jkfit (PySCF's auto-pick for sto-3g).
aux_name = "def2-universal-jkfit"
aux = make_aux_basis_set(mol, aux_name=aux_name)

print(f"MgO rocksalt, sto-3g, aux = {aux_name}")
print(f"  n_orb = {basis.nbasis}, n_aux = {aux.nbasis}")

# --- Get PySCF's Lpq for comparison ---
print("\n[A] PySCF Lpq baseline...")
cell = _pyscf_cell_from_vibeqc(system, basis.name)
P = _build_permutation_matrix(system, basis, cell)
Lpq_pyscf, mf = _lpq_tensor_from_pyscf(cell)
print(f"  Lpq shape = {Lpq_pyscf.shape} (in PySCF/libcint AO order)")

# Permute back to vibeqc AO order for elementwise comparison with native:
#   M_pyscf (in pyscf) = P @ M_vq @ P.T
#   For Lpq with two orbital indices: Lpq_pyscf[L, μ_pyscf, ν_pyscf]
#                                   = Σ_{a, b} P[μ_p, a] · Lpq_vq[L_aux_pyscf, a, b] · P[ν_p, b]
# where L_aux_pyscf is in PySCF aux order.
# AO index permutation only — aux order may also differ but the
# permutation on the aux side cancels out in J/K (sum over L), so we
# can compare the JK matrices directly without aligning aux order.

# Native Lpq @ cutoff=4 (where Cholesky should succeed)
print("\n[B] Native Lpq @ cutoff = 4 bohr (Cholesky path)...")
opts_b = vq.LatticeSumOptions()
opts_b.cutoff_bohr = 4.0
Lpq_native_b = build_lpq_native(system, basis, aux, lat_opts=opts_b)
print(f"  Lpq shape = {Lpq_native_b.shape} (in vibeqc/libint AO order)")

# Native Lpq @ cutoff=20 (SVD-fallback path)
print("\n[C] Native Lpq @ cutoff = 20 bohr (SVD fallback)...")
opts_c = vq.LatticeSumOptions()
opts_c.cutoff_bohr = 20.0
Lpq_native_c = build_lpq_native(system, basis, aux, lat_opts=opts_c)
print(f"  Lpq shape = {Lpq_native_c.shape}")


def jk_from_lpq(Lpq, D, S, madelung):
    """Compute (J, K) via DF + Madelung correction (closed-shell, Γ).
    Same algebra as periodic_rhf_gdf:run_rhf_periodic_gamma_gdf SCF body."""
    rho = np.einsum("Lab,ab->L", Lpq, D, optimize=True)
    J = np.einsum("L,Lij->ij", rho, Lpq, optimize=True)
    temp = np.einsum("Lab,bc->Lac", Lpq, D, optimize=True)
    K_DF = np.einsum("Lac,Ldc->ad", temp, Lpq, optimize=True)
    K = K_DF + madelung * (S @ D @ S)
    return J, K


# --- Evaluate iter-1 energy with each Lpq (using SAD initial density) ---
# We do this in vibeqc AO ordering — need to permute PySCF's Lpq + ops back.
PT = P.T

# Common pieces (in vibeqc layout): SAD density + native S, T, V_ne via PySCF.
S_pyscf = np.asarray(mf.get_ovlp())
T_kin_pyscf = np.asarray(cell.pbc_intor("int1e_kin", hermi=1))
Hcore_pyscf = np.asarray(mf.get_hcore())
V_ne_pyscf = Hcore_pyscf - T_kin_pyscf
e_nuc = float(cell.energy_nuc())
from pyscf.pbc.tools import madelung as _madelung
madelung = float(_madelung(cell, np.zeros(3)))

S_vq = PT @ S_pyscf @ P
T_vq = PT @ T_kin_pyscf @ P
V_ne_vq = PT @ V_ne_pyscf @ P

D_vq = np.asarray(vq.sad_density(mol, basis))
D_vq = 0.5 * (D_vq + D_vq.T)

print("\n--- iter-1 energies (SAD density, no SCF) ---")
for label, Lpq, in_vq_layout in [
    ("[A] PySCF Lpq",    Lpq_pyscf,   False),
    ("[B] Native cut=4", Lpq_native_b, True),
    ("[C] Native cut=20", Lpq_native_c, True),
]:
    if in_vq_layout:
        J, K = jk_from_lpq(Lpq, D_vq, S_vq, madelung)
        F = T_vq + V_ne_vq + J - 0.5 * K
        E = 0.5 * float(np.einsum("ij,ij->", D_vq, T_vq + V_ne_vq + F)) + e_nuc
    else:
        D_pyscf = P @ D_vq @ P.T
        J, K = jk_from_lpq(Lpq, D_pyscf, S_pyscf, madelung)
        F = T_kin_pyscf + V_ne_pyscf + J - 0.5 * K
        E = 0.5 * float(np.einsum("ij,ij->", D_pyscf, T_kin_pyscf + V_ne_pyscf + F)) + e_nuc
    print(f"  {label:24s}  E_iter1 = {E:14.6f} Ha")

# The iter-1 numbers above are already definitive:
#   [A] PySCF Lpq baseline       E_iter1 ≈ -1084 Ha (close to converged -1085).
#   [B] Native cut=4 (Cholesky)  E_iter1 ≈ -270  Ha (off by ~800 Ha).
#   [C] Native cut=20 (SVD)      E_iter1 ≈ -12,653 Ha (catastrophic).
#
# Verdict: the bare image-summed 2c metric on def2-universal-jkfit
# (PySCF's auto-pick for sto-3g) does NOT converge to a usable Lpq
# without modrho compensation. The Cholesky-ok cutoff (=4 bohr) captures
# only short-range aux interactions; the SVD-fallback at larger cutoff
# is contaminated by the divergent monopole-monopole tail.
#
# Slice-3 conclusion: modrho compensation is mandatory, not optional.
# The next session needs to either port pyscf.pbc.df.df.make_modrho_basis
# (which requires a way to construct a vibe-qc BasisSet with custom
# contraction coefficients) OR move to a different convergence
# treatment (Ewald-split (P|Q)). Both are substantial follow-ups.
