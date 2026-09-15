"""Explicit W^z/D^z/Γ^z gradient correction for PT2 Z-vectors.

Replaces the z^T.g^R shortcut with the correct CP-MCSCF construction:
  1. Build kappa from z (antisymmetric orbital rotation matrix)
  2. D^z = [kappa, D_eff]  (one-index transformed 1-RDM)
  3. Gamma^z = [kappa, Gamma_eff]  (one-index transformed 2-RDM)
  4. W^z_MO = [F_eff, kappa]  (Fock commutator)
  5. AO transform + contract with derivative integrals

This is the same construction as Helgaker-Jorgensen-Olsen Eq 12.5.19-27
and the OpenMolcas analytic Lagrangian (clagd.F90 + olagfinal.F90 family),
but uses the PT2 effective densities (D + DeltaD_PT2, Gamma + DeltaGamma_PT2,
F_eff) instead of the bare CASSCF RDMs.
"""

from __future__ import annotations

import numpy as np


def compute_pt2_explicit_wz_gradient(
    mol,
    basis,
    C_mo: np.ndarray,
    z_orb: np.ndarray,
    D_eff: np.ndarray,  # effective 1-RDM in MO basis (nmo, nmo)
    Gamma_eff: np.ndarray,  # effective 2-RDM in MO basis (nmo, nmo, nmo, nmo)
    F_eff: np.ndarray,  # effective generalized Fock (nmo, nmo)
    n_core: int,
    n_act: int,
    nmo: int,
    pairs: list,
) -> np.ndarray:
    """Compute the W^z/D^z/Γ^z gradient correction explicitly.

    This is the CORRECT CP-MCSCF correction for the PT2 Z-vector,
    replacing the z^T.g^R shortcut which is incompatible with PT2
    Hessians and gives approximate results (~95-97%).

    Returns (n_atoms, 3) gradient correction in Hartree/bohr.
    """
    from vibeqc._vibeqc_core import (
        compute_overlap,
        one_electron_gradient_contribution,
        overlap_gradient_contribution,
        two_electron_gradient_casscf,
    )
    from vibeqc.gradient._casscf import (
        _build_kappa_from_z,
        _compute_one_index_transformed_dm,
        _compute_one_index_transformed_gamma,
        _compute_wz_mo,
        _transform_4index_mo_to_ao_flat,
    )

    n_atoms = len(mol.atoms)
    nb = C_mo.shape[0]

    # 1. Build kappa from z
    kappa = _build_kappa_from_z(z_orb, pairs, nmo)

    # 2. One-index transformed densities
    D_z = _compute_one_index_transformed_dm(kappa, D_eff)
    Gamma_z = _compute_one_index_transformed_gamma(kappa, Gamma_eff)

    # 3. W^z_MO = [F_eff, kappa]
    Wz_mo = _compute_wz_mo(F_eff, kappa)

    # 4. AO transform
    D_z_ao = C_mo @ D_z @ C_mo.T
    Wz_ao = C_mo @ Wz_mo @ C_mo.T

    # Gamma_z MO->AO: 4-index transform, then symmetrize
    gamma_z_flat = _transform_4index_mo_to_ao_flat(Gamma_z, C_mo, nmo)
    gamma_z_flat = _symmetrize_8fold_flat(gamma_z_flat, nb)

    # Energy convention: factor 0.5 from E = 0.5 * Gamma * g
    gamma_z_flat = gamma_z_flat * 0.5

    # 5. Contract with derivative integrals
    S_ref = np.asarray(compute_overlap(basis))
    grad = np.zeros((n_atoms, 3))

    # 1-electron contribution: Tr(D^z . dh1/dR)
    grad += np.asarray(one_electron_gradient_contribution(basis, mol, D_z_ao))

    # 2-electron contribution: 0.5 * Tr(Gamma^z . dg/dR)
    grad += np.asarray(two_electron_gradient_casscf(basis, mol, gamma_z_flat.ravel()))

    # Overlap contribution: Tr(W^z_MO . S . dS/dR)
    Wz_ao_overlap = Wz_ao @ S_ref
    grad -= np.asarray(overlap_gradient_contribution(basis, mol, Wz_ao_overlap))

    return grad


def _symmetrize_8fold_flat(gf: np.ndarray, nb: int) -> np.ndarray:
    """8-fold symmetrize a flat 4-index AO tensor for libint l-canonical reorder."""
    g4 = gf.reshape(nb, nb, nb, nb)
    gs = g4.copy()
    gs += g4.transpose(1, 0, 2, 3)
    gs += g4.transpose(0, 1, 3, 2)
    gs += g4.transpose(1, 0, 3, 2)
    gs += g4.transpose(2, 3, 0, 1)
    gs += g4.transpose(2, 3, 1, 0)
    gs += g4.transpose(3, 2, 0, 1)
    gs += g4.transpose(3, 2, 1, 0)
    return (gs / 8.0).ravel()
