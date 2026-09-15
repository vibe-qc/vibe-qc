# PRODUCTION — CP-MCSCF Lagrangian for PT2-corrected CASSCF gradients.
"""Full CP-MCSCF Lagrangian gradient for PT2-corrected CASSCF.

Two entry points:
  1. compute_pt2_lagrangian_gradient — PT2-only relaxation correction
     (Lagrangian(D_eff) - Lagrangian(D_CASSCF)). Used as zvec_corr replacement.
  2. compute_pt2_total_lagrangian_gradient — Full OpenMolcas-equivalent
     Lagrangian including frozen-density + relaxation contributions.
     Replaces the entire grad_casscf + grad_corr + zvec_corr assembly.

This replaces the z^T·g^R shortcut with the proper CP-MCSCF construction:
  D_lag = D_CASSCF + ΔD_eff + D^z
  Γ_lag = Γ_CASSCF + ΔΓ_eff + Γ^z
  W_lag = F_eff + W^z_MO

where D^z = [κ, D_eff], Γ^z = [κ, Γ_eff], W^z_MO = [F_eff + DEPSA, κ].

References
----------
Helgaker-Jorgensen-Olsen, Sec. 12.5
OpenMolcas: cnstclag.F90 + olagfinal.F90
"""

from __future__ import annotations

import numpy as np


def compute_pt2_lagrangian_gradient(
    mol,
    basis,
    C_mo: np.ndarray,
    z_orb: np.ndarray,
    h1e_mo: np.ndarray,
    h2e_mo: np.ndarray,
    D_casscf: np.ndarray,
    Gamma_casscf: np.ndarray,
    DeltaD: np.ndarray,
    DeltaGamma: np.ndarray,
    DEPSA: np.ndarray | None,
    n_core: int,
    n_act: int,
    nmo: int,
    pairs: list,
) -> np.ndarray:
    """Compute the PT2 Lagrangian gradient correction.

    Parameters
    ----------
    z_orb : (n_pairs,) ndarray
        Z-vector from CP-MCSCF solve.
    D_casscf, Gamma_casscf : ndarray
        CASSCF 1,2-RDMs in MO basis (full nmo x nmo).
    DeltaD, DeltaGamma : ndarray
        PT2 effective density corrections in MO basis (full nmo x nmo).
    DEPSA : (n_act, n_act) ndarray or None
        Active orbital energy corrections from derfg3 chain rule.

    Returns
    -------
    grad : (n_atoms, 3) ndarray
        Gradient correction in Hartree/bohr.
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
    act_s = slice(n_core, n_core + n_act)

    # ---- Step 1: Build κ from Z-vector ----
    kappa = _build_kappa_from_z(z_orb, pairs, nmo)

    # ---- Step 2: Build Fock for both CASSCF and effective densities ----
    eri_chem = h2e_mo.transpose(0, 2, 1, 3)

    def _build_fock(D_in, extra_active=None):
        F = np.zeros((nmo, nmo))
        for p in range(nmo):
            for q in range(nmo):
                fv = h1e_mo[p, q]
                for r in range(nmo):
                    for s in range(nmo):
                        fv += D_in[r, s] * (
                            2.0 * eri_chem[p, q, r, s] - eri_chem[p, s, r, q]
                        )
                F[p, q] = fv
        if extra_active is not None:
            F[act_s, act_s] += extra_active
        return F

    F_cas = _build_fock(D_casscf)
    F_eff = _build_fock(D_casscf + DeltaD, DEPSA)

    # ---- Step 3: One-index transformed densities ----
    # D^z = [kappa, D_eff] - [kappa, D_CASSCF] = [kappa, DeltaD]
    # Since the one-index transform is linear in the density.
    Dz_cas = _compute_one_index_transformed_dm(kappa, D_casscf)
    Dz_eff = _compute_one_index_transformed_dm(kappa, D_casscf + DeltaD)
    D_z = Dz_eff - Dz_cas  # PT2-only relaxation

    Gz_cas = _compute_one_index_transformed_gamma(kappa, Gamma_casscf)
    Gz_eff = _compute_one_index_transformed_gamma(kappa, Gamma_casscf + DeltaGamma)
    Gamma_z = Gz_eff - Gz_cas  # PT2-only relaxation

    # ---- Step 4: W^z_MO = [F_eff, kappa] - [F_cas, kappa] ----
    Wz_cas = _compute_wz_mo(F_cas, kappa)
    Wz_eff = _compute_wz_mo(F_eff, kappa)
    Wz_mo = Wz_eff - Wz_cas

    # ---- Step 5: AO transform ----
    D_z_ao = C_mo @ D_z @ C_mo.T
    Wz_ao = C_mo @ Wz_mo @ C_mo.T

    gamma_z_flat = _transform_4index_mo_to_ao_flat(Gamma_z, C_mo, nmo)
    gamma_z_flat = _symmetrize_8fold_flat(gamma_z_flat, nb)
    gamma_z_flat = gamma_z_flat * 0.5

    # ---- Step 6: Contract with derivative integrals ----
    S_ref = np.asarray(compute_overlap(basis))
    grad = np.zeros((n_atoms, 3))

    grad += np.asarray(one_electron_gradient_contribution(basis, mol, D_z_ao))
    grad += np.asarray(two_electron_gradient_casscf(basis, mol, gamma_z_flat.ravel()))
    Wz_ao_overlap = Wz_ao @ S_ref
    grad -= np.asarray(overlap_gradient_contribution(basis, mol, Wz_ao_overlap))

    return grad


def compute_pt2_total_lagrangian_gradient(
    mol,
    basis,
    C_mo: np.ndarray,
    z_orb: np.ndarray,
    h1e_mo: np.ndarray,
    h2e_mo: np.ndarray,
    D_casscf: np.ndarray,
    Gamma_casscf: np.ndarray,
    DeltaD: np.ndarray,
    DeltaGamma: np.ndarray,
    DEPSA: np.ndarray | None,
    n_core: int,
    n_act: int,
    nmo: int,
    pairs: list,
) -> np.ndarray:
    """Full CP-MCSCF Lagrangian gradient — OpenMolcas-equivalent.

    Computes the total CASPT2/NEVPT2 gradient from effective densities
    and the Z-vector in one shot, replacing:
        grad_casscf + grad_corr + zvec_corr

    The construction:
        grad = D_eff · dh/dR + 0.5·\u0393_eff · dg/dR - W_eff · dS/dR       (frozen)
             + D^z · dh/dR + 0.5·\u0393^z · dg/dR - W^z · dS/dR               (relaxation)

    where D_eff = D_CASSCF + \u0394D, \u0393_eff = \u0393_CASSCF + \u0394\u0393,
    F_eff = F(D_eff) + DEPSA, and ^{z} denotes one-index transform by \u03ba.

    Parameters
    ----------
    z_orb : (n_pairs,) ndarray
        Z-vector from CP-MCSCF solve.
    D_casscf, Gamma_casscf : ndarray
        CASSCF 1,2-RDMs in MO basis (full nmo x nmo).
    DeltaD, DeltaGamma : ndarray
        PT2 effective density corrections in MO basis (full nmo x nmo).
    DEPSA : (n_act, n_act) ndarray or None
        Active orbital energy corrections from derfg3 chain rule.

    Returns
    -------
    grad : (n_atoms, 3) ndarray
        Total nuclear gradient in Hartree/bohr.
    """
    from vibeqc._vibeqc_core import (
        compute_overlap,
        nuclear_repulsion_gradient,
        one_electron_gradient_contribution,
        overlap_gradient_contribution,
        two_electron_gradient_casscf,
    )
    from vibeqc.gradient._casscf import (
        _build_full_gamma_mo,
        _build_kappa_from_z,
        _compute_one_index_transformed_dm,
        _compute_one_index_transformed_gamma,
        _compute_w_unrelaxed,
        _compute_wz_mo,
        _transform_4index_mo_to_ao_flat,
    )

    n_atoms = len(mol.atoms)
    nb = C_mo.shape[0]
    act_s = slice(n_core, n_core + n_act)

    # ---- Effective densities ----
    D_eff = D_casscf + DeltaD
    Gamma_eff = Gamma_casscf + DeltaGamma
    D_eff_act = D_eff[act_s, act_s]
    Gamma_eff_act = Gamma_eff[act_s, act_s, act_s, act_s]

    # ---- Build full MO gamma from effective active RDMs ----
    gamma_mo_eff = _build_full_gamma_mo(D_eff_act, Gamma_eff_act, n_core, n_act, nmo)

    # ---- Frozen-density gradient (effective densities only, no relaxation) ----
    D_ao = C_mo @ D_eff @ C_mo.T

    gamma_ao_flat = _transform_4index_mo_to_ao_flat(gamma_mo_eff, C_mo, nmo)
    gamma_ao_4d = gamma_ao_flat.reshape(nb, nb, nb, nb)
    gamma_sym = gamma_ao_4d.copy()
    gamma_sym += gamma_ao_4d.transpose(1, 0, 2, 3)
    gamma_sym += gamma_ao_4d.transpose(0, 1, 3, 2)
    gamma_sym += gamma_ao_4d.transpose(1, 0, 3, 2)
    gamma_sym += gamma_ao_4d.transpose(2, 3, 0, 1)
    gamma_sym += gamma_ao_4d.transpose(2, 3, 1, 0)
    gamma_sym += gamma_ao_4d.transpose(3, 2, 0, 1)
    gamma_sym += gamma_ao_4d.transpose(3, 2, 1, 0)
    gamma_ao_flat = (gamma_sym / 8.0).ravel()
    gamma_ao_flat *= 0.5

    grad = np.asarray(nuclear_repulsion_gradient(mol), dtype=float)
    grad += np.asarray(one_electron_gradient_contribution(basis, mol, D_ao))
    grad += np.asarray(two_electron_gradient_casscf(basis, mol, gamma_ao_flat))

    # Overlap: W_unrelaxed from effective densities
    n_act_elec = int(round(np.trace(D_eff_act)))
    W_unrelaxed = _compute_w_unrelaxed(
        mol,
        basis,
        C_mo,
        D_eff,
        gamma_mo_eff,
        n_core + n_act,
        n_core=n_core,
        n_act=n_act,
        n_act_elec=n_act_elec,
        ms2=mol.multiplicity - 1,
        h1e_cas=h1e_mo,
        h2e_cas=h2e_mo,
    )
    grad += np.asarray(overlap_gradient_contribution(basis, mol, W_unrelaxed))

    # ---- Relaxation gradient (D^z, Gamma^z, W^z) ----
    kappa = _build_kappa_from_z(z_orb, pairs, nmo)

    eri_chem = h2e_mo.transpose(0, 2, 1, 3)
    F_eff = np.zeros((nmo, nmo))
    for p in range(nmo):
        for q in range(nmo):
            fv = h1e_mo[p, q]
            for r in range(nmo):
                for s in range(nmo):
                    fv += D_eff[r, s] * (
                        2.0 * eri_chem[p, q, r, s] - eri_chem[p, s, r, q]
                    )
            F_eff[p, q] = fv
    if DEPSA is not None:
        F_eff[act_s, act_s] += DEPSA

    D_z = _compute_one_index_transformed_dm(kappa, D_eff)
    Gamma_z = _compute_one_index_transformed_gamma(kappa, Gamma_eff)
    Wz_mo = _compute_wz_mo(F_eff, kappa)

    D_z_ao = C_mo @ D_z @ C_mo.T
    Wz_ao = C_mo @ Wz_mo @ C_mo.T

    gamma_z_flat = _transform_4index_mo_to_ao_flat(Gamma_z, C_mo, nmo)
    gamma_z_flat = _symmetrize_8fold_flat(gamma_z_flat, nb)
    gamma_z_flat *= 0.5

    S_ref = np.asarray(compute_overlap(basis))
    grad += np.asarray(one_electron_gradient_contribution(basis, mol, D_z_ao))
    grad += np.asarray(two_electron_gradient_casscf(basis, mol, gamma_z_flat.ravel()))
    Wz_ao_overlap = Wz_ao @ S_ref
    grad -= np.asarray(overlap_gradient_contribution(basis, mol, Wz_ao_overlap))

    return grad


def _symmetrize_8fold_flat(gf: np.ndarray, nb: int) -> np.ndarray:
    """8-fold symmetrize a flat 4-index AO tensor."""
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
