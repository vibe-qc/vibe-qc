"""CASSCF molecular properties -- dipole moments, charges, etc.

Computes properties from the CASSCF one-particle density matrix,
including electric dipole moments.

Usage::

    from vibeqc.properties_casscf import compute_casscf_dipole

    mu = compute_casscf_dipole(mol, basis, C_conv, D_mo, n_core)
    print(f"Dipole = {mu} au = {np.linalg.norm(mu)*2.541746:.4f} Debye")
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .._vibeqc_core import BasisSet, Molecule, compute_dipole


def compute_casscf_dipole(
    mol: Molecule,
    basis: BasisSet,
    C_mo: np.ndarray,
    D_mo: np.ndarray,
    n_core: int,
) -> np.ndarray:
    """Electric dipole moment from the CASSCF one-particle density.

    mu = S_A Z_A R_A - S_muν D_muν <mu|r|ν>

    Parameters
    ----------
    mol : Molecule
        Molecular geometry.
    basis : BasisSet
        AO basis set.
    C_mo : (n_ao, n_mo) ndarray
        AO->MO coefficient matrix.
    D_mo : (n_mo, n_mo) ndarray
        One-particle density in MO basis (core + active).
    n_core : int
        Number of inactive (core) orbitals (unused here; D_mo already
        includes the core contribution).

    Returns
    -------
    mu : (3,) ndarray
        Dipole moment in atomic units (e.a₀).  Convert to Debye by
        multiplying by 2.541746.
    """
    D_ao = C_mo @ D_mo @ C_mo.T
    dip = compute_dipole(basis)

    # Electronic contribution: -tr(D . dipole_integrals)
    mu_elec = np.array(
        [
            -np.sum(D_ao * np.asarray(dip.x)),
            -np.sum(D_ao * np.asarray(dip.y)),
            -np.sum(D_ao * np.asarray(dip.z)),
        ]
    )

    # Nuclear contribution
    mu_nuc = np.zeros(3)
    for atom in mol.atoms:
        mu_nuc += atom.Z * np.array(atom.xyz)

    return mu_nuc + mu_elec
