"""CASPT2/NEVPT2 one-electron properties from PT2 effective densities.

Computes dipole moments and other 1e expectation values using the
chain-rule-corrected effective 1-RDM.

    <O>_PT2 = Tr(D_eff_AO · O_AO) + nuclear_contribution

where D_eff_AO = C · (D_CASSCF + DeltaD) · C^T.

References
----------
OpenMolcas: dens1_rpt2.F90, moddip.F90
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .._vibeqc_core import BasisSet, Molecule, compute_dipole
from ..properties import DipoleMoment, center_of_mass


def _build_effective_ao_density(
    C_mo: np.ndarray,
    D_casscf: np.ndarray,
    DeltaD: np.ndarray,
    n_core: int,
    n_act: int,
) -> np.ndarray:
    """Build PT2-corrected AO density matrix.

    Parameters
    ----------
    C_mo : (n_ao, n_mo) ndarray
        AO→MO coefficient matrix.
    D_casscf : (n_mo, n_mo) ndarray
        CASSCF 1-RDM in MO basis (core + active).
    DeltaD : (n_mo, n_mo) ndarray
        PT2 effective 1-RDM correction (chain-rule corrected).

    Returns
    -------
    D_ao : (n_ao, n_ao) ndarray
        Effective AO density.
    """
    D_mo = D_casscf + DeltaD
    return C_mo @ D_mo @ C_mo.T


def caspt2_dipole_moment(
    mol: Molecule,
    basis: BasisSet,
    C_mo: np.ndarray,
    D_casscf: np.ndarray,
    DeltaD: np.ndarray,
    *,
    n_core: int = 0,
    n_act: int = 0,
    origin: Optional[Sequence[float]] = None,
) -> DipoleMoment:
    """CASPT2/NEVPT2 electric dipole moment from effective density.

    Parameters
    ----------
    mol : Molecule
    basis : BasisSet
    C_mo : (n_ao, n_mo) ndarray
        AO→MO coefficient matrix from converged CASSCF.
    D_casscf : (n_mo, n_mo) ndarray
        CASSCF 1-RDM in MO basis.
    DeltaD : (n_mo, n_mo) ndarray
        PT2 effective 1-RDM correction (from compute_pt2_effective_density
        or compute_nevpt2_effective_density, chain-rule corrected).
    n_core, n_act : int
        For documentation; not used directly (D_casscf and DeltaD already
        contain full MO-basis densities).
    origin : sequence of 3 floats, optional
        Dipole origin in bohr (default: center of mass).

    Returns
    -------
    DipoleMoment with x, y, z components in e·bohr and origin.
    """
    if origin is None:
        origin_vec = center_of_mass(mol)
    else:
        origin_vec = np.asarray(origin, dtype=np.float64)

    # Dipole integrals in AO basis
    dip = compute_dipole(basis, [float(x) for x in origin_vec])
    Mx = np.asarray(dip.x)
    My = np.asarray(dip.y)
    Mz = np.asarray(dip.z)

    # Effective AO density
    P_eff = _build_effective_ao_density(C_mo, D_casscf, DeltaD, n_core, n_act)

    # Electronic contribution: -Tr(P · M)
    mu_e_x = -np.einsum("ij,ji->", P_eff, Mx)
    mu_e_y = -np.einsum("ij,ji->", P_eff, My)
    mu_e_z = -np.einsum("ij,ji->", P_eff, Mz)

    # Nuclear contribution
    mu_n_x = 0.0
    mu_n_y = 0.0
    mu_n_z = 0.0
    for atom in mol.atoms:
        z = float(atom.Z)
        mu_n_x += z * (atom.xyz[0] - origin_vec[0])
        mu_n_y += z * (atom.xyz[1] - origin_vec[1])
        mu_n_z += z * (atom.xyz[2] - origin_vec[2])

    return DipoleMoment(
        x=float(mu_e_x + mu_n_x),
        y=float(mu_e_y + mu_n_y),
        z=float(mu_e_z + mu_n_z),
        origin=(
            float(origin_vec[0]),
            float(origin_vec[1]),
            float(origin_vec[2]),
        ),
    )
