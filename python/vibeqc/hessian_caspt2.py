"""CASPT2/NEVPT2 numerical Hessian and harmonic vibrational frequencies.

Computes the nuclear Hessian by central finite differences of the
analytic CASPT2/NEVPT2 gradient (CASSCF + correlation + Z-vector),
then mass-weights and diagonalises for harmonic frequencies.

Usage::

    from vibeqc.hessian_caspt2 import compute_hessian_caspt2

    result = compute_hessian_caspt2(mol, "6-31g", active_space=(2,2))
    print(result.frequencies)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ._vibeqc_core import Atom, BasisSet, Molecule
from .solvers import build_hamiltonian_mo, get_hf_orbital_provider
from .solvers._casscf import CASSCFOptions, casscf
from .solvers._rdm import make_rdm12, make_rdm12_sa

_ATOMIC_MASSES: dict[int, float] = {
    1: 1.00782503223,
    6: 12.0000000,
    7: 14.00307400443,
    8: 15.99491461957,
    9: 18.99840316273,
    16: 31.9720711744,
    17: 34.968852682,
}


@dataclass
class PT2HessianResult:
    """CASPT2/NEVPT2 vibrational frequency result."""

    hessian: np.ndarray
    frequencies: np.ndarray
    normal_modes: np.ndarray
    mass_weights: np.ndarray


def _mass_weight(mol: Molecule) -> np.ndarray:
    weights = []
    for atom in mol.atoms:
        m = _ATOMIC_MASSES.get(int(atom.Z), float(atom.Z))
        weights.extend([np.sqrt(m)] * 3)
    return np.array(weights, dtype=float)


def compute_hessian_caspt2(
    mol: Molecule,
    basis_name: str,
    active_space: tuple[int, int],
    *,
    n_core: int = 0,
    variant: str = "caspt2",
    casscf_options: Optional[CASSCFOptions] = None,
    fd_step: float = 0.005,
    max_macro: int = 100,
) -> PT2HessianResult:
    """CASPT2/NEVPT2 harmonic frequencies by FD of analytic gradient.

    Parameters
    ----------
    variant : str
        "caspt2" (default) or "nevpt2".
    """
    from .ecp_metadata import refuse_molecular_ecp_derivative_route

    refuse_molecular_ecp_derivative_route(
        mol,
        basis_name,
        route="compute_hessian_caspt2",
    )
    n_atoms = len(list(mol.atoms))
    n_coords = 3 * n_atoms
    atoms_list = list(mol.atoms)
    opts = casscf_options or CASSCFOptions()
    n_act_elec, n_act_orb = active_space

    # Reference CASSCF
    basis0 = BasisSet(mol, basis_name)
    C_hf0 = get_hf_orbital_provider(mol, basis0)
    H0 = build_hamiltonian_mo(mol, basis0, C_hf0)
    sc0 = casscf(
        H0.h1e,
        H0.h2e,
        n_act_elec,
        n_act_orb,
        n_core=n_core,
        nuclear_repulsion=H0.nuclear_repulsion,
        max_macro=max_macro,
    )
    if not sc0.converged:
        raise RuntimeError("CASSCF did not converge at reference geometry.")

    C_conv0 = C_hf0 @ sc0.mo_rotation
    rdm1_0, rdm2_0 = make_rdm12(sc0.cas.ci_coeffs, sc0.cas.determinants, n_act_orb)

    # Gradient function at displaced geometry
    def _gradient_at(mol_disp: Molecule) -> np.ndarray:
        basis_d = BasisSet(mol_disp, basis_name)
        C_hf_d = get_hf_orbital_provider(mol_disp, basis_d)
        H_d = build_hamiltonian_mo(mol_disp, basis_d, C_hf_d)

        sc_d = casscf(
            H_d.h1e,
            H_d.h2e,
            n_act_elec,
            n_act_orb,
            n_core=n_core,
            nuclear_repulsion=H_d.nuclear_repulsion,
            max_macro=max_macro,
        )
        if not sc_d.converged:
            raise RuntimeError("CASSCF did not converge at displaced geometry.")
        C_conv_d = C_hf_d @ sc_d.mo_rotation
        rdm1_d, rdm2_d = make_rdm12(
            sc_d.cas.ci_coeffs, sc_d.cas.determinants, n_act_orb
        )

        from .gradient._caspt2 import compute_caspt2_gradient
        from .gradient._nevpt2 import compute_nevpt2_gradient

        if variant == "nevpt2":
            return compute_nevpt2_gradient(
                mol_disp,
                basis_d,
                C_conv_d,
                sc_d.h1e_cas,
                sc_d.h2e_cas,
                n_core=n_core,
                n_active_orb=n_act_orb,
                n_active_elec=n_act_elec,
                rdm1=rdm1_d,
                rdm2=rdm2_d,
                fd_eps=0.001,
                use_zvector=True,
                determinants=sc_d.cas.determinants,
                ci_coeffs=sc_d.cas.ci_coeffs,
            )
        else:
            return compute_caspt2_gradient(
                mol_disp,
                basis_d,
                C_conv_d,
                sc_d.h1e_cas,
                sc_d.h2e_cas,
                n_core=n_core,
                n_active_orb=n_act_orb,
                n_active_elec=n_act_elec,
                rdm1=rdm1_d,
                rdm2=rdm2_d,
                fd_eps=0.001,
                use_zvector=True,
                determinants=sc_d.cas.determinants,
                ci_coeffs=sc_d.cas.ci_coeffs,
            )

    grad0 = _gradient_at(mol)

    # FD of gradient -> Hessian
    H = np.zeros((n_coords, n_coords))
    for i in range(n_coords):
        a = i // 3
        c = i % 3
        xyz_p = np.array([at.xyz for at in atoms_list], dtype=float)
        xyz_m = np.array([at.xyz for at in atoms_list], dtype=float)
        xyz_p[a, c] += fd_step
        xyz_m[a, c] -= fd_step
        mol_p = Molecule(
            [Atom(int(at.Z), list(xyz)) for at, xyz in zip(atoms_list, xyz_p)]
        )
        mol_m = Molecule(
            [Atom(int(at.Z), list(xyz)) for at, xyz in zip(atoms_list, xyz_m)]
        )
        grad_p = _gradient_at(mol_p)
        grad_m = _gradient_at(mol_m)
        H[:, i] = (grad_p.ravel() - grad_m.ravel()) / (2.0 * fd_step)

    H = 0.5 * (H + H.T)

    mw = _mass_weight(mol)
    H_mw = H / np.outer(mw, mw)
    eigvals, eigvecs = np.linalg.eigh(H_mw)

    hartree_to_J = 4.3597447222071e-18
    bohr_to_m = 5.29177210903e-11
    amu_to_kg = 1.66053906660e-27
    c_cm_s = 2.99792458e10
    factor = np.sqrt(hartree_to_J / (amu_to_kg * bohr_to_m**2)) / (2.0 * np.pi * c_cm_s)

    freqs = np.where(
        eigvals > 0,
        factor * np.sqrt(np.maximum(eigvals, 0)),
        -factor * np.sqrt(np.maximum(-eigvals, 0)),
    )

    return PT2HessianResult(
        hessian=H,
        frequencies=freqs,
        normal_modes=eigvecs,
        mass_weights=mw,
    )
