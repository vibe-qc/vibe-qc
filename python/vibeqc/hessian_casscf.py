"""CASSCF numerical Hessian and harmonic vibrational frequencies.

Computes the 3N x 3N nuclear Hessian by central finite differences of the
analytic CASSCF gradient, then mass-weights and diagonalises to produce
harmonic frequencies and normal modes.

This follows the same FD-Hessian pattern as :func:`vibeqc.compute_hessian_fd`
but uses the CASSCF analytic gradient instead of energy differences,
requiring only 6N gradient evaluations (vs 12N energy evaluations for
a full CASSCF FD Hessian).

Usage::

    from vibeqc.hessian_casscf import compute_hessian_casscf

    result = compute_hessian_casscf(mol, "6-31g", active_space=(2,2))
    print(result.frequencies)       # cm⁻¹
    print(result.normal_modes)      # (3N, 3N) mass-weighted Cartesian
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ._vibeqc_core import Atom, BasisSet, Molecule
from .gradient import compute_casscf_gradient
from .solvers import build_hamiltonian_mo, get_hf_orbital_provider
from .solvers._casscf import CASSCFOptions, casscf
from .solvers._rdm import make_rdm12, make_rdm12_sa

# Atomic masses (amu) for mass-weighting -- CODATA 2018.
_ATOMIC_MASSES: dict[int, float] = {
    1: 1.00782503223,
    2: 4.00260325413,
    3: 7.0160034366,
    4: 9.012183065,
    5: 11.00930536,
    6: 12.0000000,
    7: 14.00307400443,
    8: 15.99491461957,
    9: 18.99840316273,
    10: 19.9924401762,
    11: 22.98976928,
    12: 23.985041697,
    13: 26.98153853,
    14: 27.97692653465,
    15: 30.97376199842,
    16: 31.9720711744,
    17: 34.968852682,
    18: 39.9623831237,
    19: 38.9637064864,
    20: 39.962590863,
    26: 55.93493633,
    27: 58.93319429,
    28: 57.93534241,
    29: 62.92959772,
    30: 63.92914201,
    35: 78.9183376,
    53: 126.9044719,
}


@dataclass
class CASSCFHessianResult:
    """CASSCF vibrational frequency result.

    Attributes
    ----------
    hessian : (3N, 3N) ndarray
        Cartesian force-constant matrix in Hartree/bohr^2.
    frequencies : (3N,) ndarray
        Harmonic frequencies in cm⁻¹.  Imaginary frequencies are
        reported as negative values; translational/rotational modes
        are near zero (typically < 10 cm⁻¹).
    normal_modes : (3N, 3N) ndarray
        Mass-weighted Cartesian normal modes (columns).  Each column
        is normalised such that L^T L = I.
    mass_weights : (3N,) ndarray
        sqrt(atomic_mass) array used for mass-weighting.
    """

    hessian: np.ndarray
    frequencies: np.ndarray
    normal_modes: np.ndarray
    mass_weights: np.ndarray


def _mass_weight(mol: Molecule) -> np.ndarray:
    """Build the mass-weight vector for Cartesian mass-weighting.

    Returns (3N,) array of sqrt(m_i) for each Cartesian coordinate,
    using the standard isotopic masses.
    """
    weights: list[float] = []
    for atom in mol.atoms:
        m = _ATOMIC_MASSES.get(int(atom.Z), float(atom.Z))
        weights.extend([np.sqrt(m)] * 3)
    return np.array(weights, dtype=float)


def compute_hessian_casscf(
    mol: Molecule,
    basis_name: str,
    active_space: tuple[int, int],
    *,
    n_core: int = 0,
    casscf_options: Optional[CASSCFOptions] = None,
    compute_wz: bool = False,
    fd_step: float = 0.005,
    max_macro: int = 100,
) -> CASSCFHessianResult:
    """CASSCF harmonic vibrational frequencies by FD of analytic gradient.

    Parameters
    ----------
    mol : Molecule
        Geometry at which to compute the Hessian.
    basis_name : str
        Basis-set name (e.g. ``"6-31g"``, ``"sto-3g"``).
    active_space : tuple[int, int]
        ``(n_active_elec, n_active_orb)`` for the CAS.
    n_core : int
        Number of frozen-core (inactive) orbitals (default 0).
    casscf_options : CASSCFOptions, optional
        CASSCF convergence controls.  ``compute_wz`` on this object is
        ignored -- use the top-level ``compute_wz`` parameter instead.
    compute_wz : bool
        Kept for backward compatibility. The analytic CASSCF gradient is
        already exact for a variational CASSCF, so ``True`` selects the
        same gradient as ``False`` (with a ``FutureWarning``; the former
        experimental W^z correction was retired -- GitLab #516).
    fd_step : float
        Cartesian displacement step in bohr (default 0.005).
    max_macro : int
        Maximum CASSCF macro-iterations per displaced geometry.

    Returns
    -------
    CASSCFHessianResult
    """
    from .ecp_metadata import refuse_molecular_ecp_derivative_route

    refuse_molecular_ecp_derivative_route(
        mol,
        basis_name,
        route="compute_hessian_casscf",
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
    opts_ref = CASSCFOptions(
        nroots=opts.nroots,
        weights=opts.weights,
        orbital_step=opts.orbital_step,
    )
    sc0 = casscf(
        H0.h1e,
        H0.h2e,
        n_act_elec,
        n_act_orb,
        n_core=n_core,
        nuclear_repulsion=H0.nuclear_repulsion,
        max_macro=max_macro,
        nroots=opts_ref.nroots,
        weights=opts_ref.weights,
    )
    if not sc0.converged:
        raise RuntimeError("CASSCF did not converge at reference geometry.")

    # Helper: gradient at a displaced geometry
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
            nroots=opts_ref.nroots,
            weights=opts_ref.weights,
        )
        if not sc_d.converged:
            raise RuntimeError(
                f"CASSCF did not converge at displaced geometry (atom coords modified)."
            )
        C_conv_d = C_hf_d @ sc_d.mo_rotation

        if opts_ref.weights is not None and sc_d.cas.ci_coeffs_all is not None:
            rdm1_g, rdm2_g = make_rdm12_sa(
                sc_d.cas.ci_coeffs_all,
                sc_d.cas.determinants,
                n_act_orb,
                opts_ref.weights,
            )
            _dets = None
            _civec = None
        else:
            rdm1_g, rdm2_g = make_rdm12(
                sc_d.cas.ci_coeffs,
                sc_d.cas.determinants,
                n_act_orb,
            )
            _dets = sc_d.cas.determinants
            _civec = sc_d.cas.ci_coeffs

        return compute_casscf_gradient(
            mol_disp,
            basis_d,
            C_conv_d,
            sc_d.h1e_cas,
            sc_d.h2e_cas,
            n_core=n_core,
            n_active_orb=n_act_orb,
            rdm1=rdm1_g,
            rdm2=rdm2_g,
            compute_wz=compute_wz,
            determinants=_dets,
            ci_coeffs=_civec,
        )

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

    # Symmetrize
    H = 0.5 * (H + H.T)

    # Mass-weight: H_mw = H / sqrt(m_i m_j)
    mw = _mass_weight(mol)
    H_mw = H / np.outer(mw, mw)

    # Diagonalize
    eigvals, eigvecs = np.linalg.eigh(H_mw)

    # Convert eigenvalues (Ha/bohr^2/amu) to frequencies (cm⁻¹)
    # factor = sqrt(Ha / (me * a0^2)) / (2pi c)  [in cm⁻¹]
    # Hartree = 4.3597447222071e-18 J
    # bohr = 5.29177210903e-11 m
    # amu = 1.66053906660e-27 kg
    # c = 2.99792458e10 cm/s
    # factor = sqrt(4.3597447222071e-18 / (1.66053906660e-27 * (5.29177210903e-11)^2)) / (2pi * 2.99792458e10)
    #       = 5140.485... cm⁻¹ per sqrt(Ha/bohr^2/amu)
    hartree_to_J = 4.3597447222071e-18
    bohr_to_m = 5.29177210903e-11
    amu_to_kg = 1.66053906660e-27
    c_cm_s = 2.99792458e10
    factor = np.sqrt(hartree_to_J / (amu_to_kg * bohr_to_m**2)) / (2.0 * np.pi * c_cm_s)

    # Frequencies: ν = factor * sqrt(l) for l > 0, else -factor * sqrt(-l)
    freqs = np.where(
        eigvals > 0,
        factor * np.sqrt(np.maximum(eigvals, 0)),
        -factor * np.sqrt(np.maximum(-eigvals, 0)),
    )

    return CASSCFHessianResult(
        hessian=H,
        frequencies=freqs,
        normal_modes=eigvecs,
        mass_weights=mw,
    )
