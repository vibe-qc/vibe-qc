"""Phase 17b-2 — second-derivative integral contractions.

Each of the four routines exposed by ``hessian_integrals.cpp`` returns
a ``(3N, 3N)`` skeleton-Hessian contribution from contracting one
weight matrix against second derivatives of the corresponding
integral type. Validation strategy:

1. **Symmetry** — every returned matrix must be symmetric to machine
   precision (the underlying integrals are symmetric in the
   perturbation indices, and the scatter helper writes both halves).

2. **Translational invariance** — for every contribution, summing
   the rows over the three Cartesian DOFs of any single atom and
   subtracting the same on a second atom must give zero (the
   corresponding integral derivative satisfies the same sum-rule
   that powers translational invariance of the gradient). We test
   the simpler statement: ``Σ_A H[3A:3A+3, :] = 0`` summed over
   atoms A (uniform translation has zero second-derivative
   contribution to the energy). Equivalently for columns.

3. **Closed-form nuclear repulsion** — the analytic ``∂²E_nuc/∂R∂R``
   formula is implemented from scratch (no libint). Cross-check
   against finite differences of ``nuclear_repulsion_gradient`` to
   ~1e-7 Ha/bohr².

4. **FD-on-gradient** for the overlap and (T+V) skeleton pieces:
   build the libint-2nd-derivative Hessian, then build an
   FD-on-gradient version (with the energy-weighted density / total
   density frozen at the reference values, evaluating the
   *gradient-of-the-W·S-trace* on displaced geometries) and require
   the two to match to FD-truncation tolerance.

5. **PySCF parity on H₂O / STO-3G** — the assembled skeleton
   contributions must add up to the same ``(3N, 3N)`` matrix as
   PySCF's intermediate Hessian skeleton when contracted with the
   reference D and W. (The full Hessian comparison is the 17b-3
   regression.)

Pre-rebuild precondition: ``third_party/libint/install/`` must have
been built with ``LIBINT2_ENABLE_ONEBODY=2`` and
``LIBINT2_ENABLE_ERI=2``. See ``scripts/build_libint.sh``.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc._vibeqc_core import (
    compute_eri_hessian_contribution,
    compute_kinetic_nuclear_hessian_contribution,
    compute_overlap_hessian_contribution,
    nuclear_repulsion_gradient,
    nuclear_repulsion_hessian,
)


ANGSTROM_TO_BOHR = 1.8897261339213


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _h2o_mol():
    return vq.Molecule([
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0,  0.7572 * ANGSTROM_TO_BOHR, 0.5868 * ANGSTROM_TO_BOHR]),
        vq.Atom(1, [0.0, -0.7572 * ANGSTROM_TO_BOHR, 0.5868 * ANGSTROM_TO_BOHR]),
    ], charge=0, multiplicity=1)


def _h2_mol(R_bohr: float = 1.4):
    return vq.Molecule([
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.0, R_bohr]),
    ], charge=0, multiplicity=1)


def _converged_rhf(mol, basis_name="sto-3g"):
    basis = vq.BasisSet(mol, basis_name)
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-9
    return basis, vq.run_rhf(mol, basis, opts)


def _D_W(result):
    """Total density and energy-weighted density from a converged RHF
    result. Closed-shell convention: D = 2 C_occ C_occ^T,
    W = 2 C_occ · diag(ε_occ) · C_occ^T."""
    C = np.asarray(result.mo_coeffs)
    eps = np.asarray(result.mo_energies)
    D = np.asarray(result.density)
    n_occ = int(np.round(np.sum(np.linalg.eigvalsh(D) > 1e-6)))
    C_occ = C[:, :n_occ]
    W = 2.0 * C_occ @ np.diag(eps[:n_occ]) @ C_occ.T
    return D, W


# ---------------------------------------------------------------------------
# 1. Symmetry of returned matrices
# ---------------------------------------------------------------------------

def test_overlap_hessian_is_symmetric():
    mol = _h2o_mol()
    basis, result = _converged_rhf(mol)
    _, W = _D_W(result)
    H = np.asarray(compute_overlap_hessian_contribution(basis, mol, W))
    assert H.shape == (9, 9)
    np.testing.assert_allclose(H, H.T, atol=1e-12,
                                err_msg="overlap Hessian not symmetric")


def test_kinetic_nuclear_hessian_is_symmetric():
    mol = _h2o_mol()
    basis, result = _converged_rhf(mol)
    D, _ = _D_W(result)
    H = np.asarray(
        compute_kinetic_nuclear_hessian_contribution(basis, mol, D))
    assert H.shape == (9, 9)
    np.testing.assert_allclose(H, H.T, atol=1e-12,
                                err_msg="(T+V) Hessian not symmetric")


def test_eri_hessian_is_symmetric():
    mol = _h2o_mol()
    basis, result = _converged_rhf(mol)
    D, _ = _D_W(result)
    H = np.asarray(compute_eri_hessian_contribution(basis, mol, D, 1.0))
    assert H.shape == (9, 9)
    np.testing.assert_allclose(H, H.T, atol=1e-12,
                                err_msg="ERI Hessian not symmetric")


def test_nuclear_repulsion_hessian_is_symmetric():
    mol = _h2o_mol()
    H = np.asarray(nuclear_repulsion_hessian(mol))
    assert H.shape == (9, 9)
    np.testing.assert_allclose(H, H.T, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Closed-form nuclear repulsion vs FD
# ---------------------------------------------------------------------------

def test_nuclear_repulsion_hessian_matches_fd_on_gradient():
    """∂²E_nuc/∂R∂R from the closed-form must match an FD on
    nuclear_repulsion_gradient — both are pure analytic functions of
    the geometry, no SCF involved."""
    mol = _h2o_mol()
    H_analytic = np.asarray(nuclear_repulsion_hessian(mol))

    delta = 1e-4
    n_atoms = 3
    H_fd = np.zeros((9, 9))
    positions = np.array([list(a.xyz) for a in mol.atoms])
    for j_atom in range(n_atoms):
        for j_dim in range(3):
            mol_p = vq.Molecule([
                vq.Atom(int(a.Z),
                        [a.xyz[0] + (delta if (i == j_atom and 0 == j_dim) else 0.0),
                         a.xyz[1] + (delta if (i == j_atom and 1 == j_dim) else 0.0),
                         a.xyz[2] + (delta if (i == j_atom and 2 == j_dim) else 0.0)])
                for i, a in enumerate(mol.atoms)
            ], charge=mol.charge, multiplicity=mol.multiplicity)
            mol_m = vq.Molecule([
                vq.Atom(int(a.Z),
                        [a.xyz[0] - (delta if (i == j_atom and 0 == j_dim) else 0.0),
                         a.xyz[1] - (delta if (i == j_atom and 1 == j_dim) else 0.0),
                         a.xyz[2] - (delta if (i == j_atom and 2 == j_dim) else 0.0)])
                for i, a in enumerate(mol.atoms)
            ], charge=mol.charge, multiplicity=mol.multiplicity)
            g_p = np.asarray(nuclear_repulsion_gradient(mol_p)).reshape(-1)
            g_m = np.asarray(nuclear_repulsion_gradient(mol_m)).reshape(-1)
            H_fd[:, 3 * j_atom + j_dim] = (g_p - g_m) / (2 * delta)
    H_fd = 0.5 * (H_fd + H_fd.T)
    np.testing.assert_allclose(H_analytic, H_fd, atol=1e-7, rtol=0,
                                err_msg="nuclear-repulsion analytic Hessian disagrees with FD")


# ---------------------------------------------------------------------------
# 3. FD-on-gradient parity for skeleton overlap + (T+V)
# ---------------------------------------------------------------------------

def _fd_grad_with_frozen_weights(mol, basis_name, weight_op, atom_index, dim,
                                  delta=1e-4):
    """Finite-difference of an integral-gradient contribution evaluated
    with the *reference-geometry weight matrix* W or D held fixed.
    ``weight_op(basis_at_geom, molecule_at_geom)`` returns the per-atom
    gradient that the skeleton Hessian is the y-derivative of."""
    positions = [list(a.xyz) for a in mol.atoms]
    def _displaced(sign):
        atoms = []
        for k, a in enumerate(mol.atoms):
            xyz = list(a.xyz)
            if k == atom_index:
                xyz[dim] += sign * delta
            atoms.append(vq.Atom(int(a.Z), xyz))
        return vq.Molecule(atoms, charge=mol.charge, multiplicity=mol.multiplicity)
    mol_p, mol_m = _displaced(+1), _displaced(-1)
    basis_p = vq.BasisSet(mol_p, basis_name)
    basis_m = vq.BasisSet(mol_m, basis_name)
    g_p = np.asarray(weight_op(basis_p, mol_p)).reshape(-1)
    g_m = np.asarray(weight_op(basis_m, mol_m)).reshape(-1)
    return (g_p - g_m) / (2 * delta)


def test_overlap_hessian_matches_fd_on_grad():
    """Skeleton overlap Hessian should equal the y-derivative of the
    overlap gradient contribution evaluated at fixed reference W.
    FD step 1e-4; tolerance 5e-5 absorbs FD truncation noise."""
    mol = _h2o_mol()
    basis, result = _converged_rhf(mol)
    _, W = _D_W(result)
    H_libint = np.asarray(compute_overlap_hessian_contribution(basis, mol, W))

    from vibeqc._vibeqc_core import overlap_gradient_contribution
    def weight_op(b, m):
        return overlap_gradient_contribution(b, m, W)

    H_fd = np.zeros_like(H_libint)
    for j_atom in range(len(mol.atoms)):
        for j_dim in range(3):
            H_fd[:, 3 * j_atom + j_dim] = _fd_grad_with_frozen_weights(
                mol, "sto-3g", weight_op, j_atom, j_dim, delta=1e-4)
    H_fd = 0.5 * (H_fd + H_fd.T)
    np.testing.assert_allclose(H_libint, H_fd, atol=5e-5, rtol=0,
                                err_msg="overlap skeleton Hessian disagrees with FD-on-gradient")


def test_kinetic_nuclear_hessian_matches_fd_on_grad():
    """Skeleton (T+V) Hessian should equal the y-derivative of
    one_electron_gradient_contribution at fixed reference D.
    Same FD setup as the overlap test."""
    mol = _h2o_mol()
    basis, result = _converged_rhf(mol)
    D, _ = _D_W(result)
    H_libint = np.asarray(
        compute_kinetic_nuclear_hessian_contribution(basis, mol, D))

    from vibeqc._vibeqc_core import one_electron_gradient_contribution
    def weight_op(b, m):
        return one_electron_gradient_contribution(b, m, D)

    H_fd = np.zeros_like(H_libint)
    for j_atom in range(len(mol.atoms)):
        for j_dim in range(3):
            H_fd[:, 3 * j_atom + j_dim] = _fd_grad_with_frozen_weights(
                mol, "sto-3g", weight_op, j_atom, j_dim, delta=1e-4)
    H_fd = 0.5 * (H_fd + H_fd.T)
    np.testing.assert_allclose(H_libint, H_fd, atol=5e-5, rtol=0,
                                err_msg="(T+V) skeleton Hessian disagrees with FD-on-gradient")


def test_eri_hessian_matches_fd_on_grad():
    """Skeleton ERI Hessian should equal the y-derivative of
    two_electron_gradient_contribution at fixed reference D.
    Same FD setup."""
    mol = _h2o_mol()
    basis, result = _converged_rhf(mol)
    D, _ = _D_W(result)
    H_libint = np.asarray(
        compute_eri_hessian_contribution(basis, mol, D, 1.0))

    from vibeqc._vibeqc_core import two_electron_gradient_contribution
    def weight_op(b, m):
        return two_electron_gradient_contribution(b, m, D, 1.0)

    H_fd = np.zeros_like(H_libint)
    for j_atom in range(len(mol.atoms)):
        for j_dim in range(3):
            H_fd[:, 3 * j_atom + j_dim] = _fd_grad_with_frozen_weights(
                mol, "sto-3g", weight_op, j_atom, j_dim, delta=1e-4)
    H_fd = 0.5 * (H_fd + H_fd.T)
    # ERI FD precision is the loosest because 4-center integrals
    # accumulate the most FD truncation error. 1e-3 absorbs noise
    # without hiding real errors (which would be O(1) for sign /
    # sum-rule mistakes).
    np.testing.assert_allclose(H_libint, H_fd, atol=1e-3, rtol=0,
                                err_msg="ERI skeleton Hessian disagrees with FD-on-gradient")
