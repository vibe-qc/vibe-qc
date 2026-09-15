"""Phase 17c — analytic UHF Hessian assembly.

Per-spin extension of the RHF analytic Hessian (Phase 17b-3).
Validation against PySCF UHF Hessian on triplet O₂ and OH radical.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


ANGSTROM_TO_BOHR = 1.8897261339213


def _o2_triplet_mol():
    """Triplet O₂ at near-equilibrium UHF/STO-3G geometry."""
    return vq.Molecule([
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(8, [0.0, 0.0, 1.21 * ANGSTROM_TO_BOHR]),
    ], charge=0, multiplicity=3)


def _oh_radical_mol():
    """OH radical (doublet) at HF/STO-3G geometry."""
    return vq.Molecule([
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.0, 0.96 * ANGSTROM_TO_BOHR]),
    ], charge=0, multiplicity=2)


def _converged_uhf(mol, basis_name="sto-3g"):
    basis = vq.BasisSet(mol, basis_name)
    opts = vq.UHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-9
    opts.max_iter = 300   # OH/STO-3G needs ~190 iters; bump for safety
    # These tests validate the analytic-Hessian/UCPHF kernel against the
    # reference program's Hessian AT THE SAME SCF SOLUTION. The internal
    # stability escape (UHF-ABOVE-ROHF-VARIATIONAL-INVERSION fix) moves
    # O2 triplet/STO-3G to a lower symmetry-broken UHF solution
    # (-147.6355561091 vs -147.6340485051 Ha) carrying a ~0-eigenvalue
    # internal Hessian mode, which the pinned reference numbers were not
    # computed at (and which makes the UCPHF system near-singular). Pin
    # the legacy symmetric solution so the kernel parity claim stays
    # solution-matched; UCPHF robustness at symmetry-broken minima is
    # separate follow-up scope (bug-claims.md,
    # UHF-ABOVE-ROHF-VARIATIONAL-INVERSION FIXED entry).
    opts.stability_check = False
    return basis, vq.run_uhf(mol, basis, opts)


def _pyscf_uhf_hessian(atoms, basis="sto-3g", spin=2):
    """PySCF reference UHF Hessian. ``atoms`` is a list of (symbol,
    (x_Å, y_Å, z_Å)) tuples; ``spin`` is 2S = mult−1."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, scf
    mol = gto.Mole()
    mol.atom = atoms
    mol.basis = basis
    mol.unit = "Angstrom"
    mol.spin = spin
    mol.verbose = 0
    mol.build()
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    h = mf.Hessian().kernel()
    n = mol.natm
    H = np.zeros((3*n, 3*n))
    for i in range(n):
        for j in range(n):
            H[3*i:3*i+3, 3*j:3*j+3] = h[i, j]
    return H


# ---------------------------------------------------------------------------
# 1. API + return shape
# ---------------------------------------------------------------------------

def test_compute_hessian_uhf_analytic_exposed():
    assert hasattr(vq, "compute_hessian_uhf_analytic")


def test_returns_hessian_result():
    mol = _o2_triplet_mol()
    basis, result = _converged_uhf(mol)
    r = vq.compute_hessian_uhf_analytic(mol, basis, result,
                                          basis_name="sto-3g")
    assert isinstance(r, vq.HessianResult)
    assert r.hessian.shape == (6, 6)
    assert r.is_linear is True


# ---------------------------------------------------------------------------
# 2. PySCF parity
# ---------------------------------------------------------------------------

def test_o2_triplet_hessian_matches_pyscf():
    """Triplet O₂ / STO-3G UHF Hessian vs PySCF analytic."""
    mol = _o2_triplet_mol()
    basis, result = _converged_uhf(mol)
    r = vq.compute_hessian_uhf_analytic(mol, basis, result,
                                          basis_name="sto-3g")
    H_ps = _pyscf_uhf_hessian(
        [["O", (0.0, 0.0, 0.0)], ["O", (0.0, 0.0, 1.21)]],
        basis="sto-3g", spin=2,
    )
    np.testing.assert_allclose(r.hessian, H_ps, atol=1e-7, rtol=0,
                                err_msg="O2 triplet UHF Hessian disagrees with PySCF")


def test_oh_doublet_hessian_matches_pyscf():
    """OH (doublet) / STO-3G UHF Hessian vs PySCF analytic. Spin
    asymmetry is non-trivial here (singly-occupied alpha SOMO);
    a stronger test of the per-spin code path than O2 triplet.
    Needs more CPHF iters than the default — OH has small gaps and
    GMRES converges more slowly."""
    mol = _oh_radical_mol()
    basis, result = _converged_uhf(mol)
    cphf_opts = vq.CPHFOptions(max_iter=300, tol=1e-7)
    r = vq.compute_hessian_uhf_analytic(mol, basis, result,
                                          basis_name="sto-3g",
                                          cphf_options=cphf_opts)
    H_ps = _pyscf_uhf_hessian(
        [["O", (0.0, 0.0, 0.0)], ["H", (0.0, 0.0, 0.96)]],
        basis="sto-3g", spin=1,
    )
    np.testing.assert_allclose(r.hessian, H_ps, atol=1e-6, rtol=0,
                                err_msg="OH UHF Hessian disagrees with PySCF")


# ---------------------------------------------------------------------------
# 3. Symmetry + linearity + structure
# ---------------------------------------------------------------------------

def test_returned_hessian_symmetric():
    mol = _o2_triplet_mol()
    basis, result = _converged_uhf(mol)
    r = vq.compute_hessian_uhf_analytic(mol, basis, result,
                                          basis_name="sto-3g")
    np.testing.assert_allclose(r.hessian, r.hessian.T, atol=1e-12)


def test_o2_linear_5_zero_modes():
    """Linear diatomic → 5 zero modes + 1 stretch."""
    mol = _o2_triplet_mol()
    basis, result = _converged_uhf(mol)
    r = vq.compute_hessian_uhf_analytic(mol, basis, result,
                                          basis_name="sto-3g")
    assert r.is_linear
    n_zero = int(np.sum(np.abs(r.frequencies_cm1) < 1e-3))
    assert n_zero == 5


# ---------------------------------------------------------------------------
# 4. Validation paths
# ---------------------------------------------------------------------------

def test_unconverged_uhf_rejected():
    mol = _o2_triplet_mol()
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.UHFOptions()
    opts.max_iter = 1
    opts.use_diis = False
    result = vq.run_uhf(mol, basis, opts)
    assert not result.converged
    with pytest.raises(ValueError, match="not converged"):
        vq.compute_hessian_uhf_analytic(mol, basis, result,
                                          basis_name="sto-3g")


def test_basis_name_required():
    mol = _o2_triplet_mol()
    basis, result = _converged_uhf(mol)
    with pytest.raises(ValueError, match="basis_name"):
        vq.compute_hessian_uhf_analytic(mol, basis, result)
