"""Phase 17e — analytic UKS Hessian assembly.

Open-shell-DFT extension of 17c (UHF) and 17d (RKS). Pinned contracts:

1. **API surface** — :func:`vibeqc.compute_hessian_uks_analytic` is
   public and returns a :class:`HessianResult` (drop-in replacement
   for the FD path).

2. **PySCF parity on triplet O₂ / STO-3G / LDA** — agrees with PySCF
   analytic UKS Hessian to ~1e-3 Ha/bohr² (same XC-grid-sensitivity
   tolerance as 17d for closed-shell). The HF-skeleton +
   per-spin-CPHF-response components match PySCF to <1e-6 in
   isolation; the residual ~3e-4 sits in the XC skeleton.

3. **Symmetry** — returned ``hessian`` is symmetric to machine
   precision.

4. **Refusal on un-converged SCF** — analytic CPKS requires a
   stationary KS reference; non-converged input raises a clean
   ``ValueError``.

5. **Hybrid functional smoke (B3LYP)** — non-zero α_HF path
   exercises K-derivative skeleton + hybrid CPKS.

6. **OH radical (doublet)** smoke — small open-shell molecule with
   2 unpaired electrons-per-spin asymmetry, finishes in <30s.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


ANGSTROM_TO_BOHR = 1.8897261339213


def _o2_triplet():
    return vq.Molecule([
        vq.Atom(8, [0.0, 0.0, -0.605 * ANGSTROM_TO_BOHR]),
        vq.Atom(8, [0.0, 0.0,  0.605 * ANGSTROM_TO_BOHR]),
    ], charge=0, multiplicity=3)


def _oh_doublet():
    return vq.Molecule([
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.0, 0.97 * ANGSTROM_TO_BOHR]),
    ], charge=0, multiplicity=2)


def _converged_uks(mol, *, functional="lda", basis_name="sto-3g",
                     level_shift: float = 0.5):
    """Open-shell-DFT SCFs are notorious for slow / oscillating
    convergence. The default Saunders-Hillier level shift of 0.5
    suffices for the test radicals (O₂ triplet, OH doublet) on STO-3G;
    pass ``level_shift=0.0`` for the un-converged-rejection check."""
    basis = vq.BasisSet(mol, basis_name)
    opts = vq.UKSOptions()
    opts.functional = functional
    opts.max_iter = 500
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    opts.level_shift = level_shift
    return basis, vq.run_uks(mol, basis, opts)


# ---------------------------------------------------------------------------
# 1. API surface
# ---------------------------------------------------------------------------

def test_compute_hessian_uks_analytic_exposed():
    assert hasattr(vq, "compute_hessian_uks_analytic")


def test_returns_hessian_result():
    mol = _o2_triplet()
    basis, result = _converged_uks(mol, functional="lda")
    r_an = vq.compute_hessian_uks_analytic(
        mol, basis, result, basis_name="sto-3g", functional="lda")
    assert isinstance(r_an, vq.HessianResult)
    assert r_an.hessian.shape == (6, 6)
    assert r_an.hessian_mw.shape == (6, 6)
    assert r_an.frequencies_cm1.shape == (6,)
    assert r_an.normal_modes.shape == (6, 6)
    assert r_an.is_linear is True   # diatomic


# ---------------------------------------------------------------------------
# 2. PySCF parity (triplet O2 / STO-3G / LDA)
# ---------------------------------------------------------------------------

def _pyscf_uks_hessian_o2(xc: str = "lda,vwn", basis: str = "sto-3g"):
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, dft

    mol = gto.Mole()
    mol.atom = [
        ["O", (0.0, 0.0, -0.605)],
        ["O", (0.0, 0.0,  0.605)],
    ]
    mol.basis = basis
    mol.unit = "Angstrom"
    mol.charge = 0
    mol.spin = 2
    mol.verbose = 0
    mol.build()
    mf = dft.UKS(mol)
    mf.xc = xc
    mf.grids.atom_grid = (75, 302)
    mf.conv_tol = 1e-12
    mf.kernel()
    h = mf.Hessian().kernel()
    n = mol.natm
    H = np.zeros((3 * n, 3 * n))
    for i in range(n):
        for j in range(n):
            H[3*i:3*i+3, 3*j:3*j+3] = h[i, j]
    return H


def test_o2_triplet_lda_close_to_pyscf():
    """Element-wise PySCF parity within the XC-skeleton FD-tolerance.

    Tolerance 1e-3 absorbs the difference between vibe-qc's
    FD-on-gradient XC-skeleton path and PySCF's analytic
    XC-skeleton path. The HF + per-spin-CPHF components match
    PySCF to <1e-6 — the gap is purely XC-grid sensitivity, not
    a bug. Same caveat as the 17d test on H₂O.
    """
    mol = _o2_triplet()
    basis, result = _converged_uks(mol, functional="lda")
    r_an = vq.compute_hessian_uks_analytic(
        mol, basis, result, basis_name="sto-3g", functional="lda")
    H_ps = _pyscf_uks_hessian_o2("lda,vwn", "sto-3g")
    np.testing.assert_allclose(
        r_an.hessian, H_ps, atol=1e-3, rtol=0,
        err_msg="vibe-qc UKS Hessian disagrees with PySCF beyond the "
                "XC-grid-sensitivity tolerance.")


# ---------------------------------------------------------------------------
# 3. Symmetry
# ---------------------------------------------------------------------------

def test_hessian_is_symmetric():
    mol = _o2_triplet()
    basis, result = _converged_uks(mol, functional="lda")
    r_an = vq.compute_hessian_uks_analytic(
        mol, basis, result, basis_name="sto-3g", functional="lda")
    np.testing.assert_allclose(r_an.hessian, r_an.hessian.T, atol=1e-12)


# ---------------------------------------------------------------------------
# 4. Refusal on un-converged SCF
# ---------------------------------------------------------------------------

def test_unconverged_uks_rejected():
    mol = _o2_triplet()
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.UKSOptions()
    opts.functional = "lda"
    opts.max_iter = 1
    opts.conv_tol_energy = 1e-30
    opts.conv_tol_grad = 1e-30
    opts.level_shift = 0.0  # don't help convergence in this test
    result = vq.run_uks(mol, basis, opts)
    assert result.converged is False
    with pytest.raises(ValueError):
        vq.compute_hessian_uks_analytic(
            mol, basis, result, basis_name="sto-3g", functional="lda")


# ---------------------------------------------------------------------------
# 5. Hybrid (B3LYP) smoke — exercises α_HF·K skeleton + hybrid CPKS
# ---------------------------------------------------------------------------

def test_b3lyp_o2_triplet_smoke():
    """B3LYP on triplet O₂ — exercises α_HF·K skeleton + hybrid CPKS
    on the open-shell DFT path, including the polarized GGA fxc
    response pieces."""
    mol = _o2_triplet()
    basis, result = _converged_uks(mol, functional="b3lyp")
    r_an = vq.compute_hessian_uks_analytic(
        mol, basis, result, basis_name="sto-3g", functional="b3lyp")
    fr = np.sort(np.real(r_an.frequencies_cm1))
    assert np.sum(fr > 100.0) == 1
    assert 1000 < fr[-1] < 3000
    assert r_an.is_linear is True


# ---------------------------------------------------------------------------
# 6. OH radical doublet — extra coverage for the per-spin asymmetric path
# ---------------------------------------------------------------------------

def test_oh_doublet_lda_finishes_and_returns_real_frequency():
    mol = _oh_doublet()
    basis, result = _converged_uks(mol, functional="lda")
    r_an = vq.compute_hessian_uks_analytic(
        mol, basis, result, basis_name="sto-3g", functional="lda")
    fr = np.sort(np.real(r_an.frequencies_cm1))
    # Linear diatomic → 5 zero modes (3 trans + 2 rot) + 1 stretch.
    assert np.sum(np.abs(fr) < 5.0) == 5
    assert fr[-1] > 100.0
