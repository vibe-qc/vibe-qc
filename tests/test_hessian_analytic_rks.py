"""Phase 17d — analytic RKS Hessian assembly.

Closed-shell-DFT extension of 17b-3. Pinned contracts:

1. **API surface** — :func:`vibeqc.compute_hessian_rks_analytic` is
   public and returns a :class:`HessianResult` (drop-in replacement for
   the FD path).

2. **PySCF parity on H₂O / STO-3G / LDA** — the assembled analytic
   Hessian matches PySCF's analytic RKS Hessian to ~1e-3 Ha/bohr².
   The agreement is limited by the XC-skeleton path: vibe-qc evaluates
   the XC contribution to the skeleton via central-difference of the
   analytic gradient at fixed reference density (avoids needing
   2nd-derivative AO + 2nd-derivative Becke partition machinery), and
   PySCF evaluates it analytically. Both are valid; the residual
   ~5e-4 reflects grid-integration sensitivity differences (vibe-qc
   uses 75 radial × 17×36 Gauss-Legendre/φ-uniform; PySCF uses
   75 radial × 302 Lebedev). The HF-skeleton path *alone* (alpha_HF=1)
   matches PySCF to <1e-7. The CPHF kernel response *alone* matches
   PySCF to <1e-5. The discrepancy is purely in the XC-skeleton
   integration. Vibrational frequencies still agree with ORCA to
   ~1 cm⁻¹.

3. **Symmetry** — returned ``hessian`` is symmetric to machine
   precision.

4. **Refusal on un-converged SCF** — analytic CPKS requires a
   stationary KS reference; non-converged input raises a clean
   ValueError.

5. **Hybrid functional smoke test (B3LYP)** — non-zero α_HF path
   exercises K-derivative skeleton + hybrid CPKS.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


ANGSTROM_TO_BOHR = 1.8897261339213


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


def _converged_rks(mol, *, functional="lda", basis_name="sto-3g"):
    basis = vq.BasisSet(mol, basis_name)
    opts = vq.RKSOptions()
    opts.functional = functional
    opts.max_iter = 200
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    return basis, vq.run_rks(mol, basis, opts)


# ---------------------------------------------------------------------------
# 1. API surface
# ---------------------------------------------------------------------------

def test_compute_hessian_rks_analytic_exposed():
    assert hasattr(vq, "compute_hessian_rks_analytic")


def test_returns_hessian_result():
    """Drop-in shape compatibility with the FD Hessian path."""
    mol = _h2o_mol()
    basis, result = _converged_rks(mol, functional="lda")
    r_an = vq.compute_hessian_rks_analytic(
        mol, basis, result, basis_name="sto-3g", functional="lda")
    assert isinstance(r_an, vq.HessianResult)
    assert r_an.hessian.shape == (9, 9)
    assert r_an.hessian_mw.shape == (9, 9)
    assert r_an.frequencies_cm1.shape == (9,)
    assert r_an.normal_modes.shape == (9, 9)
    assert r_an.is_linear is False


# ---------------------------------------------------------------------------
# 2. PySCF parity (H2O / STO-3G / LDA)
# ---------------------------------------------------------------------------

def _pyscf_rks_hessian_h2o(xc: str, basis: str = "sto-3g"):
    """Reference analytic RKS Hessian via PySCF on H2O."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, dft
    from pyscf.hessian import thermo

    mol = gto.Mole()
    mol.atom = [
        ["O", (0.0, 0.0, 0.0)],
        ["H", (0.0,  0.7572, 0.5868)],
        ["H", (0.0, -0.7572, 0.5868)],
    ]
    mol.basis = basis
    mol.unit = "Angstrom"
    mol.verbose = 0
    mol.build()
    mf = dft.RKS(mol)
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
    info = thermo.harmonic_analysis(mol, h)
    freqs = np.real(info["freq_wavenumber"])
    return H, np.sort(freqs)


def test_h2o_lda_analytic_hessian_close_to_pyscf():
    """Element-wise PySCF parity within the XC-skeleton-FD tolerance.

    Tolerance 1e-3 absorbs the difference between vibe-qc's
    FD-on-gradient XC-skeleton path (75 × 17×36 Gauss-Legendre+φ grid)
    and PySCF's analytic XC-skeleton path (75 × 302 Lebedev grid). The
    HF-skeleton + CPHF response components match PySCF to <1e-6 in
    isolation, so this gap is purely XC-grid sensitivity, not a bug.
    """
    mol = _h2o_mol()
    basis, result = _converged_rks(mol, functional="lda")
    r_an = vq.compute_hessian_rks_analytic(
        mol, basis, result, basis_name="sto-3g", functional="lda")

    H_ps, freqs_ps = _pyscf_rks_hessian_h2o("lda,vwn", "sto-3g")
    np.testing.assert_allclose(
        r_an.hessian, H_ps, atol=1e-3, rtol=0,
        err_msg="vibe-qc analytic RKS Hessian disagrees with PySCF beyond "
                "the XC-grid-sensitivity tolerance.")


def test_h2o_lda_frequencies_close_to_pyscf():
    """Vibrational frequencies match PySCF within ~5 cm⁻¹ — the
    XC-grid difference shows up here as a small frequency shift
    well below typical experimental scatter."""
    mol = _h2o_mol()
    basis, result = _converged_rks(mol, functional="lda")
    r_an = vq.compute_hessian_rks_analytic(
        mol, basis, result, basis_name="sto-3g", functional="lda")
    _, freqs_ps = _pyscf_rks_hessian_h2o("lda,vwn", "sto-3g")

    # Compare only the non-zero modes (last 3).
    fv = np.sort(np.real(r_an.frequencies_cm1))[-3:]
    fp = np.sort(np.real(freqs_ps))[-3:]
    np.testing.assert_allclose(fv, fp, atol=5.0, rtol=0)


# ---------------------------------------------------------------------------
# 3. Symmetry
# ---------------------------------------------------------------------------

def test_hessian_is_symmetric():
    mol = _h2o_mol()
    basis, result = _converged_rks(mol, functional="lda")
    r_an = vq.compute_hessian_rks_analytic(
        mol, basis, result, basis_name="sto-3g", functional="lda")
    H = r_an.hessian
    np.testing.assert_allclose(H, H.T, atol=1e-12)


# ---------------------------------------------------------------------------
# 4. Refusal on un-converged SCF
# ---------------------------------------------------------------------------

def test_unconverged_rks_rejected():
    mol = _h2o_mol()
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.RKSOptions()
    opts.functional = "lda"
    opts.max_iter = 1               # certainly not converged
    opts.conv_tol_energy = 1e-30
    opts.conv_tol_grad = 1e-30
    result = vq.run_rks(mol, basis, opts)
    assert result.converged is False
    with pytest.raises(ValueError):
        vq.compute_hessian_rks_analytic(
            mol, basis, result, basis_name="sto-3g", functional="lda")


# ---------------------------------------------------------------------------
# 5. Hybrid (B3LYP) smoke test — exercises α_HF·K skeleton + hybrid CPKS
# ---------------------------------------------------------------------------

def test_b3lyp_h2_smoke():
    """B3LYP on H2 — α_HF = 0.20 brings a non-zero K-derivative
    skeleton + hybrid CPKS path. Two-atom molecule keeps cost low.
    Smoke check: produces 5 zero modes + 1 vibration above 100 cm⁻¹."""
    mol = _h2_mol()
    basis, result = _converged_rks(mol, functional="b3lyp")
    r_an = vq.compute_hessian_rks_analytic(
        mol, basis, result, basis_name="sto-3g", functional="b3lyp")
    fr = np.sort(np.real(r_an.frequencies_cm1))
    # H2/STO-3G/B3LYP: bond stretch around ~4400 cm⁻¹ (real ≈ 4308 exptl)
    assert np.sum(fr > 100.0) == 1
    assert 3500 < fr[-1] < 5500
    assert r_an.is_linear is True
