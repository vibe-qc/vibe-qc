"""Phase 17b-3 — analytic RHF Hessian assembly via CPHF + skeleton-deriv.

Contracts pinned here:

1. **API surface** — :func:`vibeqc.compute_hessian_rhf_analytic` is
   public and returns a :class:`HessianResult` indistinguishable in
   shape from the FD path (:func:`compute_hessian_fd`), so all the
   downstream consumers (frequencies, IR intensities, thermo) plug
   in unchanged.

2. **PySCF parity on H₂O / STO-3G** — the assembled analytic Hessian
   matches PySCF's analytic Hessian to ~1e-8 Ha/bohr² (FD-truncation
   on the h1ao step) and frequencies to <0.01 cm⁻¹.

3. **Symmetry** — returned ``hessian`` is symmetric to machine
   precision.

4. **Refusal on un-converged SCF** — analytic CPHF requires a
   stationary HF reference; non-converged input raises a clean
   ValueError.

5. **Smoke test on H₂ / STO-3G** — the linear-molecule branch produces
   sensible 5 zero-mode + 1 vibration structure with consistent
   frequencies vs the FD path.
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


def _converged_rhf(mol, basis_name="sto-3g"):
    basis = vq.BasisSet(mol, basis_name)
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-9
    return basis, vq.run_rhf(mol, basis, opts)


# ---------------------------------------------------------------------------
# 1. API
# ---------------------------------------------------------------------------

def test_compute_hessian_rhf_analytic_exposed():
    assert hasattr(vq, "compute_hessian_rhf_analytic")


def test_returns_hessian_result_compatible_with_fd_path():
    """Both code paths return :class:`HessianResult` with the same
    fields, so frequency/IR/thermo consumers are agnostic."""
    mol = _h2o_mol()
    basis, result = _converged_rhf(mol)
    r_an = vq.compute_hessian_rhf_analytic(mol, basis, result,
                                            basis_name="sto-3g")
    assert isinstance(r_an, vq.HessianResult)
    assert r_an.hessian.shape == (9, 9)
    assert r_an.hessian_mw.shape == (9, 9)
    assert r_an.frequencies_cm1.shape == (9,)
    assert r_an.normal_modes.shape == (9, 9)
    assert r_an.is_linear is False


# ---------------------------------------------------------------------------
# 2 & 3. PySCF parity + symmetry
# ---------------------------------------------------------------------------

def _pyscf_rhf_hessian_h2o(basis: str = "sto-3g"):
    """Reference analytic Hessian via PySCF on the same H2O geometry."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, scf
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
    mf = scf.RHF(mol)
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


def test_h2o_analytic_hessian_matches_pyscf():
    """Element-by-element parity vs PySCF analytic Hessian on H2O / STO-3G.
    Tolerance ~1e-7 absorbs FD-truncation noise on h1ao + s1ao at
    step 1e-4."""
    mol = _h2o_mol()
    basis, result = _converged_rhf(mol)
    r_an = vq.compute_hessian_rhf_analytic(mol, basis, result,
                                            basis_name="sto-3g")

    H_ps, freqs_ps = _pyscf_rhf_hessian_h2o("sto-3g")
    np.testing.assert_allclose(
        r_an.hessian, H_ps, atol=1e-7, rtol=0,
        err_msg="analytic RHF Hessian disagrees with PySCF on H2O/STO-3G")


def test_h2o_analytic_frequencies_match_pyscf():
    mol = _h2o_mol()
    basis, result = _converged_rhf(mol)
    r_an = vq.compute_hessian_rhf_analytic(mol, basis, result,
                                            basis_name="sto-3g")
    _, freqs_ps = _pyscf_rhf_hessian_h2o("sto-3g")
    freqs_vq = np.sort(r_an.frequencies_cm1)[-3:]   # 3 vibrations
    np.testing.assert_allclose(freqs_vq, freqs_ps, atol=0.01,
                                err_msg="analytic frequencies disagree with PySCF")


def test_returned_hessian_is_symmetric():
    mol = _h2o_mol()
    basis, result = _converged_rhf(mol)
    r_an = vq.compute_hessian_rhf_analytic(mol, basis, result,
                                            basis_name="sto-3g")
    np.testing.assert_allclose(r_an.hessian, r_an.hessian.T, atol=1e-12)


# ---------------------------------------------------------------------------
# 4. Validation paths
# ---------------------------------------------------------------------------

def test_unconverged_scf_rejected():
    mol = _h2o_mol()
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.RHFOptions()
    opts.max_iter = 1
    opts.use_diis = False
    result = vq.run_rhf(mol, basis, opts)
    assert not result.converged
    with pytest.raises(ValueError, match="not converged"):
        vq.compute_hessian_rhf_analytic(mol, basis, result,
                                         basis_name="sto-3g")


def test_basis_name_required():
    """The FD-on-Fock h1ao path needs the basis name to rebuild
    ``BasisSet`` at displaced geometries — there's no API to extract
    that from a built ``BasisSet`` object yet."""
    mol = _h2o_mol()
    basis, result = _converged_rhf(mol)
    with pytest.raises(ValueError, match="basis_name"):
        vq.compute_hessian_rhf_analytic(mol, basis, result)


# ---------------------------------------------------------------------------
# 5. Smoke test on linear molecule
# ---------------------------------------------------------------------------

def test_h2_linear_5_zero_modes_plus_one_vibration():
    """H2 is linear → 5 zero modes + 1 stretch. Both analytic and FD
    paths should agree on the structure."""
    mol = _h2_mol(R_bohr=1.4)
    basis, result = _converged_rhf(mol)
    r_an = vq.compute_hessian_rhf_analytic(mol, basis, result,
                                            basis_name="sto-3g")
    assert r_an.is_linear
    n_zero = int(np.sum(np.abs(r_an.frequencies_cm1) < 1e-3))
    assert n_zero == 5

    # Compare H2 stretch frequency with FD path
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-9
    r_fd = vq.compute_hessian_fd(mol, "sto-3g", method="RHF",
                                  scf_options=opts)
    f_an = r_an.frequencies_cm1[-1]
    f_fd = r_fd.frequencies_cm1[-1]
    assert abs(f_an - f_fd) < 1.0   # 1 cm⁻¹ tolerance for FD vs analytic
