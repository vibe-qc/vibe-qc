"""Phase D4b-1 — imaginary-frequency dipole polarizability α(iω).

Contracts pinned here:

1. **API surface** — :func:`vibeqc.cphf.cphf_solve_dynamic_rhf` and
   :func:`vibeqc.cphf.dynamic_polarizability_rhf` are importable and
   listed in the module ``__all__``.

2. **Static limit** — ``α(i0)`` reproduces the static
   :func:`dipole_polarizability_rhf` tensor to machine precision
   (same orbital Hessian, ω² shift switched off).

3. **(A−B) consistency** — the new ``(A−B)`` orbital-Hessian action
   satisfies ``(A+B)·v − (A−B)·v = 2B·v`` against an explicit
   four-index build, and ``(A−B)`` is symmetric positive-definite
   for a stable closed-shell RHF.

4. **Monotone decay** — the isotropic ``α(iω)`` is strictly
   decreasing along the imaginary axis (the ω²-shifted operator is
   monotone in ω).

5. **High-frequency limit** — ``α(iω) → 0`` as ``ω → ∞``.

6. **Tensor symmetry** — ``α(iω)`` is a symmetric 3×3 tensor at every
   frequency.

7. **Casimir-Polder integrability** — the homo-molecular
   ``C₆ = (3/π) ∫₀^∞ α(iω)² dω`` is finite and positive (the D4
   reference-data use case — milestone D4b-2 builds on this).

8. **Input handling** — scalar vs sequence ``frequencies`` produce
   ``(3,3)`` vs ``(n,3,3)``; negative ω is rejected.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import compute_eri
from vibeqc.cphf import (
    CPHFOptions,
    cphf_solve_dynamic_rhf,
    cphf_solve_rhf,
    dipole_polarizability_rhf,
    dynamic_polarizability_rhf,
)
from vibeqc.cphf import (
    _cphf_mo_setup,
    _orbital_hessian_action,
    _orbital_hessian_minus_action,
)

ANGSTROM_TO_BOHR = 1.8897261339213


def _h2o_mol():
    return vq.Molecule([
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0,  0.7572 * ANGSTROM_TO_BOHR, 0.5868 * ANGSTROM_TO_BOHR]),
        vq.Atom(1, [0.0, -0.7572 * ANGSTROM_TO_BOHR, 0.5868 * ANGSTROM_TO_BOHR]),
    ], charge=0, multiplicity=1)


def _converged_rhf(mol, basis_name="sto-3g"):
    basis = vq.BasisSet(mol, basis_name)
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-9
    return basis, vq.run_rhf(mol, basis, opts)


# ---------------------------------------------------------------------------
# 1. API surface
# ---------------------------------------------------------------------------

def test_dynamic_api_exposed():
    import vibeqc.cphf as cphf
    assert "cphf_solve_dynamic_rhf" in cphf.__all__
    assert "dynamic_polarizability_rhf" in cphf.__all__
    assert callable(cphf.cphf_solve_dynamic_rhf)
    assert callable(cphf.dynamic_polarizability_rhf)


# ---------------------------------------------------------------------------
# 2. Static limit — α(i0) == static polarizability
# ---------------------------------------------------------------------------

def test_alpha_i0_equals_static():
    mol = _h2o_mol()
    basis, rhf = _converged_rhf(mol)
    eri = np.asarray(compute_eri(basis))
    a_static = dipole_polarizability_rhf(rhf, basis, mol, eri=eri)
    a_dynamic = dynamic_polarizability_rhf(rhf, basis, mol, 0.0, eri=eri)
    assert np.allclose(a_static, a_dynamic, atol=1e-10, rtol=0.0), (
        f"α(i0) must equal the static polarizability; "
        f"max|Δ| = {np.abs(a_static - a_dynamic).max():.3e}"
    )


def test_dynamic_solver_static_limit_doubles_static_U():
    """At ω=0 the dynamic solver's U₊ = X+Y convention carries a
    factor 2 relative to the static cphf_solve_rhf amplitude."""
    mol = _h2o_mol()
    basis, rhf = _converged_rhf(mol)
    eri = np.asarray(compute_eri(basis))
    from vibeqc import compute_dipole
    from vibeqc.properties import center_of_mass
    origin = center_of_mass(mol)
    dip = compute_dipole(basis, [float(x) for x in origin])
    M = np.stack([np.asarray(dip.x), np.asarray(dip.y), np.asarray(dip.z)])

    U_static = cphf_solve_rhf(M, rhf, eri)
    U_plus = cphf_solve_dynamic_rhf(M, rhf, eri, 0.0)
    assert np.allclose(U_plus, 2.0 * U_static, atol=1e-8, rtol=0.0)


# ---------------------------------------------------------------------------
# 3. (A−B) orbital-Hessian consistency
# ---------------------------------------------------------------------------

def test_a_minus_b_satisfies_2B_identity():
    """(A+B)·v − (A−B)·v must equal 2·B·v, with B the TD-HF coupling
    block B_{ia,jb} = 2(ia|jb) − (ib|ja)."""
    mol = _h2o_mol()
    basis, rhf = _converged_rhf(mol)
    eri = np.asarray(compute_eri(basis))
    C_occ, C_vir, _, eps_diff_inv, n_occ, n_vir = _cphf_mo_setup(rhf)

    rng = np.random.default_rng(1234)
    v = rng.standard_normal((n_occ, n_vir))

    apb = _orbital_hessian_action(v, C_occ, C_vir, eps_diff_inv, eri)
    amb = _orbital_hessian_minus_action(v, C_occ, C_vir, eps_diff_inv, eri)

    # Explicit 2B·v from the four-index ERI: 4(ia|jb)v − 2(ib|ja)v.
    mo_ovov = np.einsum("uvls,ui,va,lj,sb->iajb", eri, C_occ, C_vir,
                        C_occ, C_vir, optimize=True)              # (ia|jb)
    mo_ovvo = np.einsum("uvls,ui,vb,lj,sa->iajb", eri, C_occ, C_vir,
                        C_occ, C_vir, optimize=True)              # (ib|ja)
    two_b_v = (4.0 * np.einsum("iajb,jb->ia", mo_ovov, v)
               - 2.0 * np.einsum("iajb,jb->ia", mo_ovvo, v))

    assert np.allclose(apb - amb, two_b_v, atol=1e-10, rtol=0.0)


def test_a_minus_b_is_symmetric_positive_definite():
    mol = _h2o_mol()
    basis, rhf = _converged_rhf(mol)
    eri = np.asarray(compute_eri(basis))
    C_occ, C_vir, _, eps_diff_inv, n_occ, n_vir = _cphf_mo_setup(rhf)

    dim = n_occ * n_vir
    amb = np.zeros((dim, dim))
    for k in range(dim):
        e = np.zeros((n_occ, n_vir))
        e.flat[k] = 1.0
        amb[:, k] = _orbital_hessian_minus_action(
            e, C_occ, C_vir, eps_diff_inv, eri).ravel()

    assert np.abs(amb - amb.T).max() < 1e-10, "(A−B) must be symmetric"
    eigvals = np.linalg.eigvalsh(0.5 * (amb + amb.T))
    assert eigvals.min() > 0.0, (
        f"(A−B) must be positive-definite for a stable RHF; "
        f"lowest eigenvalue = {eigvals.min():.3e}"
    )


# ---------------------------------------------------------------------------
# 4-6. Physical behaviour along the imaginary axis
# ---------------------------------------------------------------------------

def test_alpha_iw_monotone_decreasing():
    mol = _h2o_mol()
    basis, rhf = _converged_rhf(mol)
    eri = np.asarray(compute_eri(basis))
    freqs = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
    a = dynamic_polarizability_rhf(rhf, basis, mol, freqs, eri=eri)
    iso = np.array([np.trace(t) / 3.0 for t in a])
    assert np.all(np.diff(iso) < 0.0), (
        f"α(iω) must strictly decrease along the imaginary axis; "
        f"isotropic values = {iso}"
    )


def test_alpha_iw_high_frequency_vanishes():
    mol = _h2o_mol()
    basis, rhf = _converged_rhf(mol)
    eri = np.asarray(compute_eri(basis))
    a_lo = dynamic_polarizability_rhf(rhf, basis, mol, 1.0, eri=eri)
    a_hi = dynamic_polarizability_rhf(rhf, basis, mol, 200.0, eri=eri)
    assert np.trace(a_hi) / 3.0 < 1e-3, "α(iω→∞) must vanish"
    assert np.trace(a_hi) / 3.0 < np.trace(a_lo) / 3.0


def test_alpha_iw_tensor_symmetric():
    mol = _h2o_mol()
    basis, rhf = _converged_rhf(mol)
    eri = np.asarray(compute_eri(basis))
    for omega in (0.0, 0.5, 2.0):
        a = dynamic_polarizability_rhf(rhf, basis, mol, omega, eri=eri)
        assert a.shape == (3, 3)
        assert np.abs(a - a.T).max() < 1e-8


def test_alpha_iw_positive_definite_at_each_frequency():
    """α(iω) is a response function — symmetric positive-definite at
    every point on the imaginary axis."""
    mol = _h2o_mol()
    basis, rhf = _converged_rhf(mol)
    eri = np.asarray(compute_eri(basis))
    for omega in (0.0, 0.5, 2.0, 8.0):
        a = dynamic_polarizability_rhf(rhf, basis, mol, omega, eri=eri)
        eigvals = np.linalg.eigvalsh(a)
        assert eigvals.min() > 0.0, (
            f"α(i{omega}) must be positive-definite; "
            f"eigenvalues = {eigvals}"
        )


# ---------------------------------------------------------------------------
# 7. Casimir-Polder C6 integrability — the D4 reference-data use case
# ---------------------------------------------------------------------------

def test_casimir_polder_c6_finite_and_positive():
    """C₆ = (3/π) ∫₀^∞ α(iω)² dω for a homo-pair must be finite and
    positive. Uses the standard D3/D4 substitution ω = ω₀(1−t)/(1+t)
    mapping (0,∞)→(0,1) with Gauss-Legendre nodes."""
    mol = _h2o_mol()
    basis, rhf = _converged_rhf(mol)
    eri = np.asarray(compute_eri(basis))

    # Gauss-Legendre on t∈(0,1); ω = ω0·(1−t)/(1+t), dω = −2ω0/(1+t)² dt.
    omega0 = 0.4
    n_pts = 24
    t_nodes, t_wts = np.polynomial.legendre.leggauss(n_pts)
    t = 0.5 * (t_nodes + 1.0)                  # (0,1)
    wt = 0.5 * t_wts
    omega = omega0 * (1.0 - t) / (1.0 + t)
    jac = 2.0 * omega0 / (1.0 + t) ** 2        # |dω/dt|

    a = dynamic_polarizability_rhf(rhf, basis, mol, omega, eri=eri)
    iso = np.array([np.trace(tt) / 3.0 for tt in a])     # α(iω) isotropic
    c6 = (3.0 / np.pi) * np.sum(wt * jac * iso * iso)

    assert np.isfinite(c6)
    assert c6 > 0.0, f"homo-molecular C6 must be positive; got {c6}"
    # STO-3G grossly underbinds, but C6 for a small molecule is O(1-100).
    assert 0.1 < c6 < 1.0e3, f"C6 out of physical range: {c6}"


# ---------------------------------------------------------------------------
# 8. Input handling
# ---------------------------------------------------------------------------

def test_scalar_vs_sequence_frequencies_shapes():
    mol = _h2o_mol()
    basis, rhf = _converged_rhf(mol)
    eri = np.asarray(compute_eri(basis))
    a_scalar = dynamic_polarizability_rhf(rhf, basis, mol, 0.5, eri=eri)
    assert a_scalar.shape == (3, 3)
    a_seq = dynamic_polarizability_rhf(rhf, basis, mol, [0.1, 0.5, 1.0],
                                       eri=eri)
    assert a_seq.shape == (3, 3, 3)
    # The 0.5 entry of the sequence must match the scalar call.
    assert np.allclose(a_seq[1], a_scalar, atol=1e-9)


def test_negative_frequency_rejected():
    mol = _h2o_mol()
    basis, rhf = _converged_rhf(mol)
    eri = np.asarray(compute_eri(basis))
    with pytest.raises(ValueError, match="must be >= 0"):
        dynamic_polarizability_rhf(rhf, basis, mol, -1.0, eri=eri)
    with pytest.raises(ValueError, match="omega must be >= 0"):
        cphf_solve_dynamic_rhf(np.zeros((basis.nbasis, basis.nbasis)),
                               rhf, eri, -0.5)


def test_dynamic_solver_rejects_unconverged_rhf():
    mol = _h2o_mol()
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.RHFOptions()
    opts.max_iter = 1            # force a non-converged result
    rhf = vq.run_rhf(mol, basis, opts)
    if rhf.converged:
        pytest.skip("RHF converged in 1 iteration — cannot test refusal")
    eri = np.asarray(compute_eri(basis))
    with pytest.raises(ValueError, match="not converged"):
        cphf_solve_dynamic_rhf(np.zeros((basis.nbasis, basis.nbasis)),
                               rhf, eri, 0.5)
