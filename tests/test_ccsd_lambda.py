"""Closed-shell CCSD Lambda-equation prerequisite for A-CCSD(T)."""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc import BasisSet, RHFOptions, run_rhf
from vibeqc._vibeqc_core import Atom, Molecule
from vibeqc.density_fitting import DensityFitting
from vibeqc.dlpno._ccsd_cs import (
    _blocks,
    _flatten_amplitudes,
    cs_ccsd_energy,
    cs_ccsd_energy_gradient,
    cs_ccsd_residual,
    cs_ccsd_residual_jacobian,
    cs_ccsd_residual_vjp,
    cs_lambda_triples_correction,
    cs_triples_correction,
    run_cs_ccsd,
    run_cs_ccsd_lambda,
    run_cs_ccsd_lambda_iterative,
)
from vibeqc.cc import CCSDOptions, _run_accsd_t_from_mos, run_accsd_t

ANGSTROM_TO_BOHR = 1.8897259886
AUX = "def2-svp-rifit"

H2O_ATOMS = [
    (8, [0.0, 0.0, 0.0]),
    (1, [0.0, 0.793353 * ANGSTROM_TO_BOHR, -0.613510 * ANGSTROM_TO_BOHR]),
    (1, [0.0, -0.793353 * ANGSTROM_TO_BOHR, -0.613510 * ANGSTROM_TO_BOHR]),
]


@pytest.mark.parametrize("mode", ["blocked", "direct", "disk"])
def test_accsdt_rejects_unimplemented_bounded_triples_modes(mode):
    options = CCSDOptions(triples="A-CCSD(T)", triples_memory_mode=mode)
    with pytest.raises(NotImplementedError, match="dense Lambda-triples"):
        _run_accsd_t_from_mos(None, None, None, None, 0.0, options)


def test_accsdt_rejects_native_workspace_budget():
    options = CCSDOptions(
        triples="A-CCSD(T)",
        requested_memory_bytes=1024,
    )
    with pytest.raises(NotImplementedError, match="cannot enforce"):
        _run_accsd_t_from_mos(None, None, None, None, 0.0, options)


@pytest.fixture(scope="module")
def h2o_ccsd_system():
    mol = Molecule([Atom(z, p) for z, p in H2O_ATOMS], charge=0, multiplicity=1)
    basis = BasisSet(mol, "sto-3g")
    opts = RHFOptions()
    opts.max_iter = 100
    rhf = run_rhf(mol, basis, opts)
    assert rhf.converged

    df = DensityFitting(basis, BasisSet(mol, AUX), aux_basis_name=AUX)
    C = np.asarray(rhf.mo_coeffs)
    F = np.asarray(rhf.fock)
    n_occ = mol.n_electrons() // 2
    f_mo = C.T @ F @ C
    Co, Cv = C[:, :n_occ], C[:, n_occ:]
    B_ov = np.asarray(df.mo_transform(Co, Cv))
    B_vv = np.asarray(df.mo_transform(Cv, Cv))
    B_oo = np.asarray(df.mo_transform(Co, Co))
    ccsd = run_cs_ccsd(
        f_mo,
        B_ov,
        B_vv,
        B_oo,
        n_occ,
        e_hf=rhf.energy,
        conv_tol=1e-11,
    )
    assert ccsd.converged
    return ccsd, f_mo, _blocks(B_ov, B_vv, B_oo), n_occ


def _projected_dense_lambda(t1, t2, f_oo, f_vv, f_ov, V):
    """Physical restricted Lambda solution from the dense Jacobian oracle."""
    jacobian = cs_ccsd_residual_jacobian(t1, t2, f_oo, f_vv, f_ov, V)
    gradient = _flatten_amplitudes(
        *cs_ccsd_energy_gradient(t1, t2, f_ov, V)
    )
    n_singles = t1.size
    n_total = n_singles + t2.size
    partner = np.arange(t2.size).reshape(t2.shape)
    partner = partner.transpose(1, 0, 3, 2).ravel()
    pairs = [(index, int(other)) for index, other in enumerate(partner)
             if index <= other]
    basis = np.zeros((n_total, n_singles + len(pairs)))
    basis[:n_singles, :n_singles] = np.eye(n_singles)
    for column, (first, second) in enumerate(pairs, start=n_singles):
        if first == second:
            basis[n_singles + first, column] = 1.0
        else:
            basis[n_singles + first, column] = 1.0 / np.sqrt(2.0)
            basis[n_singles + second, column] = 1.0 / np.sqrt(2.0)

    reduced = basis.T @ jacobian.T @ basis
    multiplier = basis @ np.linalg.solve(reduced, -(basis.T @ gradient))
    multiplier1 = multiplier[:n_singles].reshape(t1.shape)
    multiplier2 = multiplier[n_singles:].reshape(t2.shape)
    l1 = 0.5 * multiplier1
    l2 = (
        2.0 * multiplier2 + multiplier2.transpose(0, 1, 3, 2)
    ) / 3.0
    return l1, l2


def test_ccsd_lambda_solves_left_amplitude_equations(h2o_ccsd_system):
    ccsd, f_mo, V, n_occ = h2o_ccsd_system
    f_oo = f_mo[:n_occ, :n_occ]
    f_vv = f_mo[n_occ:, n_occ:]
    f_ov = f_mo[:n_occ, n_occ:]

    lam = run_cs_ccsd_lambda(ccsd.t1, ccsd.t2, f_oo, f_vv, f_ov, V)

    assert lam.converged
    assert lam.residual_norm < 1e-10
    assert lam.n_amplitudes == ccsd.t1.size + ccsd.t2.size
    assert lam.l1.shape == ccsd.t1.shape
    assert lam.l2.shape == ccsd.t2.shape
    assert np.linalg.norm(lam.l2) > np.linalg.norm(lam.l1)


def test_ccsd_lambda_lagrangian_is_stationary(h2o_ccsd_system):
    ccsd, f_mo, V, n_occ = h2o_ccsd_system
    f_oo = f_mo[:n_occ, :n_occ]
    f_vv = f_mo[n_occ:, n_occ:]
    f_ov = f_mo[:n_occ, n_occ:]
    lam = run_cs_ccsd_lambda(ccsd.t1, ccsd.t2, f_oo, f_vv, f_ov, V)
    lam_flat = _flatten_amplitudes(lam.l1, lam.l2)

    rng = np.random.default_rng(20260712)
    d1 = rng.normal(size=ccsd.t1.shape)
    d2 = rng.normal(size=ccsd.t2.shape)
    d = _flatten_amplitudes(d1, d2)
    d /= np.linalg.norm(d)
    d1 = d[: ccsd.t1.size].reshape(ccsd.t1.shape)
    d2 = d[ccsd.t1.size :].reshape(ccsd.t2.shape)

    def lagrangian(alpha):
        t1 = ccsd.t1 + alpha * d1
        t2 = ccsd.t2 + alpha * d2
        r1, r2 = cs_ccsd_residual(t1, t2, f_oo, f_vv, f_ov, V)
        return cs_ccsd_energy(t1, t2, f_ov, V) + float(
            np.dot(lam_flat, _flatten_amplitudes(r1, r2))
        )

    h = 1e-5
    directional_derivative = (lagrangian(h) - lagrangian(-h)) / (2.0 * h)
    assert abs(directional_derivative) < 1e-8


def test_iterative_lambda_matches_dense_oracle(h2o_ccsd_system):
    ccsd, f_mo, V, n_occ = h2o_ccsd_system
    f_oo = f_mo[:n_occ, :n_occ]
    f_vv = f_mo[n_occ:, n_occ:]
    f_ov = f_mo[:n_occ, n_occ:]

    dense_l1, dense_l2 = _projected_dense_lambda(
        ccsd.t1, ccsd.t2, f_oo, f_vv, f_ov, V
    )
    iterative = run_cs_ccsd_lambda_iterative(
        ccsd.t1,
        ccsd.t2,
        f_oo,
        f_vv,
        f_ov,
        V,
        max_iter=200,
        residual_tol=1e-9,
    )

    assert iterative.converged
    assert iterative.residual_norm < 1e-9
    np.testing.assert_allclose(iterative.l1, dense_l1, atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose(iterative.l2, dense_l2, atol=2e-7, rtol=2e-7)


def test_ccsd_residual_jacobian_transpose_is_adjoint(h2o_ccsd_system):
    ccsd, f_mo, V, n_occ = h2o_ccsd_system
    f_oo = f_mo[:n_occ, :n_occ]
    f_vv = f_mo[n_occ:, n_occ:]
    f_ov = f_mo[:n_occ, n_occ:]
    jac = cs_ccsd_residual_jacobian(ccsd.t1, ccsd.t2, f_oo, f_vv, f_ov, V)

    rng = np.random.default_rng(7)
    x = rng.normal(size=jac.shape[1])
    y = rng.normal(size=jac.shape[0])

    assert float(np.dot(y, jac @ x)) == pytest.approx(
        float(np.dot(jac.T @ y, x)), abs=1e-10
    )

    seed1 = y[: ccsd.t1.size].reshape(ccsd.t1.shape)
    seed2 = y[ccsd.t1.size :].reshape(ccsd.t2.shape)
    vjp1, vjp2 = cs_ccsd_residual_vjp(
        ccsd.t1, ccsd.t2, seed1, seed2, f_oo, f_vv, f_ov, V
    )
    np.testing.assert_allclose(
        _flatten_amplitudes(vjp1, vjp2), jac.T @ y, atol=2e-8, rtol=2e-8
    )


def test_lambda_triples_reduces_to_standard_t_when_lambda_equals_t(
    h2o_ccsd_system,
):
    ccsd, f_mo, V, n_occ = h2o_ccsd_system
    eps_o = np.diag(f_mo[:n_occ, :n_occ])
    eps_v = np.diag(f_mo[n_occ:, n_occ:])

    e_standard = cs_triples_correction(
        ccsd.t1, ccsd.t2, V["ovvv"], V["ooov"], V["ovov"], eps_o, eps_v
    )
    e_lambda_as_t = cs_lambda_triples_correction(
        ccsd.t1,
        ccsd.t2,
        ccsd.t1,
        ccsd.t2,
        V["ovvv"],
        V["ooov"],
        V["ovov"],
        eps_o,
        eps_v,
    )

    assert e_lambda_as_t == pytest.approx(e_standard, abs=1e-12)


def test_lambda_triples_uses_solved_left_amplitudes(h2o_ccsd_system):
    ccsd, f_mo, V, n_occ = h2o_ccsd_system
    f_oo = f_mo[:n_occ, :n_occ]
    f_vv = f_mo[n_occ:, n_occ:]
    f_ov = f_mo[:n_occ, n_occ:]
    eps_o = np.diag(f_oo)
    eps_v = np.diag(f_vv)
    lam = run_cs_ccsd_lambda_iterative(
        ccsd.t1, ccsd.t2, f_oo, f_vv, f_ov, V,
        residual_tol=1e-9,
    )

    e_standard = cs_triples_correction(
        ccsd.t1, ccsd.t2, V["ovvv"], V["ooov"], V["ovov"], eps_o, eps_v
    )
    e_accsdt = cs_lambda_triples_correction(
        ccsd.t1,
        ccsd.t2,
        lam.l1,
        lam.l2,
        V["ovvv"],
        V["ooov"],
        V["ovov"],
        eps_o,
        eps_v,
    )

    assert np.isfinite(e_accsdt)
    assert abs(e_accsdt) > 1e-8
    assert e_accsdt != pytest.approx(e_standard, abs=1e-10)


def test_accsdt_matches_published_psi4_anchor():
    """Crawford-Stanton stretched-water CCSD(AT)/cc-pVDZ benchmark."""
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, -0.14687455]),
            Atom(1, [0.0, 1.89407696, 1.16550258]),
            Atom(1, [0.0, -1.89407696, 1.16550258]),
        ],
        charge=0,
        multiplicity=1,
    )
    basis = BasisSet(mol, "cc-pvdz")
    rhf_options = RHFOptions()
    rhf_options.max_iter = 150
    rhf_options.conv_tol_energy = 1e-12
    rhf = run_rhf(mol, basis, rhf_options)
    options = CCSDOptions(
        density_fit=False,
        compute_triples=True,
        triples="A-CCSD(T)",
        n_frozen_core=0,
        max_iter=150,
        conv_tol_energy=1e-11,
        conv_tol_residual=1e-9,
        diis_subspace_size=8,
    )

    result = run_accsd_t(mol, basis, rhf, options)

    assert result.converged
    assert result.e_hf == pytest.approx(-75.930810791060466, abs=2e-10)
    assert result.e_ccsd == pytest.approx(-76.164158766102886, abs=2e-9)
    assert result.e_t == pytest.approx(-0.004719400975376, abs=2e-9)
    assert result.e_total == pytest.approx(-76.168878167078262, abs=2e-9)
