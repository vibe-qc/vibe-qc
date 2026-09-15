"""AO evaluation on grid: values and spatial gradients."""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom,
    BasisSet,
    GridOptions,
    Molecule,
    build_grid,
    compute_overlap,
    evaluate_ao,
    evaluate_ao_with_gradient,
)


@pytest.fixture
def h2o_mol():
    return Molecule([
        Atom(8, [0.0, 0.0, 0.0]),
        Atom(1, [0.0, 1.5, -1.16]),
        Atom(1, [0.0, -1.5, -1.16]),
    ])


@pytest.mark.parametrize("basis_name", ["sto-3g", "6-31g", "6-31g*", "cc-pvdz"])
def test_grid_integrated_overlap_matches_analytic(h2o_mol, basis_name):
    """S_μν = ∫ χ_μ χ_ν dr computed on the grid agrees with libint's
    analytic overlap. The grid accuracy sets the tolerance."""
    basis = BasisSet(h2o_mol, basis_name)
    opts = GridOptions()
    opts.n_radial = 120
    g = build_grid(h2o_mol, opts)
    chi = np.asarray(evaluate_ao(basis, g.points))
    S_grid = chi.T @ (g.weights[:, None] * chi)
    S_ana = np.asarray(compute_overlap(basis))
    err = np.abs(S_grid - S_ana).max()
    assert err < 1e-5, f"{basis_name}: |S_grid - S_analytic|_max = {err:.2e}"


def test_evaluate_ao_and_with_gradient_give_same_values(h2o_mol):
    basis = BasisSet(h2o_mol, "6-31g*")
    rng = np.random.default_rng(42)
    pts = rng.uniform(-2, 2, size=(20, 3))
    v1 = np.asarray(evaluate_ao(basis, pts))
    v2, *_ = (np.asarray(m) for m in evaluate_ao_with_gradient(basis, pts))
    np.testing.assert_allclose(v1, v2, rtol=0, atol=1e-14)


@pytest.mark.parametrize("basis_name", ["sto-3g", "6-31g*", "cc-pvdz"])
def test_ao_gradient_matches_finite_difference(h2o_mol, basis_name):
    basis = BasisSet(h2o_mol, basis_name)
    rng = np.random.default_rng(42)
    pts = rng.uniform(-2.0, 2.0, size=(15, 3))
    _, gx, gy, gz = (np.asarray(m) for m in evaluate_ao_with_gradient(basis, pts))
    h = 1e-5
    for axis, g_ana in zip((0, 1, 2), (gx, gy, gz)):
        pts_p = pts.copy(); pts_p[:, axis] += h
        pts_m = pts.copy(); pts_m[:, axis] -= h
        fd = (np.asarray(evaluate_ao(basis, pts_p))
              - np.asarray(evaluate_ao(basis, pts_m))) / (2 * h)
        err = np.abs(fd - g_ana).max()
        assert err < 1e-8, (
            f"{basis_name} axis={axis}: |analytic - FD|_max = {err:.2e}"
        )


def test_density_integral_equals_electron_count(h2o_mol):
    """∫ ρ(r) dr = n_electrons, where ρ(r) = Σ_μν D_μν χ_μ χ_ν is
    built from the converged RHF density."""
    from vibeqc import run_rhf
    basis = BasisSet(h2o_mol, "sto-3g")
    result = run_rhf(h2o_mol, basis)
    D = np.asarray(result.density)
    opts = GridOptions()
    opts.n_radial = 120
    g = build_grid(h2o_mol, opts)
    chi = np.asarray(evaluate_ao(basis, g.points))
    rho = np.einsum("gm,mn,gn->g", chi, D, chi)
    n_elec = (rho * g.weights).sum()
    assert n_elec == pytest.approx(float(h2o_mol.n_electrons()), abs=1e-5)
