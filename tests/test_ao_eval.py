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


@pytest.mark.parametrize("pure", [False, True])
def test_ao_shell_scratch_preserves_high_l_gradients_and_zero_tails(pure):
    """Reused vectors must not leak values between points or angular shells."""
    from vibeqc._vibeqc_core import ShellInfo

    origin = np.array([0.13, -0.27, 0.31])
    mol = Molecule([Atom(2, origin)])
    shells = [ShellInfo(
        atom_index=0, l=l, pure=pure, origin=origin,
        exponents=[0.2, 0.7, 2.3], coefficients=[0.6, -0.15, 0.8],
    ) for l in range(7)]
    basis = BasisSet(mol, shells, "AO scratch control", False)
    points = np.array([
        origin, origin + [0.3, -0.6, 0.9], [1000., 1000., 1000.],
        origin + [-0.7, 0.2, 0.4], origin,
    ])
    values, *gradients = (
        np.asarray(x) for x in evaluate_ao_with_gradient(basis, points)
    )
    np.testing.assert_allclose(evaluate_ao(basis, points), values, atol=1e-14)
    for matrix in [values, *gradients]:
        np.testing.assert_array_equal(matrix[2], np.zeros(basis.nbasis))
        np.testing.assert_array_equal(matrix[0], matrix[-1])
    h = 1e-5
    for axis, gradient in enumerate(gradients):
        shift = np.eye(3)[axis] * h
        fd = (evaluate_ao(basis, points + shift)
              - evaluate_ao(basis, points - shift)) / (2 * h)
        np.testing.assert_allclose(gradient, fd, rtol=0, atol=1e-8)


@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("pre_normalized", [False, True])
def test_tensor_grid_signed_shells_match_general_evaluator(pure, pre_normalized):
    """Factorization preserves contracted s through g and libint conventions."""
    from vibeqc import _vibeqc_core as core

    center = np.array([0.13, -0.27, 0.31])
    mol = Molecule([Atom(2, center)])
    shells = [core.ShellInfo(
        atom_index=0, l=l, pure=pure, origin=center,
        exponents=[0.2, 0.7, 2.3], coefficients=[0.6, -0.15, 0.8],
    ) for l in range(5)]
    basis = BasisSet(mol, shells, "Tensor grid control", pre_normalized)
    # Unequal, nonuniform axes and repeated points are valid tensor products.
    axes = [np.array([-0.8, 0.13, 0.13, 1.2, 1000.]),
            np.array([-1.1, 0.4, 0.7]), np.array([-0.3, 0.31, 0.9, 1.7])]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    shifts = np.array([[0., 0., 0.], [0.3, -0.4, 0.7], [-1.2, 0.5, -0.3]])
    expected = sum(core.evaluate_ao(basis, points - shift) for shift in shifts)
    actual = core.evaluate_ao_tensor_grid(basis, *axes, shifts)
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-15)
    assert actual.shape == (60, basis.nbasis)
    np.testing.assert_array_equal(actual[-12:], 0.)
    # Duplicate and reordered translations must be retained, not deduplicated.
    reordered = shifts[[2, 0, 2, 1]]
    expected = sum(core.evaluate_ao(basis, points - shift) for shift in reordered)
    np.testing.assert_allclose(
        core.evaluate_ao_tensor_grid(basis, *axes, reordered), expected,
        rtol=3e-14, atol=3e-15,
    )


def test_tensor_grid_empty_domains_and_nonfinite_rejection(h2o_mol):
    from vibeqc import _vibeqc_core as core

    basis = BasisSet(h2o_mol, "sto-3g")
    axis = np.array([0., 1.])
    shifts = np.zeros((1, 3))
    assert core.evaluate_ao_tensor_grid(basis, [], axis, axis, shifts).shape == (0, basis.nbasis)
    np.testing.assert_array_equal(
        core.evaluate_ao_tensor_grid(basis, axis, axis, axis, np.zeros((0, 3))),
        np.zeros((8, basis.nbasis)),
    )
    for coordinate in [np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError, match="finite"):
            core.evaluate_ao_tensor_grid(basis, [coordinate], axis, axis, shifts)
        with pytest.raises(ValueError, match="finite"):
            core.evaluate_ao_tensor_grid(basis, axis, axis, axis, [[coordinate, 0., 0.]])
    with pytest.raises(TypeError):
        core.evaluate_ao_tensor_grid(basis, axis, axis, axis, np.zeros((3, 2)))


def test_tensor_grid_image_sum_is_thread_independent(h2o_mol):
    from vibeqc import _vibeqc_core as core

    basis = BasisSet(h2o_mol, "sto-3g")
    axes = [np.linspace(-1., 2., n) for n in (7, 8, 9)]
    shifts = np.array([[0., 0., 0.], [1., -2., 3.], [-1., 2., -3.]])
    old_threads = core.get_num_threads()
    try:
        core.set_num_threads(1)
        serial = core.evaluate_ao_tensor_grid(basis, *axes, shifts)
        core.set_num_threads(2)
        parallel = core.evaluate_ao_tensor_grid(basis, *axes, shifts)
    finally:
        core.set_num_threads(old_threads)
    np.testing.assert_array_equal(serial, parallel)


@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("pre_normalized", [False, True])
def test_bloch_multi_k_reuses_general_image_values(pure, pre_normalized):
    """Sharing real panels preserves signed s-g shells and the +ik.T gauge."""
    from vibeqc import _vibeqc_core as core

    center = np.array([0.13, -0.27, 0.31])
    mol = Molecule([Atom(2, center)])
    shells = [core.ShellInfo(
        atom_index=0, l=l, pure=pure, origin=center,
        exponents=[0.2, 0.7, 2.3], coefficients=[0.6, -0.15, 0.8],
    ) for l in range(5)]
    basis = BasisSet(mol, shells, "Bloch panel control", pre_normalized)
    # An irregular skew point set crosses the native tile boundary and tail.
    points = np.random.default_rng(17).normal(size=(517, 3))
    points = points @ np.array([[2., 0., 0.], [0.7, 1.8, 0.], [0.3, 0.2, 2.1]])
    points[-1] = [1000., 1000., 1000.]
    shifts = np.array([[0.3, -0.4, 0.7], [0., 0., 0.], [-1.2, 0.5, -0.3],
                       [0.3, -0.4, 0.7]])
    kpoints = np.array([[0., 0., 0.], [0.4, -0.3, 0.2], [-0.4, 0.3, -0.2],
                        [0.4, -0.3, 0.2]])
    result = core.evaluate_bloch_ao_multi_k(basis, points, kpoints, shifts)
    for k, actual in zip(kpoints, result):
        expected = core.evaluate_bloch_ao(basis, points, k, shifts)
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-15)
        np.testing.assert_array_equal(actual[-1], 0.)
    np.testing.assert_array_equal(result[1], result[2].conj())
    np.testing.assert_array_equal(result[1], result[3])
    assert not np.shares_memory(result[1], result[3])
    # A fresh geometry/point domain cannot inherit a cached panel.
    displaced = core.evaluate_bloch_ao_multi_k(basis, points + 0.1, kpoints, shifts)
    np.testing.assert_allclose(
        displaced[1], core.evaluate_bloch_ao(basis, points + 0.1, kpoints[1], shifts),
        rtol=2e-14, atol=2e-15,
    )
    assert not np.allclose(result[1], displaced[1])


def test_bloch_multi_k_empty_domains_and_nonfinite_rejection(h2o_mol):
    from vibeqc import _vibeqc_core as core

    basis = BasisSet(h2o_mol, "sto-3g")
    points = np.zeros((2, 3))
    kpoints = np.zeros((1, 3))
    shifts = np.zeros((1, 3))
    assert core.evaluate_bloch_ao_multi_k(basis, points, np.zeros((0, 3)), shifts) == []
    assert core.evaluate_bloch_ao_multi_k(basis, np.zeros((0, 3)), kpoints, shifts)[0].shape == (0, basis.nbasis)
    np.testing.assert_array_equal(
        core.evaluate_bloch_ao_multi_k(basis, points, kpoints, np.zeros((0, 3)))[0],
        np.zeros((2, basis.nbasis)),
    )
    for slot in range(3):
        for value in [np.nan, np.inf, -np.inf]:
            args = [points.copy(), kpoints.copy(), shifts.copy()]
            args[slot][0, 0] = value
            with pytest.raises(ValueError, match="finite"):
                core.evaluate_bloch_ao_multi_k(basis, *args)
        args = [points, kpoints, shifts]
        args[slot] = np.zeros((2, 2))
        with pytest.raises(TypeError):
            core.evaluate_bloch_ao_multi_k(basis, *args)


def test_bloch_multi_k_image_order_is_thread_independent(h2o_mol):
    from vibeqc import _vibeqc_core as core

    basis = BasisSet(h2o_mol, "6-31g*")
    rng = np.random.default_rng(31)
    points, shifts, kpoints = rng.normal(size=(1031, 3)), rng.normal(size=(7, 3)), rng.normal(size=(3, 3))
    old_threads = core.get_num_threads()
    try:
        core.set_num_threads(1)
        serial = core.evaluate_bloch_ao_multi_k(basis, points, kpoints, shifts)
        core.set_num_threads(2)
        parallel = core.evaluate_bloch_ao_multi_k(basis, points, kpoints, shifts)
    finally:
        core.set_num_threads(old_threads)
    for a, b in zip(serial, parallel):
        np.testing.assert_array_equal(a, b)
