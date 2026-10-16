"""Paired RI-BUPO Coulomb energy, exact metric and amplified error ledger."""
import gc

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from tests.test_manta_bupo_fitting import _basis, _inputs, _options


def _density(orbital, auxiliary):
    d, _ = _inputs(orbital, auxiliary)
    return .5*(d+d.T)


def _dense(metric, b, d, options):
    w, u = np.linalg.eigh(metric)
    keep = w > max(options.absolute_cutoff, options.relative_cutoff*w[-1])
    factor = u[:, keep]/np.sqrt(w[keep])
    rhs = np.einsum("pij,ij->p", b, d)
    coefficients = factor@(factor.T@rhs)
    return np.einsum("pij,p->ij", b, coefficients), .5*rhs@coefficients


@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("order,tolerance", [(0, 1.), (6, 1e-8), (12, 1e-10)])
def test_paired_j_and_energy_against_independent_dense_reference(pure, order, tolerance):
    orbital, auxiliary = _basis(pure), _basis(pure, True)
    metric_options = manta.RIOptions()
    builder = manta.BupoRIJBuilder(orbital, auxiliary, _options(order, tolerance=tolerance), metric_options)
    d = _density(orbital, auxiliary)
    result = builder.build(d); r = result.receipt
    metric, coupling = vq.compute_2c_eri(auxiliary), vq.compute_3c_eri(orbital, auxiliary)
    j, energy = _dense(metric, coupling, d, metric_options)
    exact = builder.build_exact(d)
    np.testing.assert_allclose(builder.metric, metric, atol=3e-12)
    np.testing.assert_allclose(exact.J, j, atol=2e-11)
    assert exact.receipt.coulomb_energy == pytest.approx(energy, abs=2e-11)
    assert exact.receipt.source_identity == builder.exact_ri_source_identity == r.exact_ri_source_identity
    assert r.source_identity == builder.source_identity != builder.exact_ri_source_identity
    assert np.max(abs(result.J-j)) <= r.approximation_j_bound+3e-11
    assert abs(r.coulomb_energy-energy) <= r.approximation_energy_bound+3e-11
    assert builder.coupling_plan.far_blocks > 0
    if order == 0:
        assert np.max(abs(result.J-j)) > 1e-7
        assert abs(r.coulomb_energy-energy) > 1e-8
    if order == 12:
        np.testing.assert_allclose(result.J, j, atol=3e-10)
        assert r.coulomb_energy == pytest.approx(energy, abs=3e-10)
    np.testing.assert_allclose(builder.build_J(d), result.J, atol=0)
    np.testing.assert_allclose(result.J, result.J.T, atol=1e-12)
    assert r.coulomb_energy >= 0
    assert r.coulomb_energy == pytest.approx(.5*np.sum(d*result.J), abs=3e-12)
    assert r.metric_rhs_norm == pytest.approx(np.linalg.norm(result.rhs))
    assert r.metric_residual_norm == pytest.approx(np.linalg.norm(builder.metric@result.coefficients-result.rhs), abs=2e-12)
    assert r.forward.partition_identity == r.adjoint.partition_identity == builder.coupling_plan.partition_identity
    assert r.forward.source_identity == r.adjoint.source_identity
    assert r.forward.input_identity == r.density_identity
    assert r.approximation_j_bound >= r.adjoint_j_bound+r.forward_propagated_j_bound
    assert r.adjoint_j_bound == r.adjoint.approximation_bound
    vp = builder.factor@builder.factor.T
    assert r.metric_inverse_norm_bound >= np.linalg.norm(vp, ord=2)*(1-1e-13)
    assert r.forward_error_l2_bound >= np.sqrt(auxiliary.nbasis)*r.forward.approximation_bound
    np.testing.assert_allclose(result.coefficients, vp@result.rhs, atol=2e-12)
    b_exact = np.einsum("pij,ij->p", coupling, d)
    assert np.linalg.norm(result.rhs-b_exact) <= r.forward_error_l2_bound+2e-11
    # Isolate pass-two error at fixed coefficients, independently of pass one.
    j_fixed = np.einsum("pij,p->ij", coupling, result.coefficients)
    assert np.max(abs(result.J-j_fixed)) <= r.adjoint_j_bound+2e-11
    assert np.max(abs(j_fixed-j)) <= r.forward_propagated_j_bound+2e-11


@pytest.mark.parametrize("cutoff", [0., .2, 1e10])
def test_exact_near_limit_rank_selection_and_residual(cutoff):
    orbital, auxiliary = _basis(distance=2.), _basis(auxiliary=True, distance=2.)
    o = manta.RIOptions(); o.absolute_cutoff = cutoff
    builder = manta.BupoRIJBuilder(orbital, auxiliary, _options(opening=0.), o)
    reference = manta.RIJKBuilder(orbital, auxiliary, o)
    d = _density(orbital, auxiliary); result = builder.build(d)
    np.testing.assert_array_equal(builder.factor, reference.factor)
    assert builder.metric_info.rank == reference.metric_info.rank
    np.testing.assert_allclose(result.J, reference.build_J(d), atol=2e-11)
    assert result.receipt.approximation_energy_bound == result.receipt.approximation_j_bound == 0
    if cutoff == 1e10:
        assert result.receipt.retained_rank == 0
        assert np.count_nonzero(result.J) == np.count_nonzero(result.coefficients) == 0
        assert result.receipt.coulomb_energy == result.receipt.metric_inverse_norm_bound == 0
        assert result.receipt.metric_residual_norm == pytest.approx(np.linalg.norm(result.rhs))


@pytest.mark.parametrize("separation", [0., 1e-3])
def test_rank_deficient_and_ill_conditioned_metric_uses_same_exact_projector(separation):
    orbital, auxiliary = _basis(), _basis(auxiliary=True)
    shells = auxiliary.shells(); duplicate = auxiliary.shells()[0]
    duplicate.origin = (np.array(duplicate.origin)+[separation, 0., 0.]).tolist()
    shells.append(duplicate)
    mol = vq.Molecule([vq.Atom(2, s.origin) for s in shells], 0, 1)
    auxiliary = vq.BasisSet(mol, shells, "near-dependent-fitting", True)
    options = manta.RIOptions(); options.absolute_cutoff = 1e-13; options.relative_cutoff = 1e-12
    builder = manta.BupoRIJBuilder(orbital, auxiliary, _options(0, tolerance=1.), options)
    exact = manta.RIJKBuilder(orbital, auxiliary, options)
    np.testing.assert_array_equal(builder.factor, exact.factor)
    if separation == 0.: assert builder.metric_info.rank < auxiliary.nbasis
    else: assert builder.metric_info.rank == auxiliary.nbasis
    result = builder.build(_density(orbital, auxiliary)); reference = exact.build_J(_density(orbital, auxiliary))
    assert np.max(abs(result.J-reference)) <= result.receipt.approximation_j_bound+2e-9
    assert result.receipt.coulomb_energy >= 0.
    assert result.receipt.coulomb_energy == pytest.approx(.5*np.sum(_density(orbital, auxiliary)*result.J), abs=5e-9)


@pytest.mark.parametrize("order", [0, 6])
def test_linearity_reciprocity_energy_derivative_and_zero_density(order):
    orbital, auxiliary = _basis(), _basis(auxiliary=True)
    builder = manta.BupoRIJBuilder(orbital, auxiliary, _options(order, tolerance=1.))
    d = _density(orbital, auxiliary); e = np.eye(orbital.nbasis)
    a, b, combined = builder.build(d), builder.build(e), builder.build(.4*d-.7*e)
    np.testing.assert_allclose(combined.J, .4*a.J-.7*b.J, atol=3e-12)
    assert np.sum(d*b.J) == pytest.approx(np.sum(e*a.J), abs=3e-12)
    h = 1e-5
    derivative = (builder.build(d+h*e).receipt.coulomb_energy-builder.build(d-h*e).receipt.coulomb_energy)/(2*h)
    assert derivative == pytest.approx(np.sum(e*a.J), abs=2e-9)
    zero = builder.build(np.zeros_like(d))
    assert np.count_nonzero(zero.J) == np.count_nonzero(zero.rhs) == np.count_nonzero(zero.coefficients) == 0
    assert zero.receipt.approximation_energy_bound == zero.receipt.approximation_j_bound == 0.


@pytest.mark.parametrize("field,plan_field", [("memory_bytes", "modeled_peak_bytes"), ("shell_calls", "shell_calls"),
    ("primitive_quartets", "primitive_products"), ("scalar_terms", "scalar_terms"), ("kernel_terms", "kernel_terms")])
@pytest.mark.parametrize("opening", [0., .8])
def test_combined_resource_boundary_includes_exact_confirmation(field, plan_field, opening):
    orbital, auxiliary = _basis(), _basis(auxiliary=True)
    o = _options(4, opening=opening, tolerance=1e-5)
    budget = manta.ResourceBudget(); budget.other_live_bytes = 12345
    p = manta.BupoRIJBuilder(orbital, auxiliary, o, budget=budget).plan
    setattr(budget, field, getattr(p, plan_field))
    builder = manta.BupoRIJBuilder(orbital, auxiliary, o, budget=budget)
    d = _density(orbital, auxiliary)
    result, exact = builder.build(d), builder.build_exact(d)
    assert result.receipt.modeled_peak_bytes == p.modeled_peak_bytes
    assert builder.metric_info.shell_calls+result.receipt.forward.near_shell_calls+result.receipt.adjoint.near_shell_calls+exact.receipt.shell_calls <= p.shell_calls
    setattr(budget, field, getattr(p, plan_field)-1)
    with pytest.raises(ValueError, match="budget"):
        manta.BupoRIJBuilder(orbital, auxiliary, o, budget=budget)


def test_ownership_source_identity_and_invalid_density():
    orbital, auxiliary = _basis(), _basis(auxiliary=True)
    options = _options(0, tolerance=1.); metric = manta.RIOptions()
    builder = manta.BupoRIJBuilder(orbital, auxiliary, options, metric)
    d = _density(orbital, auxiliary); result = builder.build(d); saved = result.J.copy()
    np.testing.assert_array_equal(builder.build(np.asfortranarray(d)).J, saved)
    for value in [result.J, result.rhs, result.coefficients, builder.metric, builder.factor]: value[:] = 0
    options.order = 12; metric.absolute_cutoff = 1e20
    np.testing.assert_array_equal(builder.build(d).J, saved)
    np.testing.assert_array_equal(result.J, saved)
    for bad in [d.astype(np.float32), d.astype(complex), d[:, ::-1], d[:, :1], np.full_like(d, np.nan), np.triu(np.ones_like(d))]:
        with pytest.raises(ValueError): builder.build(bad)
    with pytest.raises(ValueError, match="independent exchange"): builder.build_K(d)
    with pytest.raises(ValueError, match="independent exchange"): builder.build_g_rhf(d)
    np.testing.assert_array_equal(builder.build_g_rhf(d, 0.), saved)
    assert manta.BupoRIJBuilder(orbital, auxiliary, _options(0, tolerance=1.)).source_identity == builder.source_identity
    assert manta.BupoRIJBuilder(orbital, auxiliary, _options(1, tolerance=1.)).source_identity != builder.source_identity
    del builder; gc.collect(); np.testing.assert_array_equal(result.J, saved)
