"""Cartesian multipole translation and Coulomb algebra, independent references."""
from decimal import Decimal, localcontext

import numpy as np
import pytest
from scipy.special import eval_legendre

from vibeqc import manta


def _moments(indices, points, charges):
    return np.sum(np.prod(points[None, :, :]**np.array(indices)[:, None, :], axis=2)*charges[None, :], axis=1)


@pytest.mark.parametrize("order", [0, 1, 2, 4, 6, 9, 12])
def test_translation_reproduces_point_moments_and_composes_with_adjoint(order):
    shift = np.array([.17, -.23, .11]); other = np.array([-.13, .08, .19])
    result = manta.multipole_translation(shift, order); w = result.values
    rng = np.random.default_rng(649)
    points = rng.normal(size=(15, 3))*.3; charges = rng.normal(size=15)
    source = _moments(result.indices, points, charges)
    expected = _moments(result.indices, points+shift, charges)
    np.testing.assert_allclose(w@source, expected, atol=2e-12, rtol=2e-12)
    second = manta.multipole_translation(other, order)
    summed = manta.multipole_translation(shift+other, order)
    np.testing.assert_allclose(second.values@w, summed.values, atol=2e-12, rtol=2e-12)
    reverse = manta.multipole_translation(-shift, order)
    np.testing.assert_allclose(reverse.values@w, np.eye(len(source)), atol=2e-12, rtol=2e-12)
    field = rng.normal(size=len(source))
    assert field@(w@source) == pytest.approx((w.T@field)@source, abs=2e-12)
    assert len(result.indices) == (order+1)*(order+2)*(order+3)//6
    assert result.normalization == "raw-cartesian-moments"


@pytest.mark.parametrize("order", [0, 1, 2, 4, 6, 9, 12])
def test_coulomb_matches_independent_legendre_series_and_error_bound(order):
    rng = np.random.default_rng(572)
    points_a = rng.normal(size=(7, 3)); points_a *= .7/np.maximum(1., np.linalg.norm(points_a, axis=1))[:, None]
    points_b = rng.normal(size=(9, 3)); points_b *= .6/np.maximum(1., np.linalg.norm(points_b, axis=1))[:, None]
    charges_a, charges_b = rng.normal(size=7), rng.normal(size=9)
    displacement = np.array([7., -4., 5.]); distance = np.linalg.norm(displacement)
    result = manta.multipole_coulomb(displacement, order)
    ma, mb = _moments(result.indices, points_a, charges_a), _moments(result.indices, points_b, charges_b)
    actual = ma@result.values@mb
    h = points_a[:, None, :]-points_b[None, :, :]
    lengths = np.linalg.norm(h, axis=2)
    cosines = np.einsum("ijk,k->ij", h, displacement)/(lengths*distance)
    approximate_kernel = sum((-lengths/distance)**k*eval_legendre(k, cosines)/distance for k in range(order+1))
    weights = charges_a[:, None]*charges_b[None, :]
    expected = np.sum(weights*approximate_kernel)
    assert actual == pytest.approx(expected, abs=3e-13, rel=3e-13)
    exact = np.sum(weights/np.linalg.norm(displacement+h, axis=2))
    bound = manta.compact_multipole_remainder(distance, .7, .6, np.sum(abs(charges_a)), np.sum(abs(charges_b)), order)
    roundoff = 128*np.finfo(float).eps*np.sum(abs(weights))/(distance-1.3)
    assert abs(actual-exact) <= bound+roundoff
    reversed_source = manta.multipole_coulomb(-displacement, order)
    np.testing.assert_allclose(reversed_source.values, result.values.T, atol=2e-14, rtol=2e-13)


def test_low_order_analytic_derivatives_and_rotation_of_complete_degree():
    r = np.array([1.3, -.7, 2.1]); distance = np.linalg.norm(r)
    result = manta.multipole_coulomb(r, 2)
    indices = [tuple(x) for x in result.indices]; zero = indices.index((0, 0, 0))
    assert result.values[zero, zero] == pytest.approx(1/distance, abs=1e-15)
    gradient = -r/distance**3
    hessian = 3*np.outer(r, r)/distance**5-np.eye(3)/distance**3
    for axis in range(3):
        a = [0, 0, 0]; a[axis] = 1; i = indices.index(tuple(a))
        assert result.values[i, zero] == pytest.approx(gradient[axis], abs=1e-15)
        assert result.values[zero, i] == pytest.approx(-gradient[axis], abs=1e-15)
        for axis2 in range(3):
            b = [0, 0, 0]; b[axis2] = 1; j = indices.index(tuple(b))
            assert result.values[i, j] == pytest.approx(-hessian[axis, axis2], abs=1e-15)
    angle = .61; q = np.array([[np.cos(angle), 0., -np.sin(angle)], [0., 1., 0.], [np.sin(angle), 0., np.cos(angle)]])
    points = np.array([[.1, .2, -.3], [-.3, .1, .2]])
    fields = []
    for rotation in [np.eye(3), q]:
        t = manta.multipole_coulomb(rotation@r, 6)
        a = _moments(t.indices, points@rotation.T, np.array([1., -.2]))
        b = _moments(t.indices, -.7*points@rotation.T, np.array([.4, .7]))
        fields.append(a@t.values@b)
    assert fields[0] == pytest.approx(fields[1], abs=3e-13)


@pytest.mark.parametrize("entry", ["multipole_translation", "multipole_coulomb"])
@pytest.mark.parametrize("field,plan_field", [("memory_bytes", "modeled_peak_bytes"), ("scalar_terms", "scalar_terms"), ("kernel_terms", "kernel_terms")])
def test_resource_boundaries_and_owned_results(entry, field, plan_field):
    plan = manta.plan_multipoles(6)
    budget = manta.ResourceBudget(); setattr(budget, field, getattr(plan, plan_field))
    result = getattr(manta, entry)([.7, -.3, 1.2], 6, budget)
    assert result.modeled_peak_bytes == plan.modeled_peak_bytes
    assert 0 < result.coefficients_evaluated <= plan.scalar_terms
    values = result.values; values[:] = -99.
    assert not np.any(result.values == -99.)
    assert len(result.source_identity) == 64
    setattr(budget, field, getattr(plan, plan_field)-1)
    with pytest.raises(ValueError, match="budget"):
        getattr(manta, entry)([.7, -.3, 1.2], 6, budget)


@pytest.mark.parametrize("order", [-1, 13])
def test_order_refuses_before_allocation(order):
    with pytest.raises(ValueError, match="order"):
        manta.plan_multipoles(order)


def test_source_geometry_zero_limits_and_overflow():
    for r in [[np.nan, 1., 2.], [0., np.inf, 1.]]:
        with pytest.raises(ValueError):
            manta.multipole_coulomb(r, 3)
    with pytest.raises(ValueError, match="nonzero"):
        manta.multipole_coulomb([0., 0., 0.], 3)
    with pytest.raises(OverflowError):
        manta.multipole_coulomb([1e-300, 0., 0.], 2)
    with pytest.raises(OverflowError):
        manta.multipole_translation([1e300, 0., 0.], 2)
    far = manta.multipole_coulomb([1e308, 0., 0.], 12)
    assert far.values[0, 0] == pytest.approx(1e-308, rel=1e-14, abs=0.)
    a = manta.multipole_translation([0., 0., 0.], 3)
    b = manta.multipole_translation([-0., 0., 0.], 3)
    assert a.source_identity == b.source_identity
    np.testing.assert_array_equal(a.values, np.eye(len(a.indices)))


def test_compact_remainder_extreme_scales_and_invalid_supports():
    actual = manta.compact_multipole_remainder(1e300, 1e-300, 0., 1e300, 1e300, 0)
    with localcontext() as context:
        context.prec = 70
        d, a, q = map(Decimal.from_float, [1e300, 1e-300, 1e300])
        expected = float(q*q*(a/d)/(d-a))
    assert actual > 0 and actual == pytest.approx(expected, rel=2e-12, abs=0.)
    assert manta.compact_multipole_remainder(2., 0., 0., 1., 1., 0) == 0.
    assert manta.compact_multipole_remainder(2., .1, .2, 0., 1., 4) == 0.
    for args in [(1., .5, .5, 1., 1.), (1., -.1, .2, 1., 1.), (1., .1, .2, -1., 1.), (np.inf, .1, .2, 1., 1.)]:
        with pytest.raises(ValueError):
            manta.compact_multipole_remainder(*args, 2)
