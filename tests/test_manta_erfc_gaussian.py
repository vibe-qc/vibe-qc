"""Gauge-free erfc Taylor interactions against independent Gaussian integrals."""
from itertools import product

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.special import erfc

from vibeqc import manta
from tests.test_manta_lattice_gaussian import source, moments, components, gaussian_integral


def reference(a, b, omega, quadrature=96):
    aa, bb = components(a), components(b)
    x, w = leggauss(quadrature); t = (x+1)/2
    s, weights = omega+t/(1-t), .5*w/(1-t)**2
    out = np.zeros((len(aa), len(bb)))
    for i, ai in enumerate(aa):
        for j, bj in enumerate(bb):
            for ta, tb in product(ai, bj):
                out[i, j] += 2/np.sqrt(np.pi)*np.dot(weights, gaussian_integral(ta, tb, np.zeros(3), s))
    return out


@pytest.mark.parametrize('kinds', [('aux', 'aux'), ('aux', 'pair'), ('pair', 'aux'), ('pair', 'pair')])
@pytest.mark.parametrize('order', [0, 2, 6, 12])
def test_signed_angular_sources_bound_independent_erfc_integrals(kinds, order):
    a = source(kinds[0], [3.1, -.2, .1], exponent=30., radius=1.3, angular=2, signed=True, spread=True, strict=True)
    b = source(kinds[1], [0., .1, -.1], exponent=40., radius=1.1, angular=1, signed=True, spread=True, strict=True)
    omega = .23
    result = manta.erfc_gaussian_interaction(omega, a['extent'], b['extent'], order)
    value = moments(a, order) @ result.matrix.values @ moments(b, order).T
    exact = reference(a, b, omega)
    np.testing.assert_allclose(exact, reference(a, b, omega, 128), atol=2e-20, rtol=2e-11)
    assert np.max(abs(value-exact)) <= result.error.total+2e-18
    assert result.matrix.gauge_constant == result.error.bounded_kernel_sup == 0.
    assert result.matrix.boundary == 'isolated-erfc'
    assert result.error.qualification == 'isolated-erfc-gaussian-approximation'
    assert not result.error.roundoff_included
    assert result.matrix.source_identity == result.compact.expansion_identity
    assert result.error.total >= result.compact.total+result.error.tail_penetration+result.error.full_moment_tail
    reverse = manta.erfc_gaussian_interaction(omega, b['extent'], a['extent'], order)
    np.testing.assert_array_equal(result.matrix.values, reverse.matrix.values.T)


@pytest.mark.parametrize('omega', [.01, .5, 4.])
def test_no_gauge_cancellation_and_nonharmonic_trace(omega):
    a = source('aux', [3., -.2, .1], exponent=40., radius=1.1, strict=True)
    b = source('pair', [0., 0., 0.], exponent=40., radius=1.1, strict=True)
    result = manta.erfc_gaussian_interaction(omega, a['extent'], b['extent'], 2)
    distance = np.linalg.norm(np.array(a['extent'].center)-b['extent'].center)
    matrix, indices = result.matrix.values, list(map(tuple, result.matrix.indices))
    np.testing.assert_allclose(matrix[0, 0], erfc(omega*distance)/distance, atol=0., rtol=5e-14)
    trace = sum(2*matrix[indices.index(index), 0] for index in [(2, 0, 0), (0, 2, 0), (0, 0, 2)])
    assert trace == pytest.approx(4*omega**3/np.sqrt(np.pi)*np.exp(-(omega*distance)**2), rel=3e-11, abs=1e-19)


@pytest.mark.parametrize('field,quantity', [('memory_bytes','modeled_peak_bytes'), ('scalar_terms','scalar_terms'), ('kernel_terms','kernel_terms')])
def test_joint_budget_before_evaluation(field, quantity):
    budget = manta.ResourceBudget(); p = manta.plan_erfc_gaussian_interaction(4, budget)
    a = source('aux', [3., 0., 0.]); b = source('pair', [0., 0., 0.])
    for cap, name in [('memory_bytes','modeled_peak_bytes'), ('scalar_terms','scalar_terms'), ('kernel_terms','kernel_terms')]:
        setattr(budget, cap, getattr(p, name))
    manta.erfc_gaussian_interaction(.3, a['extent'], b['extent'], 4, budget)
    setattr(budget, field, getattr(p, quantity)-1)
    with pytest.raises((RuntimeError, ValueError), match='budget'):
        manta.erfc_gaussian_interaction(.3, a['extent'], b['extent'], 4, budget)


def test_native_envelope_and_support_requirements():
    a = source('aux', [0., 0., 0.]); b = source('pair', [0., 0., 0.])
    with pytest.raises(ValueError, match='singularity'):
        manta.erfc_gaussian_interaction(.3, a['extent'], b['extent'], 4)
    for omega in [0., -1., float('inf'), float('nan')]:
        with pytest.raises(ValueError):
            manta.erfc_gaussian_interaction(omega, a['extent'], b['extent'], 4)
    b = source('pair', [3., 0., 0.], retain=False)
    with pytest.raises(ValueError, match='retained'):
        manta.erfc_gaussian_interaction(.3, a['extent'], b['extent'], 4)
    b = source('pair', [3., 0., 0.], radius=.01, strict=True)
    with pytest.raises(ValueError, match='always-near'):
        manta.erfc_gaussian_interaction(.3, a['extent'], b['extent'], 4)
