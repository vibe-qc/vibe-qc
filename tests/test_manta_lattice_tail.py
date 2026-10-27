"""Finite periodic Ewald complement: independent derivatives and charge energies."""
from itertools import product
from math import factorial

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.special import erf, erfc, eval_hermite, factorial2

from vibeqc import manta


def cube(n, *, reciprocal=False):
    indices = np.array(list(product(range(-n, n+1), repeat=3)), dtype=np.int64)
    return indices[np.any(indices != 0, axis=1)] if reciprocal else indices


def fixture(order=6, *, skew=True, real=1, reciprocal=3, near=0, eta=.63):
    cell = np.array([[5., .4, -.3], [0., 4.7, .2], [0., 0., 5.3]]) if skew else np.eye(3)*5
    return dict(cell=cell, target_minus_source=np.array([.37, -.23, .19]), eta=eta,
                real_indices=cube(real), reciprocal_indices=cube(reciprocal, reciprocal=True),
                near_indices=cube(near), order=order)


def reference_coefficients(args, indices):
    """Differentiate Gaussian integrals via SciPy Hermite polynomials.

    erf/r = 2/sqrt(pi) integral_0^eta exp(-s*s*r*r) ds;
    erfc/r integrates eta..infinity. Finite quadrature replaces the latter
    upper endpoint by eta+12/r, beyond any observable Gaussian contribution.
    This uses neither the native Boys seeds nor its radial recurrence.
    """
    a, r, eta = args['cell'], args['target_minus_source'], args['eta']
    powers = np.array(indices); order = args['order']
    denominators = np.array([np.prod([factorial(int(x)) for x in n]) for n in powers])
    nodes, weights = leggauss(100)
    near = set(map(tuple, args['near_indices']))
    values = np.zeros(len(indices), dtype=np.longdouble)
    for n in args['real_indices']:
        x = r+a@n; radius = np.linalg.norm(x)
        lo, hi, sign = (0., eta, -1.) if tuple(n) in near else (eta, eta+12/radius, 1.)
        s = (hi-lo)/2*nodes+(lo+hi)/2
        table = np.array([[(-s)**p*eval_hermite(p, s*x[axis]) for p in range(order+1)] for axis in range(3)])
        derivatives = table[0, powers[:, 0]]*table[1, powers[:, 1]]*table[2, powers[:, 2]]
        integrand = derivatives*(weights*np.exp(-(s*radius)**2))
        values += sign*(hi-lo)/np.sqrt(np.pi)*np.sum(integrand, axis=1)/denominators
    volume = abs(np.linalg.det(a))
    for n in args['reciprocal_indices']:
        g = 2*np.pi*np.linalg.solve(a.T, n); g2 = g@g
        derivatives = np.prod((1j*g[None, :])**powers, axis=1)*np.exp(1j*g@r)
        values += 4*np.pi/volume*np.exp(-g2/(4*eta**2))/g2*derivatives.real/denominators
    values[0] -= np.pi/(eta*eta*volume)
    return np.array(values, dtype=float)


def reference_matrix(coefficients, indices, order):
    lookup = {tuple(n): i for i, n in enumerate(indices)}
    result = np.zeros((len(indices), len(indices)))
    for i, a in enumerate(indices):
        for j, b in enumerate(indices):
            total = tuple(x+y for x, y in zip(a, b))
            if sum(total) <= order:
                multiplier = (-1)**sum(b)*np.prod([factorial(x+y)//factorial(x)//factorial(y) for x, y in zip(a, b)])
                result[i, j] = multiplier*coefficients[lookup[total]]
    return result


def point_moments(indices, points, charges):
    return np.sum(np.prod(points[None, :, :]**np.array(indices)[:, None, :], axis=2)*charges, axis=1)


def gaussian_moments(indices, center, exponent):
    """Unit-integral isotropic exp(-exponent*|x-center|^2), raw moments."""
    moments = []
    for powers in indices:
        value = 1.
        for axis, n in enumerate(powers):
            value *= sum(factorial(n)/factorial(2*k)/factorial(n-2*k)*center[axis]**(n-2*k)
                         * (factorial2(2*k-1) if k else 1)/(2*exponent)**k for k in range(n//2+1))
        moments.append(value)
    return np.array(moments)


def gaussian_calculation(*, skew=True, eta=.6, near=1, order=8):
    """Neutral periodic signed Gaussian charge energy, all component pairs.

    Independent reference is the analytic Gaussian Fourier Green function.
    Full moments have infinite support; the compact-tail error is NOT certified.
    The fixed tight distributions and near domain make penetration negligible
    here, as checked by the numerical reference and refinement tests.
    """
    args = fixture(order, skew=skew, real=3, reciprocal=9, near=near, eta=eta)
    args['target_minus_source'] = np.zeros(3)
    result = manta.multipole_lattice_tail(**args)
    centers = np.array([[.21, -.17, .12], [-.19, .11, -.14], [.04, .07, .02]])
    exponents = np.array([1.2, 1.7, 2.1]); charges = np.array([1., -.7, -.3])
    moments = np.array([gaussian_moments(result.indices, c, e) for c, e in zip(centers, exponents)])
    far = moments@result.values@moments.T
    shifts = args['near_indices']@args['cell'].T
    direct = np.zeros_like(far); exact = np.zeros_like(far)
    reciprocal = 2*np.pi*np.linalg.solve(args['cell'].T, cube(12, reciprocal=True).T).T
    g2 = np.einsum('gi,gi->g', reciprocal, reciprocal)
    for i in range(len(centers)):
        for j in range(len(centers)):
            beta = exponents[i]*exponents[j]/(exponents[i]+exponents[j])
            r = centers[i]-centers[j]; lengths = np.linalg.norm(r+shifts, axis=1)
            nonzero = lengths != 0
            direct[i, j] = np.sum(erf(np.sqrt(beta)*lengths[nonzero])/lengths[nonzero])
            direct[i, j] += np.count_nonzero(~nonzero)*2*np.sqrt(beta/np.pi)
            exact[i, j] = 4*np.pi/abs(np.linalg.det(args['cell']))*np.sum(np.exp(-g2/(4*beta))*np.cos(reciprocal@r)/g2)
    actual = direct+far
    return dict(energy=float(.5*charges@actual@charges), reference_energy=float(.5*charges@exact@charges),
                maximum_pair_error=float(np.max(abs(actual-exact))),
                pair_values=actual, reference_values=exact, tail=result, args=args)


@pytest.mark.parametrize('order', [0, 1, 2, 4, 6, 9, 12])
@pytest.mark.parametrize('origin', [False, True])
def test_all_coefficients_against_independent_gaussian_quadrature(order, origin):
    args = fixture(order, reciprocal=2)
    if origin:
        args['target_minus_source'][:] = 0.
    result = manta.multipole_lattice_tail(**args)
    expected = reference_matrix(reference_coefficients(args, result.indices), result.indices, order)
    np.testing.assert_allclose(result.values, expected, rtol=2e-11, atol=3e-13)
    assert result.radial_derivatives_evaluated == len(args['real_indices'])*result.plan.moments
    assert result.reciprocal_coefficients_evaluated == len(args['reciprocal_indices'])*result.plan.moments
    assert result.matrix_coefficients_evaluated == sum(sum(a)+sum(b) <= order for a in result.indices for b in result.indices)


@pytest.mark.parametrize('order', [0, 2, 6, 12])
def test_reciprocity_and_disjoint_near_transfer(order):
    args = fixture(order, near=0)
    first = manta.multipole_lattice_tail(**args)
    opposite = manta.multipole_lattice_tail(**dict(args, target_minus_source=-args['target_minus_source']))
    np.testing.assert_allclose(opposite.values, first.values.T, atol=2e-14, rtol=3e-13)
    second = manta.multipole_lattice_tail(**dict(args, near_indices=cube(1)))
    transferred = sum(manta.multipole_coulomb(args['target_minus_source']+args['cell']@n, order).values
                      for n in cube(1) if np.any(n))
    np.testing.assert_allclose(second.values+transferred, first.values, atol=4e-14, rtol=5e-13)
    assert first.kernel_identity != second.kernel_identity


@pytest.mark.parametrize('skew', [False, True])
def test_ewald_split_poisson_trace_gauge_and_refinement(skew):
    values = []
    for eta in [.42, .65, .85]:
        args = fixture(6, skew=skew, real=3, reciprocal=10, eta=eta)
        result = manta.multipole_lattice_tail(**args)
        indices = list(map(tuple, result.indices)); values.append(result.values)
        trace = sum(2*result.values[indices.index(tuple(2*np.eye(3, dtype=int)[axis])), 0] for axis in range(3))
        assert trace == pytest.approx(4*np.pi/abs(np.linalg.det(args['cell'])), abs=3e-13)
        assert result.gauge_constant == pytest.approx(-np.pi/(eta**2*result.cell_volume), abs=1e-16)
    np.testing.assert_allclose(values[0], values[1], atol=5e-13, rtol=2e-11)
    np.testing.assert_allclose(values[2], values[1], atol=5e-13, rtol=2e-11)


def test_cubic_trace_second_moments_and_constant_are_retained():
    args = fixture(2, skew=False, real=3, reciprocal=9)
    args['target_minus_source'][:] = 0
    result = manta.multipole_lattice_tail(**args)
    pa = np.array([[.12, -.07, .16], [-.08, .15, .03]])
    pb = np.array([[.01, .04, -.14], [-.11, .07, .08], [.05, .02, -.07]])
    qa, qb = np.array([.7, -.2]), np.array([.4, .6, -.3])
    ma, mb = point_moments(result.indices, pa, qa), point_moments(result.indices, pb, qb)
    quadratic = 2*np.pi/(3*result.cell_volume)
    expected = result.values[0, 0]*sum(qa)*sum(qb)+quadratic*(
        sum(qa*np.sum(pa*pa, axis=1))*sum(qb)+sum(qb*np.sum(pb*pb, axis=1))*sum(qa)-2*(qa@pa)@(qb@pb))
    assert ma@result.values@mb == pytest.approx(expected, abs=2e-15)
    assert abs(expected-result.values[0, 0]*sum(qa)*sum(qb)) > 1e-5


@pytest.mark.parametrize('skew', [False, True])
@pytest.mark.parametrize('eta', [.45, .75])
def test_gaussian_pair_and_neutral_energy_vs_independent_reciprocal_sum(skew, eta):
    row = gaussian_calculation(skew=skew, eta=eta, order=8)
    assert row['maximum_pair_error'] < 2e-10
    assert row['energy'] == pytest.approx(row['reference_energy'], abs=2e-10)


def test_gaussian_order_ladder_converges_without_claiming_a_bound():
    errors = [gaussian_calculation(order=order)['maximum_pair_error'] for order in [0, 2, 4, 6, 8, 10]]
    assert errors[0] > 1e-2 and errors[1] > 1e-6
    assert errors[-1] < 3e-13
    assert all(b < a for a, b in zip(errors[:4], errors[1:5]))


def test_complete_degree_rotation_covariance_and_scaled_units():
    args = fixture(6); result = manta.multipole_lattice_tail(**args)
    angle = .43; rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    pa = np.array([[.17, -.11, .07], [-.19, .02, .13]]); pb = -.4*pa
    qa, qb = np.array([1., -.3]), np.array([.7, .2])
    energies = []
    for rot in [np.eye(3), rotation]:
        rr = manta.multipole_lattice_tail(**dict(args, cell=rot@args['cell'], target_minus_source=rot@args['target_minus_source']))
        energies.append(point_moments(rr.indices, pa@rot.T, qa)@rr.values@point_moments(rr.indices, pb@rot.T, qb))
    assert energies[0] == pytest.approx(energies[1], abs=3e-15)
    degrees = np.sum(result.indices, axis=1)
    for scale in [1e-8, 1e8]:
        scaled = manta.multipole_lattice_tail(**dict(args, cell=args['cell']*scale,
            target_minus_source=args['target_minus_source']*scale, eta=args['eta']/scale))
        np.testing.assert_allclose(scaled.values*scale**(degrees[:, None]+degrees[None, :]+1), result.values, atol=2e-14, rtol=2e-11)


def test_changed_expansion_centers_converge_to_same_point_energy():
    args = fixture(12, real=2, reciprocal=4)
    pa = np.array([[.12, -.06, .04], [-.08, .03, .07]])
    pb = np.array([[-.05, -.04, .02], [.03, .01, -.02]])
    qa, qb = np.array([1., -.3]), np.array([.7, -.1])
    expected = 0.
    for a, q in zip(pa, qa):
        for b, charge in zip(pb, qb):
            r = args['target_minus_source']+a-b
            positions = r+args['real_indices']@args['cell'].T
            distances = np.linalg.norm(positions, axis=1)
            real = np.sum(erfc(args['eta']*distances)/distances)-1/np.linalg.norm(r)
            g = 2*np.pi*np.linalg.solve(args['cell'].T, args['reciprocal_indices'].T).T
            g2 = np.sum(g*g, axis=1); volume = abs(np.linalg.det(args['cell']))
            periodic = real+4*np.pi/volume*np.sum(np.exp(-g2/(4*args['eta']**2))*np.cos(g@r)/g2)-np.pi/(args['eta']**2*volume)
            expected += q*charge*periodic
    for u, v in [(np.zeros(3), np.zeros(3)), (np.array([.03, -.07, .01]), np.array([-.02, .04, .05]))]:
        result = manta.multipole_lattice_tail(**dict(args, target_minus_source=args['target_minus_source']+u-v))
        actual = point_moments(result.indices, pa-u, qa)@result.values@point_moments(result.indices, pb-v, qb)
        assert actual == pytest.approx(expected, abs=3e-14)


def test_insufficient_domains_have_separate_observable_errors():
    converged = fixture(2, real=3, reciprocal=9)
    target = manta.multipole_lattice_tail(**converged).values
    for changes in [dict(real_indices=cube(0), eta=.2), dict(reciprocal_indices=cube(0, reciprocal=True))]:
        actual = manta.multipole_lattice_tail(**dict(converged, **changes))
        assert np.max(abs(actual.values-target)) > 1e-3
        assert actual.qualification == 'finite-domain-coefficients'


def test_canonical_source_identity_and_owned_exports():
    args = fixture(); result = manta.multipole_lattice_tail(**args)
    permuted = dict(args)
    for key in ['real_indices', 'reciprocal_indices', 'near_indices']:
        permuted[key] = np.asfortranarray(args[key][::-1])
    other = manta.multipole_lattice_tail(**permuted)
    assert other.source_identity == result.source_identity
    np.testing.assert_array_equal(other.values, result.values)
    assert result.boundary == '3d-zero-average-conducting'
    assert result.qualification == 'finite-domain-coefficients'
    assert result.normalization == 'raw-cartesian-moments'
    exported = result.values; exported[:] = 88.
    args['cell'][:] = 0; args['near_indices'][:] = 11
    assert not np.any(result.values == 88.)
    with pytest.raises(AttributeError):
        result.source_identity = 'forged'
    with pytest.raises(AttributeError):
        result.plan.order = 3
    for changes in [dict(eta=.61), dict(order=4), dict(target_minus_source=np.zeros(3)), dict(real_indices=cube(2))]:
        changed = manta.multipole_lattice_tail(**dict(fixture(), **changes))
        assert changed.source_identity != result.source_identity
    zero = fixture(); zero['target_minus_source'][:] = 0
    assert manta.multipole_lattice_tail(**zero).source_identity == manta.multipole_lattice_tail(**dict(zero, target_minus_source=-zero['target_minus_source'])).source_identity


@pytest.mark.parametrize('field,planned', [('memory_bytes', 'modeled_peak_bytes'), ('scalar_terms', 'scalar_terms'), ('kernel_terms', 'kernel_terms')])
def test_admission_before_payload_and_exact_cap(field, planned):
    args = fixture(); p = manta.plan_lattice_tail(6, 27, 342, 1)
    budget = manta.ResourceBudget(); setattr(budget, field, getattr(p, planned))
    result = manta.multipole_lattice_tail(**args, budget=budget)
    assert result.plan.modeled_peak_bytes == p.modeled_peak_bytes
    setattr(budget, field, getattr(p, planned)-1)
    args['cell'][:] = np.nan
    with pytest.raises(ValueError, match='budget'):
        manta.multipole_lattice_tail(**args, budget=budget)
    budget = manta.ResourceBudget(); budget.other_live_bytes = 12345
    assert manta.plan_lattice_tail(6, 27, 342, 1, budget).modeled_peak_bytes == p.modeled_peak_bytes+12345


@pytest.mark.parametrize('changes,match', [
    (dict(order=-1), 'order'), (dict(order=13), 'order'),
    (dict(eta=0), 'eta'), (dict(eta=np.inf), 'eta'),
    (dict(cell=np.zeros((3, 3))), 'cell'), (dict(cell=np.eye(3)*np.nan), 'cell'),
    (dict(target_minus_source=np.array([0., np.nan, 0.])), 'finite'),
    (dict(near_indices=np.array([[1, 0, 0]], dtype=np.int64)), 'origin'),
    (dict(near_indices=cube(2)), 'near'),
    (dict(real_indices=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64)), 'inversion'),
    (dict(real_indices=np.array([[0, 0, 0], [0, 0, 0]], dtype=np.int64)), 'unique'),
    (dict(reciprocal_indices=cube(1)), 'zero'),
    (dict(reciprocal_indices=np.array([[2**40, 0, 0], [-2**40, 0, 0]], dtype=np.int64)), '31-bit'),
    (dict(real_indices=cube(1).astype(float)), 'dtype'),
    (dict(real_indices=np.zeros((3, 4), dtype=np.int64)), 'shape'),
    (dict(cell=np.zeros((2, 3))), 'shapes'),
    (dict(target_minus_source=np.zeros(4)), 'shapes'),
])
def test_invalid_domains_and_payloads_fail_closed(changes, match):
    with pytest.raises((ValueError, TypeError), match=match):
        manta.multipole_lattice_tail(**dict(fixture(), **changes))


def test_missing_near_image_singularity_and_disconnected_subset():
    args = fixture(); args['target_minus_source'] = args['cell'][:, 0].copy()
    with pytest.raises(ValueError, match='singularity'):
        manta.multipole_lattice_tail(**args)
    args = fixture(); args['near_indices'] = np.array([[0, 0, 0], [2, 0, 0], [-2, 0, 0]], dtype=np.int64)
    with pytest.raises(ValueError, match='contained'):
        manta.multipole_lattice_tail(**args)
    with pytest.raises(OverflowError):
        manta.plan_lattice_tail(12, 2**64-1, 1, 1)
    args = fixture(); args['eta'] = 1e300; args['target_minus_source'][:] = 0
    with pytest.raises(OverflowError):
        manta.multipole_lattice_tail(**args)


def test_empty_reciprocal_is_explicit_finite_source_not_convergence_claim():
    args = fixture(4, real=0, reciprocal=0)
    result = manta.multipole_lattice_tail(**args)
    reference = reference_coefficients(args, result.indices)
    np.testing.assert_allclose(result.values[:, 0], reference, atol=2e-15)
    assert result.reciprocal_coefficients_evaluated == 0
    assert result.qualification == 'finite-domain-coefficients'
