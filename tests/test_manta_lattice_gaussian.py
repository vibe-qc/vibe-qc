"""Finite Ewald Gaussian approximation ledger and independent convolutions."""
from itertools import product
from functools import lru_cache
from math import comb

import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy.special import erf, factorial2

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from tests.test_manta_lattice_tail import cube


def source(kind, center, *, exponent=.7, radius=.3, angular=0, signed=False,
           spread=False, order=12, retain=True, strict=False, scale=1.):
    center = np.asarray(center, dtype=float)
    origins = [(center+(np.array([-.07, .05, 0.]) if i == 0 else
                       np.array([.03, -.02, .01]))*spread).tolist()
               for i in range(2 if kind == 'pair' else 1)]
    atom_centers = list(dict.fromkeys(map(tuple, origins)))
    mol = vq.Molecule([vq.Atom(2, list(p)) for p in atom_centers], 0, 1)
    shells = []
    for i, origin in enumerate(origins):
        shell = core.ShellInfo(); shell.atom_index = atom_centers.index(tuple(origin))
        shell.l = angular if i == 0 else 0; shell.pure = False
        shell.origin = origin
        shell.exponents = [exponent*(1+.3*i)]
        shell.coefficients = [scale*(1-.3*i)]
        if signed:
            shell.exponents += [2.1*exponent*(1+.3*i)]
            shell.coefficients += [-.35*scale]
        shells.append(shell)
    basis = vq.BasisSet(mol, shells, 'finite-lattice-gaussian', True)
    options = manta.ExtentOptions(); options.moment_order = order
    options.retain_full_moments = retain
    options.minimum_radius = options.maximum_radius = radius
    if not strict:
        options.pointwise_tolerance = options.tail_mass_tolerance = 1e300
    if kind == 'pair':
        extent = next(p for p in manta.compute_pair_extents(basis, options).pairs if tuple(p.shells) == (1, 0))
    else:
        extent = manta.compute_auxiliary_extents(basis, options).shells[0]
    return dict(kind=kind, basis=basis, extent=extent)


def fixture(order=6, *, skew=True, eta=.35, near=0):
    cell = np.array([[8., .4, -.3], [0., 7.7, .2], [0., 0., 8.3]]) if skew else np.eye(3)*8
    return dict(cell=cell, eta=eta, real_indices=cube(1),
                reciprocal_indices=cube(2, reciprocal=True), near_indices=cube(near), order=order)


def interaction(args, a, b):
    return manta.lattice_gaussian_interaction(**args, target=a['extent'], source=b['extent'])


def moments(a, order):
    e = a['extent']
    if a['kind'] == 'pair':
        return manta.compute_pair_moments(a['basis'], e.shells, e.center, order).values
    return manta.compute_auxiliary_moments(a['basis'], e.shell, e.center, order).values


def powers(l):
    return [(x, y, l-x-y) for x in range(l, -1, -1) for y in range(l-x, -1, -1)]


@lru_cache(maxsize=2)
def hermite_rule(order):
    return hermgauss(order)


def components(a):
    """Contracted Cartesian distributions as independent primitive polynomials."""
    shells = a['basis'].shells(); result = []
    if a['kind'] == 'pair':
        shells = [shells[i] for i in a['extent'].shells]
    for cart in product(*(powers(s.l) for s in shells)):
        terms = []
        for prim in product(*(range(len(s.exponents)) for s in shells)):
            exponents = np.array([s.exponents[k] for s, k in zip(shells, prim)])
            centers = np.array([s.origin for s in shells]); p = sum(exponents)
            center = exponents@centers/p
            coefficient = np.prod([s.coefficients[k] for s, k in zip(shells, prim)])
            coefficient *= np.exp(-sum(exponents*np.sum((centers-center)**2, axis=1)))
            polynomials = []
            for axis in range(3):
                polynomial = np.ones(1)
                for origin, indices in zip(centers, cart):
                    n = indices[axis]; d = center[axis]-origin[axis]
                    factor = np.array([comb(n, k)*d**(n-k) for k in range(n+1)])
                    polynomial = np.polynomial.polynomial.polymul(polynomial, factor)
                polynomials.append(polynomial)
            terms.append((p, center, coefficient, polynomials))
        result.append(terms)
    return result


def fourier(terms, vectors):
    """Gaussian characteristic moments, independent of native Fourier panels."""
    result = np.zeros(len(vectors), complex)
    for p, center, coefficient, polynomials in terms:
        value = np.full(len(vectors), coefficient*(np.pi/p)**1.5, complex)
        value *= np.exp(-np.sum(vectors*vectors, axis=1)/(4*p)+1j*(vectors@center))
        for axis, polynomial in enumerate(polynomials):
            shift = 1j*vectors[:, axis]/(2*p); component = np.zeros(len(vectors), complex)
            for n, c in enumerate(polynomial):
                for k in range(n//2+1):
                    component += c*comb(n, 2*k)*(factorial2(2*k-1) if k else 1)*shift**(n-2*k)/(2*p)**k
            value *= component
        result += value
    return result


def gaussian_integral(ta, tb, displacement, s):
    """Bivariate Gaussian polynomial integral for exp(-s^2 |x-y|^2).

    Two-dimensional Gauss-Hermite integration is exact for the Cartesian
    polynomial degrees used here. No native moments or Taylor coefficients
    enter this reference. s is subsequently integrated by Gauss-Legendre.
    """
    p, ca, wa, pa = ta; q, cb, wb, pb = tb
    d = ca-cb+displacement; det = p*q+(p+q)*s*s
    value = wa*wb*np.pi**3/det**1.5*np.exp(-p*q*s*s*np.dot(d, d)/det)
    x, w = hermite_rule(5); weights = w[:, None]*w[None, :]/np.pi
    l00 = np.sqrt((q+s*s)/det); l10 = s*s/(det*l00); l11 = 1/np.sqrt(q+s*s)
    for axis in range(3):
        u = (-s*s*q*d[axis]/det)[:, None, None]+l00[:, None, None]*x[None, :, None]
        v = ((s*s*p*d[axis]/det)[:, None, None]+l10[:, None, None]*x[None, :, None]
             +l11[:, None, None]*x[None, None, :])
        uv = np.polynomial.polynomial.polyval(u, pa[axis])*np.polynomial.polynomial.polyval(v, pb[axis])
        value *= np.sum(uv*weights, axis=(1, 2))
    return value


def finite_reference(args, a, b, *, quadrature=72):
    """Complete finite Ewald-minus-near integrals for signed angular Gaussians."""
    aa, bb = components(a), components(b); out = np.zeros((len(aa), len(bb)))
    x, w = leggauss(quadrature); t = (x+1)/2; w = w/2
    s_near, w_near = args['eta']*t, args['eta']*w
    s_outer, w_outer = args['eta']+t/(1-t), w/(1-t)**2
    near = set(map(tuple, args['near_indices']))
    for index in args['real_indices']:
        inside = tuple(index) in near
        s, weights = (s_near, -w_near) if inside else (s_outer, w_outer)
        for i, ai in enumerate(aa):
            for j, bj in enumerate(bb):
                for ta, tb in product(ai, bj):
                    out[i, j] += 2/np.sqrt(np.pi)*np.dot(weights, gaussian_integral(ta, tb, args['cell']@index, s))
    gs = 2*np.pi*np.linalg.solve(args['cell'].T, args['reciprocal_indices'].T).T
    g2 = np.sum(gs*gs, axis=1); volume = abs(np.linalg.det(args['cell']))
    fa, fb = np.array([fourier(t, gs) for t in aa]), np.array([fourier(t, gs) for t in bb])
    out += ((fa*(4*np.pi/volume*np.exp(-g2/(4*args['eta']**2))/g2))@fb.conj().T).real
    qa = np.array([fourier(t, np.zeros((1, 3)))[0].real for t in aa])
    qb = np.array([fourier(t, np.zeros((1, 3)))[0].real for t in bb])
    out -= np.pi/(args['eta']**2*volume)*qa[:, None]*qb[None, :]
    return out


def scalar_reference(args, a, b):
    """Closed Gaussian erfc/erf convolution, also at the removable origin."""
    aa, bb = components(a), components(b); assert len(aa) == len(bb) == 1
    value = 0.; near = set(map(tuple, args['near_indices']))
    gs = 2*np.pi*np.linalg.solve(args['cell'].T, args['reciprocal_indices'].T).T
    g2 = np.sum(gs*gs, axis=1); volume = abs(np.linalg.det(args['cell']))
    def erf_over_r(exponent, r):
        return 2*np.sqrt(exponent/np.pi) if r == 0 else erf(np.sqrt(exponent)*r)/r
    for ta, tb in product(aa[0], bb[0]):
        p, ca, wa, _ = ta; q, cb, wb, _ = tb
        beta = p*q/(p+q); delta = beta*args['eta']**2/(beta+args['eta']**2)
        mass = wa*wb*(np.pi/p)**1.5*(np.pi/q)**1.5
        pair = -np.pi/(args['eta']**2*volume)
        for index in args['real_indices']:
            r = np.linalg.norm(ca-cb+args['cell']@index)
            pair -= erf_over_r(delta, r)
            if tuple(index) not in near:
                pair += erf_over_r(beta, r)
        pair += 4*np.pi/volume*sum(np.exp(-g2/(4*delta))*np.cos(gs@(ca-cb))/g2)
        value += mass*pair
    return value


def reference_ledger(args, a, b, result):
    ea, eb = a['extent'], b['extent']
    fa, fb = np.array(ea.full_moment_bounds), np.array(eb.full_moment_bounds)
    ta, tb = np.array(ea.tail_moment_bounds), np.array(eb.tail_moment_bounds)
    gs = 2*np.pi*np.linalg.solve(args['cell'].T, args['reciprocal_indices'].T).T
    g2 = np.sum(gs*gs, axis=1); volume = abs(np.linalg.det(args['cell']))
    bounded = (len(args['near_indices'])*2*args['eta']/np.sqrt(np.pi)+np.pi/(args['eta']**2*volume)
               +4*np.pi/volume*sum(np.exp(-g2/(4*args['eta']**2))/g2))
    penetration = bounded*(ta[0]*fb[0]+fa[0]*tb[0])
    penetration += (len(args['real_indices'])-len(args['near_indices']))*(ta[0]*eb.potential_bound+tb[0]*ea.potential_bound)
    degrees = np.sum(result.matrix.indices, axis=1)
    substitution = np.sum(abs(result.matrix.values)*(ta[degrees, None]*fb[None, degrees]+fa[degrees, None]*tb[None, degrees]))
    return bounded, penetration, substitution


def gaussian_calculation(order=8, *, skew=True, eta=.25, near=0, radius=.8):
    """Complete finite electrostatic energy with Gaussian self integrals.

    The independent reference analytically convolves the entire finite Green
    function. This is neither a point-charge nuclear self convention nor an
    infinite-lattice or density-aggregated production Fock certificate.
    """
    args = fixture(order, skew=skew, eta=eta, near=near)
    centers = [[.21, -.17, .12], [-.19, .11, -.14], [.04, .07, .02]]
    exponents = [16., 19., 23.]; charges = np.array([1., -.7, -.3])
    sources = [source('aux', c, exponent=p, radius=radius, scale=(p/np.pi)**1.5)
               for c, p in zip(centers, exponents)]
    actual = np.zeros((3, 3)); exact = np.zeros((3, 3)); bounds = np.zeros((3, 3))
    identities = []; parts = []
    for i, a in enumerate(sources):
        for j, b in enumerate(sources):
            result = interaction(args, a, b)
            beta = exponents[i]*exponents[j]/(exponents[i]+exponents[j])
            lengths = np.linalg.norm(np.array(centers[i])-centers[j]+args['near_indices']@args['cell'].T, axis=1)
            nonzero = lengths != 0
            direct = sum(erf(np.sqrt(beta)*lengths[nonzero])/lengths[nonzero])
            direct += np.count_nonzero(~nonzero)*2*np.sqrt(beta/np.pi)
            actual[i, j] = direct+(moments(a, order)@result.matrix.values@moments(b, order).T)[0, 0]
            exact[i, j] = direct+scalar_reference(args, a, b)
            bounds[i, j] = result.error.total
            parts.append([result.error.compact_remainder, result.error.tail_penetration, result.error.full_moment_tail])
            identities.append(result.error.source_identity)
    energy, reference = float(.5*charges@actual@charges), float(.5*charges@exact@charges)
    return dict(energy=energy, reference_energy=reference, absolute_error=abs(energy-reference),
                bound=float(.5*abs(charges)@bounds@abs(charges)), pair_values=actual, reference_values=exact,
                pair_bounds=bounds, component_bounds=np.array(parts), source_identities=identities)


@pytest.mark.parametrize('kinds', [('pair', 'pair'), ('pair', 'aux'), ('aux', 'pair'), ('aux', 'aux')])
@pytest.mark.parametrize('exponent,radius', [(.03, .1), (.7, .5), (12., 1.)])
@pytest.mark.parametrize('order', [0, 4, 12])
def test_native_envelopes_bound_closed_gaussian_convolutions(kinds, exponent, radius, order):
    args = fixture(order); a = source(kinds[0], [.37, -.23, .19], exponent=exponent, radius=radius)
    b = source(kinds[1], [0., 0., 0.], exponent=1.3*exponent, radius=radius)
    result = interaction(args, a, b)
    value = float((moments(a, order)@result.matrix.values@moments(b, order).T)[0, 0])
    exact = scalar_reference(args, a, b)
    assert abs(value-exact) <= result.error.total+2e-13*max(1., abs(exact))
    expected = reference_ledger(args, a, b, result)
    actual = [result.error.bounded_kernel_sup, result.error.tail_penetration, result.error.full_moment_tail]
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=0.)
    assert np.all(np.array(actual) >= np.array(expected)*(1-2e-13))
    assert result.error.total >= result.error.compact_remainder+sum(actual[1:])
    assert result.compact.expansion_identity == result.matrix.source_identity
    assert result.error.qualification == 'finite-domain-gaussian-approximation'
    assert not result.error.roundoff_included
    assert result.error.reciprocal_terms_evaluated == len(args['reciprocal_indices'])
    assert result.error.moment_products_evaluated <= result.plan.coefficients.scalar_terms


@pytest.mark.parametrize('kinds', [('pair', 'pair'), ('pair', 'aux'), ('aux', 'pair'), ('aux', 'aux')])
def test_signed_off_center_angular_sources_against_independent_gaussian_quadrature(kinds):
    args = fixture(8)
    a = source(kinds[0], [.37, -.23, .19], angular=1, signed=True, spread=True)
    b = source(kinds[1], [0., 0., 0.], angular=2, signed=True, spread=True, exponent=1.1)
    result = interaction(args, a, b)
    value = moments(a, 8)@result.matrix.values@moments(b, 8).T
    reference = finite_reference(args, a, b)
    refined = finite_reference(args, a, b, quadrature=104)
    np.testing.assert_allclose(reference, refined, atol=5e-13, rtol=3e-12)
    assert np.max(abs(value-refined)) <= result.error.total+5e-13
    # Every AO-component error is bounded before any signed-density contraction.
    da, db = np.linspace(-.8, 1.1, len(value)), np.linspace(.7, -.4, value.shape[1])
    assert abs(da@(value-refined)@db) <= result.error.total*sum(abs(da))*sum(abs(db))+5e-13


def test_independent_quadrature_matches_closed_gaussian_limit():
    args = fixture(); a = source('pair', [.2, -.1, .3], signed=True)
    b = source('aux', [0., 0., 0.], signed=True)
    assert finite_reference(args, a, b)[0, 0] == pytest.approx(scalar_reference(args, a, b), abs=2e-13)


@pytest.mark.parametrize('near', [0, 1])
@pytest.mark.parametrize('order', [0, 4, 8, 12])
def test_complete_finite_gaussian_energy_with_near_and_tail_partition(order, near):
    row = gaussian_calculation(order, near=near)
    assert row['absolute_error'] <= row['bound']+3e-14
    assert np.all(abs(row['pair_values']-row['reference_values']) <= row['pair_bounds']+3e-14)
    np.testing.assert_allclose(row['pair_values'], row['pair_values'].T, atol=3e-14, rtol=0.)
    assert np.all(row['component_bounds'] > 0)


def test_reciprocity_canonical_domains_and_owned_immutable_results():
    args = fixture(); a = source('pair', [.37, -.23, .19]); b = source('aux', [0., 0., 0.])
    result = interaction(args, a, b); reverse = interaction(args, b, a)
    np.testing.assert_allclose(reverse.matrix.values, result.matrix.values.T, rtol=3e-13, atol=3e-15)
    assert reverse.error.total == pytest.approx(result.error.total, rel=3e-12)
    assert reverse.error.source_identity != result.error.source_identity
    assert reverse.error.source_envelope_identity == result.error.target_envelope_identity
    permuted = {key: np.asfortranarray(value[::-1] if key.endswith('_indices') else value)
                for key, value in args.items() if isinstance(value, np.ndarray)}
    other = interaction(args|permuted, a, b)
    assert other.error.source_identity == result.error.source_identity and other.error.total == result.error.total
    copied = result.matrix.values; saved = copied.copy(); copied[:] = -777
    np.testing.assert_array_equal(result.matrix.values, saved)
    for obj, field in [(result.error, 'total'), (result.plan, 'kernel_terms'), (result.matrix, 'source_identity')]:
        with pytest.raises(AttributeError): setattr(obj, field, 0)
    matrix, error = result.matrix, result.error
    del result, a, b
    np.testing.assert_array_equal(matrix.values, saved)
    assert error.total > 0


@pytest.mark.parametrize('field', ['memory_bytes', 'scalar_terms', 'kernel_terms'])
def test_joint_admission_exact_caps_and_refusal_before_payload_read(field):
    args = fixture(8); a = source('pair', [.37, -.23, .19]); b = source('aux', [0., 0., 0.])
    p = manta.plan_lattice_gaussian_interaction(8, 27, 124, 1)
    budget = manta.ResourceBudget(); required = getattr(p, 'modeled_peak_bytes' if field == 'memory_bytes' else field)
    setattr(budget, field, required)
    result = interaction(args|dict(budget=budget), a, b)
    assert getattr(result.plan, 'modeled_peak_bytes' if field == 'memory_bytes' else field) == required
    setattr(budget, field, required-1)
    with pytest.raises(ValueError, match='budget'):
        interaction(args|dict(cell=np.full((3, 3), np.nan), budget=budget), a, b)


def test_conditional_support_missing_moments_always_near_and_strict_arrays():
    args = fixture(); a = source('pair', [.37, -.23, .19]); b = source('aux', [0., 0., 0.])
    for bad in (source('pair', [0., 0., 0.], retain=False), source('pair', [0., 0., 0.], order=2)):
        with pytest.raises(ValueError, match='full and tail moments'):
            interaction(args, bad, b)
    with pytest.raises(ValueError, match='always-near'):
        interaction(args, source('pair', [0., 0., 0.], strict=True, radius=.01), b)
    with pytest.raises(ValueError, match='support'):
        interaction(args, source('pair', [0., 0., 0.], radius=8.), b)
    broad = interaction(args|dict(near_indices=args['real_indices']), source('pair', [0., 0., 0.], radius=8.), b)
    assert np.isfinite(broad.error.total)
    for changed in (dict(cell=args['cell'].astype(np.float32)), dict(real_indices=args['real_indices'].astype(np.int32))):
        with pytest.raises(ValueError, match='dtype'): interaction(args|changed, a, b)
    with pytest.raises(TypeError):
        manta.lattice_gaussian_interaction(**args, target=dict(center=[0., 0., 0.]), source=b['extent'])
    huge = np.lib.stride_tricks.as_strided(np.zeros(3, np.int64), shape=(10**9, 3), strides=(24, 8))
    with pytest.raises(ValueError, match='budget'):
        interaction(args|dict(real_indices=huge), a, b)


def test_envelope_identity_retains_charged_components_and_radius_policy():
    args = fixture(); b = source('aux', [0., 0., 0.])
    a = source('aux', [.37, -.23, .19], angular=1)
    result = interaction(args, a, b)
    assert np.all(moments(a, 0) == 0.)  # zero signed monopoles, nonzero absolute envelopes
    assert result.error.total > 0 and result.error.tail_penetration > 0
    resized = interaction(args, source('aux', [.37, -.23, .19], angular=1, radius=.5), b)
    assert resized.matrix.source_identity == result.matrix.source_identity
    assert resized.error.source_identity != result.error.source_identity
    assert resized.error.target_envelope_identity != result.error.target_envelope_identity


def test_positive_tiny_error_is_retained_and_overflow_refuses():
    args = fixture(0)
    tiny_a = source('aux', [.37, -.23, .19], scale=1e-165)
    tiny_b = source('aux', [0., 0., 0.], scale=1e-165)
    tiny = interaction(args, tiny_a, tiny_b)
    assert 0 < tiny.error.total < np.finfo(float).tiny
    with pytest.raises(OverflowError):
        interaction(args, source('aux', [.37, -.23, .19], scale=1e200), source('aux', [0., 0., 0.], scale=1e200))


def test_empty_reciprocal_set_keeps_gauge_and_near_penetration():
    args = fixture(4, eta=.01)|dict(real_indices=cube(0), near_indices=cube(0), reciprocal_indices=cube(0, reciprocal=True))
    a = source('pair', [.37, -.23, .19]); b = source('aux', [0., 0., 0.])
    result = interaction(args, a, b); expected = reference_ledger(args, a, b, result)
    assert result.error.reciprocal_terms_evaluated == 0 and result.compact.real_remainder == 0
    assert result.error.bounded_kernel_sup >= abs(result.matrix.gauge_constant)
    np.testing.assert_allclose([result.error.bounded_kernel_sup, result.error.tail_penetration,
                               result.error.full_moment_tail], expected, rtol=2e-11, atol=0.)


def test_units_and_density_scaling_preserve_separate_error_terms():
    args = fixture(6); center = np.array([.37, -.23, .19])
    result = interaction(args, source('pair', center), source('aux', [0., 0., 0.]))
    fields = ('compact_remainder', 'tail_penetration', 'full_moment_tail', 'total')
    for length in (1e-3, 1e3):
        a = source('pair', center*length, exponent=.7/length**2, radius=.3*length, scale=length**-1.5)
        b = source('aux', [0., 0., 0.], exponent=.7/length**2, radius=.3*length, scale=length**-3)
        scaled = interaction(args|dict(cell=args['cell']*length, eta=args['eta']/length), a, b)
        for field in fields:
            assert getattr(scaled.error, field)*length == pytest.approx(getattr(result.error, field), rel=3e-11)
    charged = interaction(args, source('pair', center, scale=2.), source('aux', [0., 0., 0.], scale=3.))
    for field in fields:
        assert getattr(charged.error, field) == pytest.approx(12*getattr(result.error, field), rel=3e-11)


def test_tight_gaussian_limit_can_meet_an_explicit_error_target():
    exponent = 1e4; scale = (exponent/np.pi)**1.5
    a = source('aux', [.37, -.23, .19], exponent=exponent, radius=.08, scale=scale)
    b = source('aux', [0., 0., 0.], exponent=exponent, radius=.08, scale=scale)
    exact = scalar_reference(fixture(), a, b)
    first = interaction(fixture(0), a, b)
    assert first.error.total > 1e-8
    for order in range(1, 13):
        result = interaction(fixture(order), a, b)
        if result.error.total < 1e-8:
            value = (moments(a, order)@result.matrix.values@moments(b, order).T)[0, 0]
            assert abs(value-exact) <= result.error.total+3e-14
            assert abs(value-exact) < 1e-8
            break
    else:
        pytest.fail('No order meets the finite Gaussian approximation target in the tight-source limit')
