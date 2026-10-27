"""Full Gaussian envelopes and infinite Ewald domain errors, without compact support."""
from itertools import product

import numpy as np
import pytest
from scipy.special import erf

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from tests.test_manta_pair_moments import _basis, _cart
from tests.test_manta_lattice_tail import cube
from tests.test_manta_lattice_gaussian import (
    source, components, fourier, scalar_reference, finite_reference, gaussian_integral,
)


def majorant(s):
    e = s['extent']
    if s['kind'] == 'pair':
        return manta.compute_pair_gaussian_majorant(s['basis'], e.shells)
    return manta.compute_auxiliary_gaussian_majorant(s['basis'], e.shell)


def fixture(cutoff=2, eta=.45, skew=True):
    cell = np.array([[5., .7, -.4], [0., 4.7, .3], [0., 0., 5.3]]) if skew else np.eye(3)*5
    return dict(cell=cell, eta=eta, real_indices=cube(cutoff),
                reciprocal_indices=cube(cutoff, reciprocal=True), near_indices=cube(0))


def envelope(g, points):
    return sum(t.charge_bound*(t.exponent/np.pi)**1.5
               *np.exp(-t.exponent*np.sum((points-t.center)**2, axis=1)) for t in g.terms)


def ao_values(shell, points):
    x = points-shell.origin
    cart = _cart(shell.l)
    angular = np.prod(x[:, None, :]**cart[None, :, :], axis=2)
    radial = sum(c*np.exp(-a*np.sum(x*x, axis=1))
                 for a, c in zip(shell.exponents, shell.coefficients, strict=True))
    transform = manta.cartesian_to_spherical(shell.l) if shell.pure else np.eye(len(cart))
    return (angular*radial[:, None])@transform.T


@pytest.mark.parametrize('l', [0, 2, 6])
@pytest.mark.parametrize('pure', [False, True])
@pytest.mark.parametrize('shift', [[0., 0., 0.], [2.3, -.9, .7]])
def test_pointwise_all_signed_angular_components(l, pure, shift):
    basis = _basis(l, pure); a, b = basis.shells(); b.origin = (np.array(b.origin)+shift).tolist()
    rng = np.random.default_rng(135)
    points = rng.normal(size=(400, 3))*rng.choice([.1, 1., 5., 12.], size=(400, 1))
    ga = manta.compute_auxiliary_gaussian_majorant(basis, 0)
    gp = manta.compute_pair_gaussian_majorant(basis, [0, 1], shift)
    av, bv = ao_values(a, points), ao_values(b, points)
    assert np.all(np.max(abs(av), axis=1) <= envelope(ga, points)*(1+5e-13)+1e-300)
    assert np.all(np.max(abs(av), axis=1)*np.max(abs(bv), axis=1) <= envelope(gp, points)*(1+5e-13)+1e-300)
    assert ga.charge_bound >= sum(t.charge_bound for t in ga.terms)
    assert gp.plan.primitive_pairs == gp.plan.maximum_terms == len(gp.terms) == 4


def make_shell(*, l=0, exponent=.7, coefficient=1., center=(0., 0., 0.)):
    mol = vq.Molecule([vq.Atom(2, center)], 0, 1)
    s = core.ShellInfo(); s.atom_index = 0; s.l = l; s.pure = True
    s.origin = center; s.exponents = [exponent]; s.coefficients = [coefficient]
    return vq.BasisSet(mol, [s], 'gaussian-majorant', True)


def test_s_gaussian_mass_auxiliary_limit_high_angular_and_rounded_centers():
    b = make_shell(exponent=.7)
    g = manta.compute_auxiliary_gaussian_majorant(b, 0); t = g.terms[0]
    assert t.exponent == np.nextafter(.35, 0.)
    assert t.charge_bound == pytest.approx((np.pi/t.exponent)**1.5, rel=1e-12)
    b = make_shell(l=7); s = b.shells()[0]; g = manta.compute_auxiliary_gaussian_majorant(b, 0)
    x = np.random.default_rng(12).normal(size=(200, 3))*3
    assert np.all(np.max(abs(ao_values(s, x)), axis=1) <= envelope(g, x))
    # Translation at a large absolute coordinate makes rounding of the product
    # center material; the exported wider Gaussian still dominates every sample.
    b = _basis(0, False, exponents=(.7,), shift=(1e12, 0., 0.))
    g = manta.compute_pair_gaussian_majorant(b, [0, 1])
    x = np.array(g.terms[0].center)+np.random.default_rng(4).normal(size=(200, 3))
    a, c = b.shells()
    assert np.all(abs(ao_values(a, x)[:, 0]*ao_values(c, x)[:, 0]) <= envelope(g, x))


@pytest.mark.parametrize('pair', [False, True])
@pytest.mark.parametrize('field,planned', [('shell_calls', 'shell_calls'), ('primitive_quartets', 'primitive_pairs'),
    ('memory_bytes', 'modeled_peak_bytes'), ('scalar_terms', 'scalar_terms'), ('kernel_terms', 'kernel_terms')])
def test_producer_resource_caps(pair, field, planned):
    b = _basis(); args = (b, [0, 1]) if pair else (b, 0)
    plan = (manta.plan_pair_gaussian_majorant if pair else manta.plan_auxiliary_gaussian_majorant)(*args)
    compute = manta.compute_pair_gaussian_majorant if pair else manta.compute_auxiliary_gaussian_majorant
    budget = manta.ResourceBudget(); setattr(budget, field, getattr(plan, planned))
    assert compute(*args, budget=budget).charge_bound > 0
    setattr(budget, field, getattr(plan, planned)-1)
    with pytest.raises(ValueError, match='budget exceeded'): compute(*args, budget=budget)


def test_provenance_ownership_zero_and_invalid_envelopes():
    b = _basis(); g = manta.compute_pair_gaussian_majorant(b, [0, 1])
    for cls in [manta.GaussianMajorant, manta.GaussianMajorantTerm]:
        with pytest.raises(TypeError): cls()
    with pytest.raises(AttributeError): g.terms[0].charge_bound = 0
    copy = g.terms; copy[0].center[0] = 1e10; copy.clear()
    assert g.terms and abs(g.terms[0].center[0]) < 1
    changed = manta.compute_pair_gaussian_majorant(b, [0, 1], [1., 0., 0.])
    assert g.basis_identity == changed.basis_identity and g.source_identity != changed.source_identity
    assert g.source_identity == manta.compute_pair_gaussian_majorant(b, [0, 1]).source_identity
    for i in [-1, 2]:
        with pytest.raises((TypeError, IndexError)): manta.compute_auxiliary_gaussian_majorant(b, i)
    with pytest.raises(ValueError): manta.compute_pair_gaussian_majorant(b, [0, 1], [np.nan, 0., 0.])
    zero = manta.compute_auxiliary_gaussian_majorant(make_shell(coefficient=0.), 0)
    assert zero.charge_bound == 0 and not zero.terms
    tiny = manta.compute_auxiliary_gaussian_majorant(make_shell(coefficient=1e-300), 0)
    assert tiny.charge_bound > 0
    with pytest.raises(OverflowError):
        manta.compute_auxiliary_gaussian_majorant(make_shell(exponent=1e-300), 0)


def infinite_reference(args, a, b, cutoff=16):
    """Independent all-reciprocal zero-average Gaussian Coulomb matrix."""
    g = 2*np.pi*np.linalg.solve(args['cell'].T, cube(cutoff, reciprocal=True).T).T
    g2 = np.sum(g*g, axis=1)
    fa, fb = np.array([fourier(t, g) for t in components(a)]), np.array([fourier(t, g) for t in components(b)])
    return ((fa*(4*np.pi/abs(np.linalg.det(args['cell']))/g2))@fb.conj().T).real


def near_reference(args, a, b, quadrature=72):
    aa, bb = components(a), components(b); out = np.zeros((len(aa), len(bb)))
    x, w = np.polynomial.legendre.leggauss(quadrature); t = (x+1)/2
    s, weights = t/(1-t), w/2/(1-t)**2
    for index in args['near_indices']:
        for i, ai in enumerate(aa):
            for j, bj in enumerate(bb):
                for ta, tb in product(ai, bj):
                    out[i, j] += 2/np.sqrt(np.pi)*np.dot(weights, gaussian_integral(ta, tb, args['cell']@index, s))
    return out


@pytest.mark.parametrize('kinds', [('pair', 'pair'), ('pair', 'aux'), ('aux', 'pair'), ('aux', 'aux')])
@pytest.mark.parametrize('exponent', [.04, .7, 3.])
def test_infinite_gaussian_convolutions_all_sources(kinds, exponent):
    args = fixture(cutoff=2)
    a = source(kinds[0], [.3, -.2, .1], exponent=exponent, signed=True, spread=True)
    b = source(kinds[1], [0., 0., 0.], exponent=1.2*exponent, signed=True)
    result = manta.lattice_gaussian_domain_remainder(**args, target=majorant(a), source=majorant(b))
    finite = scalar_reference(args, a, b)+near_reference(args, a, b)[0, 0]
    expected = infinite_reference(args, a, b)[0, 0]
    assert abs(finite-expected) <= result.total+2e-11
    assert 0 < result.minimum_effective_eta < args['eta']
    assert result.physical_eta == args['eta'] and not result.roundoff_included
    assert result.qualification == 'native-gaussian-infinite-ewald-domain-remainder'


@pytest.mark.parametrize('kinds', [('pair', 'aux'), ('aux', 'pair')])
def test_signed_angular_gaussian_domain_error(kinds):
    args = fixture(cutoff=1)
    a = source(kinds[0], [.3, -.2, .1], exponent=.8, angular=1, signed=True, spread=True)
    b = source(kinds[1], [0., 0., 0.], exponent=1.1, angular=2, signed=True)
    result = manta.lattice_gaussian_domain_remainder(**args, target=majorant(a), source=majorant(b))
    finite = finite_reference(args, a, b)+near_reference(args, a, b)
    expected = infinite_reference(args, a, b)
    np.testing.assert_allclose(expected, infinite_reference(args, a, b, 20), atol=3e-13, rtol=3e-13)
    assert np.max(abs(finite-expected)) <= result.total+3e-12


@pytest.mark.parametrize('field,planned', [('memory_bytes', 'modeled_peak_bytes'), ('primitive_quartets', 'primitive_products'),
                                           ('scalar_terms', 'scalar_terms'), ('kernel_terms', 'kernel_terms')])
def test_domain_joint_admission_precedes_payload_reads(field, planned):
    args = fixture(); g = manta.compute_pair_gaussian_majorant(_basis(), [0, 1])
    p = manta.plan_lattice_gaussian_domain_remainder(len(g.terms), len(g.terms), 125, 124, 1)
    budget = manta.ResourceBudget(); setattr(budget, field, getattr(p, planned))
    assert manta.lattice_gaussian_domain_remainder(**args, target=g, source=g, budget=budget).total > 0
    setattr(budget, field, getattr(p, planned)-1); args['cell'][:] = np.nan
    with pytest.raises(ValueError, match='budget exceeded'):
        manta.lattice_gaussian_domain_remainder(**args, target=g, source=g, budget=budget)


def test_domain_identity_symmetry_near_invariance_and_refusals():
    args = fixture(); a = majorant(source('pair', [.3, -.2, .1])); b = majorant(source('aux', [0., 0., 0.]))
    def run(**kw): return manta.lattice_gaussian_domain_remainder(**dict(args, target=a, source=b, **kw))
    first = run()
    reordered = {k: np.asfortranarray(v[::-1]) for k, v in args.items() if k.endswith('indices')}
    second = manta.lattice_gaussian_domain_remainder(**dict(args, **reordered), target=a, source=b)
    assert first.source_identity == second.source_identity and first.total == second.total
    reverse = manta.lattice_gaussian_domain_remainder(**args, target=b, source=a)
    assert first.total == pytest.approx(reverse.total, rel=2e-12)
    assert first.source_identity != reverse.source_identity
    near = manta.lattice_gaussian_domain_remainder(**dict(args, near_indices=cube(1)), target=a, source=b)
    assert near.total == first.total and near.kernel_identity != first.kernel_identity
    tail = manta.multipole_lattice_tail(**args, target_minus_source=np.zeros(3), order=0)
    assert tail.kernel_identity == first.kernel_identity
    for field, value in [('eta', 0.), ('cell', np.zeros((3, 3))), ('real_indices', cube(2)[1:-1]),
                         ('near_indices', cube(3)), ('real_indices', cube(2).astype(float))]:
        with pytest.raises((ValueError, OverflowError)):
            manta.lattice_gaussian_domain_remainder(**dict(args, **{field:value}), target=a, source=b)
    far = majorant(source('aux', [100., 0., 0.]))
    with pytest.raises(ValueError, match='not separated'):
        manta.lattice_gaussian_domain_remainder(**args, target=a, source=far)
    huge = np.lib.stride_tricks.as_strided(np.zeros(3, dtype=np.int64), shape=(10**14, 3), strides=(24, 8))
    with pytest.raises((ValueError, OverflowError), match='budget|overflow'):
        manta.lattice_gaussian_domain_remainder(**dict(args, real_indices=huge), target=a, source=b)
    zero = manta.compute_auxiliary_gaussian_majorant(make_shell(coefficient=0.), 0)
    assert manta.lattice_gaussian_domain_remainder(**args, target=zero, source=b).total == 0
    with pytest.raises(ValueError):
        manta.lattice_gaussian_domain_remainder(**dict(args, eta=0.), target=zero, source=b)
    tiny = manta.compute_auxiliary_gaussian_majorant(make_shell(coefficient=1e-300), 0)
    positive = manta.lattice_gaussian_domain_remainder(**args, target=tiny, source=tiny)
    assert 0 < positive.real_remainder <= positive.total and 0 < positive.reciprocal_remainder <= positive.total


def gaussian_domain_calculation(*, cutoff=3, eta=.45, skew=True, translated=False):
    """Complete neutral smooth-charge Ewald energy with full Gaussian self."""
    args = fixture(cutoff=cutoff, eta=eta, skew=skew)
    centers = np.array([[.2, -.1, .3], [1.8, 1.6, 1.4]])
    if translated: centers[1] += args['cell'][:, 0]
    exponents = [.65, 1.1]; charges = np.array([1., -1.])
    sources = [source('aux', c, exponent=p, scale=(p/np.pi)**1.5) for c, p in zip(centers, exponents)]
    envelopes = list(map(majorant, sources))
    values = np.empty((2, 2)); reference = np.empty((2, 2)); bounds = np.empty((2, 2)); real = np.empty((2, 2)); ids = []
    for i, a in enumerate(sources):
        for j, b in enumerate(sources):
            bound = manta.lattice_gaussian_domain_remainder(**args, target=envelopes[i], source=envelopes[j])
            beta = exponents[i]*exponents[j]/(exponents[i]+exponents[j]); d = np.linalg.norm(centers[i]-centers[j])
            direct = erf(np.sqrt(beta)*d)/d if d else 2*np.sqrt(beta/np.pi)
            values[i, j] = scalar_reference(args, a, b)+direct
            reference[i, j] = infinite_reference(args, a, b)[0, 0]
            bounds[i, j] = bound.total; real[i, j] = bound.real_remainder; ids.append(bound.source_identity)
    return dict(energy=float(.5*charges@values@charges), reference_energy=float(.5*charges@reference@charges),
                bound=float(.5*abs(charges)@bounds@abs(charges)), values=values, reference=reference,
                bounds=bounds, real_bounds=real, positions=centers, charges=charges, source_identities=ids)


@pytest.mark.parametrize('skew', [False, True])
@pytest.mark.parametrize('eta', [.3, .65])
def test_full_gaussian_energies_with_declared_accuracy_target(skew, eta):
    for cutoff in range(3, 13):
        row = gaussian_domain_calculation(skew=skew, eta=eta, cutoff=cutoff, translated=True)
        assert np.all(abs(row['values']-row['reference']) <= row['bounds']+5e-13)
        if row['bound'] < 1e-8:
            assert abs(row['energy']-row['reference_energy']) < 1e-8
            break
    else: raise AssertionError('No cube met the 1e-8 Eh target')


def native_gaussian_domain_calculation():
    """Native moments/finite Ewald coefficients plus both Gaussian error ledgers."""
    from tests.test_manta_lattice_gaussian import interaction, moments
    args = fixture(cutoff=2, eta=.3); order = 8
    centers = np.array([[.2, -.1, .3], [-.15, .12, -.17]])
    exponents = [8., 11.]; charges = np.array([1., -1.])
    sources = [source('aux', c, exponent=p, radius=1.5, scale=(p/np.pi)**1.5, order=order)
               for c, p in zip(centers, exponents)]
    envelopes = list(map(majorant, sources)); arrays = [np.empty((2, 2)) for _ in range(5)]
    values, reference, finite, domain_bounds, approximation_bounds = arrays; ids = []
    for i, a in enumerate(sources):
        for j, b in enumerate(sources):
            result = interaction(dict(args, order=order), a, b)
            bound = manta.lattice_gaussian_domain_remainder(**args, target=envelopes[i], source=envelopes[j])
            assert result.matrix.kernel_identity == bound.kernel_identity
            beta = exponents[i]*exponents[j]/(exponents[i]+exponents[j]); d = np.linalg.norm(centers[i]-centers[j])
            direct = erf(np.sqrt(beta)*d)/d if d else 2*np.sqrt(beta/np.pi)
            values[i, j] = direct+(moments(a, order)@result.matrix.values@moments(b, order).T)[0, 0]
            finite[i, j] = direct+scalar_reference(args, a, b)
            reference[i, j] = infinite_reference(args, a, b, cutoff=32)[0, 0]
            domain_bounds[i, j] = bound.total; approximation_bounds[i, j] = result.error.total
            ids.append([result.matrix.kernel_identity, bound.source_identity, result.error.source_identity])
    bounds = domain_bounds+approximation_bounds
    return dict(energy=float(.5*charges@values@charges), reference_energy=float(.5*charges@reference@charges),
                bound=float(.5*abs(charges)@bounds@abs(charges)), values=values, reference=reference, finite=finite,
                bounds=bounds, domain_bounds=domain_bounds, approximation_bounds=approximation_bounds,
                positions=centers, charges=charges, source_identities=ids)


def test_native_complete_energy_combines_matching_gaussian_ledgers():
    row = native_gaussian_domain_calculation()
    assert np.all(abs(row['values']-row['finite']) <= row['approximation_bounds']+5e-13)
    assert np.all(abs(row['finite']-row['reference']) <= row['domain_bounds']+5e-13)
    assert np.all(abs(row['values']-row['reference']) <= row['bounds']+5e-13)
    assert abs(row['energy']-row['reference_energy']) <= row['bound']+5e-13
