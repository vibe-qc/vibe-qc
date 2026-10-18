"""Conditional compact-charge Taylor control for the finite Ewald complement."""
from math import factorial

import numpy as np
import pytest
from scipy.special import erfc, gamma, gammainc, gammaincc

from vibeqc import manta
from tests.test_manta_lattice_tail import cube, fixture, point_moments


def controls():
    return dict(target_radius=.19, source_radius=.13, target_norm=1.7, source_norm=1.2)


def reference_components(args, envelopes):
    """The analytic directional bound with SciPy's incomplete gamma functions.

    Native code instead uses positive upper envelopes for those integrals.
    This reference uses the input geometry without the native distance guards.
    """
    n = args['order']+1
    rho = envelopes['target_radius']+envelopes['source_radius']
    charge = envelopes['target_norm']*envelopes['source_norm']
    near = set(map(tuple, args['near_indices'])); parts = np.zeros(3)
    for index in args['real_indices']:
        distance = np.linalg.norm(args['target_minus_source']+args['cell']@index)
        lower, upper = max(0., distance-rho), distance+rho
        inside = tuple(index) in near
        assert inside or lower > 0
        for k in range(n//2+1):
            m, shape = n-k, n-k+.5
            if lower == 0:
                integral = args['eta']**(2*m+1)/(2*m+1)
            else:
                f = gammainc if inside else gammaincc
                integral = .5*gamma(shape)*f(shape, (args['eta']*lower)**2)/lower**(2*m+1)
            parts[0 if inside else 1] += 2/np.sqrt(np.pi)*rho**n*2**(n-2*k)*upper**(n-2*k)*integral/(factorial(k)*factorial(n-2*k))
    gs = 2*np.pi*np.linalg.solve(args['cell'].T, args['reciprocal_indices'].T).T
    norms = np.linalg.norm(gs, axis=1)
    parts[2] = 4*np.pi/abs(np.linalg.det(args['cell']))*sum(
        np.exp(-norms**2/(4*args['eta']**2))/norms**2*(rho*norms)**n)/factorial(n)
    return parts*charge


def finite_green(args, displacement):
    """Direct finite Ewald Green values, without native Taylor coefficients."""
    x = np.atleast_2d(displacement)
    distances = np.linalg.norm(x[:, None, :]+args['real_indices']@args['cell'].T, axis=2)
    if np.any(distances == 0):
        raise ValueError('Reference cross-charge energy requires distinct point supports')
    real = np.sum(erfc(args['eta']*distances)/distances, axis=1)
    gs = 2*np.pi*np.linalg.solve(args['cell'].T, args['reciprocal_indices'].T).T
    g2 = np.sum(gs*gs, axis=1); volume = abs(np.linalg.det(args['cell']))
    reciprocal = 4*np.pi/volume*(np.cos(x@gs.T)@(np.exp(-g2/(4*args['eta']**2))/g2))
    return real+reciprocal-np.pi/(args['eta']**2*volume)


def compact_calculation(order=6, *, skew=True, eta=.63, near=0, radius_scale=1.):
    """Complete cross electrostatic energy of two signed compact point clouds.

    These are finite measures with exact supports and raw moments, not Gaussian
    distributions or nuclei requiring a point self-energy subtraction.
    """
    args = fixture(order, skew=skew, real=1, reciprocal=3, near=near, eta=eta)
    pa = radius_scale*np.array([[.17, -.04, .07], [-.12, .13, .01], [.02, -.07, -.13]])
    pb = radius_scale*np.array([[.03, .10, -.06], [-.07, -.03, .10], [.09, -.02, -.03]])
    qa, qb = np.array([.9, -.5, .3]), np.array([.6, -.45, -.15])
    envelopes = dict(target_radius=float(np.max(np.linalg.norm(pa, axis=1))),
                     source_radius=float(np.max(np.linalg.norm(pb, axis=1))),
                     target_norm=float(sum(abs(qa))), source_norm=float(sum(abs(qb))))
    matrix = manta.multipole_lattice_tail(**args)
    bound = manta.lattice_tail_remainder(**args, **envelopes)
    differences = args['target_minus_source']+pa[:, None, :]-pb[None, :, :]
    exact = finite_green(args, differences.reshape(-1, 3)).reshape(len(qa), len(qb))
    near_values = np.sum(1/np.linalg.norm(differences[:, :, None, :]+args['near_indices']@args['cell'].T, axis=3), axis=2)
    target = np.array([point_moments(matrix.indices, a[None, :], np.ones(1)) for a in pa])
    source = np.array([point_moments(matrix.indices, b[None, :], np.ones(1)) for b in pb])
    actual = near_values+target@matrix.values@source.T
    energy, reference = float(qa@actual@qb), float(qa@exact@qb)
    return dict(energy=energy, reference_energy=reference, absolute_error=abs(energy-reference),
                pair_values=actual, reference_values=exact, bound=bound, matrix=matrix, args=args, envelopes=envelopes)


@pytest.mark.parametrize('order', [0, 1, 2, 4, 6, 9, 12])
@pytest.mark.parametrize('eta', [.07, .63, 4.])
def test_positive_integral_envelopes_cover_independent_gamma_formula(order, eta):
    args = fixture(order, eta=eta, reciprocal=2)
    result = manta.lattice_tail_remainder(**args, **controls())
    expected = reference_components(args, controls())
    actual = np.array([result.near_remainder, result.real_remainder, result.reciprocal_remainder])
    assert np.all(actual >= expected*(1-2e-13))
    assert result.total >= sum(actual)
    assert result.radial_integrals_evaluated == result.plan.radial_integrals
    assert result.reciprocal_terms_evaluated == len(args['reciprocal_indices'])
    assert result.qualification == 'finite-domain-compact-taylor'
    assert not result.roundoff_included


@pytest.mark.parametrize('order', [0, 2, 4, 6, 8, 12])
@pytest.mark.parametrize('skew', [False, True])
def test_complete_signed_compact_energy_and_charged_pairs(order, skew):
    row = compact_calculation(order, skew=skew)
    assert row['absolute_error'] <= row['bound'].total+2e-14
    pair_bound = row['bound'].total/(row['envelopes']['target_norm']*row['envelopes']['source_norm'])
    assert np.max(abs(row['pair_values']-row['reference_values'])) <= pair_bound+2e-14
    assert row['bound'].expansion_identity == row['matrix'].source_identity
    # A neutral source has nonzero absolute charge. Its remainder is not zero.
    assert row['bound'].total > 0


def test_bound_controls_adaptive_order_and_near_domain_transfers():
    energies = []
    for near in (0, 1):
        accepted = None
        for order in range(13):
            row = compact_calculation(order, near=near)
            if row['bound'].total < 1e-8:
                accepted = row; break
        assert accepted is not None and accepted['args']['order'] > 2
        assert accepted['absolute_error'] < 1e-8
        energies.append(accepted['energy'])
    assert abs(energies[0]-energies[1]) < 2e-8


def test_zero_support_and_charge_are_exact_zeros_but_still_validate_geometry():
    args = fixture(6)
    for changed in (dict(target_radius=0., source_radius=0.), dict(target_norm=0.), dict(source_norm=0.)):
        result = manta.lattice_tail_remainder(**args, **(controls()|changed))
        assert result.total == result.near_remainder == result.real_remainder == result.reciprocal_remainder == 0
        assert result.radial_integrals_evaluated == result.reciprocal_terms_evaluated == 0
    args['target_minus_source'] = args['cell'][:, 0].copy()
    with pytest.raises(ValueError, match='singularity'):
        manta.lattice_tail_remainder(**args, **(controls()|dict(target_norm=0.)))


def test_near_removable_origin_and_non_near_support_contact():
    args = fixture(12, skew=False)
    args['target_minus_source'][:] = 0
    result = manta.lattice_tail_remainder(**args, **controls())
    expected = reference_components(args, controls())
    assert result.near_remainder >= expected[0] and np.isfinite(result.total)
    for radius in (5., np.nextafter(5., 0.), 5.1):
        with pytest.raises(ValueError, match='support'):
            manta.lattice_tail_remainder(**args, **(controls()|dict(target_radius=radius, source_radius=0.)))
    # Including all supplied real cells in near removes their singularities,
    # regardless of support overlap. No omitted-image bound is inferred.
    args['near_indices'] = args['real_indices'].copy()
    result = manta.lattice_tail_remainder(**args, **(controls()|dict(target_radius=5.1)))
    assert np.isfinite(result.total) and np.isinf(result.minimum_non_near_distance)
    assert result.real_remainder == 0


def test_covariance_units_swapped_supports_and_envelope_scaling():
    args = fixture(6); env = controls()
    original = manta.lattice_tail_remainder(**args, **env)
    reverse = manta.lattice_tail_remainder(**(args|dict(target_minus_source=-args['target_minus_source'])),
        target_radius=env['source_radius'], source_radius=env['target_radius'],
        target_norm=env['source_norm'], source_norm=env['target_norm'])
    assert reverse.total == pytest.approx(original.total, rel=2e-14)
    angle = .34
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
    rotated = manta.lattice_tail_remainder(**(args|dict(cell=rotation@args['cell'], target_minus_source=rotation@args['target_minus_source'])), **env)
    assert rotated.total == pytest.approx(original.total, rel=3e-12)
    for scale in (1e-50, 1e50):
        scaled = manta.lattice_tail_remainder(**(args|dict(cell=args['cell']*scale,
            target_minus_source=args['target_minus_source']*scale, eta=args['eta']/scale)),
            **(env|dict(target_radius=env['target_radius']*scale, source_radius=env['source_radius']*scale)))
        assert scaled.total*scale == pytest.approx(original.total, rel=5e-12)
    charged = manta.lattice_tail_remainder(**args, **(env|dict(target_norm=17., source_norm=12.)))
    assert charged.total == pytest.approx(100*original.total, rel=5e-13)


def test_small_positive_contributions_do_not_underflow_to_false_zero():
    args = fixture(12, real=0, reciprocal=0)
    args['target_minus_source'][:] = 0
    tiny = manta.lattice_tail_remainder(**args, target_radius=1e-100, source_radius=0., target_norm=1., source_norm=1.)
    assert 0 < tiny.near_remainder <= tiny.total < np.finfo(float).tiny
    # Charge rescaling precedes conversion, so tiny radial factors can recover.
    scaled = manta.lattice_tail_remainder(**args, target_radius=1e-25, source_radius=0., target_norm=1e150, source_norm=1e150)
    assert scaled.total > 0 and np.isfinite(scaled.total)
    assert tiny.reciprocal_remainder == 0
    far = fixture(12, real=1, reciprocal=0, eta=100.)
    result = manta.lattice_tail_remainder(**far, **controls())
    assert 0 < result.real_remainder < np.finfo(float).tiny
    with pytest.raises(OverflowError):
        manta.lattice_tail_remainder(**fixture(0), **(controls()|dict(target_norm=1e308, source_norm=1e308)))


def test_identity_canonicalization_ownership_and_conditional_scope():
    args = fixture(); env = controls()
    original = manta.lattice_tail_remainder(**args, **env)
    permutation = {key: np.asfortranarray(value[::-1] if key.endswith('_indices') else value)
                   for key, value in args.items() if isinstance(value, np.ndarray)}
    other = manta.lattice_tail_remainder(**(args|permutation), **env)
    assert original.source_identity == other.source_identity and original.total == other.total
    for changed in (dict(target_radius=.2), dict(source_norm=2.)):
        other = manta.lattice_tail_remainder(**args, **(env|changed))
        assert other.source_identity != original.source_identity
        assert other.expansion_identity == original.expansion_identity
    for changes in (dict(eta=.7), dict(order=4), dict(near_indices=cube(1)), dict(real_indices=cube(2))):
        other = manta.lattice_tail_remainder(**(args|changes), **env)
        assert other.expansion_identity != original.expansion_identity
    args['cell'][:] = 0; args['real_indices'][:] = 777
    assert original.total > 0
    with pytest.raises(AttributeError):
        original.total = 0
    with pytest.raises(AttributeError):
        original.plan.order = 0


@pytest.mark.parametrize('field,planned', [('memory_bytes', 'modeled_peak_bytes'), ('scalar_terms', 'scalar_terms'), ('kernel_terms', 'kernel_terms')])
def test_count_only_admission_exact_and_one_short_before_payload(field, planned):
    args = fixture(); plan = manta.plan_lattice_tail_remainder(6, 27, 342, 1)
    budget = manta.ResourceBudget(); setattr(budget, field, getattr(plan, planned))
    result = manta.lattice_tail_remainder(**args, **controls(), budget=budget)
    assert result.plan.modeled_peak_bytes == plan.modeled_peak_bytes
    setattr(budget, field, getattr(plan, planned)-1)
    args['cell'][:] = np.nan; args['real_indices'][:] = 2**40
    with pytest.raises(ValueError, match='budget'):
        manta.lattice_tail_remainder(**args, **controls(), budget=budget)
    budget = manta.ResourceBudget(); budget.other_live_bytes = 12345
    assert manta.plan_lattice_tail_remainder(6, 27, 342, 1, budget).modeled_peak_bytes == plan.modeled_peak_bytes+12345


@pytest.mark.parametrize('field', ['target_radius', 'source_radius', 'target_norm', 'source_norm'])
@pytest.mark.parametrize('value', [-1., np.inf, np.nan])
def test_invalid_envelopes_fail_closed(field, value):
    with pytest.raises(ValueError):
        manta.lattice_tail_remainder(**fixture(), **(controls()|{field: value}))


def test_domains_counts_dtype_and_shape_are_admitted_separately():
    bad = [dict(order=13), dict(order=-1), dict(eta=0.), dict(cell=np.zeros((3, 3))),
           dict(real_indices=cube(1).astype(float)), dict(real_indices=np.zeros((3, 4), dtype=np.int64)),
           dict(reciprocal_indices=cube(1)), dict(near_indices=cube(2)),
           dict(real_indices=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64)),
           dict(target_minus_source=np.zeros(4)), dict(real_indices=np.zeros((2, 3), dtype=np.int64))]
    for change in bad:
        with pytest.raises((ValueError, TypeError)):
            manta.lattice_tail_remainder(**(fixture()|change), **controls())
    with pytest.raises(OverflowError):
        manta.plan_lattice_tail_remainder(12, 2**64-1, 1, 1)
    budget = manta.ResourceBudget(); budget.backend_margin_bytes = 0
    with pytest.raises(ValueError, match='margin'):
        manta.plan_lattice_tail_remainder(0, 1, 0, 1, budget)


def test_optional_high_precision_positive_integral_envelopes():
    mp = pytest.importorskip('mpmath')
    with mp.workdps(70):
        for order, eta, radius in [(0, .01, .1), (5, .7, .32), (12, 4., .32), (12, .6, 1e-25)]:
            args = fixture(order, real=0, reciprocal=0, eta=eta)
            args['target_minus_source'][:] = 0
            bound = manta.lattice_tail_remainder(**args, target_radius=radius, source_radius=0., target_norm=1e100, source_norm=1e100)
            n = order+1; rho = mp.mpf(radius); e = mp.mpf(eta)
            exact = mp.mpf(0)
            for k in range(n//2+1):
                integral = mp.quad(lambda s: s**(2*(n-k)), [0, e])
                exact += 2/mp.sqrt(mp.pi)*rho**n*2**(n-2*k)*rho**(n-2*k)*integral/(mp.factorial(k)*mp.factorial(n-2*k))
            exact *= mp.mpf(1e100)**2
            assert mp.mpf(bound.near_remainder) >= exact
            assert mp.mpf(bound.near_remainder) < exact*(1+mp.mpf('1e-10'))
        # Nonzero lower radius exercises the native positive lower series,
        # complete-gamma upper envelope and non-near upper-gamma recurrence.
        for order, eta in [(0, .03), (5, .7), (12, 4.)]:
            args = fixture(order, real=1, reciprocal=0, eta=eta)
            args['real_indices'] = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.int64)
            env = controls(); bound = manta.lattice_tail_remainder(**args, **env)
            n = order+1; rho = mp.mpf(env['target_radius'])+mp.mpf(env['source_radius'])
            exact = [mp.mpf(0), mp.mpf(0)]
            for index in args['real_indices']:
                x = [mp.mpf(args['target_minus_source'][i])+sum(mp.mpf(args['cell'][i, j])*int(index[j]) for j in range(3)) for i in range(3)]
                distance = mp.sqrt(sum(v*v for v in x)); lower = distance-rho; upper = distance+rho
                near = not np.any(index)
                assert lower > 0
                for k in range(n//2+1):
                    shape = n-k+mp.mpf('.5'); t = (mp.mpf(eta)*lower)**2
                    incomplete = mp.gammainc(shape, 0, t) if near else mp.gammainc(shape, t, mp.inf)
                    integral = incomplete/(2*lower**(2*shape))
                    exact[0 if near else 1] += 2/mp.sqrt(mp.pi)*rho**n*2**(n-2*k)*upper**(n-2*k)*integral/(mp.factorial(k)*mp.factorial(n-2*k))
            for actual, value in zip((bound.near_remainder, bound.real_remainder), exact):
                assert mp.mpf(actual) >= value*mp.mpf(env['target_norm'])*mp.mpf(env['source_norm'])
