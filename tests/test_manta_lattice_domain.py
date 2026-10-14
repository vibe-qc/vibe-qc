"""Infinite Ewald cube omissions, compact energies and periodic covariance."""
import numpy as np
import pytest
from scipy.special import erfc

from vibeqc import manta
from tests.test_manta_lattice_tail import cube


def fixture(*, skew=True, real=2, reciprocal=4, near=0, eta=.45):
    cell = np.array([[5., .4, -.3], [0., 4.7, .2], [0., 0., 5.3]]) if skew else np.eye(3)*5
    return dict(cell=cell, target_minus_source=np.array([.37, -.23, .19]), eta=eta,
                real_indices=cube(real), reciprocal_indices=cube(reciprocal, reciprocal=True),
                near_indices=cube(near), target_radius=.12, source_radius=.17,
                target_norm=1.3, source_norm=.8)


def native_value(args, r=None):
    """Finite regular Ewald Green function, adding exactly its near subtraction."""
    x = args['target_minus_source'] if r is None else r
    keys = ('cell', 'eta', 'real_indices', 'reciprocal_indices', 'near_indices')
    tail = manta.multipole_lattice_tail(**{k: args[k] for k in keys}, target_minus_source=x, order=0)
    radii = np.linalg.norm(x+args['near_indices']@args['cell'].T, axis=1)
    return float(tail.values[0, 0]+np.sum(1/radii[radii != 0]))


def reference_value(args, r=None, *, real=7, reciprocal=15):
    """Independent SciPy erfc/cosine sum, including the regular point self."""
    a, eta = args['cell'], args['eta']; x = args['target_minus_source'] if r is None else r
    radii = np.linalg.norm(x+cube(real)@a.T, axis=1); nonzero = radii != 0
    direct = np.sum(erfc(eta*radii[nonzero])/radii[nonzero])-np.count_nonzero(~nonzero)*2*eta/np.sqrt(np.pi)
    g = cube(reciprocal, reciprocal=True)@(2*np.pi*np.linalg.inv(a)); g2 = np.sum(g*g, axis=1)
    return float(direct+4*np.pi/abs(np.linalg.det(a))*np.sum(np.exp(-g2/(4*eta**2))*np.cos(g@x)/g2)
                 -np.pi/(eta**2*abs(np.linalg.det(a))))


def shell_envelopes(args, result):
    """Independent infinite shell envelope sum, evaluated to negligible tail."""
    eta = args['eta']; m = np.arange(result.real_cube_radius+1, result.real_cube_radius+1000, dtype=float)
    # Use the actual guaranteed first distance, so conservative rounding in
    # the native metric/span cannot make this independent envelope larger.
    d = result.minimum_omitted_real_distance+result.real_metric_lower*(m-m[0])
    real = np.sum((24*m*m+2)*np.exp(-(eta*d)**2)/(np.sqrt(np.pi)*eta*d*d))
    n = np.arange(result.reciprocal_cube_radius+1, result.reciprocal_cube_radius+1000, dtype=float)
    g = result.reciprocal_metric_lower*n
    reciprocal = 4*np.pi/abs(np.linalg.det(args['cell']))*np.sum((24*n*n+2)*np.exp(-g*g/(4*eta**2))/(g*g))
    return np.array([real, reciprocal])*args['target_norm']*args['source_norm']


@pytest.mark.parametrize('skew', [False, True])
@pytest.mark.parametrize('eta', [.25, .55, .85])
@pytest.mark.parametrize('cutoff', [0, 2, 4])
def test_positive_infinite_shell_sums_and_native_kernel_identity(skew, eta, cutoff):
    args = fixture(skew=skew, eta=eta, real=cutoff, reciprocal=cutoff)
    result = manta.lattice_domain_remainder(**args)
    expected = shell_envelopes(args, result)
    assert np.all(expected <= np.array([result.real_remainder, result.reciprocal_remainder])*(1+3e-14))
    assert result.total >= result.real_remainder+result.reciprocal_remainder
    actual = native_value(args); reference = reference_value(args)
    assert abs(actual-reference)*args['target_norm']*args['source_norm'] <= result.total+5e-14
    keys = ('cell', 'target_minus_source', 'eta', 'real_indices', 'reciprocal_indices', 'near_indices')
    matrix = manta.multipole_lattice_tail(**{k: args[k] for k in keys}, order=2)
    assert result.kernel_identity == matrix.kernel_identity
    assert result.real_metric_lower <= np.linalg.svd(args['cell'], compute_uv=False)[-1]
    assert result.reciprocal_metric_lower <= np.linalg.svd(2*np.pi*np.linalg.inv(args['cell']).T, compute_uv=False)[-1]
    assert not result.roundoff_included
    assert result.qualification == 'compact-infinite-ewald-domain-remainder'


@pytest.mark.parametrize('skew', [False, True])
def test_signed_compact_energy_and_periodic_translation_with_explicit_defect_bound(skew):
    args = fixture(skew=skew, real=3, reciprocal=5, eta=.24)
    pa = np.array([[.07, -.03, .04], [-.08, .05, .03]])
    pb = np.array([[.01, .04, -.11], [.02, -.06, .04]])
    qa, qb = np.array([.9, -.4]), np.array([.5, -.3])
    row = manta.lattice_domain_remainder(**args)
    actual = exact = shifted = 0.
    shift = args['cell']@np.array([1, -1, 0]); moved = dict(args, target_minus_source=args['target_minus_source']+shift)
    other = manta.lattice_domain_remainder(**moved)
    for a, q in zip(pa, qa):
        for b, charge in zip(pb, qb):
            r = args['target_minus_source']+a-b
            actual += q*charge*native_value(args, r)
            exact += q*charge*reference_value(args, r)
            shifted += q*charge*native_value(args, r+shift)
    assert abs(actual-exact) <= row.total+5e-14
    assert abs(shifted-exact) <= other.total+5e-14
    assert abs(actual-shifted) <= row.total+other.total+5e-14
    assert abs(actual-shifted) <= row.real_remainder+other.real_remainder+5e-14
    assert abs(actual-shifted) > 1e-12  # Finite real domains are not exactly periodic.


def test_near_transfer_domain_order_and_owned_metadata():
    args = fixture(); initial = manta.lattice_domain_remainder(**args)
    changed = manta.lattice_domain_remainder(**dict(args, near_indices=cube(1)))
    assert changed.total == initial.total and changed.kernel_identity != initial.kernel_identity
    assert native_value(args) == pytest.approx(native_value(dict(args, near_indices=cube(1))), abs=2e-14)
    permuted = dict(args, real_indices=np.asfortranarray(args['real_indices'][::-1]),
                    reciprocal_indices=np.asfortranarray(args['reciprocal_indices'][::-1]))
    assert manta.lattice_domain_remainder(**permuted).source_identity == initial.source_identity
    args['cell'][:] = 0
    assert initial.total > 0
    with pytest.raises(AttributeError): initial.real_remainder = 0


def test_rotation_scale_left_handed_and_skew_metrics():
    args = fixture(); original = manta.lattice_domain_remainder(**args)
    angle = .43; rot = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., -1.]])
    rotated = manta.lattice_domain_remainder(**dict(args, cell=rot@args['cell'], target_minus_source=rot@args['target_minus_source']))
    assert rotated.total == pytest.approx(original.total, rel=5e-12)
    for scale in (1e-8, 1e8):
        scaled = manta.lattice_domain_remainder(**dict(args, cell=args['cell']*scale,
                    target_minus_source=args['target_minus_source']*scale, eta=args['eta']/scale,
                    target_radius=args['target_radius']*scale, source_radius=args['source_radius']*scale))
        assert scaled.total*scale == pytest.approx(original.total, rel=5e-12)
    args = fixture(real=7); args['cell'] = np.array([[3., 2.8, 1.8], [0., 1.3, .5], [0., 0., 2.1]])
    result = manta.lattice_domain_remainder(**args)
    assert abs(native_value(args)-reference_value(args, real=18, reciprocal=22))*1.04 <= result.total+5e-13


@pytest.mark.parametrize('field,planned', [('memory_bytes', 'modeled_peak_bytes'), ('scalar_terms', 'scalar_terms'), ('kernel_terms', 'kernel_terms')])
def test_exact_resource_caps_and_payload_free_refusal(field, planned):
    args = fixture(); p = manta.plan_lattice_domain_remainder(len(args['real_indices']), len(args['reciprocal_indices']), len(args['near_indices']))
    budget = manta.ResourceBudget(); setattr(budget, field, getattr(p, planned))
    assert manta.lattice_domain_remainder(**args, budget=budget).total > 0
    setattr(budget, field, getattr(p, planned)-1)
    args['cell'][:] = np.nan
    with pytest.raises((ValueError, RuntimeError), match='budget exceeded'):
        manta.lattice_domain_remainder(**args, budget=budget)


def test_invalid_domains_numerical_inputs_support_and_count_overflow():
    args = fixture()
    for key, value in [('eta', 0.), ('target_norm', -1.), ('source_radius', np.inf),
                       ('target_radius', 100.), ('cell', np.zeros((3, 3))),
                       ('real_indices', cube(0).astype(float)), ('near_indices', cube(3))]:
        with pytest.raises((ValueError, OverflowError)):
            manta.lattice_domain_remainder(**dict(args, **{key: value}))
    # Inversion closure alone is insufficient: complete cube membership is required.
    for key in ('real_indices', 'reciprocal_indices'):
        broken = args[key][1:-1].copy()
        with pytest.raises(ValueError, match='complete centered integer cubes'):
            manta.lattice_domain_remainder(**dict(args, **{key: broken}))
    huge = np.lib.stride_tricks.as_strided(np.zeros(3, dtype=np.int64), shape=(10**14, 3), strides=(24, 8))
    with pytest.raises((ValueError, RuntimeError, OverflowError), match='budget|overflow'):
        manta.lattice_domain_remainder(**dict(args, real_indices=huge, reciprocal_indices=huge, near_indices=huge))


def test_positive_subnormal_bounds_zero_norms_and_overflow():
    args = fixture(real=4, reciprocal=4)
    tiny = manta.lattice_domain_remainder(**dict(args, target_norm=1e-300, source_norm=1e-300))
    assert 0 < tiny.real_remainder <= tiny.total and 0 < tiny.reciprocal_remainder <= tiny.total
    zero = manta.lattice_domain_remainder(**dict(args, target_norm=0.))
    assert zero.total == zero.real_remainder == zero.reciprocal_remainder == 0.
    with pytest.raises(OverflowError):
        manta.lattice_domain_remainder(**dict(args, target_norm=1e300, source_norm=1e300))


def ionic_calculation(*, skew=True, eta=.45, cutoff=3, translated=False):
    """Neutral two-ion cell, full finite Ewald energy including regular self."""
    args = fixture(skew=skew, eta=eta, real=cutoff, reciprocal=cutoff)
    positions = np.array([[0., 0., 0.], args['cell']@np.array([.5, .5, .5])]); charges = np.array([1., -1.])
    if translated: positions[1] += args['cell']@np.array([1, 0, 0])
    values = np.empty((2, 2)); reference = np.empty((2, 2)); bounds = np.empty((2, 2)); ids = []
    for i in range(2):
        for j in range(2):
            r = positions[i]-positions[j]
            local = dict(args, target_minus_source=r, target_radius=0., source_radius=0., target_norm=1., source_norm=1.)
            bound = manta.lattice_domain_remainder(**local)
            values[i, j] = native_value(args, r); reference[i, j] = reference_value(args, r)
            bounds[i, j] = bound.total; ids.append(bound.source_identity)
    return dict(energy=float(.5*charges@values@charges), reference_energy=float(.5*charges@reference@charges),
                bound=float(.5*abs(charges)@bounds@abs(charges)), values=values, reference=reference,
                bounds=bounds, positions=positions, charges=charges, source_identities=ids)


@pytest.mark.parametrize('skew', [False, True])
@pytest.mark.parametrize('eta', [.35, .65])
def test_complete_neutral_ionic_energies_and_explicit_accuracy_target(skew, eta):
    first = ionic_calculation(skew=skew, eta=eta, cutoff=3)
    for cutoff in range(4, 13):
        row = ionic_calculation(skew=skew, eta=eta, cutoff=cutoff)
        assert np.all(abs(row['values']-row['reference']) <= row['bounds']+5e-14)
        if row['bound'] < 1e-8:
            assert abs(row['energy']-row['reference_energy']) < 1e-8
            assert row['bound'] < first['bound']
            break
    else:
        raise AssertionError('No admitted cube meets the declared 1e-8 energy target')
