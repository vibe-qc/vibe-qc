"""Native q=0 pair-image sources, polynomial quadrature and weighted adjoints."""
from math import factorial
import gc

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from tests.test_manta_fourier import _basis, _quadrature as fourier_quadrature
from tests.test_manta_pair_moments import _quadrature as moment_quadrature

CELL = np.array([[3.7, .2, -.1], [0., 4.1, .3], [0., 0., 3.9]])
IMAGES = np.array([[0, 0, 0], [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=np.int64)
K = np.array([[.12, -.08, .04], [.31, .13, -.2], [-.21, .17, .11]])
WEIGHTS = np.array([.2, .3, .5])


def options(order=4, cutoff=None, radius=None):
    out = manta.PeriodicPairOptions(); out.order = order; out.pair_cutoff_bohr = cutoff
    if radius is not None:
        out.extents.minimum_radius = out.extents.maximum_radius = radius
        out.extents.pointwise_tolerance = out.extents.tail_mass_tolerance = 1e300
    return out


def source(basis=None, order=4, cutoff=None, **kwargs):
    if basis is None: basis = _basis((0, 1))
    args = dict(basis=basis, lattice=CELL, images=IMAGES, momenta=K,
                weights=WEIGHTS, options=options(order, cutoff))
    args.update(kwargs)
    return manta.PeriodicPairSource(**args)


def density(n, seed=17):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(3, n, n))+1j*rng.normal(size=(3, n, n))
    return np.ascontiguousarray(x+x.conj().transpose(0, 2, 1))


def reference_panel(basis, entry, indices):
    a = basis.shells()[entry.shells[0]]; b = basis.shells()[entry.shells[1]]
    b.origin = (np.array(b.origin)+entry.translation).tolist()
    np.testing.assert_allclose(entry.extent.center, .5*(np.array(a.origin)+b.origin), atol=1e-15)
    return moment_quadrature(a, b, entry.extent.center, indices)


def coefficients(entry, d, k=K, weights=WEIGHTS):
    i, j = entry.offsets; ni, nj = entry.sizes
    phase = np.exp(-1j*k@np.array(entry.translation))
    c = np.einsum('k,k,kij->ij', weights, phase, d[:, i:i+ni, j:j+nj]).real
    return c*(2 if entry.paired else 1)


def reference_density(basis, s, d, k=K, weights=WEIGHTS):
    values = []; norms = []
    for p, entry in enumerate(s.entries):
        c = coefficients(entry, d, k, weights).ravel()
        values.append(c@reference_panel(basis, entry, s.moment_panel(p).indices))
        norms.append(np.sum(abs(c)))
    return np.array(values), np.array(norms)


def ordered_fourier_density(basis, d, vectors, *, k=K, weights=WEIGHTS, cell=CELL, images=IMAGES, cutoff=None):
    """Unreduced ordered image sum; no pair moments or native Fourier panels."""
    sizes = [2*s.l+1 if s.pure else (s.l+1)*(s.l+2)//2 for s in basis.shells()]
    offsets = np.cumsum([0]+sizes); rho = np.zeros(len(vectors), complex)
    for index in images:
        r = cell@index; phase = np.exp(-1j*k@r)
        for ia in range(len(sizes)):
            for ib in range(len(sizes)):
                a = basis.shells()[ia]; b = basis.shells()[ib]
                b.origin = (np.array(b.origin)+r).tolist()
                if cutoff is not None and np.linalg.norm(np.array(a.origin)-b.origin) > cutoff: continue
                block = d[:, offsets[ia]:offsets[ia+1], offsets[ib]:offsets[ib+1]]
                panel = fourier_quadrature(a, b, vectors)
                rho += np.einsum('k,k,kij,gij->g', weights, phase, block, panel)
    return rho


@pytest.mark.parametrize('pure', [False, True])
@pytest.mark.parametrize('order', [0, 4, 8])
@pytest.mark.parametrize('cutoff', [None, 0., 3.9])
def test_native_pairs_and_density_moments_against_independent_polynomial_quadrature(pure, order, cutoff):
    basis = _basis((0, 1, 2), pure); s = source(basis, order, cutoff); d = density(basis.nbasis)
    actual = s.contract(d); reference, norms = reference_density(basis, s, d)
    np.testing.assert_allclose(actual.moments, reference, atol=2e-11, rtol=3e-12)
    np.testing.assert_allclose(actual.pair_coefficient_l1, norms, atol=0., rtol=2e-13)
    assert np.all(actual.pair_coefficient_l1 >= norms*(1-2e-15))
    assert actual.receipt.charge == pytest.approx(reference[:, 0].sum(), abs=2e-12)
    exact_charge = ordered_fourier_density(basis, d, np.zeros((1, 3)), cutoff=cutoff)[0]
    assert abs(exact_charge.imag) < 2e-12
    assert actual.receipt.charge == pytest.approx(exact_charge.real, abs=2e-12)
    assert actual.receipt.absolute_charge_bound >= abs(actual.receipt.charge)
    assert s.receipt.ordered_pairs+s.receipt.skipped_ordered_pairs == len(IMAGES)*basis.nshells**2
    assert s.receipt.pairs <= s.plan.maximum_pairs
    assert s.extents.receipt.source_identity == s.receipt.source_identity
    assert s.extents.receipt.center_policy == 'canonical-periodic-pair-midpoint'
    for p, e in enumerate(s.entries):
        panel = s.moment_panel(p); expected = reference_panel(basis, e, panel.indices)
        np.testing.assert_allclose(panel.values, expected, atol=2e-11, rtol=3e-12)
        bound = np.array(e.extent.full_moment_bounds)[np.sum(panel.indices, axis=1)]
        assert np.all(np.max(abs(panel.values), axis=0) <= bound*(1+2e-12))
    assert actual.receipt.phase_products == len(s.entries)*len(K)
    assert actual.receipt.shell_calls == s.receipt.pairs
    assert actual.receipt.primitive_pairs == s.receipt.primitive_pairs
    assert 2*actual.receipt.shell_calls+s.receipt.pairs <= s.plan.shell_calls


@pytest.mark.parametrize('pure', [False, True])
@pytest.mark.parametrize('weights', [WEIGHTS, np.array([0., .4, .6])])
def test_weighted_adjoint_and_energy_gradient_without_time_reversal(pure, weights):
    basis = _basis((0, 1, 2), pure); s = source(basis, weights=weights)
    d = density(basis.nbasis); forward = s.contract(d)
    rng = np.random.default_rng(31); local = rng.normal(size=forward.moments.shape)
    reverse = s.adjoint(local)
    lhs = np.sum(forward.moments*local)
    rhs = np.einsum('k,kij,kji->', weights, d, reverse.values)
    assert abs(rhs.imag) < 5e-12
    assert lhs == pytest.approx(rhs.real, abs=5e-12, rel=3e-12)
    np.testing.assert_allclose(reverse.values, reverse.values.conj().transpose(0, 2, 1), atol=2e-12, rtol=2e-12)
    assert np.max(abs(reverse.values.imag)) > 1e-3
    # A complete quadratic density functional with a positive, fixed kernel.
    # Its weighted derivative must equal the returned AO response at every k.
    n = forward.moments.size; u = rng.normal(size=(5, n))/n
    flat = forward.moments.ravel(); potential = (u.T@(u@flat)).reshape(local.shape)
    response = s.adjoint(potential).values
    delta = density(basis.nbasis, 43); delta[1:] = 0
    def energy(x):
        vector = u@s.contract(x).moments.ravel()
        return .5*np.dot(vector, vector)
    h = 2e-5
    observed = (energy(d+h*delta)-energy(d-h*delta))/(2*h)
    predicted = np.einsum('k,kij,kji->', weights, delta, response).real
    assert observed == pytest.approx(predicted, abs=3e-11, rel=2e-9)
    assert reverse.source_identity == s.receipt.source_identity


def test_canonical_pairing_reproduces_unreduced_periodic_fourier_density():
    basis = _basis((0, 1)); s = source(basis); d = density(basis.nbasis)
    vectors = IMAGES@(2*np.pi*np.linalg.inv(CELL))
    expected = ordered_fourier_density(basis, d, vectors); actual = np.zeros(len(vectors), complex)
    for e in s.entries:
        a = basis.shells()[e.shells[0]]; b = basis.shells()[e.shells[1]]
        b.origin = (np.array(b.origin)+e.translation).tolist()
        actual += np.einsum('ij,gij->g', coefficients(e, d), fourier_quadrature(a, b, vectors))
    np.testing.assert_allclose(actual, expected, atol=4e-12, rtol=4e-12)
    # A nonreciprocal Fourier argument does not have the image equivalence.
    arbitrary = vectors+np.array([.19, -.13, .07]); canonical = np.zeros(len(vectors), complex)
    for e in s.entries:
        a = basis.shells()[e.shells[0]]; b = basis.shells()[e.shells[1]]
        b.origin = (np.array(b.origin)+e.translation).tolist()
        canonical += np.einsum('ij,gij->g', coefficients(e, d), fourier_quadrature(a, b, arbitrary))
    assert np.max(abs(canonical-ordered_fourier_density(basis, d, arbitrary))) > 1e-3
    assert s.receipt.qualification == 'finite-pair-image-moments-and-adjoint'


@pytest.mark.parametrize('pure', [False, True])
def test_high_angular_signed_contractions_and_streamed_panel(pure):
    basis = _basis((6,), pure); s = source(basis, order=4, images=IMAGES[:3])
    d = density(basis.nbasis); actual = s.contract(d); expected, _ = reference_density(basis, s, d)
    np.testing.assert_allclose(actual.moments, expected, atol=3e-9, rtol=2e-11)
    assert s.receipt.pairs == 2


def test_domain_permutation_owned_snapshots_and_lifetime():
    basis = _basis((0, 1)); d = density(basis.nbasis)
    a, r, k, w = CELL.copy(), IMAGES.copy(), K.copy(), WEIGHTS.copy(); opts = options()
    s = source(basis, lattice=a, images=r, momenta=k, weights=w, options=opts)
    initial = s.contract(d); local = np.ones(initial.moments.shape); reverse = s.adjoint(local)
    permuted = source(basis, images=np.asfortranarray(IMAGES[::-1]))
    assert permuted.receipt.source_identity == s.receipt.source_identity
    np.testing.assert_array_equal(permuted.contract(d).moments, initial.moments)
    a[:] = 0; r[:] = 99; k[:] = 44; w[:] = 0; opts.order = 0
    np.testing.assert_array_equal(s.contract(np.asfortranarray(d)).moments, initial.moments)
    np.testing.assert_array_equal(s.adjoint(np.asfortranarray(local)).values, reverse.values)
    saved = initial.moments.copy(); initial.moments[:] = 9; initial.pair_coefficient_l1[:] = 7
    np.testing.assert_array_equal(initial.moments, saved)
    receipt = s.receipt; entry = s.entries[0]; extent = entry.extent; del entry, s; gc.collect()
    assert len(receipt.source_identity) == 64 and extent.radius > 0
    with pytest.raises(AttributeError): extent.radius = 0


def test_covariance_k_weights_density_identity_and_zero_source():
    basis = _basis((0, 1)); s = source(basis); d = density(basis.nbasis); expected = s.contract(d)
    permutation = [2, 0, 1]
    reordered = source(basis, momenta=K[permutation], weights=WEIGHTS[permutation])
    np.testing.assert_allclose(reordered.contract(d[permutation]).moments, expected.moments, atol=1e-13)
    tr = source(basis, momenta=-K)
    np.testing.assert_array_equal(tr.contract(d.conj()).moments, expected.moments)
    reciprocal = 2*np.pi*np.linalg.inv(CELL)[0]
    shifted_k = source(basis, momenta=K+reciprocal)
    np.testing.assert_allclose(shifted_k.contract(d).moments, expected.moments, atol=1e-13)
    shifted = source(_basis((0, 1), shift=(.3, -.2, .4)))
    np.testing.assert_allclose(shifted.contract(d).moments, expected.moments, atol=5e-13)
    assert len({x.receipt.source_identity for x in (s, tr, shifted_k, shifted, reordered)}) == 5
    changed = s.contract(2*d)
    assert changed.receipt.density_identity != expected.receipt.density_identity
    np.testing.assert_allclose(changed.moments, 2*expected.moments, atol=1e-13)
    zero = s.contract(np.zeros_like(d))
    np.testing.assert_array_equal(zero.moments, 0.)
    assert zero.receipt.coefficient_l1 == zero.receipt.absolute_charge_bound == zero.receipt.charge == 0.
    np.testing.assert_array_equal(s.adjoint(np.zeros_like(zero.moments)).values, 0.)


@pytest.mark.parametrize('field, planned', [('memory_bytes', 'modeled_peak_bytes'), ('shell_calls', 'shell_calls'),
    ('primitive_quartets', 'primitive_pairs'), ('scalar_terms', 'scalar_terms'), ('kernel_terms', 'kernel_terms')])
def test_all_joint_resource_caps_and_refusal_before_payload_read(field, planned):
    basis = _basis((0, 1)); opts = options(); p = manta.plan_periodic_pairs(basis, 5, 3, opts)
    budget = manta.ResourceBudget(); setattr(budget, field, getattr(p, planned))
    s = source(basis, options=opts, budget=budget)
    forward = s.contract(density(basis.nbasis)); s.adjoint(np.zeros_like(forward.moments))
    assert getattr(s.plan, planned) == getattr(p, planned)
    setattr(budget, field, getattr(p, planned)-1)
    with pytest.raises(ValueError, match='budget'):
        source(basis, lattice=np.full((3, 3), np.nan), options=opts, budget=budget)


def test_invalid_shapes_domains_numerical_inputs_and_early_huge_count_refusal():
    basis = _basis((0,)); s = source(basis); d = density(basis.nbasis)
    for bad in (np.zeros((5, 3), np.int64), np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], np.int64),
                np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]], np.int64)):
        with pytest.raises(ValueError, match='unique|closed|origin'): source(basis, images=bad)
    huge = np.lib.stride_tricks.as_strided(np.zeros(3, np.int64), shape=(10**9+1, 3), strides=(24, 8))
    with pytest.raises(ValueError, match='budget'): source(basis, images=huge)
    for kwargs in (dict(images=IMAGES.astype(np.int32)), dict(lattice=CELL.astype(np.float32)),
                   dict(momenta=K.tolist()), dict(weights=np.array([1., 0., 1.])),
                   dict(lattice=np.zeros((3, 3))), dict(momenta=np.full((3, 3), np.inf))):
        with pytest.raises((TypeError, ValueError)): source(basis, **kwargs)
    for bad in (d.real.copy(), np.zeros((2, 1, 1), complex), np.full_like(d, np.nan)):
        with pytest.raises((ValueError, TypeError)): s.contract(bad)
    nonhermitian = d.copy(); nonhermitian[0, 0, 0] += 1j
    with pytest.raises(ValueError, match='Hermitian'): s.contract(nonhermitian)
    shape = (s.receipt.pairs, s.plan.moments)
    for bad in (np.zeros(shape, np.float32), np.zeros((1, 1)), np.full(shape, np.inf)):
        with pytest.raises(ValueError): s.adjoint(bad)
    with pytest.raises(IndexError): s.moment_panel(s.receipt.pairs)


def test_periodic_inventory_does_not_weaken_the_molecular_tree_contract():
    s = source()
    with pytest.raises(ValueError, match='canonical shell-pair inventory'):
        manta.plan_bubbles(s.extents)


def test_subnormal_weighted_density_and_unaligned_input_buffers():
    basis = _basis((0,)); s = source(basis, images=IMAGES[:1], order=0)
    tiny = np.nextafter(0., 1.)
    d = np.full((3, 1, 1), tiny, complex)
    forward = s.contract(d)
    assert forward.pair_coefficient_l1[0] >= tiny
    assert forward.receipt.absolute_charge_bound > 0
    expected = float(np.longdouble(tiny)*np.longdouble(s.moment_panel(0).values[0, 0]))
    assert forward.moments[0, 0] == expected
    def unaligned(array):
        out = np.ndarray(array.shape, array.dtype, bytearray(array.nbytes+1), offset=1)
        out[:] = array
        return out
    normal = source(basis)
    unusual = source(basis, lattice=unaligned(CELL), images=unaligned(IMAGES),
                     momenta=unaligned(K), weights=unaligned(WEIGHTS))
    d = density(basis.nbasis); a = normal.contract(d); b = unusual.contract(unaligned(d))
    np.testing.assert_array_equal(a.moments, b.moments)
    local = np.ones_like(a.moments)
    np.testing.assert_array_equal(normal.adjoint(local).values, unusual.adjoint(unaligned(local)).values)


def finite_fourier_calculation(order=8, *, pure=True, skew=True, seed=17):
    """Complete finite Fourier Coulomb energy through the new paired source.

    Taylor density approximation is explicitly distinct from source pairing.
    Independent ordered Gaussian Fourier integrals supply the exact reference.
    """
    basis = _basis((0, 1), pure); cell = CELL if skew else np.eye(3)*3.9
    d = density(basis.nbasis, seed); s = source(basis, order, lattice=cell)
    forward = s.contract(d); indices = np.array(s.moment_panel(0).indices)
    vectors = IMAGES[1:]@(2*np.pi*np.linalg.inv(cell))
    factors = 4*np.pi/np.sum(vectors*vectors, axis=1)/abs(np.linalg.det(cell))
    panels = []
    for entry in s.entries:
        p = np.ones((len(vectors), len(indices)), complex)
        for axis in range(3):
            p *= (-1j*vectors[:, axis, None])**indices[None, :, axis]
            p /= np.array([factorial(int(i)) for i in indices[:, axis]])
        p *= np.exp(-1j*vectors@np.array(entry.extent.center))[:, None]
        panels.append(p)
    panels = np.array(panels)
    rho = np.einsum('pm,pgm->g', forward.moments, panels)
    energy = float(.5*np.dot(factors, abs(rho)**2))
    local = np.einsum('g,g,pgm->pm', factors, rho.conj(), panels).real
    j = s.adjoint(np.ascontiguousarray(local)).values
    trace_energy = float(.5*np.einsum('k,kij,kji->', WEIGHTS, d, j).real)
    reference_rho = ordered_fourier_density(basis, d, vectors, cell=cell)
    reference_energy = float(.5*np.dot(factors, abs(reference_rho)**2))
    # Independent moments also predict the same finite Taylor calculation.
    reference_moments, _ = reference_density(basis, s, d)
    same_order_rho = np.einsum('pm,pgm->g', reference_moments, panels)
    # The source stores absolute moments through p. Use the p-1 exponential
    # remainder, then account for the explicitly retained degree-p term:
    # |rho - T_p| <= sum_pairs ||c||_1 F_p |G|^p/p! + |T_p - T_(p-1)|.
    # This includes unbounded Gaussian support without a guessed energy target.
    degree_p = np.sum(indices, axis=1) == order
    last_term = np.einsum('pm,pgm->g', forward.moments[:, degree_p], panels[:, :, degree_p])
    absolute_moment = sum(norm*entry.extent.full_moment_bounds[order]
                          for norm, entry in zip(forward.pair_coefficient_l1, s.entries))
    rho_bound = absolute_moment*np.linalg.norm(vectors, axis=1)**order/factorial(order)+abs(last_term)
    energy_bound = float(np.dot(factors, abs(rho)*rho_bound+.5*rho_bound**2))
    return dict(energy=energy, trace_energy=trace_energy, reference_energy=reference_energy,
                absolute_error=abs(energy-reference_energy), same_order_error=float(np.max(abs(rho-same_order_rho))),
                rho=rho, reference_rho=reference_rho, rho_bound=rho_bound, energy_bound=energy_bound,
                j=j, density=d, moments=forward.moments,
                charge=forward.receipt.charge, source_identity=s.receipt.source_identity,
                density_identity=forward.receipt.density_identity, pairs=s.receipt.pairs, plan=s.plan)


@pytest.mark.parametrize('pure', [False, True])
@pytest.mark.parametrize('skew', [False, True])
def test_complete_finite_coulomb_energy_and_taylor_convergence(pure, skew):
    rows = [finite_fourier_calculation(order, pure=pure, skew=skew) for order in (0, 6, 12)]
    for r in rows:
        assert r['energy'] == pytest.approx(r['trace_energy'], rel=3e-12, abs=3e-12)
        assert r['same_order_error'] < 5e-12
        assert np.all(abs(r['rho']-r['reference_rho']) <= r['rho_bound']+5e-12)
        assert r['absolute_error'] <= r['energy_bound']+5e-12
    assert rows[-1]['absolute_error'] < rows[0]['absolute_error']*1e-4
    assert rows[-1]['energy_bound'] < rows[0]['energy_bound']*1e-3
