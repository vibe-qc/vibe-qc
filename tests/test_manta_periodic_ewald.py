"""Complete finite Ewald J with independent Gaussian integrals and derivatives."""
import gc
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import erf, erfc

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from tests.test_manta_periodic_pairs import CELL, IMAGES, K, WEIGHTS, source, density, coefficients, _basis, ordered_fourier_density
from tests.test_manta_lattice_tail import cube, fixture
from tests.test_manta_lattice_gaussian import components, fourier, gaussian_integral


def make_basis(angular=(0, 0), pure=True, contracted=True, shift=(0., 0., 0.)):
    b = _basis(angular, pure, shift); shells = b.shells()
    if not contracted:
        for i, s in enumerate(shells):
            s.exponents = [s.exponents[0]]; s.coefficients = [(-1 if i else 1)*.8]
    mol = vq.Molecule([vq.Atom(2, s.origin) for s in shells], 0, 1)
    return vq.BasisSet(mol, shells, 'periodic-ewald-reference', True)


def build(s, d, *, eta=.63, real=None, reciprocal=None, batch=7, budget=None):
    r = cube(1) if real is None else real
    g = cube(2, reciprocal=True) if reciprocal is None else reciprocal
    args = {} if budget is None else dict(budget=budget)
    return manta.PeriodicEwaldSource(s, eta, r, g, batch, **args).build(d)


def reference(basis, s, d, cell, eta, real, reciprocal, *, quadrature=64, k=K, weights=WEIGHTS):
    """Independent Cartesian polynomial/Gaussian convolution of every component.

    Test-only dense matrices use a Gaussian characteristic function and
    bivariate Gauss-Hermite/Laplace quadrature, not native integral tiles,
    moments, Taylor coefficients or the implementation's unordered digestion.
    """
    terms = []; transforms = []; slices = []; offset = 0
    for e in s.entries:
        a, b = [basis.shells()[i] for i in e.shells]
        b.origin = (np.array(b.origin)+e.translation).tolist()
        pair = SimpleNamespace(shells=lambda a=a, b=b: [a, b])
        cart = components(dict(kind='pair', basis=pair, extent=SimpleNamespace(shells=(0, 1))))
        terms.extend(cart)
        ta = manta.cartesian_to_spherical(a.l) if a.pure else np.eye((a.l+1)*(a.l+2)//2)
        tb = manta.cartesian_to_spherical(b.l) if b.pure else np.eye((b.l+1)*(b.l+2)//2)
        transforms.append(np.kron(ta, tb)); slices.append(slice(offset, offset+len(cart))); offset += len(cart)
    rows = sum(t.shape[0] for t in transforms); transform = np.zeros((rows, len(terms))); offset = 0
    for t, sl in zip(transforms, slices):
        transform[offset:offset+len(t), sl] = t; offset += len(t)
    rs = real@cell.T
    nodes, weights_q = np.polynomial.legendre.leggauss(quadrature)
    t = (nodes+1)/2; u = eta+t/(1-t); w = weights_q/2/(1-t)**2
    real_matrix = np.zeros((len(terms), len(terms)))
    for i, aa in enumerate(terms):
        for j in range(i, len(terms)):
            value = 0.
            for a, b in product(aa, terms[j]):
                p, ca, wa, pa = a; q, cb, wb, pb = b
                if all(len(poly) == 1 for poly in pa+pb):
                    beta = 1/(1/p+1/q); zeta = np.sqrt(1/(1/p+1/q+1/eta**2))
                    radii = np.linalg.norm(ca-cb+rs, axis=1); nonzero = radii != 0
                    v = np.empty(len(radii))
                    v[nonzero] = (erfc(zeta*radii[nonzero])-erfc(np.sqrt(beta)*radii[nonzero]))/radii[nonzero]
                    # Product centers can differ by a few ulps even for the
                    # same shell. erf avoids erfc(x)-erfc(y) cancellation at
                    # that removable origin; erfc keeps distant tails.
                    small = nonzero & (radii < 1.)
                    v[small] = (erf(np.sqrt(beta)*radii[small])-erf(zeta*radii[small]))/radii[small]
                    v[~nonzero] = 2/np.sqrt(np.pi)*(np.sqrt(beta)-zeta)
                    value += wa*wb*(np.pi/p)**1.5*(np.pi/q)**1.5*np.prod([x[0] for x in pa+pb])*sum(v)
                else:
                    for r in rs:
                        value += 2/np.sqrt(np.pi)*np.dot(w, gaussian_integral(a, b, r, u))
            real_matrix[i, j] = real_matrix[j, i] = value
    gs = 2*np.pi*np.linalg.solve(cell.T, reciprocal.T).T; g2 = np.sum(gs*gs, axis=1)
    f = np.array([fourier(a, gs) for a in terms])
    q = np.array([fourier(a, np.zeros((1, 3)))[0].real for a in terms])
    volume = abs(np.linalg.det(cell)); lam = 4*np.pi/volume*np.exp(-g2/(4*eta**2))/g2
    reciprocal_matrix = ((f*lam)@f.conj().T).real
    gauge_matrix = -np.pi/(eta**2*volume)*q[:, None]*q[None, :]
    matrices = [transform@m@transform.T for m in (real_matrix, reciprocal_matrix, gauge_matrix)]
    c = np.concatenate([coefficients(e, d, k, weights).ravel() for e in s.entries])
    v = sum(matrices)@c; energies = np.array([.5*c@m@c for m in matrices])
    j = np.zeros_like(d); offset = 0
    for e in s.entries:
        i, l = e.offsets; ni, nl = e.sizes
        value = v[offset:offset+ni*nl].reshape(ni, nl); offset += ni*nl
        phase = np.exp(1j*k@np.array(e.translation))
        j[:, i:i+ni, l:l+nl] += phase[:, None, None]*value
        if e.paired: j[:, l:l+nl, i:i+ni] += phase.conj()[:, None, None]*value.T
    return dict(values=j, energies=energies, energy=sum(energies), charge=float(c@(transform@q)),
                coefficients=c, component_matrices=np.array(matrices), potential=v)


def finite_ewald_calculation(*, pure=True, skew=True, eta=.63, angular=False, seed=17,
                             quadrature=64, real_radius=1, reciprocal_radius=2, perturbation=0.):
    basis = make_basis((0, 1) if angular else (0, 0), pure, contracted=not angular)
    cell = CELL if skew else np.eye(3)*3.9
    images = IMAGES[:3].copy() if angular else IMAGES
    real = IMAGES[:3].copy() if angular else cube(real_radius)
    reciprocal = cube(reciprocal_radius, reciprocal=True)
    s = source(basis, order=0, lattice=cell, images=images)
    d = density(basis.nbasis, seed)+perturbation*density(basis.nbasis, 83)
    actual = build(s, d, eta=eta, real=real, reciprocal=reciprocal)
    ref = reference(basis, s, d, cell, eta, real, reciprocal, quadrature=quadrature)
    energies = np.array([actual.receipt.real_energy, actual.receipt.reciprocal_energy, actual.receipt.gauge_energy])
    return dict(values=actual.values, reference_values=ref['values'], energies=energies,
                reference_energies=ref['energies'], energy=actual.receipt.energy,
                reference_energy=ref['energy'], trace_energy=actual.receipt.trace_energy,
                matrix_error=float(np.max(abs(actual.values-ref['values']))),
                energy_error=abs(actual.receipt.energy-ref['energy']), charge=actual.receipt.charge,
                reference_charge=ref['charge'], source_identity=actual.receipt.source_identity,
                density_identity=actual.receipt.density_identity, d=d, weights=WEIGHTS,
                component_matrices=ref['component_matrices'], coefficients=ref['coefficients'],
                real_indices=real, reciprocal_indices=reciprocal, cell=cell)


@pytest.mark.parametrize('skew', [False, True])
@pytest.mark.parametrize('eta', [.35, .8])
def test_signed_contracted_s_gaussian_complete_j_and_component_energies(skew, eta):
    row = finite_ewald_calculation(skew=skew, eta=eta)
    np.testing.assert_allclose(row['values'], row['reference_values'], atol=2e-12, rtol=3e-12)
    np.testing.assert_allclose(row['energies'], row['reference_energies'], atol=2e-12, rtol=3e-12)
    assert row['energy'] == pytest.approx(row['trace_energy'], abs=2e-12)
    assert row['charge'] == pytest.approx(row['reference_charge'], abs=2e-12)
    assert np.max(abs(row['values'].imag)) > 1e-6


@pytest.mark.parametrize('pure', [False, True])
def test_angular_polynomial_integrals_and_quadrature_refinement(pure):
    row = finite_ewald_calculation(pure=pure, angular=True, quadrature=48)
    refined = finite_ewald_calculation(pure=pure, angular=True, quadrature=80)
    np.testing.assert_allclose(row['reference_values'], refined['reference_values'], atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(row['values'], refined['reference_values'], atol=2e-12, rtol=3e-12)
    np.testing.assert_allclose(row['energies'], refined['reference_energies'], atol=2e-12, rtol=3e-12)


@pytest.mark.parametrize('pure', [False, True])
def test_d_shell_all_components_and_signed_angular_contractions(pure):
    for angular, contracted in [((2,), False), ((0, 1), True)]:
        b = make_basis(angular, pure, contracted=contracted); s = source(b, order=0, images=cube(0))
        d = density(b.nbasis); actual = build(s, d, real=IMAGES[:3], reciprocal=cube(1, reciprocal=True))
        ref = reference(b, s, d, CELL, .63, IMAGES[:3], cube(1, reciprocal=True))
        np.testing.assert_allclose(actual.values, ref['values'], atol=3e-12, rtol=3e-12)
        assert actual.receipt.energy == pytest.approx(ref['energy'], abs=3e-12)


@pytest.mark.parametrize('weights', [WEIGHTS, np.array([0., .4, .6])])
def test_fixed_quadratic_whole_mesh_derivative_and_reciprocal_ordered_density(weights):
    b = make_basis((0, 1)); s = source(b, order=4, weights=weights); d = density(b.nbasis); h = density(b.nbasis, 83)
    engine = manta.PeriodicEwaldSource(s, .63, IMAGES, cube(1, reciprocal=True), 3)
    result = engine.build(d); response = engine.build(h)
    expected = np.einsum('k,kij,kji->', weights, h, result.values).real
    for step in (1e-4, 5e-5):
        plus, minus = engine.build(d+step*h), engine.build(d-step*h)
        assert (plus.receipt.energy-minus.receipt.energy)/(2*step) == pytest.approx(expected, abs=2e-8)
        np.testing.assert_allclose(plus.values, result.values+step*response.values, atol=5e-13)
    gs = cube(1, reciprocal=True)@(2*np.pi*np.linalg.inv(CELL))
    rho = ordered_fourier_density(b, d, gs, weights=weights)
    lam = 4*np.pi/abs(np.linalg.det(CELL))*np.exp(-np.sum(gs*gs, axis=1)/(4*.63**2))/np.sum(gs*gs, axis=1)
    assert result.receipt.reciprocal_energy == pytest.approx(.5*np.dot(lam, abs(rho)**2), abs=2e-12)
    f = s.contract(d)
    assert result.receipt.charge == pytest.approx(f.receipt.charge, abs=2e-12)
    assert result.receipt.density_identity == f.receipt.density_identity
    assert result.receipt.pair_source_identity == manta.PeriodicBubbleTree(s).pair_source_identity
    assert result.receipt.energy == pytest.approx(result.receipt.trace_energy, abs=2e-12)
    zero = engine.build(np.zeros_like(d)); np.testing.assert_array_equal(zero.values, 0.)
    assert zero.receipt.energy == 0
    h[0] = h[2] = 0; assert np.linalg.norm(engine.build(h).values[0]) > 1e-3


def test_same_center_gaussian_self_is_finite_and_gauge_is_included_once():
    b = make_basis((0,), contracted=False); s = source(b, images=cube(0), order=0)
    d = np.ones((3, 1, 1), complex); result = build(s, d, real=cube(0), reciprocal=cube(0, reciprocal=True))
    ref = reference(b, s, d, CELL, .63, cube(0), cube(0, reciprocal=True))
    assert result.receipt.real_energy > 0
    assert result.receipt.reciprocal_energy == 0
    assert result.receipt.gauge_energy == pytest.approx(.5*result.receipt.gauge_constant*result.receipt.charge**2)
    np.testing.assert_allclose(result.values, ref['values'], atol=5e-14)
    assert result.receipt.qualification == 'finite-canonical-pair-ewald-j'


def test_finite_kernel_converges_to_eta_independent_reciprocal_limit():
    b = make_basis(); s = source(b, order=0); d = density(b.nbasis)
    gs = cube(7, reciprocal=True)@(2*np.pi*np.linalg.inv(CELL))
    rho = ordered_fourier_density(b, d, gs)
    reference_energy = .5*np.dot(4*np.pi/abs(np.linalg.det(CELL))/np.sum(gs*gs, axis=1), abs(rho)**2)
    for eta in (.45, .85):
        result = build(s, d, eta=eta, real=cube(3), reciprocal=cube(6, reciprocal=True))
        assert result.receipt.energy == pytest.approx(reference_energy, abs=3e-10)


def test_batching_identity_domain_reordering_source_ownership_and_origin_covariance():
    b = make_basis(); s = source(b, order=0); d = density(b.nbasis)
    expected = build(s, d, real=IMAGES)
    for batch in (1, 17, 1000):
        actual = build(s, np.asfortranarray(d), real=np.asfortranarray(IMAGES[::-1]), reciprocal=cube(2, reciprocal=True)[::-1].copy(), batch=batch)
        np.testing.assert_allclose(actual.values, expected.values, atol=4e-13)
        assert actual.receipt.source_identity == expected.receipt.source_identity
    r = expected.receipt
    tail = manta.multipole_lattice_tail(**(fixture(0)|dict(cell=CELL, eta=.63, real_indices=IMAGES, reciprocal_indices=cube(2, reciprocal=True))))
    assert r.real_domain_identity == tail.real_domain_identity and r.reciprocal_domain_identity == tail.reciprocal_domain_identity
    for other in (source(make_basis(shift=(.3, -.2, .1)), order=0), source(b, order=12)):
        np.testing.assert_allclose(build(other, d, real=IMAGES).values, expected.values, atol=5e-13)
    tr = build(source(b, order=0, momenta=-K), d.conj(), real=IMAGES)
    np.testing.assert_allclose(tr.values, expected.values.conj(), atol=3e-13)
    relabel = build(source(b, order=0, momenta=K+2*np.pi*np.linalg.inv(CELL)[0]), d, real=IMAGES)
    np.testing.assert_allclose(relabel.values, expected.values, atol=4e-13)
    engine = manta.PeriodicEwaldSource(s, .63, IMAGES, cube(2, reciprocal=True)); del b, s; gc.collect()
    actual = engine.build(d); export = actual.values; export[:] = 123
    np.testing.assert_allclose(actual.values, expected.values, atol=4e-13)
    assert engine.source_identity == expected.receipt.source_identity
    assert build(source(make_basis(), order=0), d, eta=.8).receipt.kernel_identity != r.kernel_identity
    for cls in (manta.PeriodicEwaldPlan, manta.PeriodicEwaldReceipt, manta.PeriodicEwaldResult):
        with pytest.raises(TypeError): cls()


@pytest.mark.parametrize('field,planned', [('shell_calls','shell_calls'), ('primitive_quartets','primitive_products'),
    ('scalar_terms','scalar_terms'), ('kernel_terms','kernel_terms'), ('memory_bytes','modeled_peak_bytes')])
def test_exact_joint_caps_and_refusal_before_domain_payload_read(field, planned):
    s = source(make_basis(), order=0); p = manta.plan_periodic_ewald(s, 5, 26, 7)
    b = manta.ResourceBudget(); setattr(b, field, getattr(p, planned))
    result = build(s, density(s.plan.n_basis), real=IMAGES, reciprocal=cube(1, reciprocal=True), budget=b)
    r = result.receipt
    assert r.real_shell_calls == p.real_shell_calls
    assert r.fourier_shell_calls == p.fourier_shell_calls
    assert r.overlap_shell_calls == p.overlap_shell_calls
    assert r.primitive_products <= p.primitive_products and r.modeled_peak_bytes == p.modeled_peak_bytes
    setattr(b, field, getattr(p, planned)-1); bad = IMAGES.copy(); bad[0] = 2**50
    with pytest.raises((ValueError, RuntimeError), match='budget'):
        manta.PeriodicEwaldSource(s, .63, bad, cube(1, reciprocal=True), 7, b)


@pytest.mark.parametrize('change,message', [
    (dict(eta=0.), 'eta'), (dict(eta=np.nan), 'eta'), (dict(eta=np.inf), 'eta'),
    (dict(real_indices=IMAGES[:2]), 'count'), (dict(real_indices=IMAGES+1), 'origin'),
    (dict(real_indices=np.zeros((3, 3), np.int64)), 'unique'),
    (dict(real_indices=np.array([[0,0,0],[1,0,0],[0,1,0]], np.int64)), 'inversion'),
    (dict(real_indices=IMAGES*2**35), 'range'),
    (dict(reciprocal_indices=IMAGES), 'count'), (dict(reciprocal_indices=IMAGES[:2]), 'zero'),
    (dict(reciprocal_batch=0), 'batch'), (dict(real_indices=IMAGES.astype(float)), 'dtype'),
])
def test_invalid_domains_refused(change, message):
    args = dict(source=source(make_basis(), order=0), eta=.63, real_indices=IMAGES, reciprocal_indices=cube(1, reciprocal=True))
    with pytest.raises((ValueError, RuntimeError), match=message): manta.PeriodicEwaldSource(**(args|change))


def test_density_validation_unaligned_buffers_and_small_density_scaling():
    s = source(make_basis(), order=0); d = density(s.plan.n_basis)
    e = manta.PeriodicEwaldSource(s, .63, cube(0), cube(1, reciprocal=True))
    expected = e.build(d)
    unaligned = np.ndarray(d.shape, dtype=d.dtype, buffer=bytearray(d.nbytes+1), offset=1); unaligned[:] = d
    np.testing.assert_array_equal(e.build(unaligned).values, expected.values)
    small = e.build(d*1e-280)
    np.testing.assert_allclose(small.values/1e-280, expected.values, atol=3e-13)
    for bad, message in [(d.astype(np.complex64),'dtype'), (d[:1].copy(),'shape'), (d.copy(),'Hermitian'), (d.copy(),'finite')]:
        if message == 'Hermitian': bad[0, 0, 1] += .1j
        if message == 'finite': bad[0, 0, 0] = np.nan
        with pytest.raises(ValueError, match=message): e.build(bad)
