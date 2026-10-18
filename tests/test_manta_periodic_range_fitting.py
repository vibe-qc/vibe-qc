"""Physical range fitting: independent finite kernels and streamed RI channels."""
import numpy as np
import pytest

from vibeqc import manta
from tests.test_manta_periodic_gdf import case, gaussian_fourier
from tests.test_manta_periodic_fitting import libint_reference, CAPS
from tests.test_manta_periodic_gdf_tiles import assemble
from tests.test_manta_periodic_ri_q import builder, densities, reference as ri_reference


def physical(kernel, omega):
    p = manta.PeriodicGDFKernelOptions(); p.kernel = kernel; p.omega = omega
    return p


def fourier_blocks(a, vectors, weights):
    # These witnesses have one primitive s function per shell. Retain each
    # supplied coefficient, including normalized SCF basis coefficients.
    aux, orb = a['auxiliary'].shells(), a['orbital'].shells()
    assert all(s.l == 0 and len(s.exponents) == 1 for s in aux+orb)
    aft, pair = gaussian_fourier(a, vectors)
    aft *= np.array([s.coefficients[0] for s in aux])[:, None]
    c = np.array([s.coefficients[0] for s in orb])
    pair *= c[None, :, None, None]*c[None, None, :, None]
    return (aft.conj()*weights)@aft.T, np.einsum('pg,kmng,g->kpmn', aft.conj(), pair, weights)


def reference(a):
    """Libint raw erfc plus independent Gaussian-product Fourier integration."""
    p = a['physical_kernel']; o = a['options']; v = a['vectors']
    squared = np.sum(v*v, axis=1); keep = squared > 1e-24
    volume = abs(np.linalg.det(a['system'].lattice))
    w = 4*np.pi/volume/squared[keep]*np.exp(-squared[keep]/(4*p.omega**2))
    if p.kernel == 'erfc':
        w = 4*np.pi/volume/squared[keep]*np.exp(-squared[keep]/(4*o.omega**2))-w
    m, t = fourier_blocks(a, v[keep], w)
    if p.kernel == 'erfc':
        sm, st = libint_reference(a); m += sm; t += st
        frac = a['system'].lattice.T@a['q']/(2*np.pi)
        if np.linalg.norm(frac-np.round(frac)) < 1e-12:
            zm, zt = fourier_blocks(a, np.zeros((1, 3)), np.array([np.pi/volume*(1/p.omega**2-1/o.omega**2)]))
            m += zm; t += zt
    return m, t


@pytest.mark.parametrize('kernel', ['erfc', 'erf'])
@pytest.mark.parametrize('omega', [.35, .65, .9])
@pytest.mark.parametrize('gamma', [False, True])
def test_dense_tiles_and_fixed_projector_match_independent_physical_source(kernel, omega, gamma):
    a = case(gamma); a['physical_kernel'] = physical(kernel, omega)
    r = manta.compute_periodic_gdf(**a); source = manta.PeriodicGDFTiles(**a); m, t = reference(a)
    np.testing.assert_allclose(r.metric, m, atol=3e-12, rtol=3e-13)
    np.testing.assert_allclose(r.three_center, t, atol=3e-12, rtol=3e-13)
    np.testing.assert_array_equal(source.metric().values, r.metric)
    np.testing.assert_array_equal(assemble(source), r.three_center)
    assert source.source_identity == r.receipt.source_identity
    assert r.receipt.kernel == kernel and r.receipt.physical_omega == omega
    assert r.receipt.zero_transfer == gamma
    assert r.receipt.physical_zero_included == (gamma and kernel == 'erfc')
    assert r.receipt.finite_zero_removed == (gamma and kernel == 'erfc')
    assert r.receipt.real_short_range == (kernel == 'erfc')
    if kernel == 'erf':
        assert r.receipt.short_range.metric_shell_calls == r.receipt.short_range.three_center_shell_calls == 0
        assert r.receipt.short_range.kernel == 'none'
    d = densities(); w = np.array([.23, .77]); b = builder(a)
    fitted = b.build(d, w, make_j=gamma)
    j, k, energy, projector, _ = ri_reference(m, t, d, w)
    np.testing.assert_allclose(fitted.K, k, atol=5e-12, rtol=3e-12)
    np.testing.assert_allclose(b.projector, projector, atol=5e-13, rtol=0)
    assert fitted.receipt.kernel == kernel and b.metric_info.zero_transfer == gamma
    assert fitted.receipt.physical_omega == omega and fitted.receipt.zero_transfer == gamma
    assert fitted.receipt.physical_zero_included == (gamma and kernel == 'erfc')
    if gamma:
        np.testing.assert_allclose(fitted.J, j, atol=5e-12, rtol=3e-12)
        assert fitted.receipt.coulomb_energy == pytest.approx(energy, abs=5e-12)


@pytest.mark.parametrize('gamma', [False, True])
def test_complement_zero_ownership_and_numerical_split_limits(gamma):
    a = case(gamma); omega = .43
    full = manta.compute_periodic_gdf(**a)
    sr = manta.compute_periodic_gdf(**dict(a, physical_kernel=physical('erfc', omega)))
    lr = manta.compute_periodic_gdf(**dict(a, physical_kernel=physical('erf', omega)))
    zm, zt = fourier_blocks(a, np.zeros((1, 3)), np.array([np.pi/abs(np.linalg.det(a['system'].lattice))/omega**2 if gamma else 0.]))
    np.testing.assert_allclose(sr.metric+lr.metric, full.metric+zm, atol=4e-12, rtol=2e-13)
    np.testing.assert_allclose(sr.three_center+lr.three_center, full.three_center+zt, atol=4e-12, rtol=2e-13)
    if gamma:
        omitted = dict(a, vectors=a['vectors'][np.linalg.norm(a['vectors'], axis=1)>1e-12])
        other = manta.compute_periodic_gdf(**dict(omitted, physical_kernel=physical('erfc', omega)))
        np.testing.assert_allclose(sr.metric, other.metric, atol=3e-13, rtol=0)
        np.testing.assert_allclose(sr.three_center, other.three_center, atol=3e-13, rtol=0)
    raw = manta.compute_periodic_gdf_short_range(**{k: v for k, v in a.items() if k != 'vectors'})
    same = manta.compute_periodic_gdf(**dict(a, physical_kernel=physical('erfc', a['options'].omega)))
    np.testing.assert_array_equal(same.metric, raw.metric)
    np.testing.assert_array_equal(same.three_center, raw.three_center)
    a['options'].omega = 1.1; a['options'].auxiliary_cutoff_bohr = 7.; a['options'].integral_screen_error = 1e-4
    changed = manta.compute_periodic_gdf(**dict(a, physical_kernel=physical('erf', omega)))
    np.testing.assert_array_equal(changed.metric, lr.metric)
    np.testing.assert_array_equal(changed.three_center, lr.three_center)
    assert changed.receipt.source_identity == lr.receipt.source_identity


@pytest.mark.parametrize('kernel', ['erfc', 'erf'])
def test_source_identity_frame_density_and_owned_physical_options(kernel):
    a = case(True); a['physical_kernel'] = physical(kernel, .43)
    source = manta.PeriodicGDFTiles(**a); expected = source.metric().values; d = densities(); w = np.array([.5, .5])
    b = builder(a); r = b.build(d, w)
    a['physical_kernel'].omega = .6
    other = builder(a).build(d, w)
    assert r.receipt.source_identity != other.receipt.source_identity
    assert r.receipt.density_identity == other.receipt.density_identity
    assert r.receipt.source_frame_identity == other.receipt.source_frame_identity
    np.testing.assert_array_equal(source.metric().values, expected)
    a['physical_kernel'].kernel = 'invalid'
    np.testing.assert_array_equal(source.metric().values, expected)


@pytest.mark.parametrize('kernel', ['erfc', 'erf'])
def test_independent_image_gauge_and_transfer_conjugacy(kernel):
    a = case(); moved = case(shift=True)
    for x in (a, moved): x['physical_kernel'] = physical(kernel, .43)
    r = manta.compute_periodic_gdf(**a); shifted = manta.compute_periodic_gdf(**moved)
    L = a['system'].lattice[:, 0]; q = a['q']; aux = np.exp(1j*np.array([0, q@L]))
    np.testing.assert_allclose(shifted.metric, r.metric*aux[:, None]*aux.conj()[None], atol=3e-12, rtol=0)
    for ik, k in enumerate(a['ket_kpoints']):
        bra = np.exp(1j*np.array([0, (k-q)@L])); ket = np.exp(-1j*np.array([0, k@L]))
        np.testing.assert_allclose(shifted.three_center[ik], r.three_center[ik]*aux[:, None, None]*bra[None, :, None]*ket[None, None, :], atol=3e-12, rtol=0)
    opposite = manta.compute_periodic_gdf(**dict(a, q=-q, ket_kpoints=-a['ket_kpoints'], vectors=-a['vectors']))
    np.testing.assert_array_equal(opposite.metric, r.metric.conj())
    np.testing.assert_array_equal(opposite.three_center, r.three_center.conj())


@pytest.mark.parametrize('kernel', ['erfc', 'erf'])
@pytest.mark.parametrize('angular,pure', [(1, True), (2, False), (3, True)])
def test_angular_signed_contractions_against_independent_polynomial_fourier(kernel, angular, pure):
    import itertools
    import vibeqc as vq
    from tests.test_aux_eri_sr_lattice_enumeration import _weighted_gradient_fixture
    from tests.test_manta_periodic_gdf import quadrature_long_range
    a = case(); system, orb, aux = _weighted_gradient_fixture(angular, orbital_angular=1)
    def change(basis):
        shells = basis.shells()
        for shell in shells: shell.pure = pure
        return vq.BasisSet(system.unit_cell_molecule(), shells, 'range-fitting-angular', True)
    a.update(system=system, orbital=change(orb), auxiliary=change(aux),
        vectors=np.array(list(itertools.product(range(-1, 2), repeat=3)))@system.reciprocal_lattice().T+a['q'],
        physical_kernel=physical(kernel, .43))
    a['options'].pair_cutoff_bohr = 4.3; a['options'].auxiliary_cutoff_bohr = 5.1
    r = manta.compute_periodic_gdf(**a); source = manta.PeriodicGDFTiles(**a)
    sm, st = libint_reference(a); lm, lt = quadrature_long_range(a)
    a['options'].omega = .43; pm, pt = quadrature_long_range(a)
    expected_m, expected_t = (sm+lm-pm, st+lt-pt) if kernel == 'erfc' else (pm, pt)
    np.testing.assert_allclose(r.metric, expected_m, atol=4e-11, rtol=3e-12)
    np.testing.assert_allclose(r.three_center, expected_t, atol=4e-11, rtol=3e-12)
    np.testing.assert_array_equal(source.metric().values, r.metric)
    np.testing.assert_array_equal(assemble(source), r.three_center)


@pytest.mark.parametrize('kernel', ['erfc', 'erf'])
@pytest.mark.parametrize('field,quantity', CAPS)
def test_dense_and_streamed_exact_and_one_short_admission(kernel, field, quantity):
    for tiled in (False, True):
        a = case(True); a['physical_kernel'] = physical(kernel, .43)
        args = dict(a); args.pop('q'); args['momenta'] = len(args.pop('ket_kpoints')); args['vectors'] = len(args['vectors'])
        p = getattr(manta, 'plan_periodic_gdf_tiles' if tiled else 'plan_periodic_gdf')(**args)
        for cap, bound in CAPS: setattr(a['budget'], cap, getattr(p, bound))
        run = manta.PeriodicGDFTiles if tiled else manta.compute_periodic_gdf
        r = run(**a)
        if tiled: r.metric(); r.evaluate(0, 0, 0)
        setattr(a['budget'], field, getattr(p, quantity)-1)
        a['q'][:] = np.nan; a['vectors'][:] = np.nan
        with pytest.raises((ValueError, RuntimeError), match='budget'): run(**a)


@pytest.mark.parametrize('kernel,omega', [('wrong', .5), ('coulomb', .5), ('erfc', 0.), ('erf', -1.), ('erfc', np.nan), ('erf', np.inf), ('erfc', 1e-200), ('erfc', 1e-160), ('erf', 1e200)])
def test_invalid_physical_requests_refuse_before_source_work(kernel, omega):
    a = case(); a['physical_kernel'] = physical(kernel, omega)
    for run in (manta.PeriodicGDFTiles, manta.compute_periodic_gdf):
        with pytest.raises(ValueError, match='physical'): run(**a)


def scf_calculation(kernel, spins, whole_mesh=False):
    """Finite functional SCF diagnostic, not a production range KS route.

    Native fixed Coulomb-J and physical-range K sources plus native XC are
    composed outside the bounded driver. A separate libint/Gaussian/NumPy
    response and SciPy fixed point independently check the complete energy.
    """
    from tests.test_manta_periodic_ri_ks_domains import arguments, explicit, function
    from tests.test_manta_periodic_ks import independent_scf
    from tests.test_manta_periodic_semilocal_xc import reference as xc_reference
    from tests.test_manta_periodic_ri_ks import reference as full_reference
    from tests.test_manta_periodic_rhf import _nuclear_reference
    from tests.test_manta_periodic_gdf import legacy
    import vibeqc as vq
    # Evaluate the explicit PBE0 semilocal weights as a pure component here.
    # This diagnostic supplies the physical-range exchange separately; it
    # does not pass a global-hybrid descriptor to the pure-XC public leaf.
    vq.define_functional('manta-range-fit-semilocal-test', [('GGA_X_PBE', .75), ('GGA_C_PBE', 1.)])
    args, a, k, old = arguments('JK', spins, True, mesh=(2, 1, 1) if whole_mesh else (1, 1, 1),
                              shift=(1, 0, 0) if whole_mesh else (0, 0, 0))
    args['options'].exchange_q0 = 'strict-zero-mode'; args['options'].ks.scf.maximum_iterations = 1
    explicit(args, a, old); initial = function(args)(**args)
    s, h, _ = full_reference(a, k, old)
    np.testing.assert_allclose(initial.overlap, s, atol=4e-12, rtol=0)
    np.testing.assert_allclose(initial.hcore, h, atol=4e-12, rtol=0)
    source = dict(orbital=a['basis'], auxiliary=a['auxiliary'], system=a['system'], q=np.zeros(3),
        ket_kpoints=k, vectors=a['reciprocal_vectors'], options=a['options'].fitting, budget=a['budget'])
    jm, jt, _ = legacy(source)
    screened = dict(source, physical_kernel=physical(kernel, .43)); km, kt = reference(screened)
    jb = builder(source); kb = builder(screened); nk = len(k); w = np.full(nk, 1./nk); fraction = .25
    pairs = {}
    if whole_mesh:
        kb = manta.PeriodicRIFockBuilder(a['basis'], a['auxiliary'], a['system'], k, w,
            a['reciprocal_vectors'], a['exchange_vectors'], a['transfer_offsets'],
            a['options'].fitting, a['options'].ri, a['budget'], False, True, screened['physical_kernel'])
        for i in range(nk):
            for j in range(nk):
                if i == j: continue
                lo, hi = a['transfer_offsets'][i*nk+j:i*nk+j+2]
                pairs[i, j] = reference(dict(screened, q=k[j]-k[i], ket_kpoints=k[j:j+1], vectors=a['exchange_vectors'][lo:hi]))
    xc = dict(basis=a['basis'], functional=vq.Functional('manta-range-fit-semilocal-test', spin=spins), lattice=np.array(a['system'].lattice),
        images=a['xc_images'], momenta=k, weights=w, points=a['points'], quadrature_weights=a['quadrature_weights'],
        budget=a['budget'], grid_batch=a['options'].ks.grid_batch)

    def response(d, native):
        spin = np.repeat(.5*d, 2, axis=0) if spins == 1 else d
        if native:
            j = jb.build(spin, w, make_k=False)
            exchange = (kb.build(np.zeros_like(initial.hcore), spin).K if whole_mesh else kb.build(spin, w, make_j=False).K)
            jmat, ej = j.J, j.receipt.coulomb_energy
            local = manta.compute_periodic_semilocal_xc(**xc, densities=d); ex, v = local.receipt.energy, local.values
            one = initial.hcore
        else:
            jmat, _, ej, _, _ = ri_reference(jm, jt, spin, w)
            _, exchange, _, _, _ = ri_reference(km, kt, spin, w)
            for (i, j), (m, t) in pairs.items():
                exchange[:, i] += ri_reference(m, t, spin[:, j:j+1], w[j:j+1])[1][:, 0]
            ex, v, _ = xc_reference(dict(xc, densities=d), a['functional']); one = h
        e1 = np.einsum('k,skij,kji->', w, d, one).real
        ek = -.5*fraction*np.einsum('k,skij,skji->', w, spin, exchange).real
        return one+jmat+v-fraction*exchange[:spins], np.array([e1, ej, ex, ek]), None

    pop = (initial.plan.n_alpha,) if spins == 1 else (initial.plan.n_alpha, initial.plan.n_beta)
    expected = independent_scf(s, h, lambda d: response(d, False), pop, spins)
    d = independent_scf(initial.overlap, initial.hcore, lambda d: response(d, True), pop, spins)
    f, components, _ = response(d, True); rf, rc, _ = response(d, False)
    np.testing.assert_allclose(d, expected, atol=5e-9, rtol=0)
    np.testing.assert_allclose(f, rf, atol=8e-11, rtol=0)
    np.testing.assert_allclose(components, rc, atol=8e-11, rtol=0)
    nuclear = sum(_nuclear_reference(old)[0])
    energy = nuclear+sum(components); expected_energy = nuclear+sum(response(expected, False)[1])
    assert energy == pytest.approx(expected_energy, abs=8e-11)
    return dict(kernel=kernel, spins=spins, momenta=nk, energy=energy, reference_energy=expected_energy,
        errors=dict(density=float(np.max(abs(d-expected))), fock=float(np.max(abs(f-rf))),
                    components=float(np.max(abs(components-rc))), energy=abs(energy-expected_energy)))


@pytest.mark.parametrize('kernel', ['erfc', 'erf'])
@pytest.mark.parametrize('spins', [1, 2])
def test_complete_finite_gamma_scf_with_physical_range_exchange(kernel, spins):
    scf_calculation(kernel, spins)
