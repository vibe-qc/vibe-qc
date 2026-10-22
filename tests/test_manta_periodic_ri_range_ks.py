"""Bounded physical range KS: separate kernel projectors and complete energy."""
import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from vibeqc.bipole_fock_ewald import probe_charge_madelung_supercell
from tests.test_manta_periodic_ri_ks import case as global_case, CAPS
from tests.test_manta_periodic_ks import oracle as semilocal_oracle, independent_scf, occupy
from tests.test_manta_periodic_gdf import legacy
from tests.test_manta_periodic_range_fitting import reference as physical_reference, physical
from tests.test_manta_periodic_exchange import _quadrature_bloch
from tests.test_manta_periodic_range_separated_ks import TERMS, MIXED_RANGE, EMPTY_RANGE
from tests.test_manta_periodic_global_hybrid_ks import components
from tests.test_manta_periodic_rhf import _nuclear_reference


def case(profile='JK', spins=1, name='cam-b3lyp', **kwargs):
    a, k, old = global_case(profile, spins, name, **kwargs)
    previous = a['options']; o = manta.PeriodicRIRangeSeparatedKSOptions()
    for key in ('ks', 'fitting', 'ri', 'exchange_q0', 'madelung_alpha', 'madelung_tolerance',
                'coulomb_algorithm', 'exchange_algorithm'):
        setattr(o, key, getattr(previous, key))
    a['options'] = o
    return a, k, old


def run(a):
    return core._manta.run_periodic_ri_range_separated_ks(**a)


def plan(a):
    return manta.plan_periodic_ri_range_separated_ks(*[a[n] for n in ('basis', 'auxiliary', 'system', 'mesh', 'functional')],
        len(a['reciprocal_vectors']), int(max(np.diff(a['transfer_offsets']))), len(a['translations']),
        len(a['xc_images']), len(a['points']), a['options'], a['budget'])


def as_fock_build(r, k):
    from vibeqc.fock_state import FockBuild, EnergyComponents, SourceProvenance
    from vibeqc.solvers.operators import OperatorFrame
    ns, nk, n = r.plan.spins, r.plan.momenta, r.plan.n_basis
    frames = []
    for i, metric in enumerate(r.overlap):
        values, vectors = np.linalg.eigh(metric); rank = r.solver_memory.retained_rank
        frames.append(OperatorFrame(basis_id=r.receipt.overlap_identity, n=n, ordering='libint-shell-axial-v1',
            image_gauge=f'native-exp(+ikR):k-index={i}', overlap=metric, X=vectors[:, -rank:]/np.sqrt(values[-rank:])))
    energy = EnergyComponents(electronic=sum(components(r)[:-1]), nuclear=r.receipt.nuclear_energy,
        components=tuple(zip(('one_electron', 'coulomb', 'xc', 'uncorrected_exchange'), components(r)[:-1])),
        corrections=(('exchange_zero_mode', r.receipt.exchange_correction_energy),))
    source = SourceProvenance.capture('manta', native_source_identity=r.receipt.source_identity,
        coulomb_identity=r.receipt.coulomb_identity, xc_identity=r.receipt.xc.source_identity,
        kernel_sources=[dict(kernel=c.kernel, coefficient=c.coefficient, omega=c.omega,
            source=c.fitted_fock.source_identity if c.fitted_fock else c.exchange.source_identity,
            projectors=[m.projector_identity for m in c.fitting_info.metrics] if c.fitting_info else [])
            for c in r.receipt.exchange_components])
    labels = ['restricted-total'] if ns == 1 else ['alpha', 'beta']
    return FockBuild.from_dense(method='RKS' if ns == 1 else 'UKS', generation=r.solver_snapshot.evaluated_iteration,
        source=source, energy=energy, densities=r.density.reshape(ns*nk, n, n), focks=r.fock.reshape(ns*nk, n, n),
        frames=frames*ns, spins=[s for s in labels for _ in range(nk)], kpoints=[tuple(p) for p in k]*ns,
        weights=[1./nk]*(ns*nk), kind='ks')


def reference(a, k, old, terms=None):
    """Independent libint/Gaussian sources, fixed NumPy factors and libxc values."""
    s, h, semilocal = semilocal_oracle(a, k, old)
    terms = TERMS[a['functional'].name.lower()] if terms is None else terms
    o = a['options']; ns = a['functional'].spin; nk = len(k); w = np.full(nk, 1./nk)
    fit_j, fit_k = o.coulomb_algorithm == 'ri', o.exchange_algorithm == 'ri'
    g = a['reciprocal_vectors']; volume = abs(np.linalg.det(a['system'].lattice))
    common = dict(orbital=a['basis'], auxiliary=a['auxiliary'], system=a['system'], options=o.fitting, budget=a['budget'])

    def factor(kernel, omega, q, ket, vectors):
        src = dict(common, q=q, ket_kpoints=ket, vectors=vectors)
        m, t = legacy(src)[:2] if kernel == 'coulomb' else physical_reference(dict(src, physical_kernel=physical(kernel, omega)))
        m = np.tril(m)+np.tril(m, -1).conj().T; m[np.diag_indices(len(m))] = m.diagonal().real
        values, vectors = np.linalg.eigh(m)
        keep = values > max(o.ri.absolute_cutoff, o.ri.relative_cutoff*max(0., values[-1]))
        return np.einsum('pl,kpmn->klmn', (vectors[:, keep]/np.sqrt(values[keep])).conj(), t)

    jgamma = factor('coulomb', 0., np.zeros(3), k, g) if fit_j else None
    exchange_sources = []
    for kernel, coefficient, omega in terms:
        gamma = factor(kernel, omega, np.zeros(3), k, g) if fit_k else None
        pairs = {}
        for i in range(nk):
            for j in range(nk):
                lo, hi = a['transfer_offsets'][i*nk+j:i*nk+j+2]; q = a['exchange_vectors'][lo:hi]
                if fit_k:
                    if i != j: pairs[i, j] = factor(kernel, omega, k[j]-k[i], k[j:j+1], q)[0]
                else:
                    if kernel == 'erfc' and i == j: q = np.vstack((q, np.zeros((1, 3))))
                    q2 = np.sum(q*q, axis=1); coeff = np.empty(len(q)); nonzero = q2 > 0
                    if kernel == 'coulomb': coeff[:] = 4*np.pi/q2
                    else:
                        coeff[nonzero] = -4*np.pi*np.expm1(-q2[nonzero]/(4*omega**2))/q2[nonzero]
                        coeff[~nonzero] = np.pi/omega**2
                    b = _quadrature_bloch(a['basis'], q, a['translations'], k[i], o.ks.pair_cutoff_bohr)
                    pairs[i, j] = b, w[j]*coeff/volume
        exchange_sources.append((coefficient, gamma, pairs))
    fj = np.stack([_quadrature_bloch(a['basis'], g, a['translations'], -point, o.ks.pair_cutoff_bohr) for point in k])
    kernel = 4*np.pi/volume/np.sum(g*g, axis=1)
    long = sum(c for kernel, c, _ in terms if kernel == 'coulomb')
    xi = 0. if not long or o.exchange_q0 == 'strict-zero-mode' else probe_charge_madelung_supercell(
        a['system'], tuple(a['mesh'].mesh), alpha=o.madelung_alpha, precision=1e-14)

    def build(d):
        f, energies, populations = semilocal(d); e1, ej, ex = energies
        spin = np.repeat(.5*d, 2, axis=0) if ns == 1 else d
        if fit_j:
            rho = np.einsum('k,klmn,skmn->l', w, jgamma.conj(), spin)
            j = np.einsum('klmn,l->kmn', jgamma, rho); ej = .5*np.vdot(rho, rho).real
            rho_direct = np.einsum('k,skmn,kgmn->g', w, d, fj)
            f += j-np.einsum('g,g,kgmn->kmn', kernel, rho_direct, fj.conj())
        ek = 0.
        for coefficient, gamma, pairs in exchange_sources:
            exchange = np.zeros_like(spin)
            if fit_k:
                exchange = np.einsum('k,klmp,skpq,klnq->skmn', w, gamma, spin, gamma.conj())
                for (i, j), b in pairs.items():
                    exchange[:, i] += w[j]*np.einsum('lmp,spq,lnq->smn', b, spin[:, j], b.conj())
            else:
                for (i, j), (b, coeff) in pairs.items():
                    exchange[:, i] += np.einsum('g,gai,sab,gbj->sij', coeff, b.conj(), spin[:, j], b)
            f -= coefficient*exchange[:ns]
            ek -= .5*coefficient*np.einsum('k,skij,skji->', w, spin, exchange).real
        correction = -long/(2 if ns == 1 else 1)*xi*(s@d@s)
        ec = .5*np.einsum('k,skij,skji->', w, d, correction).real
        return f+correction, (e1, ej, ex, ek, ec), populations
    return s, h, build


def check_scf(a, k, old, terms=None):
    r = run(a); assert r.converged and r.receipt.confirmed
    s, h, build = reference(a, k, old, terms)
    pop = (r.plan.n_alpha,) if r.plan.spins == 1 else (r.plan.n_alpha, r.plan.n_beta)
    d = independent_scf(s, h, build, pop, r.plan.spins); f, e, populations = build(r.density)
    np.testing.assert_allclose(r.density, d, atol=5e-9, rtol=0)
    np.testing.assert_allclose(r.overlap, s, atol=4e-12, rtol=0)
    np.testing.assert_allclose(r.hcore, h, atol=8e-12, rtol=0)
    np.testing.assert_allclose(r.fock, f, atol=8e-10, rtol=0)
    np.testing.assert_allclose(components(r), e, atol=8e-11, rtol=0)
    np.testing.assert_allclose(r.receipt.xc.electrons_grid, populations, atol=8e-11, rtol=0)
    expected_energy = sum(_nuclear_reference(old)[0])+sum(build(d)[1])
    assert r.receipt.energy == pytest.approx(expected_energy, abs=8e-11)
    assert r.receipt.range_separated and r.receipt.algorithm == 'finite-3d-ri-range-separated-ks'
    assert r.receipt.exchange is None and r.plan.exchange is None
    assert r.receipt.fock_calls == r.solver_snapshot.fock_calls+1
    assert r.receipt.shell_calls <= r.plan.shell_calls and r.receipt.primitive_pairs <= r.plan.primitive_pairs
    assert r.receipt.primitive_points <= r.plan.primitive_points
    retained = r.plan.fitting.retained_source_bytes if r.plan.fitting else 0
    if r.plan.fitting: assert r.plan.fitting.has_j and not r.plan.fitting.has_k
    terms = TERMS[a['functional'].name.lower()] if terms is None else terms
    for c, p, (kernel, coefficient, omega) in zip(r.receipt.exchange_components, r.plan.exchange_components, terms):
        assert c.kernel == p.kernel == kernel and c.coefficient == pytest.approx(coefficient)
        assert c.omega == pytest.approx(omega)
        if a['options'].exchange_algorithm == 'ri':
            assert not p.fitting.has_j and p.fitting.has_k
            assert c.fitted_fock.exchange_kernel == kernel and c.fitting_info.physical_omega == omega
            assert c.fitted_fock.exchange_energy == c.energy
            assert len(c.fitting_info.metrics) == 1+len(k)*(len(k)-1)
            retained += p.fitting.retained_source_bytes
        else: assert p.fitting is None and c.fitted_fock is None
    assert len(r.receipt.exchange_components) == len(terms)
    assert r.solver_memory.provider_retained_bytes >= retained
    assert r.receipt.uncorrected_exchange_energy == pytest.approx(sum(c.energy for c in r.receipt.exchange_components), abs=2e-14)
    return r, dict(energy=abs(r.receipt.energy-expected_energy), density=float(np.max(abs(r.density-d))),
                  fock=float(np.max(abs(r.fock-f))), components=float(np.max(abs(np.array(components(r))-e))))


@pytest.mark.parametrize('profile', ['J', 'K', 'JK'])
@pytest.mark.parametrize('spins', [1, 2])
@pytest.mark.parametrize('name', ['hse06', 'cam-b3lyp', MIXED_RANGE])
@pytest.mark.parametrize('mesh,shift', [((1, 1, 1), (0, 0, 0)), ((2, 1, 1), (1, 0, 0))])
@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
def test_complete_scf_uses_each_physical_metric_once(profile, spins, name, mesh, shift, correction):
    check_scf(*case(profile, spins, name, mesh=mesh, shift=shift, correction=correction))


@pytest.mark.parametrize('name', ['lc-wpbe', 'wb97x', EMPTY_RANGE])
@pytest.mark.parametrize('spins', [1, 2])
def test_other_long_range_and_empty_spin_limits(name, spins):
    check_scf(*case(spins=spins, name=name, species='H' if name == EMPTY_RANGE and spins == 2 else None))


@pytest.mark.parametrize('profile', ['J', 'K', 'JK'])
@pytest.mark.parametrize('spins', [1, 2])
def test_capped_epochs_fixed_sources_and_shared_fock(profile, spins):
    a, k, old = case(profile, spins); s, h, build = reference(a, k, old)
    pop = (2,) if spins == 1 else (2, 1)
    d = occupy(np.repeat(h[None], spins, axis=0), s, pop, spins); identities = []
    for limit in (1, 2, 3):
        a['options'].ks.scf.maximum_iterations = limit; r = run(a)
        assert not r.converged and r.receipt.fock_calls == limit
        np.testing.assert_allclose(r.density, d, atol=8e-10, rtol=0)
        np.testing.assert_allclose(r.fock, build(d)[0], atol=8e-10, rtol=0)
        np.testing.assert_allclose(components(r), build(d)[1], atol=8e-10, rtol=0)
        snapshot = as_fock_build(r, k)
        assert snapshot.energy.total == pytest.approx(r.receipt.energy, abs=2e-12)
        for channel, matrix in zip(snapshot.channels, r.fock.reshape(-1, r.plan.n_basis, r.plan.n_basis)):
            np.testing.assert_allclose(channel.operator.apply(np.eye(r.plan.n_basis, dtype=complex)), matrix, atol=2e-13, rtol=0)
            assert channel.operator.metadata.symmetry_block is None
        identities.append((r.receipt.source_identity, [c.fitting_info.source_identity for c in r.receipt.exchange_components if c.fitting_info]))
        d = occupy(build(d)[0], s, pop, spins)
    assert identities[0] == identities[1] == identities[2]


@pytest.mark.parametrize('profile', ['J', 'K', 'JK'])
@pytest.mark.parametrize('field,quantity', CAPS)
def test_all_aggregate_caps_precede_payload_copies(profile, field, quantity):
    a, _, _ = case(profile, name=MIXED_RANGE); a['options'].ks.scf.maximum_iterations = 1
    p = plan(a); setattr(a['budget'], field, getattr(p, quantity)); run(a)
    setattr(a['budget'], field, getattr(p, quantity)-1)
    a['points'][:] = np.nan; a['exchange_vectors'][:] = np.nan
    with pytest.raises((ValueError, RuntimeError), match='budget'): run(a)


@pytest.mark.parametrize('spins', [1, 2])
@pytest.mark.parametrize('profile', ['J', 'K', 'JK'])
def test_fixed_projector_variational_derivative_and_source_epoch(profile, spins):
    a, k, old = case(profile, spins, MIXED_RANGE); a['options'].ks.scf.maximum_iterations = 2
    r = run(a); _, _, build = reference(a, k, old)
    rng = np.random.default_rng(218); v = rng.normal(size=r.density.shape)+1j*rng.normal(size=r.density.shape)
    d = r.density; adj = v.conj().swapaxes(-1, -2); delta = v@d+d@adj; step = 1e-5; eye = np.eye(r.plan.n_basis)
    plus = (eye+step*v)@d@(eye+step*adj); minus = (eye-step*v)@d@(eye-step*adj)
    derivative = (sum(build(plus)[1])-sum(build(minus)[1]))/(2*step)
    assert derivative == pytest.approx(np.einsum('skij,skji->', delta, r.fock).real/len(k), abs=5e-8)
    a['options'].ks.scf.maximum_iterations = 1; first = run(a)
    a['options'].fitting.omega = .8; changed = run(a)
    np.testing.assert_array_equal(first.density, changed.density)
    assert first.receipt.overlap_identity == changed.receipt.overlap_identity
    assert first.receipt.final_density_identity == changed.receipt.final_density_identity
    assert first.receipt.source_identity != changed.receipt.source_identity


@pytest.mark.parametrize('spins', [1, 2])
@pytest.mark.parametrize('cutoff', [10., 1e9])
def test_truncated_and_empty_kernel_projectors_preserve_xc_and_correction(spins, cutoff):
    a, k, old = case(spins=spins); a['options'].ri.absolute_cutoff = cutoff
    a['options'].ks.scf.maximum_iterations = 2; r = run(a); _, _, build = reference(a, k, old)
    np.testing.assert_allclose(r.fock, build(r.density)[0], atol=8e-10, rtol=0)
    np.testing.assert_allclose(components(r), build(r.density)[1], atol=8e-11, rtol=0)
    assert abs(r.receipt.xc_energy) > .01 and r.receipt.exchange_correction_energy != 0.
    if cutoff == 1e9:
        assert r.receipt.uncorrected_exchange_energy == r.receipt.coulomb_energy == 0.
        assert all(m.rank == 0 for c in r.receipt.exchange_components for m in c.fitting_info.metrics)


@pytest.mark.parametrize('profile', ['J', 'K', 'JK'])
@pytest.mark.parametrize('fraction', [0., .2])
def test_cancelled_ranges_do_not_invent_fitted_exchange(profile, fraction):
    name = f'manta-ri-rsh-cancelled-{fraction}'
    vq.define_functional(name, [('HYB_GGA_XC_HSE06', 1.), ('HYB_GGA_XC_HSE06', -1.),
        ('GGA_X_B88', .8), ('GGA_C_LYP', 1.)], hf_exchange_fraction=fraction)
    a, k, old = case(profile, name=name)
    r, _ = check_scf(a, k, old, [('coulomb', fraction, 0.)] if fraction else [])
    if not fraction: assert r.receipt.exchange_energy == 0. and r.receipt.exchange_q0 == 'not-applicable'


@pytest.mark.parametrize('spins', [1, 2])
def test_q0_correction_is_method_owned_and_screened_zero_not_repeated(spins):
    a, _, _ = case(spins=spins, name='hse06'); corrected = run(a)
    a['options'].exchange_q0 = 'strict-zero-mode'; strict = run(a)
    np.testing.assert_array_equal(corrected.density, strict.density)
    np.testing.assert_array_equal(corrected.fock, strict.fock)
    assert corrected.receipt.source_identity == strict.receipt.source_identity
    assert corrected.receipt.exchange_correction_energy == corrected.receipt.madelung == 0.
    assert not corrected.plan.madelung.kernel_terms
    a, _, _ = case(spins=spins, correction='strict-zero-mode'); strict = run(a)
    a['options'].exchange_q0 = 'BvK-ewald'; corrected = run(a)
    np.testing.assert_allclose(corrected.density, strict.density, atol=5e-9, rtol=0)
    assert corrected.receipt.energy-strict.receipt.energy == pytest.approx(-.65*corrected.receipt.madelung*corrected.plan.electrons/2, abs=4e-10)
    assert [c.fitting_info.source_identity for c in corrected.receipt.exchange_components] == [c.fitting_info.source_identity for c in strict.receipt.exchange_components]


@pytest.mark.parametrize('name', ['pbe', 'pbe0', 'scan', 'tpssh', 'b2plyp', 'wb97x-v'])
def test_incomplete_or_wrong_functional_route_refuses_before_grid(name):
    a, _, _ = case(name=name); a['points'][:] = np.nan
    with pytest.raises(ValueError, match='range-separated|complete|component'): run(a)


@pytest.mark.parametrize('field,value', [('coulomb_algorithm', 'auto'), ('exchange_algorithm', 'none'), ('exchange_q0', 'guess')])
def test_unsupported_source_choices_refuse(field, value):
    a, _, _ = case(); setattr(a['options'], field, value)
    with pytest.raises(ValueError, match='requires|exchange_q0'): run(a)


def test_direct_options_and_ri_domains_cannot_silently_change_sources():
    a, _, _ = case(); a['options'] = manta.PeriodicRangeSeparatedKSOptions()
    with pytest.raises(TypeError): run(a)
    a, _, _ = case('J'); a['options'].exchange_algorithm = 'ri'
    with pytest.raises(ValueError, match='diagonal'): run(a)
    a, _, _ = case(); a['options'].coulomb_algorithm = a['options'].exchange_algorithm = 'exact'
    with pytest.raises(ValueError, match='at least one RI'): run(a)


@pytest.mark.parametrize('profile', ['J', 'K', 'JK'])
@pytest.mark.parametrize('spins', [1, 2])
def test_multicenter_image_transport_and_component_ownership(profile, spins):
    from vibeqc.symmetry_shared import BlockSpaceAction, Budget, SpaceIdentity
    a, k, _ = case(profile, spins, MIXED_RANGE); a['options'].ks.scf.maximum_iterations = 3
    charges = (2, 1, 1) if spins == 1 else (1, 1, 1)
    atoms = [vq.Atom(z, xyz) for z, xyz in zip(charges, ([0., 0., 0.], [.9, .2, 0.], [-.2, 1.1, .1]))]
    system = vq.PeriodicSystem(3, a['system'].lattice, atoms); system.multiplicity = spins
    shells, auxiliary = a['basis'].shells(), a['auxiliary'].shells()
    for i, shell in enumerate(shells): shell.atom_index = i; shell.origin = atoms[i].xyz
    for shell in auxiliary: shell.origin = atoms[0].xyz
    a.update(system=system, translations=a['xc_images'].copy(),
        basis=vq.BasisSet(system.unit_cell_molecule(), shells, 'ri-range-multicenter', True),
        auxiliary=vq.BasisSet(system.unit_cell_molecule(), auxiliary, 'ri-range-auxiliary', True))
    original = run(a); shifts = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])@system.lattice.T
    for shell, delta in zip(shells, shifts): shell.origin = np.array(shell.origin)+delta
    for i, (shell, delta) in enumerate(zip(auxiliary, shifts[::-1])):
        shell.atom_index = i; shell.origin = np.array(shell.origin)+delta
    for key, values in (('basis', shells), ('auxiliary', auxiliary)):
        images = vq.PeriodicSystem(3, system.lattice, [vq.Atom(z, shell.origin) for z, shell in zip(charges, values)])
        images.multiplicity = spins
        a[key] = vq.BasisSet(images.unit_cell_molecule(), values, 'ri-range-images', True)
    moved = run(a); phase = np.exp(1j*k@shifts.T)
    assert np.max(abs(moved.density.imag)) > .01
    for ik in range(len(k)):
        source = SpaceIdentity('ri-range-test', original.receipt.overlap_identity, 'AO', f'original-k{ik}')
        destination = SpaceIdentity('ri-range-test', moved.receipt.overlap_identity, 'AO', f'shifted-k{ik}')
        action = BlockSpaceAction(source, destination, np.arange(4, dtype=np.int64), np.arange(3, dtype=np.int64),
            np.ascontiguousarray(phase[ik]), antiunitary=False, budget=Budget(2**20, 10**6))
        for spin in range(spins):
            np.testing.assert_allclose(action.push_density(original.density[spin, ik]), moved.density[spin, ik], atol=3e-10, rtol=0)
            np.testing.assert_allclose(action.pull_operator(moved.fock[spin, ik]), original.fock[spin, ik], atol=3e-10, rtol=0)
    assert moved.receipt.energy == pytest.approx(original.receipt.energy, abs=2e-10)
    state = as_fock_build(moved, k); assert state.energy.total == pytest.approx(moved.receipt.energy, abs=2e-12)
    component = moved.receipt.exchange_components[-1]
    if component.fitting_info:
        metrics = component.fitting_info.metrics; metrics.clear()
        assert component.fitting_info.metrics
        assert component.fitting_info.source_identity != original.receipt.exchange_components[-1].fitting_info.source_identity
    expected = moved.density.copy(); moved.density[:] = 999
    np.testing.assert_array_equal(moved.density, expected)


@pytest.mark.parametrize('spins', [1, 2])
def test_rectangular_overlap_frames_keep_full_physical_fock(spins):
    a, k, old = case(spins=spins, name=MIXED_RANGE); a['options'].ks.scf.maximum_iterations = 3
    shells = a['basis'].shells(); shells.append(shells[0])
    a['basis'] = old['basis'] = vq.BasisSet(a['system'].unit_cell_molecule(), shells, 'ri-range-duplicate-shell', True)
    a['budget'].other_live_bytes = 123456; p = plan(a)
    for cap, field in CAPS: setattr(a['budget'], cap, getattr(p, field))
    r = run(a); s, h, build = reference(a, k, old); rank = 3
    x = []
    for metric in s:
        v, u = np.linalg.eigh(metric); x.append(u[:, -rank:]/np.sqrt(v[-rank:]))
    x = np.array(x); pop = (2,) if spins == 1 else (2, 1)
    def project(f):
        reduced = x.conj().swapaxes(-1, -2)@f@x
        return x@occupy(reduced, np.repeat(np.eye(rank)[None], len(k), axis=0), pop, spins)@x.conj().swapaxes(-1, -2)
    d = project(np.repeat(h[None], spins, axis=0))
    for _ in range(2): d = project(build(d)[0])
    np.testing.assert_allclose(r.density, d, atol=1e-9, rtol=0)
    np.testing.assert_allclose(r.fock, build(d)[0], atol=1e-9, rtol=0)
    assert r.coefficients.shape == (spins, len(k), rank+1, rank)
    assert r.solver_memory.total_numerical_bytes <= p.modeled_peak_bytes
    assert as_fock_build(r, k).energy.total == pytest.approx(r.receipt.energy, abs=2e-12)


@pytest.mark.parametrize('profile', ['J', 'K', 'JK'])
@pytest.mark.parametrize('spins', [1, 2])
def test_selected_provenance_records_actual_kernel_sources(monkeypatch, profile, spins):
    import json
    from vibeqc import output
    events = []; monkeypatch.setattr(output, 'record', lambda *args, **kw: events.append(kw['integral_provenance']))
    a, _, _ = case(profile, spins, MIXED_RANGE); a['options'].ks.scf.maximum_iterations = 2
    native = run(a); budget = a.pop('budget'); a['integral_engine'] = manta.IntegralEngineOptions(backend='manta', budget=budget)
    selected = manta.run_periodic_ri_range_separated_ks(**a); evidence = events[-1]
    for name in ('density', 'fock', 'coefficients', 'orbital_energies'): np.testing.assert_array_equal(getattr(native, name), getattr(selected, name))
    assert len(json.dumps(evidence, allow_nan=False).encode()) < evidence['python_reserved_bytes']
    assert evidence['citation_methods'][0] == 'manta_periodic_ri_range_separated_ks'
    assert evidence['operator_kind'] == 'ks' and not evidence['symmetry_reduced']
    assert (evidence['hartree'] is None) == (profile != 'K')
    assert budget.other_live_bytes == 0
    for c in evidence['exchange_components']:
        if profile == 'J': assert c['exchange']['kernel'] == c['kernel']
        else:
            assert c['exchange'] is None and c['fitted_fock']['component_only']
            assert c['energy_scale'] == 1. and c['energy'] == c['fitted_fock']['exchange_energy']
            assert c['fitting']['physical_omega'] == c['omega']
            assert c['fitting']['exchange_kernel'] == c['kernel']
            assert not c['fitting']['has_j']


@pytest.mark.parametrize('field,quantity', CAPS)
def test_selected_provenance_reservation_precedes_buffers(monkeypatch, field, quantity):
    from vibeqc import output
    events = []; monkeypatch.setattr(output, 'record', lambda *args, **kw: events.append(kw))
    a, _, _ = case(name=MIXED_RANGE); a['options'].ks.scf.maximum_iterations = 1
    budget = a.pop('budget'); a['integral_engine'] = manta.IntegralEngineOptions(backend='manta', budget=budget)
    r = manta.run_periodic_ri_range_separated_ks(**a); limit = getattr(r.plan, quantity)
    setattr(budget, field, limit); manta.run_periodic_ri_range_separated_ks(**a)
    assert len(events) == 2
    setattr(budget, field, limit-1); a['points'][:] = np.nan
    with pytest.raises((ValueError, RuntimeError), match='budget'): manta.run_periodic_ri_range_separated_ks(**a)
    assert len(events) == 2
