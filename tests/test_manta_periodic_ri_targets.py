"""Independent periodic RI targets, finite-source energies and shared operators."""
import itertools

import numpy as np
import pytest

from vibeqc import manta
from tests.test_manta_periodic_fitting import CAPS
from tests.test_manta_periodic_gdf import gaussian_fourier, independent, legacy
from tests.test_manta_periodic_ri_fock import case as whole_case
from tests.test_manta_periodic_ri_q import densities, reference


def case(target, nk=4, **kwargs):
    a = whole_case(nk, **kwargs)
    a.update(make_j='J' in target, make_k='K' in target)
    if not a['make_k']:
        a['pair_vectors'] = np.empty((0, 3), dtype=float)
        a['transfer_offsets'] = np.zeros(nk*nk+1, dtype=np.int64)
    return a


def plan(a):
    return manta.plan_periodic_ri_fock(a['orbital'], a['auxiliary'], a['system'],
        len(a['gamma_vectors']), int(max(np.diff(a['transfer_offsets']))), len(a['momenta']),
        a['options'], a['ri_options'], a['budget'], a['make_j'], a['make_k'])


def oracle(a, h, d, scale=1., source=independent):
    """Raw Gaussian/Boys/libint sums and NumPy factors; no MANTA contractions."""
    k, w, ri = a['momenta'], a['weights'], a['ri_options']
    args = dict(orbital=a['orbital'], auxiliary=a['auxiliary'], system=a['system'],
        q=np.zeros(3), ket_kpoints=k, vectors=a['gamma_vectors'], options=a['options'], budget=a['budget'])
    m, t = source(args)[:2]
    j, K, ej, _, _ = reference(m, t, d, w, ri.absolute_cutoff, ri.relative_cutoff)
    if not a['make_j']:
        j = np.zeros_like(h); ej = 0.
    if a['make_k']:
        for target in range(len(k)):
            for ket in range(len(k)):
                if target == ket:
                    continue
                lo, hi = a['transfer_offsets'][target*len(k)+ket:target*len(k)+ket+2]
                args.update(q=k[ket]-k[target], ket_kpoints=k[ket:ket+1], vectors=a['pair_vectors'][lo:hi])
                m, t = source(args)[:2]
                K[:, target] += reference(m, t, d[:, ket:ket+1], w[ket:ket+1],
                    ri.absolute_cutoff, ri.relative_cutoff)[1][:, 0]
    else:
        K = np.zeros_like(d)
    e1 = np.einsum('k,skij,kji->', w, d, h).real
    ek = -.5*scale*np.einsum('k,skij,skji->', w, d, K).real
    return j, K, h[None]+j[None]-scale*K, np.array([e1, ej, ek, e1+ej+ek])


@pytest.mark.parametrize('target', ['J', 'K', 'JK'])
@pytest.mark.parametrize('nk', [1, 2, 4])
@pytest.mark.parametrize('cutoff', [1e-12, 10., 1e9])
def test_selected_targets_independent_sources_and_component_energies(target, nk, cutoff):
    a = case(target, nk, cutoff=cutoff)
    if nk == 4:
        a['weights'] = np.array([.2, 0., .3, .5])
    b = manta.PeriodicRIFockBuilder(**a)
    h, d = densities(nk, spins=1, seed=611)[0], densities(nk)
    r = b.build(h, d, -.32)
    for source in (independent, legacy):
        j, k, f, e = oracle(a, h, d, -.32, source)
        np.testing.assert_allclose(r.values, f, atol=5e-12, rtol=2e-12)
        if a['make_j']:
            np.testing.assert_allclose(r.J, j, atol=5e-12, rtol=2e-12)
        else:
            assert r.J.shape == (0, 2, 2) and r.receipt.coulomb_energy == 0.
        if a['make_k']:
            np.testing.assert_allclose(r.K, k, atol=5e-12, rtol=2e-12)
        else:
            assert r.K.shape == (0, nk, 2, 2) and r.receipt.exchange_energy == 0.
            assert r.receipt.exchange_scale == 0.
        np.testing.assert_allclose([r.receipt.one_electron_energy, r.receipt.coulomb_energy,
            r.receipt.exchange_energy, r.receipt.electronic_energy], e, atol=6e-12, rtol=2e-12)
    for record in (b.plan, b.info, r.receipt):
        assert record.has_j == a['make_j'] and record.has_k == a['make_k']
    assert b.plan.channels == len(b.info.metrics) == len(r.components) == (1 if target == 'J' else 1+nk*(nk-1))
    assert r.components[0].has_j == a['make_j'] and r.components[0].has_k == a['make_k']
    assert all(c.has_k and not c.has_j for c in r.components[1:])
    assert r.components[0].metric_residual_evaluated == a['make_j']
    assert r.receipt.shell_calls <= b.plan.build_shell_calls
    assert r.receipt.primitive_products <= b.plan.build_primitive_products
    if target == 'K' and cutoff == 1e9:
        assert r.receipt.coupling_passes == r.receipt.shell_calls == 0


@pytest.mark.parametrize('target', ['J', 'K'])
@pytest.mark.parametrize('cutoff', [1e-12, 10., 1e9])
def test_each_selected_energy_is_variational_and_weighted_self_adjoint(target, cutoff):
    a = case(target, cutoff=cutoff); b = manta.PeriodicRIFockBuilder(**a)
    h, d = densities(4, spins=1, seed=611)[0], densities(4)
    delta = densities(4, seed=931); step = 1e-5
    r = b.build(h, d, .71)
    plus, minus = b.build(h, d+step*delta, .71), b.build(h, d-step*delta, .71)
    fd = (plus.receipt.electronic_energy-minus.receipt.electronic_energy)/(2*step)
    assert fd == pytest.approx(np.einsum('k,skij,skji->', a['weights'], delta, r.values).real, abs=4e-8)
    response = b.build(np.zeros_like(h), delta, .71)
    left = np.einsum('k,skij,skji->', a['weights'], delta, r.values-h).real
    right = np.einsum('k,skij,skji->', a['weights'], d, response.values).real
    assert left == pytest.approx(right, abs=6e-12)
    for c, p, m in zip(r.components, plus.components, minus.components):
        assert c.source_identity == p.source_identity == m.source_identity
        assert c.projector_identity == p.projector_identity == m.projector_identity


@pytest.mark.parametrize('nk', [1, 4])
@pytest.mark.parametrize('cutoff', [1e-12, 10., 1e9])
def test_joint_response_decomposes_without_changing_frames_or_density_identity(nk, cutoff):
    h, d = densities(nk, spins=1, seed=611)[0], densities(nk)
    builders = [manta.PeriodicRIFockBuilder(**case(target, nk, cutoff=cutoff)) for target in ('J', 'K', 'JK')]
    j, k, both = [b.build(h, d, .42) for b in builders]
    np.testing.assert_array_equal(j.J, both.J)
    np.testing.assert_array_equal(k.K, both.K)
    np.testing.assert_allclose(j.values+k.values-h, both.values, atol=5e-15, rtol=0)
    assert j.receipt.coulomb_energy == both.receipt.coulomb_energy
    assert k.receipt.exchange_energy == both.receipt.exchange_energy
    assert len({r.receipt.density_identity for r in (j, k, both)}) == 1
    assert len({r.receipt.frame_identity for r in (j, k, both)}) == 1
    assert len({r.receipt.source_identity for r in (j, k, both)}) == 3
    assert len({r.receipt.snapshot_identity for r in (j, k, both)}) == 3
    assert len({b.info.metrics[0].projector_identity for b in builders}) == 1
    # A numerical zero coefficient is not a source or work selector.
    zero = builders[2].build(h, d, 0.)
    assert zero.receipt.has_k and len(zero.components) == len(both.components)
    np.testing.assert_array_equal(zero.K, both.K)
    assert zero.receipt.shell_calls == both.receipt.shell_calls
    assert builders[0].build(h, d, 1.).receipt.snapshot_identity == j.receipt.snapshot_identity


@pytest.mark.parametrize('target', ['J', 'K'])
@pytest.mark.parametrize('field,quantity', CAPS)
def test_selected_target_complete_admission_before_copying_inputs(target, field, quantity):
    a = case(target); a['budget'].other_live_bytes = 12345
    p = plan(a)
    for name, value in CAPS:
        setattr(a['budget'], name, getattr(p, value))
    b = manta.PeriodicRIFockBuilder(**a)
    b.build(densities(4, spins=1)[0], densities(4))
    setattr(a['budget'], field, getattr(p, quantity)-1)
    for name in ('momenta', 'weights', 'gamma_vectors', 'pair_vectors'):
        a[name][:] = np.nan
    with pytest.raises((ValueError, RuntimeError), match='budget'):
        plan(a)
    with pytest.raises((ValueError, RuntimeError), match='budget'):
        manta.PeriodicRIFockBuilder(**a)


def test_j_only_admits_without_off_diagonal_sources_and_reduces_work():
    a = case('J'); p = plan(a); full = plan(case('JK'))
    assert p.channels == 1 and p.pair.n_basis == p.pair.modeled_peak_bytes == 0
    for _, quantity in CAPS:
        assert getattr(p, quantity) < getattr(full, quantity)
    assert p.retained_source_bytes < full.retained_source_bytes
    b = manta.PeriodicRIFockBuilder(**a); joint = manta.PeriodicRIFockBuilder(**case('JK'))
    h, d = densities(4, spins=1)[0], densities(4)
    assert b.info.metric_shell_calls < joint.info.metric_shell_calls
    assert b.build(h, d).receipt.shell_calls < joint.build(h, d).receipt.shell_calls
    a['budget'].memory_bytes = p.modeled_peak_bytes
    manta.PeriodicRIFockBuilder(**a)
    full_args = case('JK'); full_args['budget'] = a['budget']
    with pytest.raises((ValueError, RuntimeError), match='memory budget'):
        manta.PeriodicRIFockBuilder(**full_args)


@pytest.mark.parametrize('target', ['J', 'K'])
def test_diagnostic_composition_with_complementary_exact_source(target):
    """Trial-density composition; no mixed SCF or aggregate Python admission."""
    a = case(target, 2); k, w = a['momenta'], a['weights']
    lattice = a['system'].lattice; volume = abs(np.linalg.det(lattice))
    g = a['gamma_vectors']; g = g[np.sum(g*g, axis=1) > 1e-24].copy()
    cells = np.asarray(list(itertools.product(range(-3, 4), repeat=3)), float)@lattice.T
    args = dict(basis=a['orbital'], lattice=lattice, translations=cells, momenta=k, weights=w,
        reciprocal_batch=4, pair_cutoff_bohr=5., budget=a['budget'])
    # The direct exchange orientation is opposite the RI interface, with
    # primed diagonal G. Keep the complete finite source for each component.
    pieces = []
    for kt in k:
        for ks in k:
            q = a['gamma_vectors']+kt-ks
            pieces.append(q[np.sum(q*q, axis=1) > 1e-24])
    q = np.concatenate(pieces)
    offsets = np.r_[0, np.cumsum([len(p) for p in pieces])].astype(np.int64)
    negative = gaussian_fourier(dict(a, ket_kpoints=-k), g)[1]
    coef = 4*np.pi/volume/np.sum(g*g, axis=1)
    builder = manta.PeriodicRIFockBuilder(**a)
    h, d = densities(2, spins=1, seed=611)[0], densities(2)

    def build(d, use_reference=False):
        if use_reference:
            _, _, f, energies = oracle(a, h, d)
            one, ej, ek = energies[:3]
            if target == 'K':
                rho = np.einsum('k,kmn,kmng->g', w, d.sum(axis=0), negative)
                j = np.einsum('g,g,kmng->kmn', coef, rho, negative.conj())
                f += j[None]; ej = .5*np.dot(coef, np.abs(rho)**2)
            else:
                exchange = np.zeros_like(d)
                for i in range(2):
                    for j in range(2):
                        v = pieces[i*2+j]
                        b = gaussian_fourier(dict(a, ket_kpoints=k[i:i+1]), v)[1][0].transpose(2, 0, 1)
                        coefficient = 4*np.pi/volume/np.sum(v*v, axis=1)
                        exchange[:, i] += w[j]*np.einsum('g,gai,sab,gbj->sij', coefficient, b.conj(), d[:, j], b)
                f -= exchange; ek = -.5*np.einsum('k,skij,skji->', w, d, exchange).real
            return f, float(one+ej+ek)
        r = builder.build(h, d); f, energy = r.values, r.receipt.electronic_energy
        if target == 'K':
            exact = manta.compute_periodic_hartree(**args, wave_vectors=g, densities=d.sum(axis=0))
            f += exact.values[None]; energy += exact.receipt.energy
        else:
            exact = manta.compute_periodic_exchange(**args, wave_vectors=q, transfer_offsets=offsets, densities=d)
            f -= exact.values; energy += sum(exact.receipt.energies)
        return f, energy

    f, energy = build(d); expected, expected_energy = build(d, use_reference=True)
    np.testing.assert_allclose(f, expected, atol=6e-12, rtol=0)
    assert energy == pytest.approx(expected_energy, abs=6e-12)
    delta, step = densities(2, seed=813), 1e-5
    derivative = (build(d+step*delta)[1]-build(d-step*delta)[1])/(2*step)
    assert derivative == pytest.approx(np.einsum('k,skij,skji->', w, delta, f).real, abs=4e-8)


@pytest.mark.parametrize('target', ['J', 'K'])
@pytest.mark.parametrize('cutoff', [1e-12, 10.])
def test_gauge_transport_and_owned_exports_cross_actual_shared_fock_api(target, cutoff):
    from vibeqc.fock_state import FockBuild, EnergyComponents, SourceProvenance
    from vibeqc.solvers.operators import OperatorFrame
    from vibeqc.symmetry_shared import BlockSpaceAction, Budget, SpaceIdentity
    a = case(target, 2, cutoff=cutoff); moved = case(target, 2, shift=True, cutoff=cutoff)
    h, d = densities(2, spins=1, seed=611)[0], densities(2)
    L = a['system'].lattice[:, 0]
    phase = np.exp(1j*np.stack([np.zeros(2), a['momenta']@L], axis=1))
    factor = phase[:, :, None]*phase.conj()[:, None, :]
    original = manta.PeriodicRIFockBuilder(**a); shifted = manta.PeriodicRIFockBuilder(**moved)
    r, s = original.build(h, d), shifted.build(h*factor, d*factor)
    np.testing.assert_allclose(s.values, r.values*factor, atol=6e-12, rtol=0)
    assert r.receipt.electronic_energy == pytest.approx(s.receipt.electronic_energy, abs=6e-12)
    frames = []
    for ik in range(2):
        source = SpaceIdentity('finite-ri-target-test', original.info.frame_identity, 'AO', f'original-k{ik}')
        destination = SpaceIdentity('finite-ri-target-test', shifted.info.frame_identity, 'AO', f'shifted-k{ik}')
        action = BlockSpaceAction(source, destination, np.array([0, 1, 2], np.int64), np.array([0, 1], np.int64),
            np.ascontiguousarray(phase[ik]), antiunitary=False, budget=Budget(2**20, 10**6))
        for spin in range(2):
            np.testing.assert_allclose(action.push_density(d[spin, ik]), d[spin, ik]*factor[ik], atol=3e-14, rtol=0)
            np.testing.assert_allclose(action.pull_operator(s.values[spin, ik]), r.values[spin, ik], atol=6e-12, rtol=0)
        frames.append(OperatorFrame(basis_id=original.info.frame_identity, n=2,
            ordering='libint-shell-axial-v1', image_gauge=f'exp(+ikR):k-index={ik}', overlap=np.eye(2), X=np.eye(2)))
    receipt = r.receipt
    energy = EnergyComponents(electronic=receipt.electronic_energy, nuclear=0.,
        components=(('one_electron', receipt.one_electron_energy), ('coulomb', receipt.coulomb_energy),
                    ('exchange', receipt.exchange_energy)))
    provenance = SourceProvenance.capture('manta', source_identity=receipt.source_identity,
        has_j=receipt.has_j, has_k=receipt.has_k, projectors=[m.projector_identity for m in original.info.metrics])
    state = FockBuild.from_dense(method='periodic-ri-component-diagnostic', generation=7,
        source=provenance, energy=energy, densities=d.reshape(4, 2, 2), focks=r.values.reshape(4, 2, 2),
        frames=frames*2, spins=['alpha']*2+['beta']*2, kpoints=[tuple(k) for k in a['momenta']]*2,
        weights=list(a['weights'])*2)
    assert state.energy.total == receipt.electronic_energy
    trial = np.array([[1.+.1j, .6j], [.4j, -.2]])
    for index, channel in enumerate(state.channels):
        np.testing.assert_array_equal(channel.density, d.reshape(4, 2, 2)[index])
        np.testing.assert_array_equal(channel.operator.apply(trial), r.values.reshape(4, 2, 2)[index]@trial)
        assert channel.operator.metadata.role == 'physical' and channel.operator.metadata.symmetry_block is None
    expected = r.values.copy()
    for name in ('momenta', 'weights', 'gamma_vectors', 'pair_vectors'):
        a[name][:] = np.nan
    a['options'].omega = 0.; a['ri_options'].absolute_cutoff = 1e9
    np.testing.assert_array_equal(original.build(h, d).values, expected)
    r.values[:] = np.nan; d[:] = np.nan
    np.testing.assert_array_equal(state.channels[0].operator.matrix, expected[0, 0])


@pytest.mark.parametrize('fault', ['no-target', 'j-pairs', 'j-count', 'j-duplicate-k', 'j-gamma',
                                  'k-empty', 'k-diagonal', 'k-reverse'])
def test_selected_source_refuses_inconsistent_or_unused_domains(fault):
    a = case('K' if fault.startswith('k-') else 'J')
    if fault == 'no-target':
        a['make_j'] = False
    elif fault in ('j-pairs', 'j-count'):
        full = case('JK'); a['pair_vectors'] = full['pair_vectors']; a['transfer_offsets'] = full['transfer_offsets']
    elif fault == 'j-duplicate-k':
        a['momenta'][1] = a['momenta'][0]
    elif fault == 'j-gamma':
        a['gamma_vectors'] = a['gamma_vectors'][1:].copy()
    elif fault == 'k-empty':
        a['pair_vectors'] = np.empty((0, 3)); a['transfer_offsets'][:] = 0
    elif fault == 'k-diagonal':
        a['transfer_offsets'][1] = 1
    elif fault == 'k-reverse':
        a['pair_vectors'][0] += a['system'].reciprocal_lattice()[:, 0]
    with pytest.raises((ValueError, RuntimeError)):
        plan(a) if fault == 'j-count' else manta.PeriodicRIFockBuilder(**a)
