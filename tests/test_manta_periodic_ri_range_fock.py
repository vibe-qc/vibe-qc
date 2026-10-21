"""Whole-mesh physical range K with independent q sources and fixed projectors."""
import numpy as np
import pytest

from vibeqc import manta
from tests.test_manta_periodic_ri_targets import case as target_case, oracle
from tests.test_manta_periodic_ri_q import densities
from tests.test_manta_periodic_range_fitting import physical, reference, scf_calculation
from tests.test_manta_periodic_fitting import CAPS


def case(kernel='erfc', nk=2, **kw):
    a = target_case('K', nk, **kw); a['exchange_kernel'] = physical(kernel, .43)
    return a


def plan(a):
    return manta.plan_periodic_ri_fock(a['orbital'], a['auxiliary'], a['system'],
        len(a['gamma_vectors']), int(max(np.diff(a['transfer_offsets']))), len(a['momenta']),
        a['options'], a['ri_options'], a['budget'], False, True, a['exchange_kernel'])


def independent(a, h, d, scale):
    return oracle(a, h, d, scale, lambda q: reference(dict(q, physical_kernel=a['exchange_kernel'])))


@pytest.mark.parametrize('kernel', ['erfc', 'erf'])
@pytest.mark.parametrize('nk', [1, 2, 4])
@pytest.mark.parametrize('cutoff', [1e-12, 10., 1e9])
def test_all_channels_weighted_exchange_and_complete_component_energy(kernel, nk, cutoff):
    a = case(kernel, nk, cutoff=cutoff)
    if nk == 4: a['weights'] = np.array([.2, 0., .3, .5])
    b = manta.PeriodicRIFockBuilder(**a); h = densities(nk, spins=1, seed=311)[0]; d = densities(nk)
    r = b.build(h, d, -.32); _, k, f, e = independent(a, h, d, -.32)
    np.testing.assert_allclose(r.K, k, atol=6e-12, rtol=3e-12)
    np.testing.assert_allclose(r.values, f, atol=6e-12, rtol=3e-12)
    np.testing.assert_allclose([r.receipt.one_electron_energy, r.receipt.coulomb_energy,
        r.receipt.exchange_energy, r.receipt.electronic_energy], e, atol=7e-12, rtol=3e-12)
    assert r.J.shape == (0, 2, 2) and not r.receipt.has_j and r.receipt.coulomb_energy == 0.
    assert b.plan.channels == len(r.components) == 1+nk*(nk-1)
    for record in (b.info, r.receipt):
        assert record.exchange_kernel == kernel and record.physical_omega == .43
    assert all(c.kernel == kernel and c.physical_omega == .43 and not c.has_j for c in r.components)
    assert r.components[0].zero_transfer and all(not c.zero_transfer for c in r.components[1:])
    assert r.components[0].physical_zero_included == (kernel == 'erfc')
    assert max(b.info.reverse_metric_residual, b.info.reverse_projector_residual, b.info.reverse_inverse_residual) < 2e-12
    assert r.receipt.shell_calls <= b.plan.build_shell_calls


@pytest.mark.parametrize('kernel', ['erfc', 'erf'])
def test_weighted_variation_reciprocity_and_density_source_separation(kernel):
    a = case(kernel, 4); b = manta.PeriodicRIFockBuilder(**a)
    h = densities(4, spins=1, seed=91)[0]; d = densities(4); delta = densities(4, seed=92); step = 1e-5
    r = b.build(h, d, .25); rp = b.build(h, d+step*delta, .25); rm = b.build(h, d-step*delta, .25)
    derivative = (rp.receipt.electronic_energy-rm.receipt.electronic_energy)/(2*step)
    assert derivative == pytest.approx(np.einsum('k,skij,skji->', a['weights'], delta, r.values).real, abs=4e-8)
    response = b.build(np.zeros_like(h), delta, .25)
    assert np.einsum('k,skij,skji->', a['weights'], delta, r.values-h).real == pytest.approx(
        np.einsum('k,skij,skji->', a['weights'], d, response.values).real, abs=7e-12)
    a['exchange_kernel'].omega = .58; other = manta.PeriodicRIFockBuilder(**a).build(h, d, .25)
    assert other.receipt.source_identity != r.receipt.source_identity
    assert other.receipt.density_identity == r.receipt.density_identity
    assert other.receipt.frame_identity == r.receipt.frame_identity
    assert all(c.projector_identity == p.projector_identity == m.projector_identity for c, p, m in zip(r.components, rp.components, rm.components))
    np.testing.assert_array_equal(b.build(h, d, .25).values, r.values)


@pytest.mark.parametrize('kernel', ['erfc', 'erf'])
def test_image_gauge_with_shared_transport(kernel):
    from vibeqc.symmetry_shared import BlockSpaceAction, Budget, SpaceIdentity
    a = case(kernel); moved = case(kernel, shift=True)
    b = manta.PeriodicRIFockBuilder(**a); shifted = manta.PeriodicRIFockBuilder(**moved)
    h = densities(spins=1, seed=77)[0]; d = densities()
    phase = np.exp(1j*np.stack([np.zeros(2), a['momenta']@a['system'].lattice[:, 0]], axis=1))
    factors = phase[:, :, None]*phase.conj()[:, None, :]
    r = b.build(h, d); other = shifted.build(h*factors, d*factors)
    np.testing.assert_allclose(other.values, r.values*factors, atol=6e-12, rtol=0)
    assert other.receipt.electronic_energy == pytest.approx(r.receipt.electronic_energy, abs=6e-12)
    for k in range(2):
        source = SpaceIdentity('range-fit-test', b.info.frame_identity, 'AO', f'input-k{k}')
        target = SpaceIdentity('range-fit-test', shifted.info.frame_identity, 'AO', f'image-k{k}')
        action = BlockSpaceAction(source, target, np.array([0, 1, 2], np.int64), np.array([0, 1], np.int64),
            np.ascontiguousarray(phase[k]), antiunitary=False, budget=Budget(2**20, 10**6))
        for spin in range(2):
            np.testing.assert_allclose(action.push_density(d[spin, k]), d[spin, k]*factors[k], atol=2e-14, rtol=0)
            np.testing.assert_allclose(action.pull_operator(other.values[spin, k]), r.values[spin, k], atol=6e-12, rtol=0)


@pytest.mark.parametrize('kernel', ['erfc', 'erf'])
@pytest.mark.parametrize('field,quantity', CAPS)
def test_whole_mesh_caps_before_source_copies(kernel, field, quantity):
    a = case(kernel, 4); p = plan(a)
    for cap, bound in CAPS: setattr(a['budget'], cap, getattr(p, bound))
    manta.PeriodicRIFockBuilder(**a).build(densities(4, spins=1)[0], densities(4))
    setattr(a['budget'], field, getattr(p, quantity)-1)
    for name in ('momenta', 'weights', 'gamma_vectors', 'pair_vectors'): a[name][:] = np.nan
    with pytest.raises((ValueError, RuntimeError), match='budget'): manta.PeriodicRIFockBuilder(**a)


@pytest.mark.parametrize('kernel', ['erfc', 'erf'])
def test_screened_metric_cannot_silently_replace_coulomb_j(kernel):
    a = case(kernel); a['make_j'] = True
    with pytest.raises(ValueError, match='separate Coulomb J'): manta.PeriodicRIFockBuilder(**a)


@pytest.mark.parametrize('kernel', ['erfc', 'erf'])
@pytest.mark.parametrize('spins', [1, 2])
def test_complete_shifted_two_k_scf(kernel, spins):
    scf_calculation(kernel, spins, whole_mesh=True)
