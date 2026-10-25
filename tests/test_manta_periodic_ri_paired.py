"""Paired periodic RI J: finite-source adjoints, fixed factors and method state."""
import dataclasses
import itertools

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from vibeqc.guess_read import _spin_density_k_from_qvf_bloch_payload
from tests.test_manta_periodic_fitting import CAPS
from tests.test_manta_periodic_gdf import case, independent, legacy
from tests.test_manta_periodic_ri_q import builder, densities, reference


def auxiliary(system, centers):
    shells = [core.ShellInfo(i, 0, True, [e], [1.], center)
              for i, center in enumerate(centers) for e in (.5, 1.2, 3.)]
    return core.BasisSet(system.unit_cell_molecule(), shells, 'paired-ri-source', True)


def paired_case(shift=False):
    a = case(True, shift)
    a['auxiliary'] = auxiliary(a['system'], [s.origin for s in a['auxiliary'].shells()])
    # Converge the finite metric; a small auxiliary domain can be indefinite.
    a['options'].auxiliary_cutoff_bohr = 8.
    a['vectors'] = np.array(list(itertools.product(range(-2, 3), repeat=3))) @ a['system'].reciprocal_lattice().T
    return a


@pytest.mark.parametrize('spins', [1, 2])
@pytest.mark.parametrize('rank', [0, 1, 2, 3, 6])
def test_fixed_rank_route_agrees_with_independent_sources_and_retained_mode(spins, rank):
    a = paired_case(); full = builder(a, spins, 1)
    eigenvalues = np.linalg.eigvalsh(full.metric)
    cutoff = 1e-12 if rank == 6 else (2*eigenvalues[-1] if rank == 0 else .5*(eigenvalues[-rank-1]+eigenvalues[-rank]))
    b = builder(a, spins, 1, absolute_cutoff=cutoff)
    control = builder(a, spins, 6, absolute_cutoff=cutoff)
    d = densities(spins=spins); w = np.array([.27, .73])
    r = b.build(d, w, make_k=False); retained = control.build(d, w, make_k=False)
    assert b.metric_info.rank == control.metric_info.rank == rank
    assert r.receipt.coulomb_contraction == ('paired-raw' if rank > 2 else 'retained-mode')
    assert r.receipt.coupling_passes == (2 if rank > 2 else max(1, rank))
    assert retained.receipt.coulomb_contraction == 'retained-mode'
    for name in ('metric', 'factor', 'projector'):
        np.testing.assert_array_equal(getattr(b, name), getattr(control, name))
    for name in ('canonical_metric_identity', 'factor_identity', 'projector_identity', 'source_identity'):
        assert getattr(b.metric_info, name) == getattr(control.metric_info, name)
    assert r.receipt.snapshot_identity == retained.receipt.snapshot_identity
    np.testing.assert_allclose(r.J, retained.J, atol=2e-12, rtol=2e-12)
    for m, t in (independent(a), legacy(a)[:2]):
        j, _, energy, p, residual = reference(m, t, d, w, absolute=cutoff)
        np.testing.assert_allclose(r.J, j, atol=5e-12, rtol=3e-12)
        np.testing.assert_allclose(b.projector, p, atol=4e-12, rtol=0)
        assert r.receipt.coulomb_energy == pytest.approx(energy, abs=5e-12)
        assert r.receipt.metric_residual_norm == pytest.approx(residual, abs=5e-12)
    assert r.K.shape == (0, 2, 2, 2)
    assert r.receipt.shell_calls+b.metric_info.shell_calls <= b.plan.shell_calls
    assert r.receipt.primitive_products+b.metric_info.primitive_products <= b.plan.primitive_products


@pytest.mark.parametrize('weights', [[.37, .63], [0., 1.]])
def test_paired_adjoint_energy_derivative_linearity_and_density_independent_route(weights):
    b = builder(paired_case()); d = densities(); delta = densities(seed=231)
    w = np.array(weights); step = 2e-5
    def build(density):
        r = b.build(density, w, make_k=False)
        assert r.receipt.coulomb_contraction == 'paired-raw' and r.receipt.coupling_passes == 2
        return r
    r, probe = build(d), build(delta)
    derivative = (build(d+step*delta).receipt.coulomb_energy-build(d-step*delta).receipt.coulomb_energy)/(2*step)
    action = np.einsum('k,skij,kji->', w, delta, r.J).real
    assert derivative == pytest.approx(action, abs=2e-9)
    assert action == pytest.approx(np.einsum('k,skij,kji->', w, d, probe.J).real, abs=5e-12)
    assert r.receipt.coulomb_energy == pytest.approx(.5*np.einsum('k,skij,kji->', w, d, r.J).real, abs=5e-12)
    np.testing.assert_allclose(build(2*d-delta).J, 2*r.J-probe.J, atol=5e-12, rtol=0)
    np.testing.assert_array_equal(build(np.zeros_like(d)).J, np.zeros_like(r.J))
    assert build(delta).receipt.source_identity == r.receipt.source_identity


@pytest.mark.parametrize('rank', [3, 6])
def test_image_gauge_transports_paired_density_and_response_with_fixed_rank(rank):
    a = paired_case(); shifted = paired_case(True)
    e = np.linalg.eigvalsh(builder(a).metric)
    cutoff = .5*(e[2]+e[3]) if rank == 3 else 1e-12
    b = builder(a, absolute_cutoff=cutoff); moved = builder(shifted, absolute_cutoff=cutoff)
    assert b.metric_info.rank == moved.metric_info.rank == rank
    d = densities(); w = np.array([.21, .79]); moved_d = d.copy()
    phases = np.exp(1j*a['ket_kpoints'] @ a['system'].lattice[:, 0])
    for ik, phase in enumerate(phases):
        u = np.array([1., phase]); moved_d[:, ik] *= u[None, :, None]*u.conj()[None, None, :]
    r = b.build(d, w, make_k=False); other = moved.build(moved_d, w, make_k=False)
    assert r.receipt.coulomb_contraction == other.receipt.coulomb_contraction == 'paired-raw'
    for ik, phase in enumerate(phases):
        u = np.array([1., phase])
        np.testing.assert_allclose(other.J[ik], r.J[ik]*u[:, None]*u.conj()[None, :], atol=5e-12, rtol=0)
    assert other.receipt.coulomb_energy == pytest.approx(r.receipt.coulomb_energy, abs=5e-12)
    assert r.receipt.source_frame_identity != other.receipt.source_frame_identity


def test_joint_jk_and_k_only_keep_retained_mode_exchange_and_all_domain_guards():
    a = paired_case(); b = builder(a); d = densities(); w = np.array([.27, .73])
    joint = b.build(d, w); exchange = b.build(d, w, make_j=False)
    j = b.build(d, w, make_k=False)
    assert joint.receipt.coulomb_contraction == 'retained-mode'
    assert exchange.receipt.coulomb_contraction == 'none'
    assert joint.receipt.coupling_passes == exchange.receipt.coupling_passes == 6
    np.testing.assert_array_equal(joint.K, exchange.K)
    np.testing.assert_allclose(joint.J, j.J, atol=2e-12, rtol=2e-12)
    with pytest.raises(ValueError, match='sum to one'):
        b.build(d, np.array([.2, .3]), make_k=False)
    a['q'] = np.array([.17, -.08, .11]); a['vectors'] += a['q']
    nonzero = builder(a)
    with pytest.raises(ValueError, match='Gamma transfer'):
        nonzero.build(d, w, make_k=False)


@pytest.mark.parametrize('field,quantity', CAPS)
def test_paired_route_respects_existing_joint_admission(field, quantity):
    a = paired_case(); source = manta.PeriodicGDFTiles(**a)
    options = manta.RIOptions(); options.mode_block = 1
    budget = a['budget']; budget.other_live_bytes = 321
    p = manta.plan_periodic_ri_q(source, 2, options, budget)
    for cap, value in CAPS:
        setattr(budget, cap, getattr(p, value))
    b = manta.PeriodicRIQBuilder(source, 2, options, budget)
    r = b.build(densities(), np.array([.4, .6]), make_k=False)
    assert r.receipt.coupling_passes == 2 < p.maximum_coupling_passes
    assert r.receipt.modeled_peak_bytes == p.modeled_peak_bytes
    setattr(budget, field, getattr(p, quantity)-1)
    with pytest.raises((ValueError, RuntimeError), match='budget'):
        manta.PeriodicRIQBuilder(source, 2, options, budget)


def paired_job(directory, method='uhf', species='Li', limit=100, block=1):
    from tests.test_manta_periodic_ri_job import job_case as spin_job
    from tests.test_manta_periodic_ri_rhf import job_case as restricted_job
    make = restricted_job if method == 'rhf' else spin_job
    job, a, k = make(directory, (2, 1, 1), (1, 0, 0), species, 'BvK-ewald')
    a['auxiliary'] = auxiliary(a['system'], [atom.xyz for atom in a['system'].unit_cell])
    a['options'].ri.mode_block = block
    a['options'].exchange_algorithm = 'exact'; a['options'].scf.maximum_iterations = limit
    job['coulomb'] = dataclasses.replace(job['coulomb'], auxiliary_basis=a['auxiliary'], metric=a['options'].ri)
    job['exchange'] = manta.ExchangeBuildOptions(algorithm='exact'); job['max_iter'] = limit
    return job, a, k


@pytest.mark.parametrize('method,species,limit', [('rhf', 'Be', 100), ('uhf', 'H', 100), ('uhf', 'Li', 3)])
def test_complete_public_mixed_hf_keeps_shared_fock_and_evaluated_qvf_state(tmp_path, method, species, limit):
    from tests.test_manta_periodic_mixed_hf import reference as method_reference
    from tests.test_manta_periodic_ri_domains import explicit
    from tests.test_manta_periodic_ri_job import _as_fock_build
    from tests.test_manta_periodic_rhf import _nuclear_reference
    job, a, k = paired_job(tmp_path, method, species, limit)
    result = vq.run_periodic_job(**job)
    assert result.converged == (limit == 100)
    domains = getattr(manta, 'build_periodic_ri_'+method+'_domains')(**a)
    c = explicit(a, domains); _, _, build = method_reference(c, k, method)
    assert result.native.receipt.fitting_info.metrics[0].rank == 3
    np.testing.assert_allclose(result.native.fock, build(result.native.density)[0], atol=8e-11, rtol=0)
    assert result.energy == pytest.approx(sum(_nuclear_reference(c)[0])+sum(build(result.native.density)[1]), abs=8e-11)
    state = _as_fock_build(result)
    assert state.energy.total == pytest.approx(result.energy, abs=1e-13)
    assert all(ch.operator.metadata.symmetry_block is None for ch in state.channels)
    d = result.native.density
    saved = _spin_density_k_from_qvf_bloch_payload(job['output'].with_suffix('.qvf'))
    np.testing.assert_array_equal(saved, np.stack([.5*d, .5*d]) if method == 'rhf' else d)


def paired_ks_job(directory, spins=1, limit=100):
    from tests.test_manta_periodic_ri_ks_job import job_case
    job, args, a, k, old = job_case(directory, spins=spins)
    args['options'].ri.mode_block = 1
    args['options'].ks.scf.maximum_iterations = limit
    job['max_iter'] = limit
    return job, args, a, k, old


@pytest.mark.parametrize('spins', [1, 2])
def test_pure_ks_paired_j_preserves_functional_energy_and_evaluated_state(tmp_path, spins):
    from tests.test_manta_periodic_ri_ks_domains import explicit
    from tests.test_manta_periodic_ri_ks import reference as ks_reference, components, as_fock_build
    from tests.test_manta_periodic_rhf import _nuclear_reference
    job, args, a, k, old = paired_ks_job(tmp_path, spins, limit=3)
    explicit(args, a, old)
    result = vq.run_periodic_job(**job); n = result.native
    assert n.receipt.fitting_info.metrics[0].rank == 3
    assert n.receipt.fitted_fock.coupling_passes == 2
    assert not result.converged and not n.receipt.confirmed
    _, _, build = ks_reference(a, k, old)
    f, energies, _ = build(n.density)
    np.testing.assert_allclose(n.fock, f, atol=8e-11, rtol=0)
    np.testing.assert_allclose(components(n), energies, atol=8e-11, rtol=0)
    assert result.energy == pytest.approx(sum(_nuclear_reference(old)[0])+sum(energies), abs=8e-11)
    expected = n.density if spins == 2 else np.repeat(.5*n.density, 2, axis=0)
    np.testing.assert_array_equal(_spin_density_k_from_qvf_bloch_payload(job['output'].with_suffix('.qvf')), expected)
    assert as_fock_build(n, k).energy.total == pytest.approx(result.energy, abs=1e-13)
