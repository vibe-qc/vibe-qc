"""Fixed finite metric with a source-owned, bounded screened coupling."""
import dataclasses
import itertools

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from vibeqc.guess_read import _spin_density_k_from_qvf_bloch_payload
from tests.test_manta_periodic_fitting import CAPS
from tests.test_manta_periodic_gdf import case, independent, legacy, quadrature_long_range
from tests.test_manta_periodic_gdf_actions import inputs
from tests.test_manta_periodic_ri_q import builder, densities, reference
from tests.test_manta_periodic_ri_paired import paired_case, paired_job


def screened_libint(a):
    """Existing libint SR selector plus unchanged independent LR/source metric."""
    o = a['options']
    exact_m, exact_t, _ = legacy(a)  # Legacy reads only common screen, still zero.
    _, lr_t = quadrature_long_range(a)
    st = core.compute_gdf_sr_three_center(a['orbital'], a['auxiliary'], a['system'],
        a['q'], a['ket_kpoints'], o.omega, o.pair_cutoff_bohr, o.auxiliary_cutoff_bohr,
        2**28, o.image_candidate_cap, 2**28, integral_screen_error=o.coupling_screen_error)
    return exact_m, np.asarray(st)+lr_t, exact_t


@pytest.mark.parametrize('gamma', [False, True])
@pytest.mark.parametrize('screen', [1e-6, .1])
def test_ordered_complex_actions_keep_metric_and_bound_omitted_source(gamma, screen):
    a = case(gamma); exact = manta.PeriodicGDFTiles(**a)
    a['options'].coupling_screen_error = screen
    source = manta.PeriodicGDFTiles(**a); dense = manta.compute_periodic_gdf(**a)
    x, c = inputs(source)
    f, adj = source.forward(x, a['budget']), source.adjoint(c, a['budget'])
    m, t, exact_t = screened_libint(a)
    np.testing.assert_array_equal(source.metric().values, exact.metric().values)
    assert source.metric_source_identity == exact.metric_source_identity
    assert source.three_center_source_identity != exact.three_center_source_identity
    assert source.receipt.reference_three_center_source_identity == exact.three_center_source_identity
    assert dense.receipt.reference_three_center_source_identity == exact.three_center_source_identity
    np.testing.assert_allclose(dense.three_center, t, atol=3e-12, rtol=3e-13)
    np.testing.assert_allclose(f.values, np.einsum('kpij,kij->p', t.conj(), x), atol=4e-12, rtol=3e-13)
    np.testing.assert_allclose(adj.values, np.einsum('kpij,p->kij', t, c), atol=4e-12, rtol=3e-13)
    assert abs(np.vdot(c, f.values)-np.vdot(adj.values, x)) < 4e-12
    assert np.max(abs(t-exact_t)) <= screen+3e-12
    for r, expected, trial in ((f, np.einsum('kpij,kij->p', exact_t.conj(), x), x),
                               (adj, np.einsum('kpij,p->kij', exact_t, c), c)):
        assert np.max(abs(r.values-expected)) <= r.receipt.approximation_bound+4e-12
        assert r.receipt.input_l1_bound >= np.sum(abs(trial))
        assert r.receipt.reference_coupling_identity == exact.three_center_source_identity
        assert r.receipt.coupling_error_bound == screen
    calls = exact.forward(x, a['budget']).receipt.shell_calls
    assert f.receipt.shell_calls <= calls
    if screen == .1:
        assert f.receipt.shell_calls < calls
    # Source owns its controls; changing the caller cannot change either map.
    a['options'].coupling_screen_error = 0.
    np.testing.assert_array_equal(source.forward(x, a['budget']).values, f.values)


@pytest.mark.parametrize('rank', [0, 1, 3, 6])
@pytest.mark.parametrize('weights', [[.27, .73], [0., 1.]])
def test_fixed_factor_and_density_specific_j_energy_bounds(rank, weights):
    a = paired_case(); full = builder(a); eigenvalues = np.linalg.eigvalsh(full.metric)
    cutoff = 1e-12 if rank == 6 else (2*eigenvalues[-1] if rank == 0 else .5*(eigenvalues[-rank-1]+eigenvalues[-rank]))
    exact = builder(a, block=6, absolute_cutoff=cutoff)
    a['options'].coupling_screen_error = .1
    b = builder(a, block=6, absolute_cutoff=cutoff)
    d, w = densities(), np.array(weights)
    r = b.build(d, w, make_k=False); e = exact.build(d, w, make_k=False)
    assert r.receipt.coulomb_contraction == 'paired-screened'
    assert r.receipt.coupling_passes == b.plan.maximum_coupling_passes == 2
    for name in ('metric', 'factor', 'projector'):
        np.testing.assert_array_equal(getattr(b, name), getattr(exact, name))
    for name in ('metric_source_identity', 'canonical_metric_identity', 'factor_identity', 'projector_identity'):
        assert getattr(b.metric_info, name) == getattr(exact.metric_info, name)
    assert r.receipt.source_identity != e.receipt.source_identity
    assert np.max(abs(r.J-e.J)) <= r.receipt.approximation_j_bound+5e-12
    assert abs(r.receipt.coulomb_energy-e.receipt.coulomb_energy) <= r.receipt.approximation_energy_bound+5e-12
    m, t, _ = screened_libint(a)
    j, _, energy, _, _ = reference(m, t, d, w, absolute=cutoff)
    np.testing.assert_allclose(r.J, j, atol=5e-12, rtol=3e-12)
    assert r.receipt.coulomb_energy == pytest.approx(energy, abs=5e-12)
    assert r.receipt.metric_inverse_norm_bound >= np.linalg.norm(b.factor)**2
    delta = densities(seed=231); step = 2e-5
    def build(d): return b.build(d, w, make_k=False)
    derivative = (build(d+step*delta).receipt.coulomb_energy-build(d-step*delta).receipt.coulomb_energy)/(2*step)
    assert derivative == pytest.approx(np.einsum('k,skij,kji->', w, delta, r.J).real, abs=2e-9)
    assert r.receipt.coulomb_energy == pytest.approx(.5*np.einsum('k,skij,kji->', w, d, r.J).real, abs=5e-12)
    zero = build(np.zeros_like(d))
    assert zero.receipt.approximation_j_bound == zero.receipt.approximation_energy_bound == 0.
    if not rank:
        assert r.receipt.approximation_j_bound == r.receipt.approximation_energy_bound == 0.
        np.testing.assert_array_equal(r.J, np.zeros_like(r.J))


@pytest.mark.parametrize('screen', [-1., float('nan'), float('inf')])
def test_invalid_coupling_controls_refused(screen):
    a = case(); a['options'].coupling_screen_error = screen
    with pytest.raises(ValueError, match='coupling_screen_error'):
        manta.PeriodicGDFTiles(**a)
    with pytest.raises(ValueError, match='coupling_screen_error'):
        manta.PeriodicFittingOptions(.65, 8., coupling_screen_error=screen)


def test_common_screen_cannot_be_mislabeled_as_fixed_metric_and_exchange_is_refused():
    a = paired_case(); a['options'].coupling_screen_error = .1
    b = builder(a)
    with pytest.raises(ValueError, match='independent exchange'):
        b.build(densities(), np.array([.5, .5]))
    with pytest.raises(ValueError, match='independent exchange'):
        b.build(densities(), np.array([.5, .5]), make_j=False)
    a['options'].integral_screen_error = 1e-6
    with pytest.raises(ValueError, match='cannot combine'):
        manta.PeriodicGDFTiles(**a)
    with pytest.raises(ValueError, match='cannot combine'):
        manta.PeriodicFittingOptions(.65, 8., 1e-6, coupling_screen_error=.1)


@pytest.mark.parametrize('field,quantity', CAPS)
def test_screened_pair_has_joint_admission_even_for_one_mode_block(field, quantity):
    a = paired_case(); a['options'].coupling_screen_error = .1
    source = manta.PeriodicGDFTiles(**a); options = manta.RIOptions(); options.mode_block = 6
    budget = a['budget']; p = manta.plan_periodic_ri_q(source, 2, options, budget)
    for cap, value in CAPS: setattr(budget, cap, getattr(p, value))
    b = manta.PeriodicRIQBuilder(source, 2, options, budget)
    assert b.build(densities(), np.array([.5, .5]), make_k=False).receipt.coupling_passes == 2
    setattr(budget, field, getattr(p, quantity)-1)
    with pytest.raises((ValueError, RuntimeError), match='budget'):
        manta.PeriodicRIQBuilder(source, 2, options, budget)


def screened_hf_reference(a, k, method):
    from tests.test_manta_periodic_mixed_hf import reference as hf_reference
    s, h, exact = hf_reference(a, k, method)
    common = dict(orbital=a['basis'], auxiliary=a['auxiliary'], system=a['system'],
        options=a['options'].fitting, budget=a['budget'], q=np.zeros(3), ket_kpoints=k,
        vectors=a['reciprocal_vectors'])
    m, t, exact_t = screened_libint(common); w = np.full(len(k), 1./len(k))
    def build(d):
        spin = np.stack([.5*d, .5*d]) if method == 'rhf' else d
        f, energy = exact(d)
        j, _, ej, _, _ = reference(m, t, spin, w)
        original_j = reference(m, exact_t, spin, w)[0]
        return f+j-original_j, (energy[0], ej, *energy[2:])
    return s, h, build, exact, (m, t, exact_t)


def screened_job(directory, method='uhf', species='Li', limit=100, screen=.1):
    job, a, k = paired_job(directory, method, species, limit, block=6)
    a['options'].fitting.coupling_screen_error = screen
    job['periodic_fitting'] = dataclasses.replace(job['periodic_fitting'], coupling_screen_error=screen)
    return job, a, k


def screened_ks_reference(a, k, old):
    from tests.test_manta_periodic_ri_ks import reference as ks_reference
    s, h, exact = ks_reference(a, k, old)
    common = dict(orbital=a['basis'], auxiliary=a['auxiliary'], system=a['system'],
        options=a['options'].fitting, budget=a['budget'], q=np.zeros(3), ket_kpoints=k,
        vectors=a['reciprocal_vectors'])
    m, t, exact_t = screened_libint(common); w = np.full(len(k), 1./len(k))
    def build(d):
        spin = np.repeat(.5*d, 2, axis=0) if a['functional'].spin == 1 else d
        f, energies, populations = exact(d)
        j, _, ej, _, _ = reference(m, t, spin, w)
        original_j = reference(m, exact_t, spin, w)[0]
        return f+j-original_j, (energies[0], ej, *energies[2:]), populations
    return s, h, build, exact, (m, t, exact_t)


@pytest.mark.parametrize('spins', [1, 2])
def test_public_pure_ks_keeps_xc_and_screened_component_receipt(tmp_path, spins):
    from tests.test_manta_periodic_ri_paired import paired_ks_job
    from tests.test_manta_periodic_ri_ks_domains import explicit
    from tests.test_manta_periodic_ri_ks import components, as_fock_build
    from tests.test_manta_periodic_rhf import _nuclear_reference
    job, args, a, k, old = paired_ks_job(tmp_path, spins, limit=3)
    args['options'].fitting.coupling_screen_error = .1
    job['periodic_fitting'] = dataclasses.replace(job['periodic_fitting'], coupling_screen_error=.1)
    explicit(args, a, old)
    result = vq.run_periodic_job(**job); n = result.native
    _, _, build, exact, _ = screened_ks_reference(a, k, old)
    f, energies, _ = build(n.density); ef, ee, _ = exact(n.density)
    np.testing.assert_allclose(n.fock, f, atol=8e-11, rtol=0)
    np.testing.assert_allclose(components(n), energies, atol=8e-11, rtol=0)
    assert result.energy == pytest.approx(sum(_nuclear_reference(old)[0])+sum(energies), abs=8e-11)
    r = result.integral_provenance['fitted_fock']
    assert r['coupling_error_bound'] == .1
    assert np.max(abs(n.fock-ef)) <= r['approximation_j_bound']+8e-11
    assert abs(sum(energies)-sum(ee)) <= r['approximation_energy_bound']+8e-11
    assert as_fock_build(n, k).energy.total == pytest.approx(result.energy, abs=1e-13)
    saved = _spin_density_k_from_qvf_bloch_payload(job['output'].with_suffix('.qvf'))
    np.testing.assert_array_equal(saved, np.repeat(.5*n.density, 2, axis=0) if spins == 1 else n.density)


@pytest.mark.parametrize('angular,pure', [(1, True), (2, False)])
def test_signed_angular_coupling_omission_against_independent_blocks(angular, pure):
    from tests.test_aux_eri_sr_lattice_enumeration import _weighted_gradient_fixture
    from tests.test_manta_periodic_fitting import libint_reference
    a = case(); s, orb, aux = _weighted_gradient_fixture(angular, orbital_angular=1)
    def change(basis):
        shells = basis.shells()
        for sh in shells: sh.pure = pure
        return vq.BasisSet(s.unit_cell_molecule(), shells, 'screened-angular', True)
    a.update(system=s, orbital=change(orb), auxiliary=change(aux),
             vectors=np.array(list(itertools.product(range(-1, 2), repeat=3)))@s.reciprocal_lattice().T+a['q'])
    a['options'].pair_cutoff_bohr = 4.3; a['options'].auxiliary_cutoff_bohr = 5.1
    _, exact_sr = libint_reference(a); _, lr = quadrature_long_range(a)
    a['options'].coupling_screen_error = .1
    source = manta.PeriodicGDFTiles(**a); x, c = inputs(source)
    f = source.forward(x, a['budget']); adj = source.adjoint(c, a['budget'])
    assert abs(np.vdot(c, f.values)-np.vdot(adj.values, x)) < 1e-10
    assert np.max(abs(f.values-np.einsum('kpij,kij->p', (exact_sr+lr).conj(), x))) <= f.receipt.approximation_bound+1e-10
    assert np.max(abs(adj.values-np.einsum('kpij,p->kij', exact_sr+lr, c))) <= adj.receipt.approximation_bound+1e-10


def test_screened_q0_image_transport_and_nongeometric_symmetry_ownership():
    a = paired_case(); moved = paired_case(True)
    a['options'].coupling_screen_error = moved['options'].coupling_screen_error = .1
    b, other = builder(a), builder(moved)
    d = densities(); md = d.copy(); w = np.array([.31, .69])
    phases = np.exp(1j*a['ket_kpoints']@a['system'].lattice[:, 0])
    for k, phase in enumerate(phases):
        u = np.array([1., phase]); md[:, k] *= u[None, :, None]*u.conj()[None, None, :]
    r = b.build(d, w, make_k=False); mr = other.build(md, w, make_k=False)
    for k, phase in enumerate(phases):
        u = np.array([1., phase])
        np.testing.assert_allclose(mr.J[k], r.J[k]*u[:, None]*u.conj()[None, :], atol=5e-12, rtol=0)
    assert mr.receipt.coulomb_energy == pytest.approx(r.receipt.coulomb_energy, abs=5e-12)
    assert mr.receipt.source_frame_identity != r.receipt.source_frame_identity


@pytest.mark.parametrize('method,species,limit', [('rhf', 'Be', 100), ('uhf', 'H', 100), ('uhf', 'Li', 3)])
def test_public_method_keeps_unified_fock_evaluated_state_and_bound_provenance(tmp_path, method, species, limit):
    from tests.test_manta_periodic_ri_domains import explicit
    from tests.test_manta_periodic_ri_job import _as_fock_build
    from tests.test_manta_periodic_rhf import _nuclear_reference
    job, args, k = screened_job(tmp_path, method, species, limit)
    result = vq.run_periodic_job(**job); n = result.native
    assert result.converged == (limit == 100)
    domains = getattr(manta, 'build_periodic_ri_'+method+'_domains')(**args)
    a = explicit(args, domains); _, _, build, exact, _ = screened_hf_reference(a, k, method)
    f, energies = build(n.density); ef, ee = exact(n.density)
    np.testing.assert_allclose(n.fock, f, atol=8e-11, rtol=0)
    assert result.energy == pytest.approx(sum(_nuclear_reference(a)[0])+sum(energies), abs=8e-11)
    bound = result.integral_provenance['coupling_approximation']
    assert bound['coupling_error_bound'] == .1 and len(bound['reference_coupling_identity']) == 64
    assert np.max(abs(n.fock-ef)) <= bound['approximation_j_bound']+8e-11
    assert abs(sum(energies)-sum(ee)) <= bound['approximation_energy_bound']+8e-11
    state = _as_fock_build(result)
    assert all(ch.operator.metadata.symmetry_block is None for ch in state.channels)
    assert state.energy.total == pytest.approx(result.energy, abs=1e-13)
    saved = _spin_density_k_from_qvf_bloch_payload(job['output'].with_suffix('.qvf'))
    np.testing.assert_array_equal(saved, np.stack([.5*n.density]*2) if method == 'rhf' else n.density)
