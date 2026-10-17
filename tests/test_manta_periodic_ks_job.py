"""Public pure-KS jobs: independent physics, immutable epochs and output state."""
import dataclasses
import json
import tomllib
import zipfile

import numpy as np
import pytest
from scipy.linalg import eigh

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from vibeqc.guess_read import _spin_density_k_from_qvf_bloch_payload
from tests.test_manta_periodic_job import _qvf, _no_computed_artifacts
from tests.test_manta_periodic_ks_grid import case, explicit
from tests.test_manta_periodic_ks import oracle, independent_scf
from tests.test_manta_periodic_rhf import _nuclear_reference


def job_case(tmp_path, species='H2', spins=1, stratmann=False):
    args, c, k, old = case(species, spins, stratmann)
    o, g = args['options'], args['grid_options']
    job = dict(system=args['system'], basis=args['basis'], kpoints=args['mesh'],
        method='RKS' if spins == 1 else 'UKS', functional='blyp' if species == 'H' else 'pbe',
        integral_engine=manta.IntegralEngineOptions(backend='manta', budget=args['budget']),
        periodic_integrals=manta.PeriodicIntegralOptions(args['reciprocal_energy_cutoff'],
            o.pair_cutoff_bohr, o.nuclear_cutoff_bohr, alpha=o.alpha, reciprocal_batch=o.reciprocal_batch),
        periodic_xc=manta.PeriodicXCOptions(args['xc_image_cutoff_bohr'], g.radial_points,
            g.lebedev_order, g.partition_radius_bohr, g.stratmann, o.grid_batch),
        max_iter=o.scf.maximum_iterations, conv_tol_energy=o.scf.energy_change_tolerance,
        conv_tol_grad=o.scf.commutator_tolerance, use_diis=False,
        output=tmp_path/'selected'/'job', record_hostname=False, structured_log=True)
    return job, args, c, k, old


@pytest.mark.parametrize('species,spins,stratmann', [
    ('H2', 1, False), ('H2', 2, True), ('HeH', 2, False), ('H', 2, True)])
def test_complete_job_independent_scf_and_lossless_outputs(tmp_path, species, spins, stratmann):
    from vibeqc.output import validate_qvf
    job, args, c, k, old = job_case(tmp_path, species, spins, stratmann)
    grid, _ = explicit(args, c, old)
    s, h, build = oracle(c, k, old)
    result = vq.run_periodic_job(**job)
    native = result.native
    assert isinstance(result, manta.PeriodicMANTARKSResult if spins == 1 else manta.PeriodicMANTAUKSResult)
    assert result.converged and native.receipt.confirmed
    pop = (native.plan.n_alpha,) if spins == 1 else (native.plan.n_alpha, native.plan.n_beta)
    density = independent_scf(s, h, build, pop, spins)
    np.testing.assert_allclose(native.density, density, atol=3e-9, rtol=0)
    f, energies, populations = build(native.density)
    np.testing.assert_allclose(native.fock, f, atol=2e-11, rtol=0)
    np.testing.assert_allclose(native.overlap, s, atol=3e-12, rtol=0)
    np.testing.assert_allclose(native.hcore, h, atol=3e-12, rtol=0)
    np.testing.assert_allclose([native.receipt.one_electron_energy, native.receipt.coulomb_energy,
                               native.receipt.xc_energy], energies, atol=2e-11, rtol=0)
    nuclear, _ = _nuclear_reference(old)
    assert result.energy == pytest.approx(sum(nuclear)+sum(energies), abs=2e-11)
    np.testing.assert_allclose(native.receipt.xc.electrons_grid, populations, atol=2e-12, rtol=0)
    direct = manta.run_periodic_ks_with_quadrature(**args)
    assert native.receipt.source_identity == direct.receipt.source_identity
    np.testing.assert_array_equal(native.density, direct.density)

    path = job['output'].with_suffix('.qvf')
    expected = native.density if spins == 2 else np.repeat(.5*native.density, 2, axis=0)
    np.testing.assert_array_equal(_spin_density_k_from_qvf_bloch_payload(path), expected)
    assert validate_qvf(path)['valid']
    qvf = _qvf(path)
    assert qvf['provenance']['method'].upper() == job['method']
    assert qvf['provenance']['functional'] == job['functional']
    assert qvf['provenance']['exchange_q0'] == 'not-applicable'
    assert qvf['provenance']['run_status'] == 'converged'
    section = next(s for s in qvf['sections'] if s['kind'] == 'x_vibeqc.periodic_integral_engine')
    with zipfile.ZipFile(path) as archive:
        evidence = json.loads(archive.read(next(iter(section['members'].values()))['path']))
        assert json.loads(archive.read('scf_history/iterations.json'))['iterations'] == result.scf_history
    assert evidence == result.integral_provenance
    assert evidence['operator_kind'] == 'ks'
    assert evidence['operator_generation'] == result.iterations
    assert evidence['basis_frame_identity'] == native.receipt.overlap_identity
    assert evidence['accuracy_qualification'] == 'finite-source-only'
    assert evidence['quadrature']['source_identity'] == grid.receipt.source_identity
    assert evidence['xc']['component_ids'] == native.receipt.xc.component_ids
    assert evidence['exchange_q0_applicability'] == 'inactive'
    assert evidence['coulomb_algorithm'] == 'exact' and evidence['exchange_algorithm'] == 'none'
    if spins == 2:
        assert len(evidence['spin_scf_history']) == result.iterations
        if species == 'H':
            assert evidence['spin_scf_history'][-1]['spin_homo'][1] is None
            assert evidence['spin_scf_history'][-1]['spin_band_gaps'][1] is None
    manifest = tomllib.loads(job['output'].with_suffix('.system').read_text())
    for key in ('quadrature_source_identity', 'xc_source_identity', 'exchange_q0_applicability',
                'grid_radial_points', 'grid_lebedev_order', 'functional', 'symmetry_reduced'):
        assert manifest['run'][key] == result.run_fields[key]
    references = job['output'].with_suffix('.references').read_text()
    for name in ('Perdew' if species != 'H' else 'Lee', 'Becke', 'Treutler', 'Lebedev'):
        assert name in references
    if stratmann:
        assert 'Stratmann' in references
    records = [json.loads(line) for line in job['output'].with_suffix('.scf.jsonl').read_text().splitlines()]
    assert next(r['integral_provenance'] for r in records if r.get('event') == 'manta_periodic_ks') == evidence
    assert len([r for r in records if r.get('event') == 'scf_iteration']) == result.iterations
    assert job['integral_engine'].budget.other_live_bytes == 0

    properties = ['density_k', 'density_alpha_k', 'density_beta_k', 'overlap_k', 'hcore_k']
    properties += ([f'{key}_k' for key in ('fock', 'mo_coeffs', 'mo_energies', 'occupations')] if spins == 1 else
        [f'{key}_{spin}_k' for key in ('fock', 'mo_coeffs', 'mo_energies', 'occupations') for spin in ('alpha', 'beta')])
    for name in properties:
        before = np.array(getattr(result, name))
        getattr(result, name)[0][:] = 777
        np.testing.assert_array_equal(getattr(result, name), before)
    result.integral_provenance['xc']['component_ids'][0] = -1
    assert result.integral_provenance == evidence
    for spin, suffix in enumerate(('',) if spins == 1 else ('_alpha', '_beta')):
        coeff = np.asarray(getattr(result, f'mo_coeffs{suffix}_k'))
        occupations = np.asarray(getattr(result, f'occupations{suffix}_k'))
        count = (2 if spins == 1 else 1)*pop[spin]
        np.testing.assert_array_equal(occupations.sum(axis=1), np.full(len(k), count))
        projector = (coeff*occupations[:, None, :])@coeff.conj().swapaxes(-1, -2)
        np.testing.assert_allclose(projector, native.density[spin], atol=3e-9, rtol=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.mesh = (1, 1, 1)


@pytest.mark.parametrize('spins', [1, 2])
def test_capped_epochs_keep_density_distinct_from_next_projector(tmp_path, spins):
    job, args, c, k, old = job_case(tmp_path, 'H2' if spins == 1 else 'HeH', spins)
    if spins == 1:
        shells = job['basis'].shells(); shells[1].exponents = [.6]
        job['basis'] = vq.BasisSet(job['system'].unit_cell_molecule(), shells, 'ks-job-asymmetric', False)
        args['basis'] = c['basis'] = old['basis'] = job['basis']
    explicit(args, c, old); _, _, build = oracle(c, k, old)
    previous_density_id = None
    for cap in (1, 2, 3):
        job.update(max_iter=cap, output=tmp_path/f'epoch-{cap}'/'job')
        result = vq.run_periodic_job(**job); r = result.native
        assert not result.converged
        assert len(result.scf_history) == result.iterations == cap
        f, energies, _ = build(r.density)
        np.testing.assert_allclose(r.fock, f, atol=3e-11, rtol=0)
        assert result.energy == pytest.approx(r.receipt.nuclear_energy+sum(energies), abs=3e-11)
        counts = (r.plan.n_alpha,) if spins == 1 else (r.plan.n_alpha, r.plan.n_beta)
        next_density = []
        for spin, count in enumerate(counts):
            coeff = r.coefficients[spin, :, :, :count]
            next_density.append((2 if spins == 1 else 1)*coeff@coeff.conj().swapaxes(-1, -2))
            for point in range(len(k)):
                # Independent generalized eigenproblem: no k weight inside S
                # normalization and no reinterpretation of F as a new epoch.
                eps, vectors = eigh(r.fock[spin, point], r.overlap[point])
                np.testing.assert_allclose(r.orbital_energies[spin, point], eps, atol=3e-11, rtol=0)
                cc = r.coefficients[spin, point]
                np.testing.assert_allclose(cc.conj().T@r.overlap[point]@cc, np.eye(cc.shape[1]), atol=3e-12, rtol=0)
        assert np.max(abs(np.array(next_density)-r.density)) > 1e-9
        expected = r.density if spins == 2 else np.repeat(.5*r.density, 2, axis=0)
        np.testing.assert_array_equal(_spin_density_k_from_qvf_bloch_payload(job['output'].with_suffix('.qvf')), expected)
        assert _qvf(job['output'].with_suffix('.qvf'))['provenance']['run_status'] == 'failed'
        assert r.receipt.final_density_identity != previous_density_id
        previous_density_id = r.receipt.final_density_identity


@pytest.mark.parametrize('spins', [1, 2])
def test_gamma_lda_job_preserves_complete_functional(tmp_path, spins):
    job, args, c, _, old = job_case(tmp_path, spins=spins)
    job.update(functional='lda', kpoints=None)
    args['functional'] = c['functional'] = vq.Functional('lda', spin=spins)
    args['mesh'] = c['mesh'] = old['mesh'] = manta.RegularKMesh((1, 1, 1), (0, 0, 0))
    explicit(args, c, old); s, h, build = oracle(c, np.zeros((1, 3)), old)
    result = vq.run_periodic_job(**job)
    density = independent_scf(s, h, build, (1,) if spins == 1 else (1, 1), spins)
    assert result.converged and result.mesh == (1, 1, 1)
    np.testing.assert_allclose(result.native.density, density, atol=3e-9, rtol=0)
    f, energies, _ = build(result.native.density)
    np.testing.assert_allclose(result.native.fock, f, atol=2e-11, rtol=0)
    assert result.energy == pytest.approx(result.native.receipt.nuclear_energy+sum(energies), abs=2e-11)


@pytest.mark.parametrize('spins', [1, 2])
def test_job_image_gauge_and_qvf_nuclear_frame(tmp_path, spins):
    job, *_ = job_case(tmp_path/'original', spins=spins)
    job['periodic_integrals'] = dataclasses.replace(job['periodic_integrals'], pair_cutoff_bohr=7., nuclear_cutoff_bohr=9.)
    original = vq.run_periodic_job(**job)
    shells = job['basis'].shells(); shells[0].origin = np.array(shells[0].origin)+job['system'].lattice[:, 0]
    job['basis'] = vq.BasisSet(vq.Molecule([vq.Atom(1, s.origin) for s in shells], 0, 1), shells, 'ks-job-image', False)
    job['output'] = tmp_path/'moved'/'job'
    moved = vq.run_periodic_job(**job)
    assert moved.energy == pytest.approx(original.energy, abs=3e-11)
    phase = np.ones((len(job['kpoints']), 2), complex)
    phase[:, 0] = np.exp(2j*np.pi*moved.kpoints_fractional[:, 0])
    action = phase[:, :, None]*phase.conj()[:, None, :]
    for name in ('overlap', 'hcore', 'density', 'fock'):
        np.testing.assert_allclose(getattr(moved.native, name), getattr(original.native, name)*action, atol=3e-9, rtol=0)
    assert moved.integral_provenance['basis_frame_identity'] != original.integral_provenance['basis_frame_identity']
    restored = _spin_density_k_from_qvf_bloch_payload(job['output'].with_suffix('.qvf'))
    expected = original.native.density if spins == 2 else np.repeat(.5*original.native.density, 2, axis=0)
    np.testing.assert_allclose(restored, expected, atol=3e-9, rtol=0)


@pytest.mark.parametrize('field,value', [
    ('functional', None), ('functional', ''), ('functional', 'wb97x-v'), ('functional', 'SCAN'),
    ('periodic_xc', None), ('exchange', manta.ExchangeBuildOptions(algorithm='exact')),
    ('coulomb', manta.CoulombBuildOptions(algorithm='ri')),
    ('symmetry', 'attach'), ('symmetry_reduce_k', True), ('reduce_to_primitive', True),
    ('bipole_cutoff_bohr', 7.), ('bipole_historical_domain', True),
    ('use_diis', True), ('smearing', .01), ('solver', 'davidson'), ('method', 'ROKS')])
def test_unsupported_source_compositions_refuse_before_generation(tmp_path, monkeypatch, field, value):
    job, *_ = job_case(tmp_path); job[field] = value
    def fail(*a, **kw): raise AssertionError('Unadmitted requests must not generate a numerical source')
    monkeypatch.setattr(core._manta, 'run_periodic_ks_with_quadrature', fail)
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        vq.run_periodic_job(**job)
    _no_computed_artifacts(job)


@pytest.mark.parametrize('field,value', [
    ('xc_image_cutoff_bohr', -1.), ('xc_image_cutoff_bohr', float('nan')),
    ('partition_radius_bohr', 0.), ('partition_radius_bohr', float('inf')),
    ('radial_points', True), ('radial_points', 10001), ('lebedev_order', 0),
    ('grid_batch', 0), ('grid_batch', 1.5), ('stratmann', 1)])
def test_xc_controls_are_explicit_finite_and_typed(field, value):
    with pytest.raises((ValueError, TypeError)):
        controls = dict(xc_image_cutoff_bohr=11.)
        controls[field] = value
        manta.PeriodicXCOptions(**controls)


@pytest.mark.parametrize('field', ['memory_bytes', 'shell_calls', 'primitive_quartets', 'scalar_terms', 'kernel_terms'])
def test_job_aggregate_budget_refuses_before_numerical_source(tmp_path, monkeypatch, field):
    job, *_ = job_case(tmp_path)
    setattr(job['integral_engine'].budget, field, 1)
    def fail(*a, **kw): raise AssertionError('Budget refusal must precede numerical generation')
    monkeypatch.setattr(core._manta, 'run_periodic_ks_with_quadrature', fail)
    with pytest.raises((ValueError, RuntimeError), match='budget'):
        vq.run_periodic_job(**job)
    _no_computed_artifacts(job)


def test_inactive_exchange_controls_and_unsupported_grid(tmp_path):
    job, *_ = job_case(tmp_path)
    for update in (dict(periodic_integrals=dataclasses.replace(job['periodic_integrals'], madelung_alpha=.6)),
                   dict(periodic_integrals=dataclasses.replace(job['periodic_integrals'], madelung_tolerance=1e-8)),
                   dict(periodic_xc=dataclasses.replace(job['periodic_xc'], lebedev_order=12))):
        with pytest.raises(ValueError):
            vq.run_periodic_job(**dict(job, **update))
        _no_computed_artifacts(job)
    a = vq.run_periodic_job(**job)
    b = vq.run_periodic_job(**dict(job, exchange_exxdiv='none', exchange=manta.ExchangeBuildOptions(algorithm='none')))
    assert a.energy == b.energy and a.exchange_q0 == b.exchange_q0 == 'not-applicable'
    assert a.native.receipt.source_identity == b.native.receipt.source_identity


def test_dry_run_admits_grid_but_never_builds_and_delivery_failure_is_visible(tmp_path, monkeypatch):
    from vibeqc.output import OutputWriter
    from vibeqc.periodic_runner import _PERIODIC_OUTPUT_WRITER
    job, *_ = job_case(tmp_path/'dry'); job['dry_run'] = True
    def fail(*a, **kw): raise AssertionError('Dry run must not generate or solve')
    with monkeypatch.context() as patch:
        patch.setattr(core._manta, 'run_periodic_ks_with_quadrature', fail)
        assert vq.run_periodic_job(**job) is None
    assert list(job['output'].parent.iterdir()) == [job['output'].with_suffix('.system')]
    job, *_ = job_case(tmp_path/'delivery')
    dispatch = OutputWriter.dispatch_role
    def fail_delivery(self, role, **kwargs):
        if role == 'qvf': raise OSError('synthetic KS delivery failure')
        return dispatch(self, role, **kwargs)
    monkeypatch.setattr(OutputWriter, 'dispatch_role', fail_delivery)
    before = _PERIODIC_OUTPUT_WRITER.get()
    with pytest.raises(OSError, match='synthetic KS delivery'):
        vq.run_periodic_job(**job)
    assert _PERIODIC_OUTPUT_WRITER.get() is before
    manifest = tomllib.loads(job['output'].with_suffix('.system').read_text())
    assert manifest['outputs']['status'] == 'crashed'
    assert manifest['outputs']['result_status'] == 'converged'
