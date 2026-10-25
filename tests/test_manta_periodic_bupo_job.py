"""Public BUPO RHF/UHF jobs preserve native sources and evaluated QVF state."""
import dataclasses
import json
import tomllib
import zipfile

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from tests.test_manta_periodic_bupo_domains import fixture
from tests.test_manta_periodic_bupo_preparation import preparation_args
from tests.test_manta_periodic_bupo_hf import as_fock_build


def job_case(path, **settings):
    a, c = fixture(**settings)
    args = preparation_args(a, c, 900. if settings.get('z') == 3 else 500.)
    source = args['source_options']
    bupo = manta.BupoOptions()
    bupo.order = source.pairs.order
    bupo.extents = source.pairs.extents
    bupo.bubbles = source.bubbles
    bupo.opening = source.bupo.opening
    bupo.integral_tolerance = source.bupo.integral_tolerance
    domains = manta.PeriodicBupoIntegralOptions(args['reciprocal_energy_cutoff'],
        pair_cube_radius=source.pair_cube_radius, real_cube_radius=source.real_cube_radius,
        reciprocal_cube_radius=source.reciprocal_cube_radius, near_cube_radius=source.near_cube_radius,
        eta=source.eta, pair_cutoff_bohr=source.pairs.pair_cutoff_bohr)
    job = dict(system=a['system'], basis=c['basis'], kpoints=a['mesh'],
        method='RHF' if a['options'].spin_mode == 'restricted' else 'UHF',
        integral_engine=manta.IntegralEngineOptions(backend='manta', budget=a['budget']),
        coulomb=manta.CoulombBuildOptions(algorithm='bupo', bupo=bupo),
        exchange=manta.ExchangeBuildOptions(algorithm='exact'), periodic_integrals=domains,
        use_diis=False, convergence='off', max_iter=a['options'].scf.maximum_iterations,
        conv_tol_energy=a['options'].scf.energy_change_tolerance, conv_tol_grad=a['options'].scf.commutator_tolerance,
        exchange_exxdiv='none' if a['options'].exchange_q0 == 'strict-zero-mode' else 'ewald',
        output=path/'job', record_hostname=False, structured_log=True)
    return job, args, c


def read_outputs(job):
    stem = job['output']
    manifest = tomllib.loads(stem.with_suffix('.system').read_text())
    rows = [json.loads(line) for line in stem.with_suffix('.scf.jsonl').read_text().splitlines()]
    with zipfile.ZipFile(stem.with_suffix('.qvf')) as archive:
        qvf = json.loads(archive.read('manifest.json'))
    return manifest, rows, qvf


@pytest.mark.parametrize('spin,z,shifted,iterations', [('restricted', 2, False, 40),
    ('restricted', 2, True, 1), ('unrestricted', 1, True, 40), ('unrestricted', 3, True, 1)])
def test_public_job_native_state_qvf_and_fock(tmp_path, spin, z, shifted, iterations):
    from vibeqc.guess_read import _load_periodic_read_density_k_closed, _spin_density_k_from_qvf_bloch_payload
    from vibeqc.output import validate_qvf
    job, args, context = job_case(tmp_path, spin=spin, z=z, shifted=shifted, iterations=iterations, q0='BvK-ewald')
    expected = manta.prepare_periodic_bupo_hf(**args).run()
    result = vq.run_periodic_job(**job)
    assert result.converged == expected.converged == (iterations > 1)
    for name in ('density', 'fock', 'hcore', 'overlap'):
        np.testing.assert_array_equal(getattr(result.native, name), getattr(expected, name))
    assert result.energy == expected.receipt.energy
    assert result.native.receipt.preparation.identity == expected.receipt.preparation.identity
    assert len(result.density_k) == len(args['mesh'])
    np.testing.assert_array_equal(result.density_k, expected.density.sum(axis=0))
    np.testing.assert_array_equal(result.density_alpha_k, expected.density[0]*(.5 if spin == 'restricted' else 1.))
    manifest, rows, qvf = read_outputs(job)
    assert manifest['run']['coulomb_algorithm'] == 'bupo' and manifest['run']['exchange_algorithm'] == 'exact'
    assert manifest['run']['jk_method_resolved'] == 'bupo-exact-reciprocal'
    assert 'nuclear_cutoff_bohr' not in manifest['run'] and manifest['run']['ewald_eta'] == args['source_options'].eta
    evidence = result.integral_provenance
    assert evidence['preparation']['identity'] == expected.receipt.preparation.identity
    assert evidence['exchange_accuracy_passed'] and not evidence['symmetry_reduced']
    assert evidence['job_scalar_terms'] > 0 and evidence['job_kernel_terms'] > 0
    assert evidence['modeled_peak_bytes'] <= job['integral_engine'].budget.memory_bytes
    assert evidence['scalar_terms'] <= job['integral_engine'].budget.scalar_terms
    assert evidence['kernel_terms'] <= job['integral_engine'].budget.kernel_terms
    assert job['integral_engine'].budget.other_live_bytes == 0
    terminal = next(row for row in rows if row.get('event') == 'scf_terminal')
    assert terminal['integral_provenance']['confirmed'] is result.converged
    assert qvf['provenance']['run_status'] == ('converged' if result.converged else 'failed')
    assert qvf['provenance']['scf_energy']['value'] == result.energy
    assert validate_qvf(job['output'].with_suffix('.qvf'))['valid']
    if spin == 'restricted':
        restored = _load_periodic_read_density_k_closed(read_from=job['output'].with_suffix('.qvf'),
            expected_n_k=len(args['mesh']), n_basis=context['basis'].nbasis)
        np.testing.assert_allclose(restored, result.density_k, atol=3e-12)
    else:
        da, db = _spin_density_k_from_qvf_bloch_payload(job['output'].with_suffix('.qvf'),
            expected_n_k=len(args['mesh']), n_basis=context['basis'].nbasis)
        np.testing.assert_allclose(da, result.density_alpha_k, atol=3e-12)
        np.testing.assert_allclose(db, result.density_beta_k, atol=3e-12)
        if z == 1:
            np.testing.assert_array_equal(db, np.zeros_like(db))
    assert as_fock_build(result.native, context['k']).energy.total == pytest.approx(result.energy, abs=3e-12)
    evidence['preparation']['identity'] = 'changed'
    assert result.integral_provenance['preparation']['identity'] == expected.receipt.preparation.identity
    assert 'Periodic BUPO' in job['output'].with_suffix('.out').read_text()


@pytest.mark.parametrize('environment', [False, True])
def test_dry_run_never_constructs_source(tmp_path, monkeypatch, environment):
    job, _, _ = job_case(tmp_path, iterations=1)
    if environment:
        monkeypatch.setenv('VIBEQC_DRY_RUN', '1')
    else:
        job['dry_run'] = True
    def forbidden(*args, **kwargs):
        raise AssertionError('Dry-run invoked scientific source construction')
    monkeypatch.setattr(core._manta, 'prepare_periodic_bupo_hf', forbidden)
    assert vq.run_periodic_job(**job) is None
    assert [p.suffix for p in tmp_path.iterdir()] == ['.system']
    manifest = tomllib.loads(job['output'].with_suffix('.system').read_text())
    assert manifest['memory']['estimate_bytes'] == job['integral_engine'].budget.memory_bytes


@pytest.mark.parametrize('field,value', [('reciprocal_energy_cutoff', True), ('pair_cube_radius', False),
    ('pair_cube_radius', 1.5), ('real_cube_radius', -1), ('reciprocal_cube_radius', 2**31),
    ('near_cube_radius', 5), ('eta', 0.), ('pair_cutoff_bohr', -1.), ('reciprocal_batch', 0),
    ('madelung_alpha', -1.), ('madelung_tolerance', 1.), ('maximum_exchange_energy_error', float('nan'))])
def test_typed_domains_reject_invalid_values(field, value):
    with pytest.raises(ValueError, match='PeriodicBupoIntegralOptions'):
        manta.PeriodicBupoIntegralOptions(**(dict(reciprocal_energy_cutoff=500.) | {field: value}))


def test_route_does_not_reinterpret_radial_or_molecular_settings(tmp_path):
    job, _, _ = job_case(tmp_path, iterations=1)
    original = job['periodic_integrals']
    job['periodic_integrals'] = manta.PeriodicIntegralOptions(500., 1., 4.)
    with pytest.raises(TypeError, match='PeriodicBupoIntegralOptions'):
        vq.run_periodic_job(**job)
    job['periodic_integrals'] = original
    job['coulomb'].bupo.collect_timings = True
    with pytest.raises(NotImplementedError, match='collect_timings=False'):
        vq.run_periodic_job(**job)
    job['coulomb'].bupo.collect_timings = False
    job['exchange'] = manta.ExchangeBuildOptions(algorithm='ri')
    with pytest.raises(NotImplementedError, match='exact reciprocal K'):
        vq.run_periodic_job(**job)
    assert [p.suffix for p in tmp_path.iterdir()] == ['.system']
    refused = tomllib.loads(job['output'].with_suffix('.system').read_text())
    assert refused['outputs']['result_status'] == 'refused'


def test_joint_job_caps_include_preparation_and_output(tmp_path):
    job, _, _ = job_case(tmp_path/'initial', iterations=1)
    result = vq.run_periodic_job(**job)
    p = result.integral_provenance['preparation_admission']
    limits = dict(memory_bytes=p['modeled_peak_bytes'], shell_calls=p['shell_calls'],
        primitive_quartets=p['primitive_pairs'], scalar_terms=p['scalar_terms'], kernel_terms=p['kernel_terms'])
    budget = job['integral_engine'].budget
    for name, value in limits.items():
        setattr(budget, name, value)
    job['output'] = tmp_path/'exact'/'job'
    assert vq.run_periodic_job(**job).energy == result.energy
    for name, value in limits.items():
        setattr(budget, name, value-1)
        job['output'] = tmp_path/name/'job'
        with pytest.raises((ValueError, OverflowError), match='budget'):
            vq.run_periodic_job(**job)
        assert not job['output'].with_suffix('.qvf').exists()
        setattr(budget, name, value)


def test_failed_accuracy_does_not_publish_qvf(tmp_path):
    job, _, _ = job_case(tmp_path, iterations=1)
    job['periodic_integrals'] = dataclasses.replace(job['periodic_integrals'], reciprocal_energy_cutoff=3.)
    with pytest.raises(ValueError, match='exchange accuracy target'):
        vq.run_periodic_job(**job)
    assert not job['output'].with_suffix('.qvf').exists()


def test_basis_image_gauge_round_trip_keeps_evaluated_density(tmp_path):
    from vibeqc.guess_read import _load_periodic_read_density_k_closed
    job, _, c = job_case(tmp_path/'ordinary', shifted=True, iterations=1, q0='BvK-ewald')
    atoms = [vq.Atom(1, [.1, .2, -.1]), vq.Atom(1, [1.5, .2, -.1])]
    job['system'] = vq.PeriodicSystem(3, c['cell'], atoms)
    shells = c['basis'].shells()
    for i, shell in enumerate(shells):
        shell.atom_index = i
        shell.origin = atoms[i].xyz
    job['basis'] = vq.BasisSet(job['system'].unit_cell_molecule(), shells, 'bupo-job-gauge', False)
    job['periodic_integrals'] = dataclasses.replace(job['periodic_integrals'], pair_cube_radius=2, pair_cutoff_bohr=6.1)
    ordinary = vq.run_periodic_job(**job)
    shells = job['basis'].shells()
    shells[0].origin = np.asarray(shells[0].origin)+c['cell'][:, 0]
    atoms[0] = vq.Atom(1, shells[0].origin)
    basis_cell = vq.PeriodicSystem(3, c['cell'], atoms)
    job['basis'] = vq.BasisSet(basis_cell.unit_cell_molecule(), shells, 'bupo-job-image', False)
    job['output'] = tmp_path/'moved'/'job'
    moved = vq.run_periodic_job(**job)
    assert moved.energy == pytest.approx(ordinary.energy, abs=3e-11)
    assert np.max(abs(np.asarray(moved.density_k)-ordinary.density_k)) > 1e-5
    assert moved.integral_provenance['source_ao_image_shifts'] == [[1, 0, 0], [0, 0, 0]]
    restored = _load_periodic_read_density_k_closed(read_from=job['output'].with_suffix('.qvf'),
        expected_n_k=2, n_basis=job['basis'].nbasis)
    np.testing.assert_allclose(restored, ordinary.density_k, atol=3e-10)
