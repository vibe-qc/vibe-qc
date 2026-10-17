"""The selected periodic front door preserves the native source and output state."""
import dataclasses
import json
import tomllib
import zipfile

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from tests.test_manta_periodic_domains import _arguments


def _job(tmp_path, shape=(2, 1, 1), shift=(1, 0, 0)):
    args, _ = _arguments(shape, shift)
    return dict(system=args['system'], basis=args['basis'], kpoints=args['mesh'],
        integral_engine=manta.IntegralEngineOptions(backend='manta', budget=args['budget']),
        periodic_integrals=manta.PeriodicIntegralOptions(2., 3., 4., alpha=.6),
        use_diis=False, output=tmp_path/'selected'/'job', record_hostname=False,
        structured_log=True, max_iter=40, conv_tol_energy=1e-12)


def _native(job):
    p = job['periodic_integrals']; o = manta.PeriodicRHFOptions()
    o.alpha = p.alpha; o.pair_cutoff_bohr = p.pair_cutoff_bohr
    o.nuclear_cutoff_bohr = p.nuclear_cutoff_bohr; o.reciprocal_batch = p.reciprocal_batch
    o.scf.maximum_iterations = job['max_iter']; o.scf.energy_change_tolerance = job['conv_tol_energy']
    o.exchange_q0 = 'BvK-ewald' if job.get('exchange_exxdiv', 'ewald') == 'ewald' else 'strict-zero-mode'
    return core._manta.run_periodic_rhf_from_cutoffs(job['basis'], job['system'], job['kpoints'],
        p.reciprocal_energy_cutoff, o, job['integral_engine'].budget)


def _qvf(path):
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read('manifest.json'))


def _no_computed_artifacts(job):
    directory = job['output'].parent
    if directory.exists():
        assert list(directory.iterdir()) == [job['output'].with_suffix('.system')]
        manifest = tomllib.loads(job['output'].with_suffix('.system').read_text())
        assert manifest['outputs']['result_status'] == 'refused'


@pytest.mark.parametrize('shape,shift', [((1, 1, 1), (0, 0, 0)),
    ((2, 1, 1), (0, 0, 0)), ((2, 1, 1), (1, 0, 0))])
def test_job_matches_native_and_standard_outputs_preserve_executed_source(tmp_path, shape, shift):
    from vibeqc.output import validate_qvf
    from vibeqc.guess_read import _load_periodic_read_density_k_closed
    job = _job(tmp_path, shape, shift)
    expected = _native(job)
    result = vq.run_periodic_job(**job)
    assert isinstance(result, manta.PeriodicMANTAResult) and result.converged
    assert result.energy == expected.receipt.energy
    assert result.native.receipt.source_identity == expected.receipt.source_identity
    assert result.exchange_q0 == 'BvK-ewald'
    np.testing.assert_array_equal(result.density_k, expected.density)
    np.testing.assert_array_equal(result.fock_k, expected.fock)
    assert job['integral_engine'].budget.other_live_bytes == 0
    stem = job['output']
    manifest = tomllib.loads(stem.with_suffix('.system').read_text())
    assert manifest['run']['integral_backend_executed'] == 'manta'
    assert manifest['run']['integral_source_identity'] == expected.receipt.source_identity
    assert manifest['run']['initial_guess_executed'] == 'HCORE'
    assert manifest['run']['coulomb_algorithm'] == manifest['run']['exchange_algorithm'] == 'exact'
    assert 'Finite periodic RHF: manta' in stem.with_suffix('.out').read_text()
    assert 'SCF status: converged' in stem.with_suffix('.out').read_text()
    assert all(stem.with_suffix(s).is_file() for s in ('.xyz', '.cif', '.xsf', '.bibtex', '.references', '.scf.jsonl', '.qvf'))
    assert 'Neese' in stem.with_suffix('.references').read_text()
    records = [json.loads(line) for line in stem.with_suffix('.scf.jsonl').read_text().splitlines()]
    source = next(row['integral_provenance'] for row in records if row.get('event') == 'manta_periodic_rhf')
    assert source['domain_identity'] == result.native.receipt.domain_identity
    steps = [row for row in records if row.get('event') == 'scf_iteration']
    assert len(steps) == result.iterations
    assert [row['iteration'] for row in steps] == list(range(1, result.iterations+1))
    assert all(row['delivery'] == 'after-solve' for row in steps)
    assert steps[-1]['energy'] == result.native.scf_history[-1].energy_per_cell
    assert source['scf_history_steps'] == result.iterations
    assert source['scf_history_reserved_bytes'] == result.native.plan.history_live_bytes
    qvf = _qvf(stem.with_suffix('.qvf'))
    assert qvf['provenance']['run_status'] == 'converged'
    assert qvf['provenance']['exchange_q0'] == 'BvK-ewald'
    assert qvf['provenance']['scf_energy']['value'] == result.energy
    assert qvf['provenance']['basis'] == job['basis'].name
    assert qvf['provenance']['n_scf_iterations'] == result.iterations
    assert qvf['provenance']['run_fields']['integral_backend_executed'] == 'manta'
    assert any(s['kind'] == 'scf_history' for s in qvf['sections'])
    section = next(s for s in qvf['sections'] if s['kind'] == 'x_vibeqc.periodic_integral_engine')
    with zipfile.ZipFile(stem.with_suffix('.qvf')) as z:
        evidence = json.loads(z.read(next(iter(section['members'].values()))['path']))
        history = json.loads(z.read('scf_history/iterations.json'))['iterations']
    assert evidence == result.integral_provenance
    assert history == result.scf_history
    assert history[-1]['energy_eh'] == pytest.approx(result.energy, abs=1e-11)
    assert 'delta_e' not in history[0]
    assert history[-1]['delta_e'] == result.native.scf_history[-1].energy_change
    result.scf_history[0]['energy_eh'] = 777.
    assert result.scf_history == history
    assert evidence['restart_density_encoding'] == 'equal-alpha-beta-evaluated-density'
    restored = _load_periodic_read_density_k_closed(read_from=stem.with_suffix('.qvf'), expected_n_k=len(job['kpoints']), n_basis=job['basis'].nbasis)
    np.testing.assert_array_equal(restored, result.density_k)
    issues = validate_qvf(stem.with_suffix('.qvf'))
    assert issues['valid'] and not issues['errors']
    for name in ('density_k', 'density_alpha_k', 'density_beta_k', 'fock_k', 'overlap_k', 'hcore_k',
                 'mo_coeffs_k', 'mo_energies_k', 'occupations_k'):
        before = np.array(getattr(result, name))
        getattr(result, name)[0][:] = 777
        np.testing.assert_array_equal(getattr(result, name), before)
    changed = result.integral_provenance; changed['source_identity'] = 'changed'
    assert result.integral_provenance['source_identity'] == expected.receipt.source_identity
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.mesh = (3, 1, 1)


def test_iteration_limit_qvf_preserves_evaluated_density_not_next_orbitals(tmp_path):
    from vibeqc.guess_read import _load_periodic_read_density_k_closed
    job = _job(tmp_path)
    # Unequal exponents break the symmetric two-orbital stationary HCORE guess.
    shells = job['basis'].shells(); shells[0].exponents = [.4]
    job['basis'] = vq.BasisSet(job['system'].unit_cell_molecule(), shells, 'manta-job-unconverged', False)
    job['max_iter'] = 1
    result = vq.run_periodic_job(**job)
    assert not result.converged and not result.native.receipt.confirmed
    coefficients = np.asarray(result.mo_coeffs_k)[:, :, :1]
    next_density = 2*coefficients @ coefficients.conj().transpose(0, 2, 1)
    assert np.max(abs(next_density-np.asarray(result.density_k))) > 1e-3
    path = job['output'].with_suffix('.qvf')
    restored = _load_periodic_read_density_k_closed(read_from=path, expected_n_k=2, n_basis=2)
    np.testing.assert_array_equal(restored, result.native.density)
    qvf = _qvf(path)
    assert qvf['provenance']['run_status'] == 'failed'
    assert qvf['provenance']['scf_converged'] is False
    assert len(result.scf_history) == 1
    with zipfile.ZipFile(path) as z:
        assert json.loads(z.read('scf_history/iterations.json'))['iterations'] == result.scf_history
    assert 'SCF status: not_converged' in job['output'].with_suffix('.out').read_text()
    manifest = tomllib.loads(job['output'].with_suffix('.system').read_text())
    assert manifest['run']['scf_converged'] is False
    assert result.native.receipt.final_density_identity == _native(job).receipt.final_density_identity


@pytest.mark.parametrize('representation', ['none', 'scalar', 'tuple', 'list', 'kpoints', 'bloch'])
def test_supported_mesh_inputs_preserve_regular_mesh_and_shift(tmp_path, representation):
    job = _job(tmp_path)
    if representation == 'none':
        job['kpoints'] = None; expected_shape, expected_shift = (1, 1, 1), (0, 0, 0)
    elif representation == 'scalar':
        job['kpoints'] = 1; expected_shape, expected_shift = (1, 1, 1), (0, 0, 0)
    elif representation in ('tuple', 'list'):
        job['kpoints'] = (2, 1, 1) if representation == 'tuple' else [2, 1, 1]
        expected_shape, expected_shift = (2, 1, 1), (0, 0, 0)
    elif representation == 'kpoints':
        job['kpoints'] = vq.KPoints.monkhorst_pack(job['system'], (2, 1, 1))
        expected_shape, expected_shift = (2, 1, 1), (1, 0, 0)
    else:
        job['kpoints'] = core.monkhorst_pack(job['system'], [2, 1, 1], [1, 0, 0], False)
        expected_shape, expected_shift = (2, 1, 1), (1, 0, 0)
    result = vq.run_periodic_job(**job)
    assert result.mesh == expected_shape and result.is_shift == expected_shift
    assert result.converged


@pytest.mark.parametrize('change', ['weights', 'order', 'lattice', 'cartesian', 'reduced', 'smearing'])
def test_inconsistent_kpoints_refused_before_outputs(tmp_path, change):
    job = _job(tmp_path)
    k = vq.KPoints.monkhorst_pack(job['system'], (2, 1, 1))
    if change == 'weights':
        k.weights = np.array([.3, .7])
    elif change == 'order':
        k.kpoints_frac = k.kpoints_frac[::-1].copy(); k.kpoints_cart = k.kpoints_cart[::-1].copy()
    elif change == 'lattice':
        job['system'] = vq.PeriodicSystem(3, 1.1*job['system'].lattice, job['system'].unit_cell)
    elif change == 'cartesian':
        k.kpoints_cart[0, 0] += .01
    elif change == 'reduced':
        k.ir_mapping = np.array([0, 0])
    else:
        k.bz_integration = 'gilat'
    job['kpoints'] = k
    with pytest.raises((ValueError, NotImplementedError), match='KPoints|mesh|smearing|metadata'):
        vq.run_periodic_job(**job)
    _no_computed_artifacts(job)


@pytest.mark.parametrize('field,value', [
    ('method', 'UKS'), ('functional', 'PBE'), ('use_diis', True), ('solver', 'davidson'),
    ('initial_guess', 'SAD'), ('convergence', 'auto'), ('damping', .2), ('level_shift', .1),
    ('symmetry', 'attach'), ('symmetry_reduce_k', True), ('reduce_to_primitive', True),
    ('optimize', True), ('hessian', True), ('dispersion', 'd3'), ('tddft', True),
    ('write_molden_file', True), ('write_population_file', True), ('write_density', True),
    ('checkpoint_every', 2), ('read_from', 'prior.qvf'), ('jk_method', 'gdf'),
    ('ewald_omega', .4), ('bipole_cutoff_bohr', 7.), ('rsgdf_ke_cutoff', 40.),
    ('cutoff_ha', 400.), ('memory_override', True), ('progress', True), ('conv_tol_energy', 0.),
    ('max_iter', True), ('max_iter', 0), ('exchange_exxdiv', 'unknown'),
    ('periodic_integrals', None), ('integral_engine', manta.IntegralEngineOptions(backend='libint')),
    ('coulomb', manta.CoulombBuildOptions(algorithm='bupo')),
    ('exchange', manta.ExchangeBuildOptions(algorithm='ri')),
])
def test_unsupported_requests_refused_before_output_creation(tmp_path, field, value):
    job = _job(tmp_path); job[field] = value
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        vq.run_periodic_job(**job)
    _no_computed_artifacts(job)


def test_ecp_refusal_and_budget_before_output_creation(tmp_path, monkeypatch):
    job = _job(tmp_path)
    monkeypatch.setattr('vibeqc.periodic_runner._resolve_ecp_data', lambda *a: ([], [], [], 2))
    with pytest.raises(NotImplementedError, match='all-electron'):
        vq.run_periodic_job(**job)
    _no_computed_artifacts(job)
    monkeypatch.setattr('vibeqc.periodic_runner._resolve_ecp_data', lambda *a: ([], [], [], 0))
    job['output'] = tmp_path/'budget'/'job'
    job['integral_engine'].budget.memory_bytes = 1
    with pytest.raises(ValueError, match='budget'):
        vq.run_periodic_job(**job)
    assert not job['output'].parent.exists()


@pytest.mark.parametrize('iterations', [10**8, np.int64(2**62)])
def test_job_trace_reservation_precedes_native_admission(tmp_path, monkeypatch, iterations):
    job = _job(tmp_path)
    job['max_iter'] = iterations
    monkeypatch.setattr(core._manta, 'plan_periodic_rhf_from_cutoffs',
                        lambda *a: pytest.fail('native admission before Python trace reservation'))
    with pytest.raises(ValueError, match='output memory budget'):
        vq.run_periodic_job(**job)
    assert not job['output'].parent.exists()


def test_dry_run_plans_without_native_calculation(tmp_path, monkeypatch):
    job = _job(tmp_path); job['dry_run'] = True
    monkeypatch.setattr(core._manta, 'run_periodic_rhf_from_cutoffs', lambda *a: pytest.fail('SCF in dry run'))
    assert vq.run_periodic_job(**job) is None
    assert list(job['output'].parent.iterdir()) == [job['output'].with_suffix('.system')]


def test_explicit_zero_mode_and_output_opt_outs(tmp_path):
    job = _job(tmp_path)
    job.update(exchange_exxdiv='none', citations=False, output_qvf=False,
               write_xyz_file=False, write_xsf_structure_file=False, write_cif_file=False,
               structured_log=False, initial_guess='HCORE', convergence='off')
    result = vq.run_periodic_job(**job)
    assert result.converged and result.exchange_q0 == 'strict-zero-mode'
    assert result.native.receipt.exchange_correction_energy == 0.
    assert {p.suffix for p in job['output'].parent.iterdir()} == {'.out', '.system'}


def test_qvf_transports_independent_shell_images_to_its_nuclear_center_basis(tmp_path):
    from vibeqc.guess_read import _load_periodic_read_density_k_closed
    original_job = _job(tmp_path/'ordinary')
    original = vq.run_periodic_job(**original_job)
    moved_job = _job(tmp_path/'moved')
    shells = moved_job['basis'].shells()
    shells[0].origin = np.asarray(shells[0].origin)+moved_job['system'].lattice[:, 0]
    atoms = list(moved_job['system'].unit_cell)
    atoms[0] = vq.Atom(1, shells[0].origin)
    basis_cell = vq.PeriodicSystem(3, moved_job['system'].lattice, atoms)
    moved_job['basis'] = vq.BasisSet(basis_cell.unit_cell_molecule(), shells, 'manta-job-image', False)
    moved = vq.run_periodic_job(**moved_job)
    assert moved.converged
    assert moved.energy == pytest.approx(original.energy, abs=3e-11)
    assert np.max(abs(np.asarray(moved.density_k)-np.asarray(original.density_k))) > .1
    restored = _load_periodic_read_density_k_closed(read_from=moved_job['output'].with_suffix('.qvf'), expected_n_k=2, n_basis=2)
    np.testing.assert_allclose(restored, original.density_k, atol=3e-10)
    assert moved.integral_provenance['source_ao_image_shifts'] == [[1, 0, 0], [0, 0, 0]]
    assert moved.integral_provenance['qvf_density_gauge'] == 'nuclear-centers'


@pytest.mark.parametrize('field,value', [('reciprocal_energy_cutoff', True), ('pair_cutoff_bohr', 0.),
    ('nuclear_cutoff_bohr', float('inf')), ('alpha', float('nan')), ('reciprocal_batch', False),
    ('reciprocal_batch', 1.5), ('madelung_alpha', -1.), ('madelung_tolerance', 1.)])
def test_domain_options_validate_physical_controls(field, value):
    kwargs = dict(reciprocal_energy_cutoff=2., pair_cutoff_bohr=3., nuclear_cutoff_bohr=4.)
    kwargs[field] = value
    with pytest.raises(ValueError, match='PeriodicIntegralOptions'):
        manta.PeriodicIntegralOptions(**kwargs)


def test_selected_runtime_error_marks_output_crashed_and_restores_channels(tmp_path, monkeypatch):
    from vibeqc.output import active_channel
    job = _job(tmp_path)
    def fail(*args, **kwargs):
        raise RuntimeError('synthetic native failure')
    monkeypatch.setattr(core._manta, 'run_periodic_rhf_from_cutoffs', fail)
    before = active_channel()
    with pytest.raises(RuntimeError, match='synthetic native failure'):
        vq.run_periodic_job(**job)
    assert active_channel() is before
    manifest = tomllib.loads(job['output'].with_suffix('.system').read_text())
    assert manifest['outputs']['status'] == 'crashed'
    assert manifest['outputs']['result_status'] == 'crashed'


@pytest.mark.parametrize('error_type', [OSError, NotImplementedError])
def test_delivery_failure_preserves_scf_verdict_and_restores_writer(tmp_path, monkeypatch, error_type):
    from vibeqc.output import OutputWriter
    from vibeqc.periodic_runner import _PERIODIC_OUTPUT_WRITER
    job = _job(tmp_path)
    dispatch = OutputWriter.dispatch_role
    def fail(self, role, **kwargs):
        if role == 'qvf':
            raise error_type('synthetic delivery failure')
        return dispatch(self, role, **kwargs)
    monkeypatch.setattr(OutputWriter, 'dispatch_role', fail)
    previous = _PERIODIC_OUTPUT_WRITER.get()
    with pytest.raises(error_type, match='synthetic delivery failure'):
        vq.run_periodic_job(**job)
    assert _PERIODIC_OUTPUT_WRITER.get() is previous
    manifest = tomllib.loads(job['output'].with_suffix('.system').read_text())
    assert manifest['outputs']['status'] == 'crashed'
    assert manifest['outputs']['result_status'] == 'converged'
    assert manifest['run']['scf_converged'] is True
