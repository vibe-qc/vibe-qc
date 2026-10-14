"""Public fitted UHF: independent SCF, explicit selection and saved spin state."""
import dataclasses
import json
import tomllib
import zipfile

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from vibeqc.guess_read import _spin_density_k_from_qvf_bloch_payload
from tests.test_manta_periodic_job import _qvf, _no_computed_artifacts
from tests.test_manta_periodic_ri_domains import arguments, explicit
from tests.test_manta_periodic_ri_uhf import reference
from tests.test_manta_periodic_uhf import _scf_reference
from tests.test_manta_periodic_rhf import _nuclear_reference


def job_case(directory, mesh=(2, 1, 1), shift=(1, 0, 0), species='Li', correction='BvK-ewald'):
    a, k = arguments(mesh, shift, species, correction)
    o = a['options']
    job = dict(system=a['system'], basis=a['basis'], kpoints=a['mesh'], method='UHF',
        integral_engine=manta.IntegralEngineOptions(backend='manta', budget=a['budget']),
        coulomb=manta.CoulombBuildOptions(algorithm='ri', auxiliary_basis=a['auxiliary'], metric=o.ri),
        exchange=manta.ExchangeBuildOptions(algorithm='ri'),
        periodic_integrals=manta.PeriodicIntegralOptions(a['reciprocal_energy_cutoff'],
            o.pair_cutoff_bohr, o.nuclear_cutoff_bohr, alpha=o.alpha, reciprocal_batch=o.reciprocal_batch),
        periodic_fitting=manta.PeriodicFittingOptions(o.fitting.omega, o.fitting.auxiliary_cutoff_bohr),
        use_diis=False, output=directory/'selected'/'job', record_hostname=False,
        structured_log=True, max_iter=100, conv_tol_energy=1e-12,
        exchange_exxdiv='ewald' if correction == 'BvK-ewald' else 'none')
    return job, a, k


@pytest.mark.parametrize('species', ['H', 'Li'])
@pytest.mark.parametrize('mesh,shift', [((1, 1, 1), (0, 0, 0)), ((2, 1, 1), (1, 0, 0))])
@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
def test_public_job_independent_scf_components_and_lossless_output(tmp_path, species, mesh, shift, correction):
    from vibeqc.output import validate_qvf
    job, a, k = job_case(tmp_path, mesh, shift, species, correction)
    domains = manta.build_periodic_ri_uhf_domains(**a); c = explicit(a, domains)
    s, h, build, _ = reference(c, k)
    populations = (1, 0) if species == 'H' else (2, 1)
    d = _scf_reference(s, h, build, populations); nuclear = sum(_nuclear_reference(c)[0])
    result = vq.run_periodic_job(**job)
    assert result.converged and isinstance(result, manta.PeriodicMANTAUHFResult)
    assert result.energy == pytest.approx(nuclear+sum(build(d)[1]), abs=8e-11)
    np.testing.assert_allclose(result.native.density, d, atol=5e-9, rtol=0)
    np.testing.assert_allclose(result.native.fock, build(result.native.density)[0], atol=8e-11, rtol=0)
    e = result.integral_provenance; json.dumps(e, allow_nan=False)
    assert e['method'] == 'UHF' and e['coulomb_algorithm'] == e['exchange_algorithm'] == 'ri'
    assert e['accuracy_qualification'] == 'finite-fitted-source-only' and not e['symmetry_reduced']
    assert e['domain_identity'] == domains.receipt.identity
    assert e['fitting']['controls']['omega'] == job['periodic_fitting'].omega
    assert e['fitting']['controls']['pair_cutoff_bohr'] == job['periodic_integrals'].pair_cutoff_bohr
    assert e['fitting']['projection']['mode_block'] == job['coulomb'].metric.mode_block
    assert e['auxiliary_basis'] == a['auxiliary'].name
    assert e['auxiliary_basis_functions'] == a['auxiliary'].nbasis
    assert e['restart_density_encoding'] == 'spin-resolved-evaluated-density'
    assert e['spin_electrons_per_cell'] == list(populations)
    fields = ('one_electron_energy', 'coulomb_energy', 'uncorrected_exchange_energy', 'exchange_correction_energy')
    np.testing.assert_allclose([e[name] for name in fields], build(result.native.density)[1], atol=8e-11, rtol=0)
    assert e['energy'] == pytest.approx(e['nuclear_energy']+sum(e[name] for name in fields), abs=1e-13)
    path = job['output'].with_suffix('.qvf')
    np.testing.assert_array_equal(_spin_density_k_from_qvf_bloch_payload(path), result.native.density)
    assert validate_qvf(path)['valid']
    qvf = _qvf(path); section = next(s for s in qvf['sections'] if s['kind'] == 'x_vibeqc.periodic_integral_engine')
    with zipfile.ZipFile(path) as archive:
        saved = json.loads(archive.read(next(iter(section['members'].values()))['path']))
        assert saved == e
        assert json.loads(archive.read('scf_history/iterations.json'))['iterations'] == result.scf_history
    manifest = tomllib.loads(job['output'].with_suffix('.system').read_text())
    assert manifest['run']['jk_method_resolved'] == 'ri'
    assert manifest['run']['jk_method_executed'] == 'finite-3d-ri-uhf'
    assert manifest['run']['fitting_source_identity'] == result.native.receipt.fitting_info.source_identity
    assert manifest['run']['coulomb_algorithm'] == manifest['run']['exchange_algorithm'] == 'ri'
    assert manifest['run']['auxiliary_basis'] == a['auxiliary'].name
    assert manifest['run']['fitting_metric_absolute_cutoff'] == job['coulomb'].metric.absolute_cutoff
    assert manifest['run']['fitting_mode_block'] == job['coulomb'].metric.mode_block
    assert qvf['provenance']['run_fields'] == result.run_fields
    assert qvf['provenance']['run_status'] == 'converged'
    output = job['output'].with_suffix('.out').read_text()
    assert 'J=RI; K=RI' in output and 'Finite periodic RI UHF: manta' in output
    refs = job['output'].with_suffix('.references').read_text()
    assert 'Berkelbach' in refs and 'Neese' in refs
    records = [json.loads(line) for line in job['output'].with_suffix('.scf.jsonl').read_text().splitlines()]
    event = next(row for row in records if row.get('event') == 'manta_periodic_ri_uhf')
    assert event['integral_provenance']['fitting']['metrics'] == e['fitting']['metrics']
    assert len([r for r in records if r.get('event') == 'scf_iteration']) == result.iterations
    assert job['integral_engine'].budget.other_live_bytes == 0
    result.integral_provenance['fitting']['metrics'].clear()
    assert result.integral_provenance == saved
    for spin in ('alpha', 'beta'):
        for field in ('density', 'fock', 'mo_coeffs', 'mo_energies', 'occupations'):
            name = f'{field}_{spin}_k'; original = np.array(getattr(result, name))
            getattr(result, name)[0][:] = 777
            np.testing.assert_array_equal(getattr(result, name), original)


@pytest.mark.parametrize('iterations', [1, 3])
def test_capped_jobs_save_evaluated_density_and_fixed_metric_source(tmp_path, iterations):
    job, _, _ = job_case(tmp_path)
    job['max_iter'] = iterations
    r = vq.run_periodic_job(**job)
    assert not r.converged and not r.native.receipt.confirmed
    next_density = []
    for spin in ('alpha', 'beta'):
        c = np.asarray(getattr(r, f'mo_coeffs_{spin}_k')); occ = np.asarray(getattr(r, f'occupations_{spin}_k'))
        next_density.append((c*occ[:, None, :]) @ c.conj().swapaxes(-1, -2))
    assert np.max(abs(np.array(next_density)-r.native.density)) > 1e-5
    np.testing.assert_array_equal(_spin_density_k_from_qvf_bloch_payload(job['output'].with_suffix('.qvf')), r.native.density)
    assert _qvf(job['output'].with_suffix('.qvf'))['provenance']['run_status'] == 'failed'
    assert len(r.scf_history) == len(r.integral_provenance['spin_scf_history']) == iterations
    job['max_iter'] = 100; job['output'] = tmp_path/'converged'/'job'
    done = vq.run_periodic_job(**job)
    assert done.converged and done.native.receipt.source_identity == r.native.receipt.source_identity
    assert [m['projector_identity'] for m in r.integral_provenance['fitting']['metrics']] == [
        m['projector_identity'] for m in done.integral_provenance['fitting']['metrics']]


@pytest.mark.parametrize('fault', ['RHF', 'RKS', 'UKS',
    'missing-aux', 'named-aux', 'metric', 'missing-fit', 'fit-type', 'unused-fit', 'spin', 'image-cap'])
def test_invalid_or_unqualified_selections_refuse_before_computation(tmp_path, fault):
    job, _, _ = job_case(tmp_path)
    if fault in ('RHF', 'RKS', 'UKS'):
        job['method'] = fault
        if fault.endswith('KS'):
            job['functional'] = 'lda'; job['periodic_xc'] = manta.PeriodicXCOptions(3.)
    elif fault == 'missing-aux': job['coulomb'] = dataclasses.replace(job['coulomb'], auxiliary_basis=None)
    elif fault == 'named-aux': job['coulomb'] = dataclasses.replace(job['coulomb'], auxiliary_basis='synthetic-aux-name')
    elif fault == 'metric': job['coulomb'] = dataclasses.replace(job['coulomb'], metric={})
    elif fault == 'missing-fit': job['periodic_fitting'] = None
    elif fault == 'fit-type': job['periodic_fitting'] = core._manta.PeriodicGDFShortRangeOptions()
    elif fault == 'unused-fit':
        job['coulomb'] = manta.CoulombBuildOptions(); job['exchange'] = manta.ExchangeBuildOptions()
    elif fault == 'spin': job['system'].multiplicity = 1
    else: job['periodic_fitting'] = dataclasses.replace(job['periodic_fitting'], image_candidate_cap=1)
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        vq.run_periodic_job(**job)
    _no_computed_artifacts(job)


@pytest.mark.parametrize('field,value', [('omega', 0.), ('omega', float('inf')), ('omega', True),
    ('auxiliary_cutoff_bohr', -1.), ('auxiliary_cutoff_bohr', float('nan')),
    ('integral_screen_error', -.01), ('integral_screen_error', float('inf')),
    ('image_candidate_cap', True), ('image_candidate_cap', 1.5), ('image_candidate_cap', 0),
    ('image_candidate_cap', 2**64)])
def test_fitting_controls_validate_scalar_source_options(field, value):
    kwargs = dict(omega=.6, auxiliary_cutoff_bohr=8.); kwargs[field] = value
    with pytest.raises(ValueError, match='PeriodicFittingOptions'):
        manta.PeriodicFittingOptions(**kwargs)


@pytest.mark.parametrize('field,planned', [('memory_bytes', 'modeled_peak_bytes'),
    ('shell_calls', 'shell_calls'), ('primitive_quartets', 'primitive_pairs'),
    ('scalar_terms', 'scalar_terms'), ('kernel_terms', 'kernel_terms')])
def test_joint_job_caps_include_generated_source_and_output(tmp_path, field, planned):
    job, a, _ = job_case(tmp_path)
    r = vq.run_periodic_job(**job)
    p = manta.plan_periodic_ri_uhf_from_cutoffs(**a)
    bound = getattr(p, planned)
    if field == 'memory_bytes':
        bound += r.integral_provenance['python_reserved_bytes']+r.integral_provenance['job_reserved_bytes']
    setattr(job['integral_engine'].budget, field, bound)
    job['output'] = tmp_path/'exact'/'job'
    assert vq.run_periodic_job(**job).converged
    setattr(job['integral_engine'].budget, field, bound-1)
    job['output'] = tmp_path/'refused'/'job'
    with pytest.raises(ValueError, match='budget'):
        vq.run_periodic_job(**job)
    _no_computed_artifacts(job)


def test_dry_run_and_oversized_output_admission_precede_native_execution(tmp_path, monkeypatch):
    job, _, _ = job_case(tmp_path/'dry'); job['dry_run'] = True
    monkeypatch.setattr(core._manta, 'run_periodic_ri_uhf_from_cutoffs', lambda *a: pytest.fail('SCF in dry run'))
    assert vq.run_periodic_job(**job) is None
    assert list(job['output'].parent.iterdir()) == [job['output'].with_suffix('.system')]
    job, _, _ = job_case(tmp_path/'huge'); job['max_iter'] = 10**8
    monkeypatch.setattr(core._manta, 'plan_periodic_ri_uhf_from_cutoffs', lambda *a: pytest.fail('native plan before output admission'))
    with pytest.raises(ValueError, match='output memory budget'):
        vq.run_periodic_job(**job)
    assert not job['output'].parent.exists()


def test_qvf_converts_independent_ao_images_and_preserves_auxiliary_gauge(tmp_path):
    job, _, _ = job_case(tmp_path/'original')
    atoms = [vq.Atom(1, p) for p in ([0., 0., 0.], [.9, .2, 0.], [-.2, 1.1, .1])]
    job['system'] = vq.PeriodicSystem(3, job['system'].lattice, atoms); job['system'].multiplicity = 2
    shells = job['basis'].shells()
    for i, shell in enumerate(shells):
        shell.atom_index = i; shell.origin = atoms[i].xyz
    job['basis'] = vq.BasisSet(job['system'].unit_cell_molecule(), shells, 'ri-job-gauge', True)
    job['max_iter'] = 3
    original = vq.run_periodic_job(**job)
    shifts = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]) @ job['system'].lattice.T
    for shell, delta in zip(shells, shifts): shell.origin = np.array(shell.origin)+delta
    image = vq.PeriodicSystem(3, job['system'].lattice, [vq.Atom(1, s.origin) for s in shells]); image.multiplicity = 2
    job['basis'] = vq.BasisSet(image.unit_cell_molecule(), shells, 'ri-job-images', True)
    auxiliary = job['coulomb'].auxiliary_basis.shells()
    for i, (shell, delta) in enumerate(zip(auxiliary, shifts[::-1])):
        shell.atom_index = i; shell.origin = np.array(shell.origin)+delta
    aux_image = vq.PeriodicSystem(3, job['system'].lattice, [vq.Atom(1, s.origin) for s in auxiliary]); aux_image.multiplicity = 2
    job['coulomb'] = dataclasses.replace(job['coulomb'], auxiliary_basis=vq.BasisSet(
        aux_image.unit_cell_molecule(), auxiliary, 'ri-job-aux-images', True))
    job['output'] = tmp_path/'moved'/'job'
    moved = vq.run_periodic_job(**job)
    assert moved.energy == pytest.approx(original.energy, abs=8e-11)
    assert np.max(abs(moved.native.density-original.native.density)) > .01
    saved = _spin_density_k_from_qvf_bloch_payload(job['output'].with_suffix('.qvf'))
    np.testing.assert_allclose(saved, original.native.density, atol=8e-11, rtol=0)
    assert moved.integral_provenance['source_ao_image_shifts'] == [[1, 0, 0], [0, -1, 0], [0, 0, 0]]


@pytest.mark.parametrize('cutoff,rank', [(10., 1), (1e9, 0)])
def test_public_metric_controls_change_the_scientific_source(tmp_path, cutoff, rank):
    job, a, k = job_case(tmp_path, species='H')
    metric = job['coulomb'].metric
    metric.absolute_cutoff = cutoff
    # Keep the independent native case explicit; pybind value properties
    # return copies when they are installed in another options owner.
    a['options'].ri = metric
    c = explicit(a, manta.build_periodic_ri_uhf_domains(**a))
    s, h, build, _ = reference(c, k)
    d = _scf_reference(s, h, build, (1, 0))
    r = vq.run_periodic_job(**job)
    assert r.converged and [m.rank for m in r.native.receipt.fitting_info.metrics] == [rank]*3
    np.testing.assert_allclose(r.native.density, d, atol=5e-9, rtol=0)
    np.testing.assert_allclose(r.native.fock, build(r.native.density)[0], atol=8e-11, rtol=0)
    before = r.integral_provenance
    metric.absolute_cutoff = 1e-12
    assert r.integral_provenance == before


def test_failed_qvf_delivery_preserves_the_computed_method_status(tmp_path, monkeypatch):
    from vibeqc.output import OutputWriter
    from vibeqc.periodic_runner import _PERIODIC_OUTPUT_WRITER
    job, _, _ = job_case(tmp_path)
    dispatch = OutputWriter.dispatch_role
    def fail(self, role, **kwargs):
        if role == 'qvf':
            raise OSError('synthetic fitted density delivery failure')
        return dispatch(self, role, **kwargs)
    monkeypatch.setattr(OutputWriter, 'dispatch_role', fail)
    before = _PERIODIC_OUTPUT_WRITER.get()
    with pytest.raises(OSError, match='synthetic fitted density delivery'):
        vq.run_periodic_job(**job)
    assert _PERIODIC_OUTPUT_WRITER.get() is before
    manifest = tomllib.loads(job['output'].with_suffix('.system').read_text())
    assert manifest['outputs']['status'] == 'crashed'
    assert manifest['outputs']['result_status'] == 'converged'


def test_output_opt_outs_and_immutable_fitting_configuration(tmp_path):
    job, _, _ = job_case(tmp_path)
    job.update(citations=False, output_qvf=False, structured_log=False, write_xyz_file=False,
               write_cif_file=False, write_xsf_structure_file=False)
    assert vq.run_periodic_job(**job).converged
    assert {p.suffix for p in job['output'].parent.iterdir()} == {'.out', '.system'}
    with pytest.raises(dataclasses.FrozenInstanceError):
        job['periodic_fitting'].omega = .7


def _as_fock_build(result):
    """Explicit owned exports into the shared API, outside native admission."""
    from vibeqc.fock_state import FockBuild, EnergyComponents, SourceProvenance
    from vibeqc.solvers.operators import OperatorFrame
    r = result.native; receipt = r.receipt; info = receipt.fitting_info
    nk, n = r.plan.momenta, r.plan.n_basis
    spins = ['restricted-total'] if r.density.ndim == 3 else ['alpha', 'beta']
    frames = []
    for ik, s in enumerate(r.overlap):
        values, vectors = np.linalg.eigh(s)
        values, vectors = values[-r.solver_memory.retained_rank:], vectors[:, -r.solver_memory.retained_rank:]
        frames.append(OperatorFrame(basis_id=info.frame_identity, n=n,
            ordering='libint-shell-axial-v1', image_gauge=f'native-exp(+ikR):k-index={ik}',
            overlap=s, X=vectors/np.sqrt(values)))
    components = (('one_electron', receipt.one_electron_energy), ('coulomb', receipt.coulomb_energy),
                  ('uncorrected_exchange', receipt.uncorrected_exchange_energy))
    energy = EnergyComponents(electronic=sum(v for _, v in components), nuclear=receipt.nuclear_energy,
        components=components, corrections=(('exchange_zero_mode', receipt.exchange_correction_energy),))
    source = SourceProvenance.capture('manta', native_source_identity=receipt.source_identity,
        coulomb_algorithm=receipt.coulomb_algorithm, exchange_algorithm=receipt.exchange_algorithm,
        coulomb_identity=receipt.coulomb_identity, exchange_identity=receipt.exchange_identity,
        overlap_identity=receipt.overlap_identity, fit_identity=info.source_identity,
        frame_identity=info.frame_identity, weight_identity=info.weight_identity,
        projectors=[m.projector_identity for m in info.metrics],
        exchange_correction_identity=receipt.exchange_correction_identity)
    return FockBuild.from_dense(method=result.integral_provenance['method'],
        generation=r.solver_snapshot.evaluated_iteration, source=source, energy=energy,
        densities=r.density.reshape(len(spins)*nk, n, n), focks=r.fock.reshape(len(spins)*nk, n, n),
        frames=frames*len(spins), spins=[spin for spin in spins for _ in range(nk)],
        kpoints=[tuple(k) for k in r.momenta]*len(spins), weights=[1./nk]*(len(spins)*nk))


@pytest.mark.parametrize('species', ['H', 'Li'])
@pytest.mark.parametrize('mesh,shift', [((1, 1, 1), (0, 0, 0)), ((2, 1, 1), (1, 0, 0))])
@pytest.mark.parametrize('correction', ['strict-zero-mode', 'BvK-ewald'])
def test_public_exports_preserve_shared_fock_build_contract_across_epochs(tmp_path, species, mesh, shift, correction):
    source_ids = []; densities = []
    for limit in (1, 3, 100):
        job, _, _ = job_case(tmp_path/str(limit), mesh, shift, species, correction)
        job['max_iter'] = limit
        result = vq.run_periodic_job(**job); r = result.native
        state = _as_fock_build(result); source_ids.append(state.source.identity); densities.append(state.density_id)
        nk, n = r.plan.momenta, r.plan.n_basis
        assert state.energy.total == pytest.approx(result.energy, abs=1e-13)
        assert state.generation == r.solver_snapshot.evaluated_iteration
        np.testing.assert_allclose(state.metric_electron_counts, [r.plan.n_alpha]*nk+[r.plan.n_beta]*nk, atol=1e-12, rtol=0)
        s, d, f = r.overlap, r.density, r.fock
        residual = f@d@s-s@d@f
        assert state.commutator_norm == pytest.approx(np.linalg.norm(residual)/np.sqrt(nk), abs=1e-12)
        trial = np.random.default_rng(894).normal(size=(n, 3))+1j*np.random.default_rng(895).normal(size=(n, 3))
        for index, channel in enumerate(state.channels):
            np.testing.assert_array_equal(channel.density, d.reshape(2*nk, n, n)[index])
            np.testing.assert_allclose(channel.operator.apply(trial), f.reshape(2*nk, n, n)[index]@trial, atol=0, rtol=0)
            assert channel.operator.metadata.role == 'physical'
            assert channel.operator.metadata.symmetry_block is None
            shifted = state.solver_operator(index, channel.operator.matrix+.1*channel.operator.overlap,
                                            modification='diagnostic test shift')
            assert shifted.metadata.role == 'solver' and shifted.metadata.build_id == state.identity
            assert shifted.metadata.density_id == state.density_id
        expected = state.channels[0].operator.matrix.copy(); f[:] = np.nan; d[:] = np.nan
        np.testing.assert_array_equal(state.channels[0].operator.matrix, expected)
    assert len(set(source_ids)) == 1
    if species == 'Li':
        assert len(set(densities)) == 3
