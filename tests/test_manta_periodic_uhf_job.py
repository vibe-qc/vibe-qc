"""Public UHF jobs preserve physical spin snapshots and lossless QVF state."""
import json
import tomllib
import zipfile

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from vibeqc.guess_read import _spin_density_k_from_qvf_bloch_payload
from tests.test_manta_periodic_job import _qvf, _no_computed_artifacts
from tests.test_manta_periodic_uhf_domains import _arguments
from tests.test_manta_periodic_domains import _explicit
from tests.test_manta_periodic_uhf import _reference, _scf_reference, _nuclear_reference


def _job(tmp_path, mesh=(2, 1, 1), shift=(1, 0, 0), species='Li'):
    args, k = _arguments(mesh, shift, species)
    job = dict(system=args['system'], basis=args['basis'], kpoints=args['mesh'], method='UHF',
        integral_engine=manta.IntegralEngineOptions(backend='manta', budget=args['budget']),
        periodic_integrals=manta.PeriodicIntegralOptions(2., 3., 4., alpha=.6),
        use_diis=False, output=tmp_path/'selected'/'job', record_hostname=False,
        structured_log=True, max_iter=100, conv_tol_energy=1e-12)
    return job, args, k


@pytest.mark.parametrize('species', ['H', 'Li'])
@pytest.mark.parametrize('mesh,shift', [((1, 1, 1), (0, 0, 0)), ((2, 1, 1), (1, 0, 0))])
def test_full_job_matches_independent_spin_scf_and_qvf_roundtrip(tmp_path, species, mesh, shift):
    from vibeqc.output import validate_qvf
    job, args, k = _job(tmp_path, mesh, shift, species)
    domains = manta.build_periodic_uhf_domains(**args)
    case = _explicit(args, domains)
    s, h, build, _ = _reference(case, k)
    populations = (1, 0) if species == 'H' else (2, 1)
    density = _scf_reference(s, h, build, populations)
    nuclear, _ = _nuclear_reference(case)
    result = vq.run_periodic_job(**job)
    assert isinstance(result, manta.PeriodicMANTAUHFResult) and result.converged
    assert result.energy == pytest.approx(sum(nuclear)+sum(build(density)[1]), abs=4e-11)
    np.testing.assert_allclose(result.native.density, density, atol=5e-10, rtol=0)
    np.testing.assert_allclose(result.native.fock, build(result.native.density)[0], atol=4e-12, rtol=0)
    np.testing.assert_array_equal(result.density_k, result.native.density.sum(axis=0))
    path = job['output'].with_suffix('.qvf')
    pair = _spin_density_k_from_qvf_bloch_payload(path, expected_n_k=len(k), n_basis=job['basis'].nbasis)
    np.testing.assert_array_equal(pair, result.native.density)
    assert validate_qvf(path)['valid']
    qvf = _qvf(path)
    assert qvf['provenance']['method'].upper() == 'UHF'
    assert qvf['provenance']['run_status'] == 'converged'
    section = next(s for s in qvf['sections'] if s['kind'] == 'x_vibeqc.periodic_integral_engine')
    with zipfile.ZipFile(path) as archive:
        evidence = json.loads(archive.read(next(iter(section['members'].values()))['path']))
        assert json.loads(archive.read('scf_history/iterations.json'))['iterations'] == result.scf_history
    assert evidence == result.integral_provenance
    assert evidence['restart_density_encoding'] == 'spin-resolved-evaluated-density'
    assert evidence['spin_electrons_per_cell'] == list(populations)
    assert len(evidence['spin_scf_history']) == result.iterations
    if species == 'H':
        assert evidence['spin_scf_history'][-1]['spin_band_gaps'][1] is None
    records = [json.loads(line) for line in job['output'].with_suffix('.scf.jsonl').read_text().splitlines()]
    steps = [r for r in records if r.get('event') == 'scf_iteration']
    assert len(steps) == result.iterations
    for step, snapshot in zip(steps, result.native.scf_history):
        assert step['energy'] == snapshot.energy_per_cell
        assert step['spin_electrons_per_cell'] == list(populations)
    manifest = tomllib.loads(job['output'].with_suffix('.system').read_text())
    assert manifest['run']['jk_method_executed'] == 'finite-3d-reciprocal-uhf'
    assert 'Finite periodic UHF: manta' in job['output'].with_suffix('.out').read_text()
    assert 'Neese' in job['output'].with_suffix('.references').read_text()
    for spin, count in zip(('alpha', 'beta'), populations):
        occupations = np.asarray(getattr(result, f'occupations_{spin}_k'))
        np.testing.assert_array_equal(occupations.sum(axis=1), np.full(len(k), count))
        for field in ('density', 'fock', 'mo_coeffs', 'mo_energies', 'occupations'):
            name = f'{field}_{spin}_k'
            original = np.array(getattr(result, name))
            getattr(result, name)[0][:] = 777
            np.testing.assert_array_equal(getattr(result, name), original)
    assert not hasattr(result, 'mo_coeffs_k') and not hasattr(result, 'fock_k')
    assert job['integral_engine'].budget.other_live_bytes == 0


def test_capped_job_stores_evaluated_spin_density_not_later_orbitals(tmp_path):
    job, _, _ = _job(tmp_path)
    job['max_iter'] = 1
    result = vq.run_periodic_job(**job)
    assert not result.converged
    next_density = []
    for spin in ('alpha', 'beta'):
        c = np.asarray(getattr(result, f'mo_coeffs_{spin}_k'))
        occ = np.asarray(getattr(result, f'occupations_{spin}_k'))
        next_density.append((c*occ[:, None, :]) @ c.conj().swapaxes(-1, -2))
    assert np.max(abs(np.asarray(next_density)-result.native.density)) > 1e-3
    path = job['output'].with_suffix('.qvf')
    np.testing.assert_array_equal(_spin_density_k_from_qvf_bloch_payload(path), result.native.density)
    assert _qvf(path)['provenance']['run_status'] == 'failed'
    assert len(result.scf_history) == len(result.integral_provenance['spin_scf_history']) == 1


@pytest.mark.parametrize('fault', ['spin', 'memory', 'trace', 'method'])
def test_invalid_or_unadmitted_job_refuses_before_computation(tmp_path, fault):
    job, _, _ = _job(tmp_path)
    if fault == 'spin': job['system'].multiplicity = 1
    elif fault == 'memory': job['integral_engine'].budget.memory_bytes = 1
    elif fault == 'trace': job['max_iter'] = 10**8
    else: job['method'] = 'UKS'
    with pytest.raises((ValueError, NotImplementedError), match='spin|budget|method'):
        vq.run_periodic_job(**job)
    _no_computed_artifacts(job)


def test_qvf_transports_both_spin_channels_to_nuclear_basis(tmp_path):
    job, _, _ = _job(tmp_path/'original')
    atoms = [vq.Atom(1, p) for p in ([0., 0., 0.], [.9, .2, 0.], [-.2, 1.1, .1])]
    job['system'] = vq.PeriodicSystem(3, job['system'].lattice, atoms)
    job['system'].multiplicity = 2
    shells = job['basis'].shells()
    for index, shell in enumerate(shells):
        shell.atom_index = index; shell.origin = atoms[index].xyz
    job['basis'] = vq.BasisSet(job['system'].unit_cell_molecule(), shells, 'manta-job-gauge', False)
    job['max_iter'] = 3
    original = vq.run_periodic_job(**job)
    shifts = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]) @ job['system'].lattice.T
    for shell, delta in zip(shells, shifts): shell.origin = np.array(shell.origin)+delta
    basis_cell = vq.PeriodicSystem(3, job['system'].lattice, [vq.Atom(1, s.origin) for s in shells])
    basis_cell.multiplicity = 2
    job['basis'] = vq.BasisSet(basis_cell.unit_cell_molecule(), shells, 'manta-job-image', False)
    job['output'] = tmp_path/'moved'/'job'
    moved = vq.run_periodic_job(**job)
    assert moved.energy == pytest.approx(original.energy, abs=4e-11)
    assert np.max(abs(moved.native.density-original.native.density)) > .01
    restored = _spin_density_k_from_qvf_bloch_payload(job['output'].with_suffix('.qvf'))
    np.testing.assert_allclose(restored, original.native.density, atol=4e-10, rtol=0)
    assert moved.integral_provenance['source_ao_image_shifts'] == [[1, 0, 0], [0, -1, 0], [0, 0, 0]]


def test_dry_run_and_delivery_failure_keep_job_lifecycle(tmp_path, monkeypatch):
    from vibeqc.output import OutputWriter
    from vibeqc.periodic_runner import _PERIODIC_OUTPUT_WRITER
    job, _, _ = _job(tmp_path/'dry')
    job['dry_run'] = True
    def fail(*a, **kw): raise AssertionError('Dry run must not solve')
    with monkeypatch.context() as patch:
        patch.setattr(core._manta, 'run_periodic_uhf_from_cutoffs', fail)
        assert vq.run_periodic_job(**job) is None
    assert not job['output'].with_suffix('.qvf').exists()
    job, _, _ = _job(tmp_path/'delivery')
    dispatch = OutputWriter.dispatch_role
    def fail_delivery(self, role, **kwargs):
        if role == 'qvf': raise OSError('synthetic spin delivery failure')
        return dispatch(self, role, **kwargs)
    monkeypatch.setattr(OutputWriter, 'dispatch_role', fail_delivery)
    before = _PERIODIC_OUTPUT_WRITER.get()
    with pytest.raises(OSError, match='synthetic spin delivery'):
        vq.run_periodic_job(**job)
    assert _PERIODIC_OUTPUT_WRITER.get() is before
    manifest = tomllib.loads(job['output'].with_suffix('.system').read_text())
    assert manifest['outputs']['status'] == 'crashed'
    assert manifest['outputs']['result_status'] == 'converged'
