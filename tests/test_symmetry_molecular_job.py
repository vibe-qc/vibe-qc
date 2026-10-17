"""Qualified sources survive ordinary jobs and declarative QVF execution."""
import copy
import json
import tomllib
import zipfile

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import runner
from vibeqc._qvf_options import hydrate_option
from vibeqc.qvf_job import load_job_container, run_container, write_pending_qvf
from vibeqc.symmetry_molecular_job import source_request
from tests.test_symmetry_molecular_scf import config
from tests.test_symmetry_quartets import molecular_case, dense_jk


def system():
    mol, basis, r, t = molecular_case('water')
    return mol, basis, config(r, t)


def job(mol, reduction, stem, **kwargs):
    return vq.run_job(mol, basis='sto-3g', method=kwargs.pop('method', 'rhf'),
        symmetry_reduction=reduction, output=stem, name_molecule=False,
        progress=False, record_hostname=False, crash_dump=False, **kwargs)


@pytest.mark.parametrize('method', ['rhf', 'uhf'])
def test_real_job_prepared_source_outputs_and_owned_state(tmp_path, monkeypatch, method):
    mol, basis, reduction = system()
    prepared = vq.prepare_molecular_symmetry(mol, basis, symmetry_reduction=reduction)
    def legacy(*args, **kwargs):
        raise AssertionError('qualified source entered a legacy SCF/guess route')
    for name in ('run_rhf', 'run_uhf', '_prepare_molecular_special_guess'):
        monkeypatch.setattr(runner, name, legacy)
    opts = vq.RHFOptions() if method == 'rhf' else vq.UHFOptions()
    opts.initial_guess = vq.InitialGuess.HCORE
    opts.conv_tol_energy, opts.conv_tol_grad = 1e-11, 1e-9
    result = job(mol, prepared, tmp_path/'qualified', method=method,
        **{method+'_options': opts}, structured_log=True)
    assert result.converged and result.orbital_kind == 'natural'
    assert opts.scf_mode == vq.SCFMode.AUTO
    assert result.symmetry_provenance['source_audit_work']['shell_calls'] == 0
    eri = np.asarray(vq.compute_eri(basis)).reshape((basis.nbasis,)*4)
    h = vq.compute_kinetic(basis)+vq.compute_nuclear(basis, mol)
    channels = result.fock_build.channels
    j = dense_jk(eri, sum(c.density for c in channels))[0]
    energy = mol.nuclear_repulsion()
    for channel in channels:
        d, f = channel.density, channel.operator.matrix
        expected = h+j-(.5 if method == 'rhf' else 1.)*dense_jk(eri, d)[1]
        np.testing.assert_allclose(f, expected, atol=2e-11, rtol=0)
        energy += .5*np.einsum('pq,qp', d, h+expected)
    assert abs(energy-result.energy) < 2e-11
    with zipfile.ZipFile(tmp_path/'qualified.qvf') as archive:
        manifest = json.loads(archive.read('manifest.json'))
        fields = manifest['provenance']['run_fields']
        assert fields['symmetry_source'] == result.run_fields['symmetry_source']
        assert fields['evaluated_density_id'] == result.fock_build.density_id
        assert fields['orbital_kind'] == 'natural'
        localized = [s for s in manifest['sections'] if s.get('id', '').startswith('wf_localized')]
        assert len(localized) == (3 if method == 'rhf' else 0)
    spec = load_job_container(tmp_path/'qualified.qvf').spec
    assert spec['options']['symmetry_reduction'] == source_request(reduction)
    text = (tmp_path/'qualified.out').read_text()
    assert 'FAILED' not in text and 'POSSIBLY CONDUCTING' not in text
    assert 'physical Fock Rayleigh quotients' in text
    assert all(label not in text for label in ('HOMO', 'LUMO', 'HOAMO', 'HOBMO', 'gap:'))
    from vibeqc.output.formats.cube import requested_mo_indices
    with pytest.raises(ValueError, match='explicit MO indices'):
        requested_mo_indices(['homo'], result, mol)
    if method == 'rhf':
        assert requested_mo_indices([0], result, mol) == [(0, 'mo_0')]
    population = json.loads((tmp_path/'qualified.population.json').read_text())
    assert not population['errors']
    citations = (tmp_path/'qualified.bibtex').read_text()
    assert 'valeev_libint' in citations and 'haser_ahlrichs_schwarz' not in citations
    assert (tmp_path/'qualified.molden').is_file()


def test_pending_container_fresh_qualification_and_config_preservation(tmp_path):
    mol, _, reduction = system()
    request = source_request(reduction)
    path = write_pending_qvf(mol, tmp_path/'pending', method='rhf', basis='sto-3g',
        options={'symmetry_reduction': request, 'rhf_options': {
            'initial_guess': 'hcore', 'conv_tol_energy': 1e-11, 'conv_tol_grad': 1e-9}})
    result = run_container(path, progress=False, record_hostname=False, name_molecule=False)
    assert result.converged
    assert not result.symmetry_provenance['prepared_source']['reused']
    assert result.symmetry_provenance['source_audit_work']['shell_calls'] > 0
    assert load_job_container(path).spec['options']['symmetry_reduction'] == request


@pytest.mark.parametrize('kwargs', [
    {'method': 'mp2'}, {'method': 'rks'}, {'optimize': True},
    {'geom_opt': 'lbfgs'}, {'hessian': True}, {'tddft': True}, {'nto': True},
    {'atomization': True}, {'qtaim': True}, {'solvent': 'water'}, {'dispersion': 'd4'},
    {'density_fit': True}, {'aux_basis': 'sto-3g'}, {'cosx': True},
    {'integral_engine': object()}, {'citype': 'ccsd'}, {'triples': 't0'},
    {'smearing_temperature': .01}, {'read_from': 'restart.qvf'}, {'fragments': []},
])
def test_unqualified_requests_refuse_before_basis(tmp_path, monkeypatch, kwargs):
    mol, _, reduction = system()
    def unexpected(*args, **kwargs):
        raise AssertionError('unqualified request reached basis construction')
    monkeypatch.setattr(runner, 'BasisSet', unexpected)
    with pytest.raises(NotImplementedError, match='Qualified molecular symmetry'):
        job(mol, reduction, tmp_path/'refused', **kwargs)
    assert [p.suffix for p in tmp_path.iterdir()] == ['.system']
    with (tmp_path/'refused.system').open('rb') as stream:
        assert tomllib.load(stream)['outputs']['result_status'] == 'refused'


def test_failed_job_never_retries_an_unqualified_source(tmp_path, monkeypatch):
    mol, _, reduction = system()
    def unexpected(*args, **kwargs):
        raise AssertionError('qualified source entered a legacy SAD retry')
    monkeypatch.setattr(runner, '_rhf_tail_sad_retry_supported', unexpected)
    opts = vq.RHFOptions(); opts.initial_guess = vq.InitialGuess.HCORE; opts.max_iter = 0
    with pytest.raises(RuntimeError, match='did not converge after 0 iterations'):
        job(mol, reduction, tmp_path/'stopped', rhf_options=opts)
    assert not (tmp_path/'stopped.qvf').exists()


def test_omitted_options_choose_hcore_but_auto_cannot_replace_correlation(tmp_path):
    mol = vq.Molecule([vq.Atom(1, [0, 0, -.7]), vq.Atom(1, [0, 0, .7])])
    reduction = config(np.array([np.eye(3), -np.eye(3)]), np.zeros((2, 3)))
    result = job(mol, reduction, tmp_path/'default')
    assert result.converged
    assert 'initial_guess = HCORE' in (tmp_path/'default.out').read_text()
    with pytest.raises(NotImplementedError, match="resolved to 'fci'"):
        job(mol, reduction, tmp_path/'auto', method='auto')
    with pytest.raises(ValueError, match='HCORE or supplied-density READ'):
        job(mol, reduction, tmp_path/'sad', initial_guess='sad')


@pytest.mark.parametrize('mutate', [
    lambda v: v.update(source_identity='untrusted-seal'),
    lambda v: v.update(rotations=[np.eye(3).tolist()]),
    lambda v: v['rotations'][0][0].__setitem__(0, True),
    lambda v: v['translations'][0].__setitem__(0, float('inf')),
    lambda v: v.update(expected_builds=True),
    lambda v: v['budget'].update(maximum_bytes=True),
    lambda v: v['budget'].update(maximum_bytes=1),
    lambda v: v.pop('source_tolerance'),
])
def test_json_records_cannot_supply_authority_or_malformed_operations(mutate):
    _, _, reduction = system()
    request = source_request(reduction)
    mutate(request)
    with pytest.raises((ValueError, RuntimeError, MemoryError), match='budget|options.symmetry_reduction|resource'):
        hydrate_option('symmetry_reduction', request)


def test_json_roundtrip_preserves_resources_and_cost_policy():
    _, _, reduction = system()
    request = source_request(reduction)
    request.update(executor='auto', expected_builds=80)
    original = copy.deepcopy(request)
    decoded = hydrate_option('symmetry_reduction', request)
    assert source_request(decoded) == original == request
    assert decoded.executor == 'auto' and decoded.expected_builds == 80


@pytest.mark.parametrize('method,kind', [('rhf', 'water'), ('uhf', 'ammonia')])
def test_ri_job_and_executable_request(tmp_path, method, kind):
    from tests.test_symmetry_ri import case
    from tests.test_symmetry_molecular_scf import ri_config
    source, mol, basis, _, r, t = case(kind)
    reduction = ri_config(r, t, source.irreps.characters, executor='auto', expected_builds=2)
    prepared = vq.prepare_molecular_symmetry(mol, basis, symmetry_reduction=reduction)
    result = job(mol, prepared, tmp_path/'ri', method=method)
    assert result.converged
    request = source_request(prepared)
    assert request['source_kind'] == 'abelian-ri'
    assert load_job_container(tmp_path/'ri.qvf').spec['options']['symmetry_reduction'] == request
    path = write_pending_qvf(mol, tmp_path/'fresh', method=method, basis='sto-3g',
        options={'symmetry_reduction': request})
    fresh = run_container(path, progress=False, record_hostname=False, name_molecule=False)
    assert fresh.converged
    assert abs(fresh.energy-result.energy) < 2e-11
    # QVF geometry unit conversion can change last bits. Fresh qualification
    # must describe that actual source, never borrow the previous digest.
    for a, b in zip(fresh.fock_build.channels, result.fock_build.channels):
        np.testing.assert_allclose(a.operator.matrix, b.operator.matrix, atol=2e-10, rtol=0)
    assert fresh.symmetry_provenance['fitting_setup_work']['metric_shell_calls'] > 0
    assert not any(result.symmetry_provenance['fitting_setup_work'].values())
    assert fresh.symmetry_provenance['one_electron_qualification']['one_electron_build_calls'] == 3
    assert not fresh.symmetry_provenance['prepared_source']['reused']
    with zipfile.ZipFile(path) as archive:
        fields = json.loads(archive.read('manifest.json'))['provenance']['run_fields']
        assert fields['symmetry_source'] == fresh.fock_build.source.as_dict()
        assert fields['evaluated_density_id'] == fresh.fock_build.density_id
        assert fields['orbital_kind'] == 'natural'
    text = (tmp_path/'fresh.out').read_text()
    assert 'physical Fock Rayleigh quotients' in text and 'Abelian RI' in text
    citations = (tmp_path/'fresh.bibtex').read_text()
    assert 'melega' in citations.lower() and 'kitsaras' in citations.lower()


@pytest.mark.parametrize('mutate', [
    lambda v: v.update(source_identity='untrusted'),
    lambda v: v.update(source_kind='periodic-ri'),
    lambda v: v.update(characters_imag=[[0.]]),
    lambda v: v['characters_real'][0].__setitem__(0, True),
    lambda v: v['characters_imag'][0].__setitem__(0, float('nan')),
    lambda v: v.update(mode_block=True),
    lambda v: v.update(relative_cutoff=-1.),
    lambda v: v.update(factor_export='unqualified-shortcut'),
    lambda v: v.update(auxiliary_basis=''),
    lambda v: v.update(maximum_omitted_norm=0.),
    lambda v: v.update(budget={'maximum_bytes': 1, 'maximum_work': 100}),
])
def test_ri_request_rejects_malformed_or_unqualified_domains(mutate):
    from tests.test_symmetry_ri import case
    from tests.test_symmetry_molecular_scf import ri_config
    source, _, _, _, r, t = case('water')
    request = source_request(ri_config(r, t, source.irreps.characters))
    mutate(request)
    with pytest.raises((ValueError, TypeError, MemoryError)):
        hydrate_option('symmetry_reduction', request)


@pytest.mark.parametrize('exporter', ['producer', 'ordered-tiles'])
def test_ri_native_policy_and_exporter_survive_job_request(exporter):
    from tests.test_symmetry_ri import case
    from tests.test_symmetry_molecular_scf import ri_config
    source, _, _, _, r, t = case('water')
    config = ri_config(r, t, source.irreps.characters, factor_export=exporter,
        absolute_cutoff=3e-13, relative_cutoff=2e-9,
        negative_relative_tolerance=7e-11, mode_block=3)
    request = source_request(config)
    restored = hydrate_option('symmetry_reduction', request)
    assert source_request(restored) == request
    assert restored.factor_export == exporter
    assert restored.mode_block == 3
