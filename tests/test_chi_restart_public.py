"""Public restart wiring on fake integrals; diagnostic, not native acceptance."""
from dataclasses import replace
import copy
import json
import tomllib

import numpy as np
import pytest

from vibeqc.periodic.chi import rohf
from vibeqc.periodic.chi import checkpoint as cp
from vibeqc.periodic.chi import reference as refs


def restart_driver_case(monkeypatch, mesh):
    """Attach the real orbital-basis identity to the synthetic source boundary."""
    from tests.test_chi_rohf_reference import synthetic_driver_case
    from vibeqc.aux_basis import _basis_fingerprint
    from vibeqc.periodic_fitted_support import ChiFittedOperator
    system, basis, controls, options = synthetic_driver_case(monkeypatch, mesh)
    original = rohf._build_source

    def source(*args, **kwargs):
        values = list(original(*args, **kwargs))
        op = values[3]
        physical = dict(geometry=cp._geometry_record(system), basis=cp._basis_record(basis))
        descriptor = replace(op.descriptor, orbital_basis_sha256=_basis_fingerprint(basis),
            source_signature=cp._physical_source_signature(physical, op.descriptor.auxiliary_basis_sha256))
        values[3] = ChiFittedOperator(op, descriptor, op._weights)
        return tuple(values)

    monkeypatch.setattr(rohf, '_build_source', source)
    return system, basis, controls, options


def forbidden_source(*args, **kwargs):
    raise AssertionError('restart must not rebuild integrals or rerun SCF')


def runner(system, basis, controls, mesh, stem, **options):
    import vibeqc as vq
    kwargs = dict(method='aiccm', variant='chi', correlation='mp2', scf_reference='rohf',
        aiccm_backend='ri', fit_source='classic', chi_coulomb_controls=controls,
        conv_tol_energy=1e-11, conv_tol_grad=1e-9, max_iter=160,
        kpoints=(mesh, 1, 1), output=stem, output_qvf=False,
        write_density=False, record_hostname=False)
    kwargs.update(options)
    return vq.run_periodic_job(system, basis, **kwargs)


@pytest.mark.parametrize('mesh', [1, 3])
def test_runner_restores_hf_and_completed_rmp2_without_source_rebuild(monkeypatch, tmp_path, mesh):
    system, basis, controls, _ = restart_driver_case(monkeypatch, mesh)
    hf_path, corr_path = tmp_path/'reference.vqr', tmp_path/'correlation.vqr'
    fresh_stem, resumed_stem = tmp_path/'fresh', tmp_path/'resumed'
    fresh = runner(system, basis, controls, mesh, fresh_stem,
        chi_reference_checkpoint=hf_path, chi_correlation_checkpoint=corr_path,
        chi_checkpoint_interval=3)
    fresh_ref = refs._seal_chi_hf_result(fresh.hf_result, 'ROHF')
    portable = cp.portable_reference_identity(fresh_ref)
    monkeypatch.setattr(rohf, '_build_source', forbidden_source)
    monkeypatch.setattr(rohf, 'optimize_rohf', forbidden_source)
    resumed = runner(system, basis, controls, mesh, resumed_stem,
        chi_reference_from=hf_path, chi_correlation_resume=corr_path)
    assert resumed.converged
    assert resumed.e_total_correlated == pytest.approx(fresh.e_total_correlated, abs=2e-13)
    assert resumed.hf_result.energy == pytest.approx(fresh.hf_result.energy, abs=1e-12)
    resumed_ref = refs._seal_chi_hf_result(resumed.hf_result, 'ROHF')
    assert cp.portable_reference_identity(resumed_ref) == portable
    assert resumed_ref.validate(resumed_ref.request_identity, 'ROHF') is resumed.hf_result
    assert resumed.correlation.source_identity == fresh.correlation.source_identity
    assert resumed.correlation.canonicalization_identity == fresh.correlation.canonicalization_identity
    assert resumed.correlation.restart_identity == fresh.correlation.restart_identity
    assert resumed.correlation.restart_mode == 'restored-completed'
    assert resumed.correlation.resumed_work_units == resumed.correlation.total_work_units
    for stem, mode in ((fresh_stem, 'fresh'), (resumed_stem, 'restored-completed')):
        record = tomllib.loads(stem.with_suffix('.system').read_text())['run']
        assert record['chi_correlation']['method'] == 'ROHF-RMP2'
        assert record['chi_correlation']['restart']['restart_mode'] == mode
        assert record['chi_correlation']['quantitative_status'] == 'not-qualified'
        assert record['chi_rohf']['n_open'] == 1
        assert record['chi_correlation']['energy_total_per_cell'] == pytest.approx(fresh.e_total_correlated)
        assert str(tmp_path) not in json.dumps(record['chi_correlation']['restart'])
    text = resumed_stem.with_suffix('.out').read_text()
    assert 'restored-validated-reference' in text and 'restored-completed' in text
    assert '0 new iterations' in text
    restart = tomllib.loads(resumed_stem.with_suffix('.system').read_text())['run']['chi_restart']
    assert restart['new_scf_iterations'] == 0
    assert restart['original_scf_iterations'] == fresh.hf_result.n_iter
    if mesh == 1:
        assert resumed_stem.with_suffix('.molden').is_file()


def test_runner_resumes_unfinished_rmp2_singles_and_doubles(monkeypatch, tmp_path):
    from vibeqc.output import checkpoint as storage
    from vibeqc.periodic.chi.correlation import run_chi_mp2
    from tests.test_chi_mp2_checkpoint import ForcedInterruption, interrupt_after
    system, basis, controls, _ = restart_driver_case(monkeypatch, 3)
    hf_path, corr_path = tmp_path/'reference.vqr', tmp_path/'correlation.vqr'
    writer = interrupt_after(monkeypatch, 1)  # Alpha singles published, beta and doubles pending.
    with pytest.raises(ForcedInterruption):
        runner(system, basis, controls, 3, tmp_path/'interrupted',
            chi_reference_checkpoint=hf_path, chi_correlation_checkpoint=corr_path)
    monkeypatch.setattr(storage, 'write_restart_state', writer)
    monkeypatch.setattr(rohf, '_build_source', forbidden_source)
    monkeypatch.setattr(rohf, 'optimize_rohf', forbidden_source)
    expected = run_chi_mp2(cp.load_chi_hf_reference(hf_path)).correlation
    resumed = runner(system, basis, controls, 3, tmp_path/'resumed',
        chi_reference_from=hf_path, chi_correlation_resume=corr_path)
    assert resumed.correlation.restart_mode == 'resumed-correlation'
    assert resumed.correlation.resumed_work_units == 1
    assert resumed.correlation.energy == pytest.approx(expected.energy, abs=2e-13)
    assert resumed.correlation.e_singles_per_cell == pytest.approx(expected.e_singles_per_cell, abs=2e-13)
    assert resumed.correlation.e_doubles_per_cell == pytest.approx(expected.e_doubles_per_cell, abs=2e-13)
    record = tomllib.loads((tmp_path/'resumed.system').read_text())['run']['chi_correlation']
    assert record['restart']['restart_mode'] == 'resumed-correlation'
    assert record['restart']['resumed_work_units'] == 1
    assert record['restart']['completed_work_units'] == record['restart']['total_work_units']


@pytest.mark.parametrize('change', ['controls', 'mesh', 'source', 'spin'])
def test_runner_incompatible_hf_restart_refuses_without_scf_fallback(monkeypatch, tmp_path, change):
    system, basis, controls, _ = restart_driver_case(monkeypatch, 1)
    path = tmp_path/'reference.vqr'
    runner(system, basis, controls, 1, tmp_path/'original', chi_reference_checkpoint=path)
    monkeypatch.setattr(rohf, '_build_source', forbidden_source)
    monkeypatch.setattr(rohf, 'optimize_rohf', forbidden_source)
    kwargs = dict(chi_reference_from=path)
    if change == 'controls':
        controls = replace(controls, pair_cutoff_bohr=controls.pair_cutoff_bohr+1)
    elif change == 'mesh':
        kwargs['kpoints'] = (3, 1, 1)
    elif change == 'source':
        kwargs['fit_source'] = 'sr'
    else:
        kwargs['scf_reference'] = 'uhf'
    with pytest.raises((ValueError, NotImplementedError)):
        runner(system, basis, controls, 1, tmp_path/'incompatible', **kwargs)


@pytest.mark.parametrize('change', ['shared-path', 'missing-reference', 'interval', 'unrelated-method'])
def test_restart_options_fail_before_source_or_output(monkeypatch, tmp_path, change):
    system, basis, controls, _ = restart_driver_case(monkeypatch, 1)
    monkeypatch.setattr(rohf, '_build_source', forbidden_source)
    kwargs = {}
    if change == 'shared-path':
        kwargs = dict(chi_reference_checkpoint=tmp_path/'state.vqr',
                      chi_correlation_checkpoint=tmp_path/'.'/'state.vqr')
    elif change == 'missing-reference':
        kwargs = dict(chi_correlation_resume=tmp_path/'correlation.vqr')
    elif change == 'interval':
        kwargs = dict(chi_checkpoint_interval=0, chi_reference_checkpoint=tmp_path/'state.vqr')
    else:
        kwargs = dict(method='pbe', chi_reference_checkpoint=tmp_path/'state.vqr')
    with pytest.raises((ValueError, NotImplementedError)):
        runner(system, basis, controls, 1, tmp_path/'invalid', **kwargs)
    assert not list(tmp_path.glob('*.vqr'))


def test_library_prepare_checkpoint_and_restore_uses_exact_request(monkeypatch, tmp_path):
    system, basis, controls, options = restart_driver_case(monkeypatch, 3)
    path = tmp_path/'library.vqr'
    kwargs = dict(method='ROHF', backend='ri', fit_source='classic', coulomb_controls=controls)
    original = refs.prepare_chi_hf_reference(system, basis, (3, 1, 1), options,
        checkpoint_path=path, **kwargs)
    monkeypatch.setattr(rohf, '_build_source', forbidden_source)
    monkeypatch.setattr(rohf, 'optimize_rohf', forbidden_source)
    restored = refs.prepare_chi_hf_reference(system, basis, (3, 1, 1), options,
        restart_from=path, **kwargs)
    assert restored.validate(restored.request_identity, 'ROHF').energy == pytest.approx(original.hf_result.energy)
    assert cp.portable_reference_identity(restored) == cp.portable_reference_identity(original)
    kwargs['coulomb_controls'] = replace(controls, pair_cutoff_bohr=controls.pair_cutoff_bohr+1)
    with pytest.raises(ValueError, match='request|controls'):
        refs.prepare_chi_hf_reference(system, basis, (3, 1, 1), options, restart_from=path, **kwargs)


@pytest.mark.parametrize('rewrite_physical_controls', [False, True])
def test_rechecksummed_request_cannot_relabel_the_executed_source(monkeypatch, tmp_path,
                                                                  rewrite_physical_controls):
    """An opaque request digest cannot certify a different source cutoff."""
    from vibeqc.output.checkpoint import read_restart_state, write_restart_state
    system, basis, controls, options = restart_driver_case(monkeypatch, 1)
    path = tmp_path/'original.vqr'
    kwargs = dict(method='ROHF', backend='ri', fit_source='classic', coulomb_controls=controls)
    original = refs.prepare_chi_hf_reference(system, basis, (1, 1, 1), options,
        checkpoint_path=path, **kwargs)
    metadata, arrays = read_restart_state(path)
    physical = original.hf_result._chi_restart_physical
    changed = copy.deepcopy(physical)
    changed['controls']['coulomb_controls']['pair_cutoff_bohr'] += 1.
    g = changed['geometry']
    geometry = (g['dimension'], np.asarray(g['lattice_bohr']),
                [(atom['Z'], tuple(atom['xyz_bohr'])) for atom in g['nuclei']],
                g['charge'], g['multiplicity'])
    forged_request = refs._digest((geometry, changed['basis']['sha256'], tuple(changed['mesh']),
        changed['options'], 'ROHF', changed['controls'], 'finite-character/full-Coulomb/BvK-ewald'))
    forged = replace(original, request_identity=forged_request)
    metadata['request_identity'] = forged_request
    if rewrite_physical_controls:
        # Even internally consistent requested controls must agree with the
        # executed source/support; no factors or source descriptor are changed.
        original.hf_result._chi_restart_physical = changed
        metadata['physical'] = cp._encode(changed, arrays)
    try:
        metadata['physical_identity'] = cp.portable_reference_identity(forged)
    finally:
        original.hf_result._chi_restart_physical = physical
    tampered = write_restart_state(tmp_path/'tampered.vqr', metadata, arrays)
    monkeypatch.setattr(rohf, '_build_source', forbidden_source)
    monkeypatch.setattr(rohf, 'optimize_rohf', forbidden_source)
    kwargs['coulomb_controls'] = replace(controls, pair_cutoff_bohr=controls.pair_cutoff_bohr+1.)
    with pytest.raises(ValueError, match='request|controls|support'):
        refs.prepare_chi_hf_reference(system, basis, (1, 1, 1), options,
            restart_from=tampered, **kwargs)
