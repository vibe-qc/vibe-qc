"""Scalar chi ECP front-door and output checks; no scientific acceptance."""
import json
from types import SimpleNamespace
import tomllib

import numpy as np
import pytest

from vibeqc.periodic.chi.ecp import ecp_contract
from vibeqc.periodic.chi.fitted import ChiCoulombControls
from vibeqc.periodic_runner import _attach_periodic_ecp_data, _chi_ecp_request_contract


def scalar_case(method='RHF'):
    import vibeqc as vq
    system = vq.PeriodicSystem(3, np.eye(3)*8., [vq.Atom(6, [1., 1., 1.])],
                               0, 1 if method == 'RHF' else 3)
    basis = vq.BasisSet(system.unit_cell_molecule(), 'sto-3g')
    block = vq.ECPPrimitiveBlock()
    block.n_primitive = 2
    block.exponents = [.9, 1.25]
    block.coefficients = [.7, -2.]
    block.ams = [0, 1]
    block.ns = [2, 2]
    data = ([block], [[1., 1., 1.]], [4.], 2)
    controls = ChiCoulombControls(6.2, 6.2, 18., 8., 5., .6)
    return system, basis, data, controls


def request(system, data, controls, **changes):
    kwargs = dict(method='RHF', backend='ri', fit_source='classic',
                  gdf_method='rsgdf', support_policy=None, coulomb_controls=controls)
    kwargs.update(changes)
    return _chi_ecp_request_contract(system, data, **kwargs)


def test_runner_and_library_bind_identical_complete_ecp_payload():
    import vibeqc as vq
    system, _, data, controls = scalar_case()
    options = _attach_periodic_ecp_data(vq.PeriodicRHFOptions(), data)
    record = request(system, data, controls)
    assert record == ecp_contract(system, options)
    assert options.ecp_total_ncore == 2
    assert list(options.ecp_effective_charges) == [4.]
    assert record['explicit_electrons'] == 4
    assert record['physical_electrons'] == 6
    assert record['projectors'][0]['primitives'] == [[2, 0, .9, .7], [2, 1, 1.25, -2.]]
    changed = scalar_case()[2]
    changed[0][0].coefficients = [.8, -2.]
    assert request(system, changed, controls)['identity'] != record['identity']


@pytest.mark.parametrize('changes', [dict(method='RKS'), dict(method='UKS'),
    dict(backend='four_center'), dict(fit_source='sr'), dict(fit_source='auto'),
    dict(gdf_method='mdf'), dict(support_policy='historical'), dict(coulomb_controls=None)])
def test_ecp_admission_preserves_unimplemented_route_gates(changes):
    system, _, data, controls = scalar_case()
    with pytest.raises(NotImplementedError, match='chi scalar ECP'):
        request(system, data, controls, **changes)


def test_charge_only_metadata_cannot_bypass_ecp_preflight():
    system, _, _, controls = scalar_case()
    with pytest.raises(ValueError, match='complete explicit operator'):
        request(system, ([], [], [4.], 0), controls)
    assert request(system, ([], [], [6.], 0), None, fit_source='auto') is None
    data = scalar_case()[2]
    with pytest.raises(ValueError, match='removed core count'):
        request(system, (*data[:3], 2.5), controls)


def run_case(monkeypatch, tmp_path, method='RHF', **changes):
    import vibeqc as vq
    from vibeqc import periodic_runner as runner
    system, basis, data, controls = scalar_case(method)
    monkeypatch.setattr(runner, '_resolve_ecp_data', lambda *args: data)
    args = dict(method='aiccm', variant='chi', scf_reference=method.lower(),
        aiccm_backend='ri', fit_source='classic', chi_coulomb_controls=controls,
        conv_tol_energy=1e-11, conv_tol_grad=1e-9, kpoints=(3, 1, 1),
        output=tmp_path/'scalar-ecp', output_qvf=False, write_density=False,
        write_molden_file=False, record_hostname=False, initial_guess='hcore')
    args.update(changes)
    return vq.run_periodic_job(system, basis, **args)


@pytest.mark.parametrize('method', ['RHF', 'UHF', 'ROHF'])
@pytest.mark.parametrize('correlation', [None, 'mp2'])
def test_ecp_dry_run_records_requested_payload_without_claiming_execution(
        monkeypatch, tmp_path, method, correlation):
    run_case(monkeypatch, tmp_path, method, dry_run=True, correlation=correlation)
    record = tomllib.loads((tmp_path/'scalar-ecp.system').read_text())
    assert record['outputs']['status'] == 'dry_run'
    request_record = record['run']['requested_chi_ecp']
    assert request_record['ecp_core_electrons'] == 2
    assert request_record['explicit_electrons'] == 4
    assert 'chi_ecp' not in record['run']
    assert 'executed_support' not in request_record


@pytest.mark.parametrize('method', ['RHF', 'UHF', 'ROHF'])
def test_runner_delivers_primitive_operator_to_selected_hf_driver(monkeypatch, tmp_path, method):
    from vibeqc import periodic_runner as runner
    observed = {}

    class ReachedSource(RuntimeError):
        pass

    def capture(system, basis, mesh, options, **kwargs):
        observed.update(contract=ecp_contract(system, options), mesh=mesh, kwargs=kwargs)
        assert options.ecp_primitive_blocks[0].coefficients == [.7, -2.]
        raise ReachedSource('actual source boundary reached')

    monkeypatch.setattr(runner, 'run_aiccm2026dev_b_'+method.lower(), capture)
    with pytest.raises(ReachedSource, match='actual source boundary'):
        run_case(monkeypatch, tmp_path, method)
    assert observed['contract']['explicit_electrons'] == 4
    assert observed['mesh'] == (3, 1, 1)
    assert observed['kwargs']['fit_source'] == 'classic'
    assert observed['kwargs']['coulomb_controls'].support_policy == 'physical_pairs'


@pytest.mark.parametrize('changes', [dict(fit_source='sr'), dict(fit_source='auto'),
    dict(aiccm_backend='four_center'), dict(chi_coulomb_controls=None),
    dict(functional='pbe')])
def test_ecp_unsupported_requests_do_not_publish_dry_run_success(monkeypatch, tmp_path, changes):
    with pytest.raises((ValueError, NotImplementedError)):
        run_case(monkeypatch, tmp_path, dry_run=True, **changes)
    for path in tmp_path.glob('*.system'):
        assert tomllib.loads(path.read_text())['outputs']['status'] != 'dry_run'


def test_runner_rechecks_complete_payload_before_source_dispatch(monkeypatch, tmp_path):
    import vibeqc as vq
    from vibeqc import periodic_runner as runner
    system, basis, data, controls = scalar_case()
    monkeypatch.setattr(runner, '_resolve_ecp_data', lambda *args: data)
    original = runner._chi_ecp_request_contract
    snapshots = []

    def changed_after_preflight(*args, **kwargs):
        record = original(*args, **kwargs)
        snapshots.append(record)
        if len(snapshots) == 1:
            # The guess selector validates before this snapshot as well as
            # after it. Mutate at the actual completed-contract boundary.
            data[0][0].coefficients = [.8, -2.]
        return record

    def forbidden_driver(*args, **kwargs):
        pytest.fail('a changed ECP request must fail before source dispatch')

    monkeypatch.setattr(runner, '_chi_ecp_request_contract', changed_after_preflight)
    monkeypatch.setattr(runner, 'run_aiccm2026dev_b_rhf', forbidden_driver)
    with pytest.raises(ValueError, match='ECP payload changed after preflight'):
        vq.run_periodic_job(system, basis, method='aiccm', variant='chi',
            scf_reference='rhf', aiccm_backend='ri', fit_source='classic',
            chi_coulomb_controls=controls, kpoints=(3, 1, 1),
            initial_guess='HCORE', output=tmp_path/'changed-ecp',
            output_qvf=False, write_density=False, record_hostname=False)
    assert len(snapshots) == 2
    assert snapshots[0]['projectors'][0]['primitives'][0][3] == .7
    assert snapshots[1]['projectors'][0]['primitives'][0][3] == .8


def test_output_reports_actual_components_and_separate_removed_and_frozen_counts():
    from vibeqc.output.chi_correlation import (
        chi_ecp_record, write_chi_ecp, chi_correlation_record, write_chi_correlation)
    from vibeqc.periodic.chi.canonical_mp2 import CanonicalMP2Result
    from vibeqc.output.citations import load_default_database
    system, _, data, controls = scalar_case()
    ecp = request(system, data, controls)
    ecp.update(executed_support={'projector_images': [[0, 0, 0], [1, 0, 0]]},
        energy_components=dict(kinetic=2., ionic_coulomb=-4., ecp_local=.3,
                               ecp_nonlocal=.2, ionic_ewald=-.7))
    hf = SimpleNamespace(chi_ecp=ecp)
    copied = chi_ecp_record(hf)
    assert copied == ecp and copied is not ecp
    counts = dict(physical_electrons=6, ecp_core_electrons=2, explicit_electrons=4,
        frozen_explicit_electrons=2, correlated_electrons=2,
        occupied_per_spin=[2, 2], frozen_explicit_per_spin=[1, 1], correlated_per_spin=[1, 1],
        frozen_convention='explicit occupied orbitals only; ECP cores already removed')
    result = CanonicalMP2Result(-2., -.01, -.02, -.03, ((1, 1, 3), (1, 1, 3)),
        'source', 'reference', (2,), 'MP2', '{}', '{}',
        ecp_json=json.dumps(ecp), electron_counts_json=json.dumps(counts))
    record = chi_correlation_record(result)
    assert record['ecp'] == ecp
    assert record['electron_counts'] == counts
    assert {c['doi'] for c in record['ecp_citations']} == {'10.1063/1.4986887', '10.1021/cr2001383'}
    rendered = []
    write_chi_ecp(rendered.append, hf)
    write_chi_correlation(rendered.append, result)
    text = ''.join(rendered)
    for token in ('scalar-semilocal', ecp['identity'], 'E(ecp_local)', 'E(ecp_nonlocal)',
                  'E(ionic_ewald)', 'frozen_explicit_electrons', 'correlated_electrons', 'not-qualified',
                  '10.1063/1.4986887', '10.1021/cr2001383'):
        assert token in text
    keys = {entry.key for entry in load_default_database().assemble(
        method=result.backend, uses_ecp=True).citations}
    assert {'shaw_gilbert_libecpint', 'dolg_cao_pseudopotentials_2011'} <= keys
    with pytest.raises(ValueError, match='complete executed'):
        chi_ecp_record(SimpleNamespace(chi_ecp=request(system, data, controls)))
