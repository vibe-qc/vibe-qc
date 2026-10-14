"""ROHF runner, public gates and output contracts; no native acceptance."""
from dataclasses import replace
import json
import tomllib

import numpy as np
import pytest

from vibeqc.periodic.chi.correlation import (
    require_public_mp2_admission, require_public_cc_admission,
    validate_public_mp2_options,
)
from vibeqc.periodic.chi.fitted import ChiCoulombControls


def test_rohf_public_admission_is_named_and_keeps_other_guards():
    assert 'Knowles RMP2' in require_public_mp2_admission('mp2', 'ROHF', 'classic')
    assert validate_public_mp2_options('ROHF', 'classic', 1, 3) == ((1, 1), 3)
    with pytest.raises(ValueError, match='restricted frozen cores'):
        validate_public_mp2_options('ROHF', 'classic', (1, 0), 3)
    for source in ('sr', 'auto'):
        with pytest.raises(NotImplementedError):
            require_public_mp2_admission('mp2', 'ROHF', source)
    with pytest.raises(NotImplementedError, match='ROHF'):
        require_public_cc_admission('ROHF', 'classic')


def _runner_case(tmp_path, **changes):
    import vibeqc as vq
    cell = vq.PeriodicSystem(3, np.eye(3)*8, [vq.Atom(3, [1., 1., 1.])], 0, 2)
    basis = vq.BasisSet(cell.unit_cell_molecule(), 'sto-3g')
    args = dict(method='aiccm', variant='chi', correlation='mp2',
        scf_reference='rohf', fit_source='classic', aiccm_backend='ri',
        chi_coulomb_controls=ChiCoulombControls(6.2, 6.2, 18., 8., 5., .6),
        chi_frozen_core=1, conv_tol_energy=1e-11, conv_tol_grad=1e-9,
        kpoints=(1, 1, 1), dry_run=True, output=tmp_path/'rohf-preflight',
        output_qvf=False, record_hostname=False)
    args.update(changes)
    return vq.run_periodic_job(cell, basis, **args)


@pytest.mark.parametrize('mesh', [(1, 1, 1), (3, 1, 1)])
@pytest.mark.parametrize('correlated', [False, True])
def test_runner_rohf_preflight(tmp_path, mesh, correlated):
    changes = dict(kpoints=mesh)
    if not correlated:
        changes.update(correlation=None, chi_frozen_core=None)
    _runner_case(tmp_path, **changes)
    record = tomllib.loads((tmp_path/'rohf-preflight.system').read_text())
    assert record['outputs']['status'] == 'dry_run'


@pytest.mark.parametrize('change', [dict(correlation='ccsd'), dict(fit_source='sr'),
    dict(fit_source='auto'), dict(aiccm_backend='four_center'),
    dict(chi_coulomb_controls=None), dict(smearing_temperature=.001),
    dict(chi_frozen_core=(1, 0)), dict(functional='pbe')])
def test_runner_refuses_unsupported_rohf_before_execution(tmp_path, change):
    with pytest.raises((ValueError, NotImplementedError)):
        _runner_case(tmp_path, **change)


def test_rohf_result_output_separates_singles_and_records_definition():
    from vibeqc.periodic.chi.canonical_rohf_mp2 import CanonicalROHFMP2Result
    from vibeqc.output.chi_correlation import chi_correlation_record, write_chi_correlation
    from vibeqc.output.citations import load_default_database
    result = CanonicalROHFMP2Result(-2., -.01, -.02, -.03, ((1, 2, 3), (1, 1, 4)),
        'source', 'reference', (2,), 'ROHF-RMP2', '{}', '{}',
        e_singles_alpha_per_cell=-.004, e_singles_beta_per_cell=-.005,
        canonicalization_identity='frame')
    record = chi_correlation_record(result)
    assert record['schema'] == 'vibeqc.chi.same-source-rohf-rmp2/v1'
    assert record['reference_method'] == 'ROHF'
    assert record['energy_singles_per_cell'] == pytest.approx(-.009)
    assert record['energy_doubles_per_cell'] == pytest.approx(-.06)
    assert record['energy_total_per_cell'] == pytest.approx(-2.069)
    assert record['canonicalization_identity'] == 'frame'
    assert 'spin-semicanonical' in record['canonicalization']
    assert 'shared-lowest-closed' in record['frozen_core_convention']
    assert record['quantitative_status'] == 'not-qualified'
    output = []
    write_chi_correlation(output.append, result)
    rendered = ''.join(output)
    for text in ('ROHF-RMP2', 'E(singles)', 'E(doubles)', 'frozen convention', 'not-qualified'):
        assert text in rendered
    db = load_default_database()
    keys = [entry.key for entry in db.assemble(method=result.backend).citations]
    assert 'knowles_rmp2_1991' in keys
    assert 'guest_saunders_rohf_1974' in keys


@pytest.mark.parametrize('mesh,frozen', [(1, 0), (3, 1)])
def test_public_wrapper_transports_one_rohf_source(monkeypatch, mesh, frozen):
    """Synthetic native schema exercises dispatch, not physical native fitting."""
    from dataclasses import asdict
    from tests.test_chi_rohf_mp2 import fixture, reseal
    from vibeqc.periodic_fitted_support import ChiFittedOperator
    from vibeqc.periodic.chi.correlation import run_chi_mp2
    from vibeqc.output.chi_correlation import chi_correlation_record
    ref, _ = fixture(monkeypatch, nk=mesh, nc=2, no=1, nv=1)
    hf = ref.hf_result
    op = hf.fitted_operator
    controls = ChiCoulombControls(6.2, 6.2, 18., 8., 5., .6)
    domains = json.loads(op.descriptor.executed_domains_json)
    domains.update(schema='vibeqc.chi.fitted-domains/v1', controls=asdict(controls))
    descriptor = replace(op.descriptor, engine='native-chi-classic-finite-contract',
                         executed_domains_json=json.dumps(domains))
    hf.fitted_operator = ChiFittedOperator(op, descriptor, op._weights)
    hf.executed_fit_source = replace(descriptor, executed_domains_json=json.dumps(
        dict(fitted_operator_identity=hf.fitted_operator.identity)))
    ref = reseal(ref)
    result = run_chi_mp2(ref, frozen_core=frozen, auxiliary_batch_size=1)
    assert result.hf_result is hf
    assert ref.validate(ref.request_identity, 'ROHF') is hf
    record = chi_correlation_record(result.correlation)
    assert record['method'] == 'ROHF-RMP2'
    assert 'Knowles RMP2' in record['public_admission']
    assert record['source_identity'] == hf.fitted_operator.identity
    assert record['reference_identity'] == ref.state_identity
    assert record['energy_singles_per_cell'] < 0
    assert record['energy_correlation_per_cell'] == pytest.approx(
        record['energy_singles_per_cell'] + record['energy_doubles_per_cell'])
    assert result.e_total_correlated == pytest.approx(hf.energy+result.e_correlation)


@pytest.mark.parametrize('mesh', [1, 3])
def test_runner_executes_retained_rohf_rmp2_and_writes_manifest(monkeypatch, tmp_path, mesh):
    """Native integrals are replaced; exercise actual SCF/correlation/output wiring."""
    import vibeqc as vq
    from tests.test_chi_rohf_reference import synthetic_driver_case
    system, basis, controls, _ = synthetic_driver_case(monkeypatch, mesh)
    stem = tmp_path/'executed-rohf'
    result = vq.run_periodic_job(system, basis,
        method='aiccm', variant='chi', correlation='mp2', scf_reference='rohf',
        aiccm_backend='ri', fit_source='classic', chi_coulomb_controls=controls,
        conv_tol_energy=1e-11, conv_tol_grad=1e-9, max_iter=160,
        kpoints=(mesh, 1, 1), output=stem, output_qvf=False,
        write_density=False, record_hostname=False)
    record = tomllib.loads(stem.with_suffix('.system').read_text())
    assert result.converged
    assert result.correlation.method == 'ROHF-RMP2'
    assert result.hf_result.mo_coeffs is not None
    assert result.hf_result.runtime_backend == 'chi-classic-shared-spatial-rohf'
    run = record['run']
    assert run['chi_rohf']['n_open'] == 1
    assert run['chi_rohf']['effective_fock'] == result.rohf_convention
    assert run['chi_correlation']['canonicalization'] == result.correlation.canonicalization
    assert run['chi_correlation']['energy_total_per_cell'] == pytest.approx(result.e_total_correlated)
    assert 'ROHF-RMP2' in stem.with_suffix('.out').read_text()
    if mesh == 1:  # Existing chi sidecar admission exports Gamma-only jobs.
        assert stem.with_suffix('.molden').is_file()
