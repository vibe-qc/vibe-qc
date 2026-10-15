"""Independent scalar-ECP source/count/restart checks, not native acceptance."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc.output import checkpoint as storage
from vibeqc.periodic.chi import checkpoint as cp, reference as refs
from vibeqc.periodic.chi import canonical_mp2 as mp
from vibeqc.periodic.chi import canonical_rohf_mp2 as rmp
from vibeqc.periodic.chi.ecp import ecp_contract, primitive_record, validate_ecp_reference
from vibeqc.periodic_fit_source import factor_digest
from vibeqc.periodic_fitted_support import ChiFittedOperator
from tests.test_chi_hf_checkpoint import fixture as ae_fixture
from tests.test_chi_canonical_mp2 import spin_orbital_oracle


def inputs():
    system = SimpleNamespace(dim=3, charge=0, multiplicity=1,
        unit_cell=[SimpleNamespace(Z=4, xyz=[0.,0.,0.])], n_electrons=lambda:4)
    block = dict(n_primitive=3, exponents=[1.2,.7,1.], coefficients=[.6,-.3,0.],
                 ams=[0,1,2], ns=[2,2,2])
    options = dict(ecp_primitive_blocks=[block], ecp_home_centers=[[0.,0.,0.]],
                   ecp_effective_charges=[2.], ecp_total_ncore=2)
    return system, options


def ecp_fixture(monkeypatch, method='RHF', nk=3):
    reference, system, basis = ae_fixture(monkeypatch, method, nk)
    hf = reference.hf_result
    explicit = system.n_electrons()
    system.unit_cell[0].Z += 2
    system.n_electrons = lambda: explicit+2
    _, options = inputs()
    options['ecp_effective_charges'] = [explicit]
    record = ecp_contract(system, options)
    record['executed_support'] = dict(selector='independent finite model', cells=[[0,0,0]])
    hf.chi_ecp = record
    hf.ecp_total_ncore = 2
    hf.effective_n_electrons = explicit
    hf.ecp_effective_charges = [explicit]
    hf.aiccm2026dev_b = cp._AttributeRecord(dict(hf.aiccm2026dev_b, ecp_total_ncore=2))
    hf.hcore_components = dict(kinetic=.5*hf.hcore, ionic_coulomb=.4*hf.hcore,
        ecp_local=.04*hf.hcore, ecp_nonlocal=.06*hf.hcore)
    _energies(hf)
    physical = hf._chi_restart_physical
    cp.bind_chi_restart_inputs(hf, system, basis, (nk,1,1), options, method, physical['controls'])
    _source(hf)
    hf._chi_hf_request_identity = cp._physical_request_identity(hf._chi_restart_physical)
    if method == 'ROHF':
        from vibeqc.periodic.chi.rohf import validate_rohf_state
        hf.chi_scf_acceptance['generalized_rohf_gradient_norm'] = max(validate_rohf_state(hf).values())
    return refs._seal_chi_hf_result(hf,method), system, basis


def _energies(hf):
    d = np.asarray(hf.density) if hasattr(hf,'density') else np.asarray(hf.density_alpha)+np.asarray(hf.density_beta)
    # Scalar AO trace sums, independently of production contraction helpers.
    hf.chi_ecp['energy_components'] = {
        name: float(sum(w*sum((d[k,p,q]*a[k,q,p]).real for p in range(a.shape[1])
            for q in range(a.shape[2])) for k,w in enumerate(hf.kpoint_weights)))
        for name,a in hf.hcore_components.items()}
    hf.chi_ecp['energy_components']['ionic_ewald'] = hf.e_nuclear


def _source(hf):
    old = hf.fitted_operator
    desc = replace(old.descriptor, source_signature=cp._physical_source_signature(
        hf._chi_restart_physical, old.descriptor.auxiliary_basis_sha256),
        factor_sha256=factor_digest(old))
    hf.fitted_operator = ChiFittedOperator(dict(old), desc, hf.kpoint_weights)
    source_ecp = {k:v for k,v in hf.chi_ecp.items() if k != 'energy_components'}
    hf.executed_fit_source = replace(desc, executed_domains_json=json.dumps(dict(
        fitted_operator_identity=hf.fitted_operator.identity, one_electron=dict(ecp=source_ecp))))


def test_contract_preserves_zero_local_channel_and_each_primitive():
    system, options = inputs()
    record = ecp_contract(system, options)
    assert record['projectors'][0]['local_l'] == 2
    assert record['projectors'][0]['primitives'][-1] == [2,2,1.,0.]
    assert record['atoms'][0]['core_electrons'] == 2
    assert record['physical_electrons'] == 4 and record['explicit_electrons'] == 2
    changed = deepcopy(options); changed['ecp_primitive_blocks'][0]['coefficients'][0] += .01
    assert ecp_contract(system,changed)['identity'] != record['identity']
    assert primitive_record(SimpleNamespace(**options['ecp_primitive_blocks'][0])) == options['ecp_primitive_blocks'][0]
    assert ecp_contract(SimpleNamespace(), None) is None


@pytest.mark.parametrize('fault', ['missing','orphan','fractional','excess','wrong-core','duplicate',
    'displaced','ambiguous','missing-center','nan','negative-exponent','fractional-am','spin','so','xml'])
def test_incomplete_or_invalid_hamiltonians_refuse(fault):
    system, options = inputs()
    if fault == 'missing': options['ecp_primitive_blocks'] = []
    elif fault == 'orphan': options = dict(ecp_total_ncore=2)
    elif fault == 'fractional': options['ecp_total_ncore'] = 2.5
    elif fault == 'excess': options['ecp_effective_charges'] = [5]
    elif fault == 'wrong-core': options['ecp_total_ncore'] = 1
    elif fault == 'duplicate':
        options['ecp_primitive_blocks'] *= 2; options['ecp_home_centers'] *= 2
    elif fault == 'displaced': options['ecp_home_centers'] = [[1.,0.,0.]]
    elif fault == 'ambiguous':
        system.unit_cell *= 2; system.n_electrons = lambda:8
        options['ecp_effective_charges'] = [2,4]
    elif fault == 'missing-center': options['ecp_home_centers'] = []
    elif fault == 'nan': options['ecp_primitive_blocks'][0]['coefficients'][0] = float('nan')
    elif fault == 'negative-exponent': options['ecp_primitive_blocks'][0]['exponents'][0] = -1.
    elif fault == 'fractional-am': options['ecp_primitive_blocks'][0]['ams'][0] = .5
    elif fault == 'spin': system.multiplicity = 2
    elif fault == 'so': options['ecp_spin_orbit'] = True
    elif fault == 'xml': options['ecp_library'] = 'synthetic-xml'
    with pytest.raises((ValueError,NotImplementedError)): ecp_contract(system,options)


@pytest.mark.parametrize('method',['RHF','UHF','ROHF'])
@pytest.mark.parametrize('nk',[1,3])
def test_ecp_physical_roundtrip_restores_explicit_counts(tmp_path,monkeypatch,method,nk):
    ref,system,basis = ecp_fixture(monkeypatch,method,nk)
    path = cp.save_chi_hf_reference(tmp_path/'ecp.vqr',ref,system=system,basis=basis)
    metadata,_ = storage.read_restart_state(path)
    assert metadata['schema'] == 'vibeqc.chi.hf-restart/v2'
    restored = cp.load_chi_hf_reference(path,system=system,basis=basis)
    assert restored.hf_result.effective_n_electrons == system.n_electrons()-2
    assert restored.hf_result.ecp_total_ncore == 2
    assert cp.portable_reference_identity(restored) == cp.portable_reference_identity(ref)
    assert validate_ecp_reference(restored.hf_result) == ref.hf_result.chi_ecp
    assert restored.hf_result.restart['new_scf_iterations'] == 0


@pytest.mark.parametrize('field', ['parameters','source','component','energy','core','explicit','charges','occupations','so'])
def test_ecp_reference_mutation_and_incomplete_source_refuse(monkeypatch,field):
    ref,_,_ = ecp_fixture(monkeypatch)
    hf = ref.hf_result
    if field == 'parameters': hf.chi_ecp['projectors'][0]['primitives'][0][-1] += .1
    elif field == 'source': hf.executed_fit_source = replace(hf.executed_fit_source,executed_domains_json='{}')
    elif field == 'component': hf.hcore_components['ecp_local'] *= 2
    elif field == 'energy': hf.chi_ecp['energy_components']['ecp_local'] += .1
    elif field == 'core': hf.ecp_total_ncore = 4
    elif field == 'explicit': hf.effective_n_electrons += 2
    elif field == 'charges': hf.ecp_effective_charges = [4]
    elif field == 'occupations': hf.occupations *= 2
    elif field == 'so': hf._chi_restart_physical['options']['ecp_spin_orbit'] = True
    with pytest.raises(ValueError,match='state/source/factors'): ref.validate(ref.request_identity,'RHF')
    with pytest.raises((ValueError,NotImplementedError)): validate_ecp_reference(hf)


def test_ecp_changed_request_and_semantic_downgrade_refuse(tmp_path,monkeypatch):
    ref,_,_ = ecp_fixture(monkeypatch)
    path = cp.save_chi_hf_reference(tmp_path/'original.vqr',ref)
    changed = deepcopy(ref.hf_result._chi_restart_physical)
    changed['options']['ecp_primitive_blocks'][0]['coefficients'][0] += .1
    with pytest.raises(ValueError,match='compatibility'):
        cp.load_chi_hf_reference(path,expected_request_identity=cp._physical_request_identity(changed))
    meta,arrays = storage.read_restart_state(path)
    meta['schema'] = 'vibeqc.chi.hf-restart/v1'
    storage.write_restart_state(tmp_path/'downgraded.vqr',meta,arrays)
    with pytest.raises(ValueError,match='semantic version'): cp.load_chi_hf_reference(tmp_path/'downgraded.vqr')


@pytest.mark.parametrize('method',['RHF','UHF','ROHF'])
def test_ecp_fresh_process_restore_does_not_import_native(tmp_path,monkeypatch,method):
    ref,_,_ = ecp_fixture(monkeypatch,method)
    path = cp.save_chi_hf_reference(tmp_path/'ecp.vqr',ref)
    script = r'''
import importlib.abc,json,pathlib,sys,types
root=pathlib.Path(sys.argv[1])
class NoNative(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*args):
  if '_vibeqc_core' in fullname: raise ImportError('native import forbidden')
sys.meta_path.insert(0,NoNative())
for name,path in [('vibeqc','python/vibeqc'),('vibeqc.periodic','python/vibeqc/periodic'),('vibeqc.periodic.chi','python/vibeqc/periodic/chi'),('vibeqc.output','python/vibeqc/output')]:
 m=types.ModuleType(name);m.__path__=[str(root/path)];sys.modules[name]=m
from vibeqc.periodic.chi import checkpoint as cp,reference as refs
refs.loaded_process_provenance=lambda:{'fresh':'source-only'}
ref=cp.load_chi_hf_reference(sys.argv[2])
print(json.dumps(dict(identity=cp.portable_reference_identity(ref),ecp=ref.hf_result.chi_ecp,
 explicit=ref.hf_result.effective_n_electrons)))
'''
    result = subprocess.run([sys.executable,'-c',script,str(Path(__file__).resolve().parents[1]),str(path)],
                            text=True,capture_output=True,check=True)
    output = json.loads(result.stdout)
    assert output['identity'] == cp.portable_reference_identity(ref)
    assert output['ecp'] == ref.hf_result.chi_ecp
    assert output['explicit'] == ref.hf_result.effective_n_electrons


@pytest.mark.parametrize('method',['RHF','UHF','ROHF'])
@pytest.mark.parametrize('core',[0,1])
def test_ecp_core_is_distinct_from_frozen_explicit_orbitals(tmp_path,monkeypatch,method,core):
    ref,_,_ = ecp_fixture(monkeypatch,method)
    evaluate = rmp.evaluate_same_source_rohf_mp2 if method == 'ROHF' else mp.evaluate_same_source_mp2
    path = tmp_path/'mp2.vqr'
    actual = evaluate(ref,frozen_core=core,checkpoint_path=path)
    counts = json.loads(actual.electron_counts_json)
    assert counts['ecp_core_electrons'] == 2
    assert counts['frozen_explicit_electrons'] == 2*core
    assert counts['explicit_electrons'] == ref.hf_result.effective_n_electrons
    assert counts['correlated_electrons'] == counts['explicit_electrons']-2*core
    assert json.loads(actual.ecp_json) == ref.hf_result.chi_ecp
    restored = evaluate(ref,frozen_core=core,resume_from=path)
    assert restored.e_corr_per_cell == actual.e_corr_per_cell
    with pytest.raises(ValueError): evaluate(ref,frozen_core=3)


def test_ecp_same_source_nonzero_mp2_independent_contraction(monkeypatch):
    ref,_,_ = ecp_fixture(monkeypatch,'RHF',3)
    hf = ref.hf_result; op = hf.fitted_operator
    rng = np.random.default_rng(591)
    b = rng.normal(size=(2,3,3))*.05
    b += b.transpose(0,2,1)
    t = rng.normal(size=b.shape)*.04 + 1j*rng.normal(size=b.shape)*.04
    by_q = {0:b,1:t,2:t.conj().transpose(0,2,1)}
    factors = {(i,j):by_q[(j-i)%3] for i in range(3) for j in range(3)}
    desc = replace(op.descriptor,factor_sha256=factor_digest(factors))
    hf.fitted_operator = ChiFittedOperator(factors,desc,hf.kpoint_weights)
    density = np.asarray(hf.density)/2
    j = np.zeros_like(hf.hcore); k = np.zeros_like(hf.hcore)
    for ki,kj,p,q,r,s in np.ndindex(3,3,3,3,3,3):
        j[ki,p,q] += hf.kpoint_weights[kj]*sum(factors[ki,ki][a,p,q]*factors[kj,kj][a,r,s].conjugate() for a in range(2))*2*density[kj,r,s]
        k[ki,p,q] += hf.kpoint_weights[kj]*sum(factors[ki,kj][a,p,r]*factors[ki,kj][a,q,s].conjugate() for a in range(2))*density[kj,r,s]
    h = hf.fock-j+k+.125*density
    hf.hcore = h
    hf.hcore_components = dict(kinetic=.5*h,ionic_coulomb=.4*h,ecp_local=.04*h,ecp_nonlocal=.06*h)
    hf.energy = hf.e_nuclear+float(np.einsum('k,kpq,kqp->',hf.kpoint_weights,density,h+hf.fock).real)
    _energies(hf); _source(hf)
    ref = refs._seal_chi_hf_result(hf,'RHF')
    channels = [(np.asarray(hf.mo_coeffs),np.asarray(hf.mo_energies),1)]*2
    expected = spin_orbital_oracle(hf.fitted_operator,channels)
    result = mp.evaluate_same_source_mp2(ref)
    assert abs(expected) > 1e-8
    assert result.e_corr_per_cell == pytest.approx(expected,abs=2e-13)
    assert cp._validate_physical(hf,'RHF',hf._chi_restart_physical)['fock_error'] < 1e-12


@pytest.mark.parametrize('method',['RHF','UHF'])
def test_ecp_cc_completed_restart_preserves_hamiltonian_and_explicit_core(tmp_path,monkeypatch,method):
    from vibeqc.periodic.chi.canonical_cc import evaluate_same_source_cc
    ref,_,_ = ecp_fixture(monkeypatch,method,1)
    frozen = (1,0) if method == 'UHF' else 0
    path = tmp_path/'cc.vqr'
    original = evaluate_same_source_cc(ref,compute_triples=False,frozen_core=frozen,checkpoint_path=path)
    loaded = evaluate_same_source_cc(ref,compute_triples=False,frozen_core=frozen,resume_from=path)
    assert loaded.restart_mode == 'restored-completed-result'
    assert loaded.e_total_per_cell == original.e_total_per_cell
    np.testing.assert_array_equal(loaded.t1,original.t1)
    np.testing.assert_array_equal(loaded.t2,original.t2)
    counts = json.loads(loaded.electron_counts_json)
    assert counts['ecp_core_electrons'] == 2
    assert counts['explicit_electrons'] == ref.hf_result.effective_n_electrons
    assert counts['frozen_explicit_electrons'] == (1 if method == 'UHF' else 0)
    assert json.loads(loaded.ecp_json) == ref.hf_result.chi_ecp


def test_changed_ecp_refuses_correlation_resume_even_with_identical_retained_factors(tmp_path,monkeypatch):
    ref,system,basis = ecp_fixture(monkeypatch)
    path = tmp_path/'mp2.vqr'
    mp.evaluate_same_source_mp2(ref,checkpoint_path=path)
    hf = ref.hf_result
    physical = deepcopy(hf._chi_restart_physical)
    physical['options']['ecp_primitive_blocks'][0]['coefficients'][0] += .1
    contract = ecp_contract(system,physical['options'])
    physical['ecp'] = deepcopy(contract)
    contract['executed_support'] = hf.chi_ecp['executed_support']
    contract['energy_components'] = hf.chi_ecp['energy_components']
    hf.chi_ecp = contract
    hf._chi_restart_physical = physical
    _source(hf)
    hf._chi_hf_request_identity = cp._physical_request_identity(physical)
    changed = refs._seal_chi_hf_result(hf,'RHF')
    with pytest.raises(ValueError,match='incompatible'):
        mp.evaluate_same_source_mp2(changed,resume_from=path)


def test_native_primitive_object_is_losslessly_bound_by_reference_request():
    _, options = inputs()
    block = type('ECPPrimitiveBlock',(),{})()
    for key,value in options['ecp_primitive_blocks'][0].items(): setattr(block,key,value)
    assert refs._value(block) == options['ecp_primitive_blocks'][0]
    changed = refs._digest(dict(ecp_primitive_blocks=[block]))
    block.coefficients = list(block.coefficients); block.coefficients[0] += .1
    assert refs._digest(dict(ecp_primitive_blocks=[block])) != changed
