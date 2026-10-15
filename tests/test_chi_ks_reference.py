"""Process-local KS seals and refusal controls without native calculation."""
import ast
from pathlib import Path
import dataclasses
import hashlib
import json
import os
from collections.abc import Mapping
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def reference():
    source = Path(__file__).parents[1] / 'python/vibeqc/periodic/chi/reference.py'
    names = {'_value', '_digest', '_hf_state_identity', '_admit_ks_state', 'ChiKSReference'}
    nodes = [n for n in ast.parse(source.read_text()).body if getattr(n, 'name', None) in names]
    scope = dict(np=np, os=os, hashlib=hashlib, json=json, Mapping=Mapping,
                 dataclass=dataclasses.dataclass, fields=dataclasses.fields,
                 is_dataclass=dataclasses.is_dataclass,
                 loaded_process_provenance=lambda: {'runtime': 'synthetic'},
                 factor_digest=lambda operator: hashlib.sha256(operator.factor_bytes.tobytes()).hexdigest())
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), scope)
    return scope


def make_ref(r):
    @dataclasses.dataclass
    class Diagnostic:
        electronic_method: str = 'RKS/pbe'
    state = SimpleNamespace(converged=True, energy=-1., e_xc=-.4,
        executed_support={'policy': 'physical_pairs'}, executed_chi_xc={
            'components': [[101, 1.], [130, 1.]], 'libxc_version': 'synthetic',
            'density_image_radius_bohr': 18., 'grid_hash': 'synthetic'},
        density=np.eye(2)[None], fock=np.diag([-1., 2.])[None],
        occupations=[[2., 0.]], accepted_density_k=np.eye(2)[None],
        aiccm2026dev_b=Diagnostic(), process_provenance={'runtime': 'synthetic'},
        chi_scf_acceptance={'final_rebuild': True, 'stationary': True})
    ref = r['ChiKSReference'](state, os.getpid(), 'request', r['_hf_state_identity'](state), 'RKS')
    return dataclasses.replace(ref, seal=r['_digest']((ref.owner_pid, ref.request_identity,
        ref.state_identity, ref.method, ref.quantitative_status)))


def test_ks_reuse_is_same_sealed_state(reference):
    ref = make_ref(reference)
    assert ref.validate('request', 'RKS') is ref.ks_result
    assert ref.quantitative_status == 'not-qualified'


@pytest.mark.parametrize('field,value', [
    ('executed_chi_exchange', {'full_range_coefficient': .5}),
    ('e_hf_exchange_components', {'full_zero': .1}),
    ('e_xc', -.5), ('occupations', [[1., 1.]]),
    ('executed_chi_xc', {'components': [[1, 1.]]}),
    ('accepted_density_k', np.zeros((1, 2, 2))),
    ('chi_scf_acceptance', {'stationary': False}),
    ('executed_support', {'policy': 'changed'})])
def test_mutated_ks_reference_rejects(reference, field, value):
    ref = make_ref(reference)
    setattr(ref.ks_result, field, value)
    with pytest.raises(ValueError, match='state/source changed'):
        ref.validate('request', 'RKS')


def test_functional_grid_spin_and_source_request_mismatch_rejects(reference):
    ref = make_ref(reference)
    for request, method in [('changed', 'RKS'), ('request', 'UKS'), ('request', 'RHF')]:
        with pytest.raises(ValueError, match='mismatch'):
            ref.validate(request, method)
    with pytest.raises(ValueError, match='seal'):
        dataclasses.replace(ref, owner_pid=-1).validate('request', 'RKS')


@pytest.mark.parametrize("mutation", ["factor_bytes","identity","assembly"])
def test_hybrid_reference_binds_actual_kernel_factors(reference,mutation):
    ref=make_ref(reference)
    source=SimpleNamespace(identity="erfc-source",factor_bytes=np.ones((1,2,2)))
    assembly={"short_range_coefficient":-.46,"omega_screen":.33}
    exchange=SimpleNamespace(factors={"erfc":source},as_dict=lambda:dict(assembly))
    ref.ks_result.fitted_exchange_operator=exchange
    ref=dataclasses.replace(ref,state_identity=reference['_hf_state_identity'](ref.ks_result))
    ref=dataclasses.replace(ref,seal=reference['_digest']((ref.owner_pid,ref.request_identity,
        ref.state_identity,ref.method,ref.quantitative_status)))
    assert ref.validate('request','RKS') is ref.ks_result
    if mutation=="factor_bytes": source.factor_bytes[0,0,0]+=1e-8
    elif mutation=="identity": source.identity="other-source"
    else: assembly["omega_screen"]+=.01
    with pytest.raises(ValueError,match='state/source changed'):
        ref.validate('request','RKS')
