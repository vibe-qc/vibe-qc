"""Explicit source controls, immutable identities, and dispatch before numerics."""
from dataclasses import asdict, replace
from types import SimpleNamespace
import json
import numpy as np
import pytest

from vibeqc.periodic.chi.fitted import ChiCoulombControls, build_chi_fitted_operator


def controls(**kwargs):
    return ChiCoulombControls(5.,6.,20.,12.,5.,.6, **kwargs)


@pytest.mark.parametrize('field,value', [('pair_cutoff_bohr',0),('metric_threshold',-1),
    ('reciprocal_cutoff_ha',np.nan),('memory_byte_cap',2.5),('support_record_byte_cap',True)])
def test_invalid_control_cannot_reach_native(field,value):
    with pytest.raises(ValueError): replace(controls(),**{field:value})


def test_source_semantics_and_historical_are_explicit():
    c=controls();c.require_source('classic')
    with pytest.raises(ValueError):c.require_source('auto')
    with pytest.raises(ValueError):c.require_source('sr')
    with pytest.raises(NotImplementedError):controls(support_policy='historical_index_ball',auxiliary_cutoff_bohr=10.,omega_bohr_inv=.6).require_source('sr')
    with pytest.raises(ValueError):controls(tail_reciprocal_cutoff_ha=10.).require_source('classic')
    with pytest.raises(ValueError):controls(omega_bohr_inv=.6).require_source('classic')


def test_tail_ladder_fixes_one_electron_and_real_domain():
    base=controls()
    rows=base.ladder('tail_reciprocal_cutoff_ha',[20.,40.,80.])
    for c in rows:
        a,b=asdict(base),asdict(c);a.pop('tail_reciprocal_cutoff_ha');b.pop('tail_reciprocal_cutoff_ha');assert a==b
    with pytest.raises(ValueError):base.ladder('omega_bohr_inv',[.5,.7])


@pytest.mark.parametrize('source',['classic','sr'])
def test_operator_uses_exact_selected_domains_and_binds_reuse(monkeypatch,source):
    import vibeqc.aux_basis as ab
    c=controls() if source=='classic' else controls(auxiliary_cutoff_bohr=9.,omega_bohr_inv=.7)
    sy=SimpleNamespace(dim=3,lattice=np.eye(3)*8.,unit_cell=[])
    basis=SimpleNamespace(nbasis=2,name='test',shells=lambda:[])
    aux=SimpleNamespace(nbasis=2,name='aux',shells=lambda:[])
    monkeypatch.setattr(ab,'_range_separated_gdf_source_signature',lambda *a:'geometry-bases')
    monkeypatch.setattr(ab,'_basis_fingerprint',lambda b:b.name)
    calls=[]
    def build(*args,**kw):
        calls.append(kw)
        record=json.dumps({'actual_vectors':[[1.,0.,0.]],'metric_eigenvalues':[1.,2.], 'keep':[0,1]})
        kw['_support_recorder'](record)
        f=np.ones((len(args[3]),2,2,2),complex)
        return list(f) if source=='classic' else SimpleNamespace(factors=f,executed_support=record)
    monkeypatch.setattr(ab,'build_lpq_bloch_native_fft_shared_q',build)
    monkeypatch.setattr(ab,'_build_lpq_range_separated_shared_q',build)
    k=np.array([[0.,0.,0.],[2*np.pi/24,0.,0.],[-2*np.pi/24,0.,0.]])
    op=build_chi_fitted_operator(sy,basis,aux,k,np.ones(3)/3,source=source,controls=c)
    assert len(op)==9 and len(calls)==3
    for call in calls:
        assert call['ke_cutoff']==20. and call['linear_dep_thr']==1e-9
        if source=='classic':assert call['lat_opts'].cutoff_bohr==6. and call['support_policy']=='physical_pairs'
        else:assert call['pair_cutoff']==6. and call['auxiliary_cutoff']==9. and call['omega']==.7
    assert op.require_identity(op.identity) is op
    with pytest.raises(ValueError):op.require_identity('different-support')
    with pytest.raises(ValueError):op[(0,0)].setflags(write=True)
    with pytest.raises(AttributeError):op._weights=(1.,)
    record=op.as_dict();record['source']['source']='wrong'
    assert op.descriptor.source==source
    with pytest.raises(ValueError):op.exchange(np.zeros((2,2,2)))
    # Rebuilding the same geometry with a refined domain must invalidate reuse,
    # even when this controlled builder returns numerically identical factors.
    refined=build_chi_fitted_operator(sy,basis,aux,k,np.ones(3)/3,source=source,controls=replace(c,pair_cutoff_bohr=7.))
    assert refined.identity != op.identity


def test_budget_and_invalid_weights_refuse_before_source(monkeypatch):
    import vibeqc.aux_basis as ab
    def forbidden(*a,**k):raise AssertionError('native source must not run')
    monkeypatch.setattr(ab,'build_lpq_bloch_native_fft_shared_q',forbidden)
    sy=SimpleNamespace(dim=3);b=SimpleNamespace(nbasis=100)
    with pytest.raises(MemoryError):build_chi_fitted_operator(sy,b,b,[[0.,0.,0.]],[1.],source='classic',controls=controls(memory_byte_cap=1024))
    with pytest.raises(ValueError):build_chi_fitted_operator(sy,b,b,[[0.,0.,0.]],[-1.],source='classic',controls=controls())


@pytest.mark.parametrize('source,equal_tail',[('classic',False),('classic',True),('sr',False)])
@pytest.mark.parametrize('spin', ['rhf', 'uhf'])
def test_library_transports_control_contract_before_integrals(monkeypatch,source,equal_tail,spin):
    import vibeqc as vq
    from vibeqc.periodic.chi import scf
    c=controls(tail_reciprocal_cutoff_ha=20. if equal_tail else None) if source=='classic' else controls(auxiliary_cutoff_bohr=9.,omega_bohr_inv=.7)
    sy=vq.PeriodicSystem(3,np.eye(3)*8.,[vq.Atom(2,[0.,0.,0.])])
    opts=vq.PeriodicRHFOptions()
    class Reached(Exception):pass
    def inner(*args,**kw):
        assert kw['fitted_support'] is c
        assert kw['fit_source']==source
        assert kw['rsgdf_ke_cutoff']==20.
        assert kw['rsgdf_tail_ke_cutoff']==c.tail_reciprocal_cutoff_ha
        raise Reached
    monkeypatch.setattr(scf,'run_k'+spin+'_periodic_gdf',inner)
    monkeypatch.setattr('vibeqc.periodic.chi.reference._hf_request_identity', lambda *a: 'test-request')
    with pytest.raises(Reached):getattr(scf, 'run_aiccm2026dev_b_'+spin)(sy,object(),(3,1,1),opts,backend='ri',fit_source=source,coulomb_controls=c)
    with pytest.raises(ValueError,match='conflicts'):getattr(scf, 'run_aiccm2026dev_b_'+spin)(sy,object(),(3,1,1),opts,backend='ri',fit_source=source,coulomb_controls=c,rsgdf_ke_cutoff=31.)
    with pytest.raises(NotImplementedError):getattr(scf, 'run_aiccm2026dev_b_'+spin)(sy,object(),(3,1,1),opts,backend='rijcosx',fit_source=source,coulomb_controls=c)


@pytest.mark.parametrize('source',['auto','classic'])
def test_posthf_rejects_explicit_domain_before_any_rebuild(source):
    from vibeqc.periodic.chi.posthf import _selected_lpq_cache, _selected_lov_cache
    reference=SimpleNamespace(executed_fit_source=SimpleNamespace(executed_domains_json='{}'))
    kw=dict(reference=reference,fit_source=source,rsgdf_tail_ke_cutoff=None,
            lattice_cutoff_bohr=6.,rsgdf_ke_cutoff=20.,gdf_linear_dep_threshold=1e-9)
    with pytest.raises(NotImplementedError,match='not admitted'):
        _selected_lpq_cache(None,None,None,None,**kw)
    with pytest.raises(NotImplementedError,match='not admitted'):
        _selected_lov_cache(None,None,None,None,None,None,**kw)


def test_legacy_descriptor_identity_is_preserved():
    import hashlib
    from vibeqc.periodic_fit_source import RSGDFFitSource
    old=dict(engine='old',engine_version=1,source='classic',source_signature='s',
        orbital_basis='b',orbital_basis_sha256='b',auxiliary_basis='a',auxiliary_basis_sha256='a',
        numerical_support=(),metric_threshold=1e-9,auxiliary_frame='canonical',q_grouping=(),
        kpoints_cart=(),base_reciprocal_cutoff_ha=20.,executed_tail_reciprocal_cutoff_ha=None,
        factor_shapes=(),factor_sha256='f',factor_space='ao-pair',schema='vibeqc.rsgdf.executed-source/v1')
    desc=RSGDFFitSource(**old)
    assert desc.as_dict()==old
    assert desc.identity==hashlib.sha256(json.dumps(old,sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize('extra',[{'retain_fit_cache':True},{'dft_plus_u_sites':[object()]}])
def test_shared_driver_rejects_unsupported_export_and_plus_u_before_integrals(extra):
    from vibeqc.periodic_k_gdf import run_krhf_periodic_gdf
    with pytest.raises(NotImplementedError):
        run_krhf_periodic_gdf(SimpleNamespace(dim=3),None,(1,1,1),
            fit_source='classic',fitted_support=controls(),**extra)


def test_nuclear_fourier_cache_budget_refuses_before_native_work(monkeypatch):
    from vibeqc.periodic_fitted_support import fitted_nuclear_operators
    import vibeqc.pbc_bipole_common as common
    def forbidden(*args, **kwargs):
        raise AssertionError('nuclear integration must not precede complete FT-cache reservation')
    monkeypatch.setattr(common, '_compute_nuclear_lattice_ewald_reciprocal_ft', forbidden)
    # Nuclear trace count alone fits the 16 MiB record cap, but the AO-square
    # reciprocal FT cache exceeds 512 MiB. No integrals should be reached.
    system = SimpleNamespace(dim=3, lattice=np.eye(3)*8., unit_cell=[object(), object()])
    basis = SimpleNamespace(nbasis=16, shells=lambda: [SimpleNamespace(origin=[0., 0., 0.])])
    options = SimpleNamespace(cutoff_bohr=5., pair_complete_1e=True)
    with pytest.raises(MemoryError, match='Fourier cache'):
        fitted_nuclear_operators(system,basis,options,None,controls())


@pytest.mark.parametrize('change', [dict(image_candidate_cap=1),
    dict(reciprocal_candidate_cap=1),dict(support_record_byte_cap=1)])
def test_early_nuclear_enclosure_caps_are_not_support_truncation(change):
    from vibeqc.periodic_fitted_support import preflight_fitted_one_electron
    system=SimpleNamespace(dim=3,lattice=np.eye(3)*8.,unit_cell=[object()])
    basis=SimpleNamespace(nbasis=1,shells=lambda:[SimpleNamespace(origin=[0.,0.,0.])])
    options=SimpleNamespace(cutoff_bohr=5.,pair_complete_1e=True)
    before=(options.cutoff_bohr,options.pair_complete_1e)
    with pytest.raises(MemoryError):
        preflight_fitted_one_electron(system,basis,options,controls(**change))
    assert (options.cutoff_bohr,options.pair_complete_1e)==before


def test_retained_one_electron_arrays_are_charged_to_source_budget(monkeypatch):
    import vibeqc.aux_basis as ab
    def forbidden(*args,**kwargs):raise AssertionError('source integration must not run')
    monkeypatch.setattr(ab,'build_lpq_bloch_native_fft_shared_q',forbidden)
    system=SimpleNamespace(dim=3)
    basis=SimpleNamespace(nbasis=1)
    c=controls()
    with pytest.raises(MemoryError,match='complete source budget'):
        build_chi_fitted_operator(system,basis,basis,[[0.,0.,0.]],[1.],source='classic',
                                  controls=c,_retained_bytes=c.memory_byte_cap)


def test_complete_support_wrapper_counts_against_record_cap(monkeypatch):
    import vibeqc.aux_basis as ab
    system=SimpleNamespace(dim=3,lattice=np.eye(3)*8.,unit_cell=[])
    basis=SimpleNamespace(nbasis=1,name='test',shells=lambda:[])
    monkeypatch.setattr(ab,'_range_separated_gdf_source_signature',lambda *args:'source')
    monkeypatch.setattr(ab,'_basis_fingerprint',lambda *args:'basis')
    def build(*args,**kwargs):
        kwargs['_support_recorder']('{}')
        return [np.ones((1,1,1),complex)]
    monkeypatch.setattr(ab,'build_lpq_bloch_native_fft_shared_q',build)
    with pytest.raises(MemoryError,match='wrapper'):
        build_chi_fitted_operator(system,basis,basis,[[0.,0.,0.]],[1.],source='classic',
                                  controls=controls(support_record_byte_cap=64))
