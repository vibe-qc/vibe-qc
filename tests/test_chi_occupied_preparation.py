"""Bounded reference phases and exact greedy reuse; source checks only."""
import weakref

import numpy as np
import pytest

from vibeqc.dlpno import translation_preparation as prep
from benchmarks.chi_occupied_preparation import model


@pytest.mark.parametrize('mesh', [(1,1,1),(3,1,1),(4,1,1),(2,3,1),(2,2,2)])
@pytest.mark.parametrize('reorder', [False, True])
def test_bounded_phases_match_complete_audits(mesh, reorder):
    rng = np.random.default_rng(584)
    n, b, m = int(np.prod(mesh)), 5, 2
    cells = np.array(list(np.ndindex(mesh)))
    fractions = cells/mesh
    raw = rng.normal(size=(b,b))
    s = np.eye(b)+.03*raw@raw.T
    w,v = np.linalg.eigh(s)
    c = (v/np.sqrt(w))@v.T@np.linalg.qr(rng.normal(size=(b,b)))[0][:,:m]
    coeff = np.tile(c,(n,1,1))*np.exp(.37j*np.arange(n*m).reshape(n,1,m))
    overlap = np.tile(s,(n,1,1))
    if reorder:
        order = rng.permutation(n)
        fractions, coeff, overlap = fractions[order], coeff[order], overlap[order]
    dense = np.exp(2j*np.pi*(fractions@cells.T))
    phase = prep.ReferencePhases(mesh, fractions, 'fixture')
    for mode in ('auto','exhaustive'):
        a, ar = prep.real_occupied_projector(coeff,overlap,dense,mesh,'active',1e-10,kfrac=fractions,audit=mode)
        z, zr = prep.real_occupied_projector(coeff,overlap,phase,mesh,'active',1e-10,kfrac=fractions,audit=mode)
        np.testing.assert_allclose(z.blocks,a.blocks,atol=4e-15,rtol=0)
        for key in ('orthogonality','projector_error','translation_error','imaginary'):
            assert zr[key] == pytest.approx(ar[key],abs=4e-15)
    assert phase.record()['phase_table_bytes'] == 0
    assert phase.record()['cache_bytes'] == 0
    if n > 1:
        with pytest.raises(ValueError): phase.block(0,n,0,n)
    with pytest.raises(AttributeError): phase.mesh = (1,1,1)
    with pytest.raises(ValueError): phase.fractions.flags.writeable = True
    changed = prep.ReferencePhases(mesh,fractions,'different-source')
    assert changed.identity != phase.identity


@pytest.mark.parametrize('n', [1,3,4,8])
@pytest.mark.parametrize('tolerance', [1e-10,1e-15])
def test_greedy_cache_matches_original_and_releases(n,tolerance):
    source, mesh, nf, _, _ = model(n)
    for label, q, rank in [('active',source.reference.occupied_projectors.active,4*n),
                           ('frozen',source.reference.occupied_projectors.frozen,2*n)]:
        original = prep.cyclic_projected_frame(q,source.overlap,rank,tolerance,1e-10,label,reuse=False)
        work = {}
        result = prep.cyclic_projected_frame(q,source.overlap,rank,tolerance,1e-10,label,work=work)
        assert result[1:] == original[1:]
        np.testing.assert_array_equal(result[0].home,original[0].home)
        assert work['primitive_gram_builds'] == n
        assert work['released_cache_bytes'] == work['cache_bytes'] == 16*n*16**2
        assert work['binding'][0] == q.identity


def test_compact_source_preflights_and_never_builds_complete_phases(monkeypatch):
    from tests.test_chi_translation_mp2 import fixture
    from vibeqc.periodic.chi import local_mp2 as local
    ref,_,_ = fixture(monkeypatch,(4,1,1),True)
    bound = prep.selection_workspace((4,1,1),4,2,10**8)['accounted_peak_bytes']
    original = np.exp
    def bounded(value, *args, **kwargs):
        assert np.shape(value) != (4,4), 'complete phase table recreated'
        return original(value,*args,**kwargs)
    with monkeypatch.context() as guard:
        guard.setattr(np,'exp',bounded)
        source = local.build_chi_translation_source(ref,frozen_core=1,compact_occupied=True,max_selection_bytes=bound)
        assert source.provenance['storage']['occupied_preparation']['reference_phases']['phase_table_bytes'] == 0
        with pytest.raises(MemoryError):
            local.build_chi_translation_source(ref,frozen_core=1,compact_occupied=True,max_selection_bytes=bound-1)


def test_phase_failure_does_not_retain_cache(monkeypatch):
    phase = prep.ReferencePhases((3,1,1),np.array([[0,0,0],[1/3,0,0],[2/3,0,0]]),'failure')
    def fail(*args, **kwargs): raise RuntimeError('injected transform failure')
    monkeypatch.setattr(np,'einsum',fail)
    with pytest.raises(RuntimeError): prep.phase_inverse(phase,np.ones((3,2,2)))
    assert phase.record()['cache_bytes'] == 0
    reference = weakref.ref(phase)
    del phase
    assert reference() is None


def test_greedy_operations_and_failure_cleanup(monkeypatch):
    source,mesh,_,_,_ = model(4)
    q = source.reference.occupied_projectors.active
    original, cached = {}, {}
    prep.cyclic_projected_frame(q,source.overlap,16,1e-10,1e-10,'active',reuse=False,work=original)
    prep.cyclic_projected_frame(q,source.overlap,16,1e-10,1e-10,'active',work=cached)
    assert cached['original_gram_builds']+cached['primitive_gram_builds'] < original['original_gram_builds']
    refs = []
    empty = np.empty
    def track(shape,*args,**kwargs):
        result = empty(shape,*args,**kwargs)
        if shape == (4,16,16): refs.append(weakref.ref(result))
        return result
    def fail(*args,**kwargs): raise RuntimeError('injected candidate failure')
    monkeypatch.setattr(np,'empty',track)
    monkeypatch.setattr(np.linalg,'eigvalsh',fail)
    work = {}
    with pytest.raises(RuntimeError):
        prep.cyclic_projected_frame(q,source.overlap,16,1e-10,1e-10,'active',work=work)
    assert refs and all(r() is None for r in refs)
    assert work['released_cache_bytes'] == work['cache_bytes']


@pytest.mark.parametrize('offset', [-2e-14,0.,2e-14])
def test_near_ties_and_dependent_seeds_replay(offset):
    from vibeqc.dlpno.translation_operators import CyclicAOOperator
    mesh=(3,1,1)
    vector=np.array([np.sqrt(.5+offset),np.sqrt(.5-offset),0.])
    q=np.zeros((3,3,3)); q[0]=np.outer(vector,vector)
    s=np.zeros_like(q); s[0]=np.eye(3)
    Q,S=CyclicAOOperator(q,mesh),CyclicAOOperator(s,mesh)
    work={}
    a=prep.cyclic_projected_frame(Q,S,3,1e-14,1e-10,'active',reuse=False)
    b=prep.cyclic_projected_frame(Q,S,3,1e-14,1e-10,'active',work=work)
    assert a[1:] == b[1:]
    np.testing.assert_array_equal(a[0].home,b[0].home)
    assert work['boundary_replays'] > 0
    # A rank-two request cannot be rescued by the cache or a new cutoff.
    for reuse in (False,True):
        with pytest.raises(ValueError,match='full-rank'):
            prep.cyclic_projected_frame(Q,S,6,1e-14,1e-10,'active',reuse=reuse)


def test_phase_mesh_and_order_rebinding_refused():
    phase=prep.ReferencePhases((3,1,1),np.array([[0,0,0],[1/3,0,0],[2/3,0,0]]),'source')
    c=np.tile(np.eye(2)[:,:1],(3,1,1))
    s=np.tile(np.eye(2),(3,1,1))
    for mesh,order in [((1,3,1),phase.fractions),((3,1,1),phase.fractions[::-1])]:
        with pytest.raises(ValueError,match='binding'):
            prep.real_occupied_projector(c,s,phase,mesh,'frozen',1e-10,kfrac=order)


@pytest.mark.parametrize('amplitude',[.8e-10,1e-10,1.2e-10])
def test_bounded_phase_threshold_decisions(amplitude):
    mesh=(3,1,1); fractions=np.array([[0,0,0],[1/3,0,0],[2/3,0,0]])
    cells=np.array(list(np.ndindex(mesh)))
    dense=np.exp(2j*np.pi*(fractions@cells.T))
    bounded=prep.ReferencePhases(mesh,fractions,'threshold')
    c=np.tile(np.eye(2,dtype=complex)[:,:1],(3,1,1))
    c[-1] *= np.sqrt(1+amplitude)
    s=np.tile(np.eye(2),(3,1,1))
    results=[]
    for phase in (dense,bounded):
        try:
            _,record=prep.real_occupied_projector(c,s,phase,mesh,'active',1e-10,kfrac=fractions)
            results.append(record)
        except ValueError:
            results.append(None)
    assert (results[0] is None) == (results[1] is None)
    if results[0] is not None:
        for key in ('orthogonality','projector_error','translation_error','imaginary'):
            assert results[0][key] == pytest.approx(results[1][key],abs=2e-15)


def test_audit_scope_rebinding_and_failure_cleanup():
    from vibeqc.dlpno import translation_audit as audit
    source,mesh,_,_,_ = model(3)
    with pytest.raises(RuntimeError):
        with audit.occupied_audit_context() as state:
            audit.bind_source('first',10**6)
            audit.operator_symbols(source.overlap)
            audit.measured(('first',),lambda:0.)
            assert state['arrays'] and state['values']
            audit.bind_source('second',10**6)
            assert not state['arrays'] and not state['values']
            audit.operator_symbols(source.overlap)
            audit.bind_source('second',0)
            assert not state['arrays'] and not state['values']
            audit.measured(('second',),lambda:0.)
            raise RuntimeError('injected failure')
    assert not state['arrays'] and not state['values']
    assert state['array_bytes'] == 0


def test_selection_cache_binding_tracks_source_and_partitions():
    from vibeqc.dlpno.translation_selection import select_translation_occupied
    source,mesh,nf,_,_ = model(2)
    first=select_translation_occupied(source,mesh,n_frozen=nf,compact_storage=True)
    source.identity += '-rebound'
    second=select_translation_occupied(source,mesh,n_frozen=nf,compact_storage=True)
    for name in ('active','frozen'):
        before=first.provenance['greedy_gram_work'][name]
        after=second.provenance['greedy_gram_work'][name]
        assert before['binding'] != after['binding']
        assert before['primitive_gram_builds'] == after['primitive_gram_builds'] == 2
    assert first.provenance['greedy_gram_work']['active']['binding'] != first.provenance['greedy_gram_work']['frozen']['binding']
    np.testing.assert_array_equal(first.occupied.home,second.occupied.home)
    np.testing.assert_array_equal(first.frozen_occupied.home,second.frozen_occupied.home)
