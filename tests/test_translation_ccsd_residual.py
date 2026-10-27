"""Internal translation CCSD algebra; source checks, not scientific acceptance."""
from itertools import combinations_with_replacement
from math import prod

import numpy as np
import pytest
from scipy.linalg import null_space

from tests.test_dlpno_source import _automatic_translation_source, _translation_block_source
from vibeqc.dlpno.translation_factors import TranslationHomeFactors
from vibeqc.dlpno.translation_builder import build_translation_pair_pnos
from vibeqc.dlpno.translation_ccsd import evaluate_translation_ccsd
from vibeqc.dlpno.translation_operators import matrix_apply


def model(mesh=(2,1,1), frozen=True, cutoff=0.):
    source, active, core, home = _automatic_translation_source(mesh,frozen)
    if sum(n>1 for n in mesh)>1:
        # The reused fixture's flattened ring hopping is one-dimensional.
        # Replace it with explicit commuting mesh translations in this model.
        cells=prod(mesh); bands=active.shape[1]//cells
        hop=np.roll(np.eye(cells),1,axis=0)
        coupling=.025*np.kron(hop+hop.T,np.eye(bands))
        grid=np.arange(cells).reshape(mesh)
        desired=np.zeros((cells,cells))
        for axis,n in enumerate(mesh):
            if n>1:
                shift=np.eye(cells)[np.roll(grid,1,axis=axis).ravel()]
                desired+=.025*(shift+shift.T)
        source.reference.fock+=source.overlap@active@(np.kron(desired,np.eye(bands))-coupling)@active.T@source.overlap
    source.factors = TranslationHomeFactors(home,mesh)
    source = _translation_block_source(source,mesh)
    built = build_translation_pair_pnos(source,mesh,active,frozen_occupied=core,
                                       compact_storage=True,tcut_pno=cutoff)
    return source,built


def amplitudes(built, scale=.013):
    s = built.spaces
    rng = np.random.default_rng(586)
    singles = [scale*rng.normal(size=s.pair_ranks[s.resolve(i,i).row]) for i in range(s._n_bands)]
    doubles = []
    for row, ((i,j),rank) in enumerate(zip(s.representative_pairs,s.pair_ranks)):
        value = scale*rng.normal(size=(rank,rank))
        if i == j:
            value = .5*(value+value.T)
        if row in s.stabilizers:
            U = s.stabilizers[row]
            value = .5*(value+U@value.T@U.T)
        doubles.append(value)
    return singles,doubles


def unfolded_oracle(source,built,singles,doubles,terms=True):
    """Explicit finite-system folding is confined to this test oracle."""
    from vibeqc import _vibeqc_core as native
    s = built.spaces
    C = np.asarray(s._occupied)
    frozen = np.asarray(s._frozen_occupied)
    # Arbitrary orthonormal virtual gauge; use identical placed PNO spaces.
    S = matrix_apply(s._overlap,np.eye(C.shape[0]))
    Q = null_space(np.column_stack((C,frozen)).T@S)
    ev,U = np.linalg.eigh(Q.T@S@Q)
    V = Q@(U/np.sqrt(ev))@U.T
    foo = s.occupied_fock
    fov = C.T@matrix_apply(s._fock,V)
    fvv = V.T@matrix_apply(s._fock,V)
    Bov = source.factors.mo_transform(C,V)
    Boo = source.factors.mo_transform(C,C)
    Bvv = source.factors.mo_transform(V,V)
    pairs = list(combinations_with_replacement(range(C.shape[1]),2))
    frames = [V.T@S@s.coefficients(i,j) for i,j in pairs]
    K = [source.factors.pair_exchange(C[:,[i,j]],s.coefficients(i,j)) for i,j in pairs]
    T = [s.orient_tensor(i,j,doubles) for i,j in pairs]
    t1 = [singles[i%s._n_bands] for i in range(C.shape[1])]
    result = native._dlpno_ccsd_pno_evaluate(
        *(np.ascontiguousarray(x.reshape(x.shape[0],-1)) for x in (Bov,Boo,Bvv)),
        foo,fov,fvv,[i for i,j in pairs],[j for i,j in pairs],frames,
        [np.diag(u.T@fvv@u) for u in frames],K,T,[list(range(C.shape[1])) for p in pairs],t1,terms)
    return result,dict(pairs=pairs,frames=frames,C=C,V=V,S=S,foo=foo,fov=fov,fvv=fvv,
                       Bov=Bov,Boo=Boo,Bvv=Bvv,t1=t1,T=T)


@pytest.mark.parametrize('mesh,frozen,cutoff',[
    ((1,1,1),False,0.),((2,1,1),True,0.),((3,1,1),True,1e-4),((2,2,1),True,0.),((3,2,1),True,0.)])
def test_translation_ccsd_arbitrary_amplitude_terms(mesh,frozen,cutoff):
    source,built = model(mesh,frozen,cutoff)
    singles,doubles = amplitudes(built)
    actual = evaluate_translation_ccsd(source,built,singles,doubles,terms=True)
    oracle,data = unfolded_oracle(source,built,singles,doubles)
    s = built.spaces
    for i in range(s._n_bands):
        np.testing.assert_allclose(actual['R1'][i],oracle['R1'][i],atol=2e-11,rtol=2e-11)
        assert actual['singles_terms'][i].keys() == oracle['singles_terms'][i].keys()
        for name,value in actual['singles_terms'][i].items():
            np.testing.assert_allclose(value,oracle['singles_terms'][i][name],atol=2e-11,rtol=2e-11)
    for p,pair in enumerate(s.representative_pairs):
        q = data['pairs'].index(pair)
        np.testing.assert_allclose(actual['R2'][p],oracle['R2'][q],atol=2e-11,rtol=2e-11)
        assert actual['doubles_terms'][p].keys() == oracle['doubles_terms'][q].keys()
        for name,value in actual['doubles_terms'][p].items():
            np.testing.assert_allclose(value,oracle['doubles_terms'][q][name],atol=2e-11,rtol=2e-11)
    assert actual['energy'] == pytest.approx(oracle['energy']/prod(mesh),abs=2e-12)
    assert actual['pno_mp2_correction_per_cell'] == built.pno_correction_per_cell
    assert actual['provenance']['cache_bytes_after_evaluation'] == 0
    # Independent canonical transcription with all amplitudes in one gauge.
    if cutoff == 0.:
        from vibeqc.dlpno._ccsd_cs import _blocks,cs_ccsd_residual,cs_ccsd_energy
        no,nv = data['C'].shape[1],data['V'].shape[1]
        t1 = np.array([data['frames'][data['pairs'].index((i,i))]@data['t1'][i] for i in range(no)])
        t2 = np.zeros((no,no,nv,nv))
        for (i,j),u,t in zip(data['pairs'],data['frames'],data['T']):
            t2[i,j] = u@t@u.T
            t2[j,i] = t2[i,j].T
        blocks = _blocks(data['Bov'],data['Bvv'],data['Boo'])
        r1,r2 = cs_ccsd_residual(t1,t2,data['foo'],data['fvv'],data['fov'],blocks)
        for i in range(s._n_bands):
            u = data['frames'][data['pairs'].index((i,i))]
            np.testing.assert_allclose(actual['R1'][i],r1[i]@u,atol=2e-11,rtol=2e-11)
        for p,pair in enumerate(s.representative_pairs):
            u = data['frames'][data['pairs'].index(pair)]
            np.testing.assert_allclose(actual['R2'][p],u.T@r2[pair]@u,atol=2e-11,rtol=2e-11)
        assert actual['energy'] == pytest.approx(cs_ccsd_energy(t1,t2,data['fov'],blocks)/prod(mesh),abs=2e-12)


@pytest.mark.parametrize('singles_only',[False,True])
def test_translation_ccsd_nonlinear_singles_and_doubles(singles_only):
    source,built = model((2,1,1),True)
    singles,doubles = amplitudes(built,.027)
    if singles_only:
        doubles = [np.zeros_like(x) for x in doubles]
    else:
        singles = [np.zeros_like(x) for x in singles]
    actual = evaluate_translation_ccsd(source,built,singles,doubles,terms=True)
    oracle,data = unfolded_oracle(source,built,singles,doubles)
    for p,pair in enumerate(built.spaces.representative_pairs):
        q = data['pairs'].index(pair)
        np.testing.assert_allclose(actual['R2'][p],oracle['R2'][q],atol=2e-11,rtol=2e-11)
    for i,r in enumerate(actual['R1']):
        np.testing.assert_allclose(r,oracle['R1'][i],atol=2e-11,rtol=2e-11)
    assert actual['energy'] == pytest.approx(oracle['energy']/2,abs=2e-12)
    if singles_only:
        assert np.linalg.norm(actual['doubles_terms'][0]['virtual_ladder']-actual['doubles_terms'][0]['occupied_ladder']) > 1e-8


def test_translation_ccsd_budget_cleanup_and_no_dense_execution(monkeypatch):
    from dataclasses import replace
    from vibeqc import _vibeqc_core as native
    from vibeqc.dlpno.translation_ccsd import _Operands
    from vibeqc.dlpno.translation_storage import TranslationOccupiedColumns
    from vibeqc.dlpno.translation_operators import CyclicAOOperator
    source,built = model((2,1,1),True)
    singles,doubles = amplitudes(built)
    expected = evaluate_translation_ccsd(source,built,singles,doubles)
    def forbidden(*args,**kwargs):
        raise AssertionError('hidden full-supercell execution')
    monkeypatch.setattr(TranslationOccupiedColumns,'__array__',forbidden)
    with pytest.raises(TypeError,match='implicit dense conversion is forbidden'):
        np.asarray(built.spaces._overlap)
    monkeypatch.setattr(native,'_dlpno_ccsd_pno_evaluate',forbidden)
    cap = expected['provenance']['required_bytes']
    actual = evaluate_translation_ccsd(source,built,singles,doubles,max_workspace_bytes=cap)
    for a,b in zip(actual['R2'],expected['R2']):
        np.testing.assert_array_equal(a,b)
    assert actual['provenance']['peak_cache_bytes'] < actual['provenance']['python_workspace_bytes']
    with pytest.raises(MemoryError,match='workspace'):
        evaluate_translation_ccsd(source,built,singles,doubles,max_workspace_bytes=cap-1)
    with pytest.raises(ValueError,match='source does not match'):
        evaluate_translation_ccsd(source,replace(built,source_identity='different'),singles,doubles)
    calls = []
    old_compute,old_clear = _Operands.compute,_Operands.clear
    def fail(self,name,args):
        if name == 'bov':
            raise RuntimeError('injected operand failure')
        return old_compute(self,name,args)
    def clear(self):
        old_clear(self)
        calls.append((len(self.cache),len(self.frames)))
    monkeypatch.setattr(_Operands,'compute',fail)
    monkeypatch.setattr(_Operands,'clear',clear)
    with pytest.raises(RuntimeError,match='injected operand failure'):
        evaluate_translation_ccsd(source,built,singles,doubles)
    assert calls == [(0,0)]
    monkeypatch.setattr(_Operands,'compute',old_compute)
    replay = evaluate_translation_ccsd(source,built,singles,doubles)
    assert replay['energy'] == expected['energy']
    assert calls[-1] == (0,0)
    singles[0][0] = np.nan
    with pytest.raises(ValueError,match='singles'):
        evaluate_translation_ccsd(source,built,singles,doubles)


def test_translation_ccsd_native_budget_precedes_operand_access():
    from vibeqc import _vibeqc_core as native
    def forbidden(*args):
        raise AssertionError('operand allocated before budget refusal')
    from vibeqc.dlpno.translation_transport import prepare_translation_transport
    source,built = model((1,1,1),True)
    transport=prepare_translation_transport(built,backend='bounded')
    rank=max(built.pno_ranks);aux=source.factors.n_aux
    cap = native._translation_ccsd_workspace(1,aux,rank,1,False)
    with pytest.raises(ValueError,match='workspace exceeds'):
        native._translation_ccsd_evaluate(transport._payload,built.spaces.occupied_fock,aux,rank,forbidden,cap-1,False)
    with pytest.raises(ValueError,match='Fock differs'):
        native._translation_ccsd_evaluate(transport._payload,built.spaces.occupied_fock+.1,aux,rank,forbidden,cap,False)



@pytest.mark.parametrize('mesh,core',[((3,1,1),0),((2,2,1),1)])
def test_translation_ccsd_complex_bloch_source(monkeypatch,mesh,core):
    from tests.test_chi_translation_mp2 import fixture
    from vibeqc.periodic.chi.local_mp2 import build_chi_translation_source
    from vibeqc.dlpno.translation_selection import select_translation_occupied
    ref,_,_ = fixture(monkeypatch,mesh)
    source = build_chi_translation_source(ref,frozen_core=core)
    selection = select_translation_occupied(source,mesh,n_frozen=core*prod(mesh),compact_storage=True)
    built = build_translation_pair_pnos(source,mesh,selection.occupied,
        frozen_occupied=selection.frozen_occupied,compact_storage=True)
    singles,doubles = amplitudes(built)
    actual = evaluate_translation_ccsd(source,built,singles,doubles,terms=True)
    oracle,data = unfolded_oracle(source,built,singles,doubles)
    for p,pair in enumerate(built.spaces.representative_pairs):
        q = data['pairs'].index(pair)
        for name,value in actual['doubles_terms'][p].items():
            np.testing.assert_allclose(value,oracle['doubles_terms'][q][name],atol=2e-11,rtol=2e-11)
        np.testing.assert_allclose(actual['R2'][p],oracle['R2'][q],atol=2e-11,rtol=2e-11)
    for i,r in enumerate(actual['R1']):
        np.testing.assert_allclose(r,oracle['R1'][i],atol=2e-11,rtol=2e-11)
    assert actual['energy'] == pytest.approx(oracle['energy']/prod(mesh),abs=2e-12)


@pytest.mark.parametrize('asymmetric',[False,True])
def test_translation_ccsd_stable_ill_conditioned_domains(asymmetric):
    from dataclasses import replace
    from tests.test_dlpno_mp2 import _translation_pno_source
    from vibeqc.dlpno.translation_transport import prepare_translation_transport
    mesh=(2,1,1)
    finite,source,active,core,_=_translation_pno_source(mesh,frozen=True)
    # Same complete covariant factors, sealed by their home auxiliary block.
    source=replace(source,factors=TranslationHomeFactors(finite.ao_factors[:1],mesh))
    if asymmetric:
        metric=source.overlap.copy()
        for cell in range(2):
            metric[4*cell,4*cell+1]+=1e-15
        source=replace(source,overlap=metric)
    source=_translation_block_source(source,mesh)
    domains=[np.arange(4),np.arange(8)]
    built=build_translation_pair_pnos(source,mesh,active,domains,
        frozen_occupied=core,compact_storage=True,tcut_pno=0.)
    transport=prepare_translation_transport(built,backend='bounded')
    assert transport.stable_links>0
    singles,doubles=amplitudes(built)
    actual=evaluate_translation_ccsd(source,built,singles,doubles,transport=transport,terms=True)
    oracle,data=unfolded_oracle(source,built,singles,doubles)
    for i,value in enumerate(actual['R1']):
        np.testing.assert_allclose(value,oracle['R1'][i],atol=2e-11,rtol=2e-11)
    for pair,value in zip(built.spaces.representative_pairs,actual['R2']):
        np.testing.assert_allclose(value,oracle['R2'][data['pairs'].index(pair)],atol=2e-11,rtol=2e-11)
    assert actual['energy']==pytest.approx(oracle['energy']/2,abs=2e-12)


def test_translation_ccsd_archive_replay_payload(monkeypatch,tmp_path):
    import json
    import runpy
    import sys
    from pathlib import Path
    from tests.test_chi_translation_mp2 import fixture
    from vibeqc.periodic.chi import checkpoint
    ref,_,_=fixture(monkeypatch,(3,1,1))
    # Exercise the payload after archive loading; this synthetic fixture does
    # not claim the physical/archive acceptance of a real chi calculation.
    monkeypatch.setattr(checkpoint,'load_chi_hf_reference',lambda path:ref)
    script=Path(__file__).resolve().parents[1]/'examples/periodic/chi_translation_ccsd_residual.py'
    output=tmp_path/'residual.json'
    monkeypatch.setattr(sys,'argv',[str(script),'--reference','synthetic.vqr','--output',str(output)])
    runpy.run_path(str(script),run_name='__main__')
    result=json.loads(output.read_text())
    assert result['singles_difference']<2e-10
    assert result['doubles_difference']<2e-10
    assert not result['scientific_acceptance'] and not result['iterative_ccsd']
