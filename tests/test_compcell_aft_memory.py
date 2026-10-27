"""Bounded reciprocal contraction against the unchanged independent dense formula.

Extract the production function so these source checks need no native rebuild;
actual affected vq acceptance is separately required before scientific closure.
"""
import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import weakref

import numpy as np
import pytest

SOURCE = Path(__file__).parents[1] / 'python/vibeqc/aux_basis.py'


def fixture(monkeypatch, block_points):
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_compcell_aft_correction_3c')
    module = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
    g = np.arange(1,40,dtype=float).reshape(13,3)/11
    cells = np.array([[0.,0.,0.],[1.,2.,-1.],[-1.,-2.,1.]])
    scales = np.array([1.2,.7])
    calls=[];refs=[]
    def pair(basis,vectors,actual_cells,*,k_cart,pair_cutoff_bohr):
        assert all(r() is None for r in refs), 'Previous native output remains live'
        np.testing.assert_array_equal(actual_cells,cells)
        assert pair_cutoff_bohr == 7.5
        calls.append((vectors.copy(),k_cart.copy()))
        # Exact Bloch cell sum with nontrivial complex phase at each G.
        a=np.exp(1j*(vectors @ cells.T + cells @ k_cart)).sum(axis=1)
        result=np.arange(1,5).reshape(2,2,1)*a[None,None,:]
        refs.append(weakref.ref(result));return result
    def aux(basis,vectors):
        return np.exp(-np.arange(1,6)[:,None]*np.sum(vectors**2,axis=1)[None,:]/10)*(1+1j*vectors[:,0])[None,:]
    package=ModuleType('_aft_test');package.__path__=[]
    native=ModuleType('_aft_test._aopair_ft');native.ao_pair_fourier_transform_bloch=pair
    monkeypatch.setitem(sys.modules,'_aft_test',package);monkeypatch.setitem(sys.modules,'_aft_test._aopair_ft',native)
    ns=dict(__name__='_aft_test.aux_basis',__package__='_aft_test',np=np,_AFT_3C_WORK_BYTES=block_points*(16*(3*4+5+2*3)+64),_aft_g_mesh=lambda *a,**kw:g.copy(),_physical_pair_ft_cells=lambda *a:cells,_aft_fourier_transform_libcint_convention=aux,rsgdf_aux_fourier_transform=aux,_per_ao_pair_libint_to_libcint_scale=lambda b:scales,_ao_scales_for_rsgdf=lambda b:scales,LatticeSumOptions=lambda:SimpleNamespace(cutoff_bohr=7.5))
    exec(compile(ast.fix_missing_locations(module),str(SOURCE),'exec'),ns)
    return ns,g,cells,scales,calls,aux


@pytest.mark.parametrize('block_points',[1,4,100])
@pytest.mark.parametrize('convention',['libint','libcint'])
@pytest.mark.parametrize('q,ket',[(None,None),([.1,-.2,.3],[.3,.1,-.1]),(None,[.2,0.,0.])])
def test_exact_mesh_cells_phase_and_dense_contraction(monkeypatch,block_points,convention,q,ket):
    ns,g,cells,scales,calls,aux=fixture(monkeypatch,block_points)
    system=SimpleNamespace(lattice=np.diag([3.,4.,5.]))
    fused=SimpleNamespace(nbasis=5);orb=SimpleNamespace(nbasis=2)
    result=ns['_compcell_aft_correction_3c'](fused,orb,2,system,eta=1.,q=q,k_ket=ket,ft_convention=convention)
    shift=np.zeros(3) if q is None else np.array(q);k=shift if ket is None else np.array(ket);gq=g+shift
    np.testing.assert_array_equal(np.concatenate([a for a,k in calls]),gq)
    assert max(len(a) for a,k in calls)<=block_points
    for a,actual_k in calls:np.testing.assert_array_equal(actual_k,k)
    rho=np.arange(1,5).reshape(2,2,1)*np.exp(1j*(gq@cells.T+cells@k)).sum(axis=1)[None,None,:]
    factor=scales if convention=='libint' else 1/scales
    rho=rho*factor[:,None,None]*factor[None,:,None]
    expected=(np.conj(aux(fused,gq)[2:])*(4*np.pi/np.sum(gq*gq,axis=1)/60))@rho.reshape(4,13).T
    expected=expected.reshape(3,2,2)
    if q is None and ket is None:expected=expected.real
    np.testing.assert_allclose(result,expected,atol=1e-12,rtol=1e-13)
    assert np.iscomplexobj(result)==np.iscomplexobj(expected)


def test_empty_mesh_and_invalid_convention(monkeypatch):
    ns,*_=fixture(monkeypatch,4);system=SimpleNamespace(lattice=np.eye(3));fused=SimpleNamespace(nbasis=5);orb=SimpleNamespace(nbasis=2)
    with pytest.raises(ValueError,match='ft_convention'):
        ns['_compcell_aft_correction_3c'](fused,orb,2,system,eta=1.,ft_convention='invalid')
    ns['_aft_g_mesh']=lambda *a,**kw:np.empty((0,3))
    np.testing.assert_array_equal(ns['_compcell_aft_correction_3c'](fused,orb,2,system,eta=1.),np.zeros((3,2,2)))
