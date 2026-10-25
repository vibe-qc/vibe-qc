"""Immutable, bounded shell source for periodic fitted consumers."""
import itertools

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from tests.test_manta_periodic_gdf import case, independent, legacy
from tests.test_manta_periodic_fitting import CAPS
from tests.test_aux_eri_sr_lattice_enumeration import _weighted_gradient_fixture


def plan(a):
    args=dict(a);args.pop('q');args['vectors']=len(args['vectors']);args['momenta']=len(args.pop('ket_kpoints'))
    return manta.plan_periodic_gdf_tiles(**args)


def assemble(source):
    p=source.plan;out=np.zeros((p.momenta,p.n_aux,p.n_orb,p.n_orb),complex)
    counts=np.zeros(2,dtype=np.uint64)
    for indices in itertools.product(*(range(n) for n in source.shell_extents)):
        t=source.evaluate(*indices);s=t.shape;o=t.offsets
        assert t.values.size<=p.maximum_tile_elements
        out[:,o[1]:o[1]+s[1],o[2]:o[2]+s[2],o[3]:o[3]+s[3]]=t.values
        counts+=np.array([t.receipt.shell_calls,t.receipt.primitive_products],dtype=np.uint64)
        assert t.receipt.three_center_source_identity==source.three_center_source_identity
    assert counts[0]<=p.coupling_shell_calls and counts[1]<=p.coupling_primitive_products
    return out


@pytest.mark.parametrize('gamma',[False,True])
@pytest.mark.parametrize('screen',[0.,1e-4])
def test_tiles_reconstruct_same_dense_source_and_independent_gaussian_blocks(gamma,screen):
    a=case(gamma);a['options'].integral_screen_error=screen
    s=manta.PeriodicGDFTiles(**a);r=manta.compute_periodic_gdf(**a)
    metric=s.metric();tensor=assemble(s);m,t,ng=legacy(a);im,it=independent(a)
    np.testing.assert_array_equal(metric.values,r.metric)
    np.testing.assert_array_equal(tensor,r.three_center)
    np.testing.assert_allclose(metric.values,m,atol=3e-12,rtol=2e-13)
    np.testing.assert_allclose(tensor,t,atol=3e-12,rtol=2e-13)
    np.testing.assert_allclose(metric.values,im,atol=screen+3e-12,rtol=2e-13)
    np.testing.assert_allclose(tensor,it,atol=screen+3e-12,rtol=2e-13)
    assert metric.receipt.reciprocal_vectors==ng and metric.receipt.finite_zero_removed==gamma
    assert metric.receipt.shell_calls<=s.plan.metric_shell_calls
    assert metric.receipt.primitive_products<=s.plan.metric_primitive_products
    for name in ('metric_source_identity','three_center_source_identity','source_identity'):
        assert getattr(s,name)==getattr(r.receipt,name)
    for name in ('orbital_basis_identity','auxiliary_basis_identity'):
        assert getattr(s,name)==getattr(r.receipt.short_range,name)


@pytest.mark.parametrize('angular,pure',[(1,True),(2,False),(3,True)])
def test_angular_signed_contraction_tiles_are_the_identical_finite_source(angular,pure):
    a=case();system,b,aux=_weighted_gradient_fixture(angular,orbital_angular=1)
    def change(basis):
        shells=basis.shells()
        for sh in shells:sh.pure=pure
        return vq.BasisSet(system.unit_cell_molecule(),shells,'gdf-tile-angular',True)
    indices=np.array(list(itertools.product(range(-1,2),repeat=3)))
    a.update(system=system,orbital=change(b),auxiliary=change(aux),vectors=indices@system.reciprocal_lattice().T+a['q'])
    a['options'].pair_cutoff_bohr=4.3;a['options'].auxiliary_cutoff_bohr=5.1
    source=manta.PeriodicGDFTiles(**a);dense=manta.compute_periodic_gdf(**a)
    np.testing.assert_array_equal(source.metric().values,dense.metric)
    np.testing.assert_array_equal(assemble(source),dense.three_center)


def test_source_snapshots_and_exports_are_owned_and_immutable():
    a=case(True);s=manta.PeriodicGDFTiles(**a);m=s.metric().values;t=s.evaluate(0,1,0).values;identity=s.source_identity
    for key in ('q','ket_kpoints','vectors'):a[key][:]=np.nan
    a['options'].omega=0.;a['options'].compute_metric=False
    a['system'].lattice=np.eye(3)*99.
    s.metric().values[:]=np.nan;s.evaluate(0,1,0).values[:]=np.nan
    np.testing.assert_array_equal(s.metric().values,m);np.testing.assert_array_equal(s.evaluate(0,1,0).values,t)
    assert s.source_identity==identity
    exported=s.evaluate(0,1,0);exported.shape[0]=999;exported.offsets[0]=999
    assert exported.shape[0]==2 and exported.offsets[0]==0


@pytest.mark.parametrize('field,quantity',CAPS)
@pytest.mark.parametrize('metric',[False,True])
def test_joint_construction_and_one_sweep_caps_before_numerical_copies(field,quantity,metric):
    a=case(True);a['options'].compute_metric=metric;a['budget'].other_live_bytes=23456;p=plan(a)
    for cap,value in CAPS:setattr(a['budget'],cap,getattr(p,value))
    source=manta.PeriodicGDFTiles(**a)
    if metric:source.metric()
    source.evaluate(0,0,0)
    setattr(a['budget'],field,getattr(p,quantity)-1)
    for key in ('q','ket_kpoints','vectors'):a[key][:]=np.nan
    with pytest.raises((ValueError,RuntimeError),match='budget'):plan(a)
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.PeriodicGDFTiles(**a)


def test_tile_admission_does_not_require_the_full_auxiliary_ao_pair_tensor():
    a=case(True);system=vq.PeriodicSystem(3,8*np.eye(3),[vq.Atom(2,[0.,0.,0.])])
    def basis(count):
        shells=[core.ShellInfo(0,0,True,[.3+.02*i],[1.],[0.,0.,0.]) for i in range(count)]
        return vq.BasisSet(system.unit_cell_molecule(),shells,'tile-memory-census',True)
    a.update(system=system,orbital=basis(24),auxiliary=basis(80),vectors=np.zeros((1,3)),ket_kpoints=np.zeros((4,3)))
    for field,_ in CAPS:setattr(a['budget'],field,2**64-1)
    p=plan(a);a['budget'].memory_bytes=p.modeled_peak_bytes
    source=manta.PeriodicGDFTiles(**a)
    assert source.plan.maximum_tile_elements==4
    assert 16*4*80*24**2>10**6
    # Only construction is needed to prove the dense allocation is absent;
    # no large numerical traversal is started for this admission witness.
    with pytest.raises((ValueError,RuntimeError),match='memory budget'):
        manta.compute_periodic_gdf(**a)


def test_layout_caps_metric_reuse_and_optional_metric_keep_source_identities():
    a=case();s=manta.PeriodicGDFTiles(**a);t=s.evaluate(0,0,0).values
    for name in ('q','ket_kpoints','vectors'):
        same=manta.PeriodicGDFTiles(**dict(a,**{name:np.asfortranarray(a[name])}))
        assert same.source_identity==s.source_identity
        np.testing.assert_array_equal(same.evaluate(0,0,0).values,t)
    p=plan(a);a['budget'].other_live_bytes+=1234
    assert plan(a).modeled_peak_bytes==p.modeled_peak_bytes+1234
    a['options'].image_candidate_cap*=2
    assert manta.PeriodicGDFTiles(**a).source_identity==s.source_identity
    a['options'].compute_metric=False;only=manta.PeriodicGDFTiles(**a)
    assert only.metric_source_identity=='' and only.three_center_source_identity==s.three_center_source_identity
    with pytest.raises(ValueError,match='metric'):only.metric()
    np.testing.assert_array_equal(only.evaluate(0,0,0).values,t)
    a['options'].compute_metric=True;a['options'].pair_cutoff_bohr=4.9
    a['ket_kpoints']=a['ket_kpoints'][:1].copy();changed=manta.PeriodicGDFTiles(**a)
    assert changed.metric_source_identity==s.metric_source_identity and changed.three_center_source_identity!=s.three_center_source_identity
    np.testing.assert_array_equal(changed.metric().values,s.metric().values)


@pytest.mark.parametrize('fault',['domain','nonfinite','dtype','strides','empty','shape','index'])
def test_source_and_tile_refusals(fault):
    a=case()
    if fault=='domain':a['vectors'][0,0]+=.01
    elif fault=='nonfinite':a['q'][0]=np.nan
    elif fault=='dtype':a['vectors']=a['vectors'].astype(np.float32)
    elif fault=='strides':a['ket_kpoints']=a['ket_kpoints'][::-1]
    elif fault=='empty':a['vectors']=np.empty((0,3))
    elif fault=='shape':a['vectors']=np.zeros((2,4))
    if fault=='index':
        with pytest.raises(IndexError):manta.PeriodicGDFTiles(**a).evaluate(999,0,0)
    else:
        with pytest.raises((ValueError,RuntimeError,TypeError)):manta.PeriodicGDFTiles(**a)
