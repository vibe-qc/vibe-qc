"""Compressed complex paired RI actions against independently materialized maps."""
import dataclasses
import itertools

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core
from tests.test_manta_periodic_gdf_actions import inputs, check_actions
from tests.test_manta_periodic_gdf_tiles import assemble
from tests.test_manta_periodic_ri_multipole import multipole_case, controls
from tests.test_manta_periodic_ri_preparation import prepared_job
from tests.test_manta_periodic_ri_hierarchy import hierarchy_ks_job
from tests.test_manta_periodic_ri_q import builder, densities, reference
from tests.test_manta_periodic_fitting import CAPS


def compressed_case(gamma=False, order=2, reuse=True, **kwargs):
    a=multipole_case(gamma, order=order, **kwargs)
    a['options'].coupling_multipole_hierarchy=True
    a['options'].coupling_multipole_compress=True
    a['options'].coupling_multipole_reuse=reuse
    return a


def check_compressed_receipt(result):
    r=result.receipt
    assert r.multipole_compressed_nodes==r.multipole_far_blocks>0
    assert r.multipole_compressed_pairs>0 and r.multipole_action_products>0
    assert r.multipole_action_translations>0 if r.direction=='forward' else r.multipole_action_translations==0
    assert r.algorithm=='ordered-periodic-gdf-compressed-v1'


@pytest.mark.parametrize('gamma',[False,True])
@pytest.mark.parametrize('order',[0,2,4])
@pytest.mark.parametrize('reuse',[False,True])
def test_compressed_and_materialized_maps_have_the_same_partition_and_complex_adjoint(gamma,order,reuse):
    a=compressed_case(gamma,order,reuse);source=manta.PeriodicGDFTiles(**a)
    t=assemble(source);x,c=inputs(source)
    f,rev=check_actions(source,t,x,c,a['budget'])
    for r in (f,rev):
        check_compressed_receipt(r)
        assert r.receipt.maximum_coupling>=np.max(abs(t))
    assert f.receipt.maximum_coupling==rev.receipt.maximum_coupling
    a['options'].coupling_multipole_compress=False;materialized=manta.PeriodicGDFTiles(**a)
    np.testing.assert_array_equal(assemble(materialized),t)
    np.testing.assert_array_equal(materialized.metric().values,source.metric().values)
    assert materialized.source_identity==source.source_identity
    assert materialized.plan.compressed_action_bytes==materialized.plan.compressed_action_scalar_terms==0
    assert source.plan.compressed_action_bytes>0 and source.plan.compressed_action_scalar_terms>0
    for action,arg,result in [('forward',x,f),('adjoint',c,rev)]:
        ref=getattr(materialized,action)(arg,a['budget'])
        assert ref.receipt.multipole_compressed_nodes==ref.receipt.multipole_compressed_pairs==0
        np.testing.assert_allclose(result.values,ref.values,atol=4e-12,rtol=3e-13)
        for name in ('multipole_tree_nodes','multipole_far_blocks','multipole_images','coupling_error_bound','approximation_bound'):
            assert getattr(result.receipt,name)==getattr(ref.receipt,name)
        assert result.receipt.snapshot_identity!=ref.receipt.snapshot_identity
        zero=getattr(source,action)(0*arg,a['budget']);check_compressed_receipt(zero)
        np.testing.assert_array_equal(zero.values,np.zeros_like(zero.values))
        assert zero.receipt.multipole_compressed_nodes==result.receipt.multipole_compressed_nodes
        assert zero.receipt.maximum_coupling==result.receipt.maximum_coupling
        np.testing.assert_allclose(getattr(source,action)((2-3j)*arg,a['budget']).values,(2-3j)*result.values,atol=1e-11,rtol=2e-13)


@pytest.mark.parametrize('gamma',[False,True])
def test_every_operator_column_and_reverse_column_against_materialized_tensor(gamma):
    a=compressed_case(gamma);source=manta.PeriodicGDFTiles(**a);t=assemble(source)
    n=source.plan.n_orb;nk=source.plan.momenta;na=source.plan.n_aux
    forward=np.column_stack([source.forward(e.reshape(nk,n,n).astype(complex),a['budget']).values
                             for e in np.eye(nk*n*n)])
    reverse=np.column_stack([source.adjoint(e.astype(complex),a['budget']).values.ravel() for e in np.eye(na)])
    dense=np.moveaxis(t,1,0).reshape(na,-1)
    np.testing.assert_allclose(forward,dense.conj(),atol=3e-12,rtol=2e-13)
    np.testing.assert_allclose(reverse,forward.conj().T,atol=3e-12,rtol=2e-13)


def test_compressed_actions_preserve_atom_image_and_reciprocal_gauges():
    a=compressed_case();moved=compressed_case(shift=True)
    source=manta.PeriodicGDFTiles(**a);other=manta.PeriodicGDFTiles(**moved);x,c=inputs(source)
    lattice=a['system'].lattice[:,0]
    auxiliary=np.exp(1j*np.array([0.,a['q']@lattice]));pair=[]
    for k in a['ket_kpoints']:
        bra=np.exp(1j*np.array([0.,(k-a['q'])@lattice]));ket=np.exp(-1j*np.array([0.,k@lattice]))
        pair.append(bra[:,None]*ket[None,:])
    pair=np.asarray(pair)
    np.testing.assert_allclose(other.forward(pair*x,a['budget']).values,auxiliary.conj()*source.forward(x,a['budget']).values,atol=5e-12,rtol=0)
    np.testing.assert_allclose(other.adjoint(auxiliary.conj()*c,a['budget']).values,pair*source.adjoint(c,a['budget']).values,atol=5e-12,rtol=0)
    reciprocal=a['system'].reciprocal_lattice()@np.array([1,-1,2])
    relabel=manta.PeriodicGDFTiles(**dict(a,q=a['q']+reciprocal,ket_kpoints=a['ket_kpoints']+reciprocal))
    for action,arg in [('forward',x),('adjoint',c)]:
        np.testing.assert_allclose(getattr(relabel,action)(arg,a['budget']).values,getattr(source,action)(arg,a['budget']).values,atol=7e-12,rtol=0)


def angular_case(pure):
    a=compressed_case();system=vq.PeriodicSystem(3,np.diag([6.,6.2,6.4]),[vq.Atom(2,[.2,.3,.4])]);mol=system.unit_cell_molecule()
    a.update(system=system,
        orbital=vq.BasisSet(mol,[core.ShellInfo(0,2,pure,[4.,7.],[1.,-.3],[.2,.3,.4])],'compressed-angular-pair',True),
        auxiliary=vq.BasisSet(mol,[core.ShellInfo(0,2,pure,[6.,9.],[1.,-.25],[.2,.3,.4])],'compressed-angular-aux',True),
        vectors=np.array(list(itertools.product(range(-1,2),repeat=3)))@system.reciprocal_lattice().T+a['q'])
    a['options'].pair_cutoff_bohr=1.;a['options'].auxiliary_cutoff_bohr=14.;controls(a['options'],1.,2)
    return a


@pytest.mark.parametrize('pure',[False,True])
def test_signed_angular_charge_compression_with_independent_integrals(pure):
    from tests.test_manta_periodic_gdf import quadrature_long_range
    from tests.test_manta_periodic_fitting import libint_reference
    a=angular_case(pure);source=manta.PeriodicGDFTiles(**a);x,c=inputs(source);t=assemble(source)
    f,rev=check_actions(source,t,x,c,a['budget']);check_compressed_receipt(f);check_compressed_receipt(rev)
    exact=libint_reference(a)[1]+quadrature_long_range(a)[1]
    for r,ref in [(f,np.einsum('kpij,kij->p',exact.conj(),x)),(rev,np.einsum('kpij,p->kij',exact,c))]:
        assert np.max(abs(r.values-ref))<=r.receipt.approximation_bound+5e-12
        assert r.receipt.maximum_coupling>=np.max(abs(t))


@pytest.mark.parametrize('field,quantity',CAPS)
def test_compressed_workspace_and_work_are_admitted_before_bad_input(field,quantity):
    a=compressed_case();source=manta.PeriodicGDFTiles(**a);x,c=inputs(source);b=a['budget'];p=source.plan_coupling(b)
    for name,value in CAPS:setattr(b,name,getattr(p,value))
    source.forward(x,b);source.adjoint(c,b)
    setattr(b,field,getattr(p,quantity)-1);x[:]=np.nan;c[:]=np.nan
    for action,arg in [('forward',x),('adjoint',c)]:
        with pytest.raises((ValueError,RuntimeError),match='budget'):getattr(source,action)(arg,b)


@pytest.mark.parametrize('rank',[0,1,6])
def test_compressed_paired_ri_keeps_finite_factor_weights_energy_and_derivative(rank):
    a=compressed_case(True,fitted=True);full=builder(a);e=np.linalg.eigvalsh(full.metric)
    cutoff=1e-12 if rank==6 else (2*e[-1] if rank==0 else .5*(e[-2]+e[-1]))
    b=builder(a,block=6,absolute_cutoff=cutoff);d=densities();w=np.array([0.,1.]);t=assemble(manta.PeriodicGDFTiles(**a))
    build=lambda density:b.build(density,w,make_k=False)
    result=build(d);j,_,energy,_,_=reference(b.metric,t,d,w,absolute=cutoff)
    assert result.receipt.coupling_multipole_compressed_nodes>0 and result.receipt.coupling_passes==2
    np.testing.assert_allclose(result.J,j,atol=5e-12,rtol=2e-12)
    assert result.receipt.coulomb_energy==pytest.approx(energy,abs=5e-12)
    delta=densities(seed=231);step=2e-5
    derivative=(build(d+step*delta).receipt.coulomb_energy-build(d-step*delta).receipt.coulomb_energy)/(2*step)
    assert derivative==pytest.approx(np.einsum('k,skij,kji->',w,delta,result.J).real,abs=2e-10)
    a['options'].coupling_multipole_compress=False;other=builder(a,block=6,absolute_cutoff=cutoff)
    for name in ('metric','factor','projector'):np.testing.assert_array_equal(getattr(b,name),getattr(other,name))


@pytest.mark.parametrize('mode',['exact','leaf','reciprocal'])
def test_inactive_modes_do_not_reserve_or_execute_compressed_hierarchy(mode):
    a=compressed_case()
    if mode=='exact':a['options'].coupling_multipole_error=0.
    if mode=='leaf':a['options'].coupling_multipole_hierarchy=False
    if mode=='reciprocal':
        kernel=manta.PeriodicGDFKernelOptions();kernel.kernel='erf';kernel.omega=.3;a['physical_kernel']=kernel
    source=manta.PeriodicGDFTiles(**a);assert source.plan.compressed_action_bytes==0
    for r in check_actions(source,assemble(source),*inputs(source),a['budget']):
        assert r.receipt.multipole_compressed_pairs==r.receipt.multipole_compressed_nodes==0


@pytest.mark.parametrize('direction',['forward','adjoint'])
@pytest.mark.parametrize('value',[complex(np.nan,0),complex(1e308,1e308)])
def test_compressed_nonfinite_and_overflow_fail_without_projection(direction,value):
    a=compressed_case();source=manta.PeriodicGDFTiles(**a);x,c=inputs(source);arg=x if direction=='forward' else c;arg[:]=value
    with pytest.raises((ValueError,RuntimeError,OverflowError)):getattr(source,direction)(arg,a['budget'])


@pytest.mark.parametrize('ks',[False,True])
def test_complete_public_job_exposes_compressed_shared_fock_receipts(tmp_path,ks):
    job=hierarchy_ks_job(tmp_path,2)[0] if ks else prepared_job(tmp_path)[0]
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],coupling_multipole_compress=True)
    result=vq.run_periodic_job(**job);assert result.converged
    r=result.native.receipt.fitted_fock
    assert r.coupling_multipole_compressed_nodes==r.coupling_multipole_far_blocks>0
    assert r.coupling_multipole_compressed_pairs>0 and r.coupling_multipole_action_translations>0
    evidence=result.integral_provenance.get('coupling_approximation',result.integral_provenance.get('fitted_fock'))
    for name in ('coupling_multipole_compressed_pairs','coupling_multipole_compressed_nodes','coupling_multipole_action_products','coupling_multipole_action_translations'):
        assert evidence[name]==getattr(r,name)
    assert result.integral_provenance['fitting']['controls']['coupling_multipole_compress']


def test_typed_compression_control():
    for value in (1,'true',None):
        with pytest.raises(TypeError,match='boolean'):manta.PeriodicFittingOptions(.2,8.,coupling_multipole_compress=value)
    assert not manta.PeriodicFittingOptions(.2,8.).coupling_multipole_compress
    assert not manta.PeriodicFittingOptions(.2,8.,coupling_multipole_compress=False).coupling_multipole_compress



@pytest.mark.parametrize('field,quantity',CAPS)
def test_ri_parent_reserves_compressed_producer_workspace_and_work(field,quantity):
    a=compressed_case(True,fitted=True);source=manta.PeriodicGDFTiles(**a)
    options=manta.RIOptions();options.mode_block=6
    b=a['budget'];p=manta.plan_periodic_ri_q(source,2,options,b)
    for name,value in CAPS:setattr(b,name,getattr(p,value))
    owner=manta.PeriodicRIQBuilder(source,2,options,b)
    owner.build(densities(),np.array([.27,.73]),make_k=False)
    setattr(b,field,getattr(p,quantity)-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.PeriodicRIQBuilder(source,2,options,b)



@pytest.mark.parametrize('global_actions',[False,True])
@pytest.mark.parametrize('gamma,order',[(False,0),(False,2),(True,2),(True,4)])
def test_verified_compressed_column_norms_use_same_source(global_actions,gamma,order):
    a=compressed_case(gamma,order);a['options'].coupling_multipole_global_actions=global_actions
    source=manta.PeriodicGDFTiles(**a);x,c=inputs(source)
    before=source.forward(x,a['budget']);audit=source.coupling_norm_bounds(a['budget'])
    assert audit.metric_norm_qualified and audit.receipt.multipole_compressed_nodes>0
    if global_actions:assert audit.receipt.multipole_global_pair_reuses>0
    assert audit.receipt.coupling_identity==before.receipt.coupling_identity
    np.testing.assert_array_equal(audit.values,np.zeros_like(audit.values))
    dense=np.moveaxis(assemble(source),1,0).reshape(source.plan.n_aux,-1)
    norms=np.asarray(audit.metric_column_norm_bounds)
    assert norms.shape==(x.size,) and np.all(np.linalg.norm(dense,axis=0)<=norms)
    assert np.linalg.norm(before.values)<=np.dot(norms,abs(x).ravel())
    assert np.all(abs(source.adjoint(c,a['budget']).values).ravel()<=norms*np.linalg.norm(c))
    after=source.forward(x,a['budget']);np.testing.assert_array_equal(after.values,before.values)
    assert after.receipt.snapshot_identity==before.receipt.snapshot_identity
    assert not after.metric_norm_qualified and after.metric_column_norm_bounds==[]
    assert audit.receipt.shell_calls<=audit.plan.shell_calls
    assert audit.receipt.primitive_products<=audit.plan.primitive_products


@pytest.mark.parametrize('field,quantity',CAPS)
def test_verified_compressed_norm_resource_caps(field,quantity):
    a=compressed_case(True,0);source=manta.PeriodicGDFTiles(**a)
    plan=source.plan_coupling_norm_bounds(a['budget']);value=getattr(plan,quantity)
    setattr(a['budget'],field,value-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):source.coupling_norm_bounds(a['budget'])
    setattr(a['budget'],field,value);assert source.coupling_norm_bounds(a['budget']).metric_norm_qualified


def test_pair_tree_metric_norm_audit_remains_refused():
    from tests.test_manta_periodic_ri_pair_tree import pair_actions_case
    a=pair_actions_case(True);source=manta.PeriodicGDFTiles(**a)
    with pytest.raises(ValueError,match='pair-tree'):source.coupling_norm_bounds(a['budget'])


@pytest.mark.parametrize('pure',[False,True])
def test_verified_column_norms_cover_signed_angular_panels(pure):
    a=angular_case(pure);a['options'].coupling_multipole_global_actions=True
    source=manta.PeriodicGDFTiles(**a);audit=source.coupling_norm_bounds(a['budget'])
    dense=np.moveaxis(assemble(source),1,0).reshape(source.plan.n_aux,-1)
    assert audit.metric_norm_qualified and audit.receipt.multipole_compressed_nodes>0
    assert np.all(np.linalg.norm(dense,axis=0)<=np.asarray(audit.metric_column_norm_bounds))
