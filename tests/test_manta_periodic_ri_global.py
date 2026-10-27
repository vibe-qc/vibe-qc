"""Call-wide ordered pair charges/fields without changing the finite coupling."""
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import threading

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from tests.test_manta_periodic_ri_compressed import compressed_case, angular_case
from tests.test_manta_periodic_gdf_actions import inputs
from tests.test_manta_periodic_gdf_tiles import assemble, plan as source_plan
from tests.test_manta_periodic_ri_q import builder, densities, reference
from tests.test_manta_periodic_ri_preparation import prepared_job
from tests.test_manta_periodic_ri_hierarchy import hierarchy_ks_job
from tests.test_manta_periodic_fitting import CAPS


def global_case(gamma=False,order=2,**kw):
    a=compressed_case(gamma,order,**kw)
    a['options'].coupling_multipole_global_actions=True
    return a


def check_global(result,source):
    r=result.receipt
    assert r.algorithm=='ordered-periodic-gdf-global-panels-v1'
    assert 0<r.multipole_global_pairs<=source.plan.global_pair_slots
    assert r.multipole_global_pairs+r.multipole_global_pair_reuses==r.multipole_compressed_pairs
    if r.direction=='forward':
        assert r.multipole_global_auxiliary==r.multipole_global_auxiliary_projections>0 and r.multipole_global_projections==0
    else:
        assert 0<r.multipole_global_auxiliary<=source.plan.global_auxiliary_slots
        assert r.multipole_global_projections==r.multipole_global_pairs


@pytest.mark.parametrize('gamma',[False,True])
@pytest.mark.parametrize('order',[0,2,4])
@pytest.mark.parametrize('capacity',[0,64*1024*1024])
def test_call_wide_maps_preserve_local_and_materialized_coupling(gamma,order,capacity):
    a=global_case(gamma,order);a['options'].coupling_multipole_partition_bytes=capacity
    source=manta.PeriodicGDFTiles(**a);x,c=inputs(source);t=assemble(source)
    f=source.forward(x,a['budget']);rev=source.adjoint(c,a['budget'])
    for r in (f,rev):
        check_global(r,source)
        assert r.receipt.multipole_global_pair_reuses>0
        assert r.receipt.maximum_coupling>=np.max(abs(t))
    np.testing.assert_allclose(f.values,np.einsum('kpij,kij->p',t.conj(),x),atol=4e-12,rtol=3e-13)
    np.testing.assert_allclose(rev.values,np.einsum('kpij,p->kij',t,c),atol=4e-12,rtol=3e-13)
    assert abs(np.vdot(c,f.values)-np.vdot(rev.values,x))<4e-12
    a['options'].coupling_multipole_global_actions=False;local=manta.PeriodicGDFTiles(**a)
    assert local.source_identity==source.source_identity and local.plan.global_action_bytes==0
    np.testing.assert_array_equal(local.metric().values,source.metric().values)
    for name,arg,result in [('forward',x,f),('adjoint',c,rev)]:
        old=getattr(local,name)(arg,a['budget'])
        np.testing.assert_allclose(result.values,old.values,atol=4e-12,rtol=3e-13)
        assert result.receipt.snapshot_identity!=old.receipt.snapshot_identity
        assert result.receipt.multipole_action_products<old.receipt.multipole_action_products
        for field in ('coupling_error_bound','maximum_coupling','multipole_images','multipole_far_blocks','multipole_compressed_nodes'):
            assert getattr(result.receipt,field)==getattr(old.receipt,field)
        np.testing.assert_allclose(getattr(source,name)((2-3j)*arg,a['budget']).values,(2-3j)*result.values,atol=1e-11,rtol=3e-13)
        zero=getattr(source,name)(0*arg,a['budget']);check_global(zero,source)
        np.testing.assert_array_equal(zero.values,0*result.values)
        assert zero.receipt.multipole_global_pairs==result.receipt.multipole_global_pairs


@pytest.mark.parametrize('pure',[False,True])
def test_global_angular_projection_preserves_signed_cartesian_and_spherical_panels(pure):
    a=angular_case(pure);a['options'].coupling_multipole_global_actions=True
    source=manta.PeriodicGDFTiles(**a);x,c=inputs(source);t=assemble(source)
    f=source.forward(x,a['budget']);rev=source.adjoint(c,a['budget'])
    check_global(f,source);check_global(rev,source)
    np.testing.assert_allclose(f.values,np.einsum('kpij,kij->p',t.conj(),x),atol=4e-12,rtol=3e-13)
    np.testing.assert_allclose(rev.values,np.einsum('kpij,p->kij',t,c),atol=4e-12,rtol=3e-13)


@pytest.mark.parametrize('gamma',[False,True])
def test_global_action_exhaustive_columns_and_adjoint(gamma):
    a=global_case(gamma);source=manta.PeriodicGDFTiles(**a);t=assemble(source)
    shape=(source.plan.momenta,source.plan.n_orb,source.plan.n_orb);m=np.prod(shape);na=source.plan.n_aux
    f=np.column_stack([source.forward(x.reshape(shape).astype(complex),a['budget']).values for x in np.eye(m)])
    rev=np.column_stack([source.adjoint(c.astype(complex),a['budget']).values.ravel() for c in np.eye(na)])
    np.testing.assert_allclose(f,np.moveaxis(t,1,0).reshape(na,-1).conj(),atol=4e-12,rtol=3e-13)
    np.testing.assert_allclose(rev,f.conj().T,atol=4e-12,rtol=3e-13)


def test_global_actions_preserve_atom_image_and_reciprocal_gauges():
    a=global_case();moved=global_case(shift=True)
    source=manta.PeriodicGDFTiles(**a);other=manta.PeriodicGDFTiles(**moved);x,c=inputs(source)
    cell=a['system'].lattice[:,0];shift=np.array([0.,1.]);aux=np.exp(1j*shift*(a['q']@cell))
    phase=np.array([np.exp(1j*shift*((k-a['q'])@cell))[:,None]*np.exp(-1j*shift*(k@cell))[None,:] for k in a['ket_kpoints']])
    np.testing.assert_allclose(other.forward(x*phase,a['budget']).values,source.forward(x,a['budget']).values/aux,atol=8e-12,rtol=3e-13)
    np.testing.assert_allclose(other.adjoint(c/aux,a['budget']).values,source.adjoint(c,a['budget']).values*phase,atol=8e-12,rtol=3e-13)
    reciprocal=a['system'].reciprocal_lattice()@np.array([1,-1,2])
    shifted=dict(a,q=a['q']+reciprocal,ket_kpoints=a['ket_kpoints']+reciprocal)
    np.testing.assert_allclose(manta.PeriodicGDFTiles(**shifted).forward(x,a['budget']).values,source.forward(x,a['budget']).values,atol=8e-12,rtol=3e-13)


def test_concurrent_global_inputs_have_no_shared_charge_or_field_state():
    a=global_case();source=manta.PeriodicGDFTiles(**a);x,c=inputs(source);barrier=threading.Barrier(4)
    def run(i):
        barrier.wait();return source.forward((i+1)*x,a['budget']) if i%2==0 else source.adjoint((i+1)*c,a['budget'])
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(run,range(4)))
    for i,r in enumerate(results):
        check_global(r,source)
        expected=source.forward((i+1)*x,a['budget']) if i%2==0 else source.adjoint((i+1)*c,a['budget'])
        np.testing.assert_array_equal(r.values,expected.values)


@pytest.mark.parametrize('rank',[0,1,6])
def test_global_fixed_metric_rank_and_weighted_energy_derivative(rank):
    a=global_case(True,fitted=True);full=builder(a);e=np.linalg.eigvalsh(full.metric)
    cutoff=1e-12 if rank==6 else (2*e[-1] if rank==0 else .5*(e[-2]+e[-1]))
    b=builder(a,block=6,absolute_cutoff=cutoff);d=densities();w=np.array([.27,.73]);r=b.build(d,w,make_k=False)
    t=assemble(manta.PeriodicGDFTiles(**a));j,_,energy,_,_=reference(b.metric,t,d,w,absolute=cutoff)
    np.testing.assert_allclose(r.J,j,atol=8e-12,rtol=3e-13);assert r.receipt.coulomb_energy==pytest.approx(energy,abs=8e-12)
    direction=.13*d[::-1].copy();step=1e-5
    derivative=(b.build(d+step*direction,w,make_k=False).receipt.coulomb_energy-b.build(d-step*direction,w,make_k=False).receipt.coulomb_energy)/(2*step)
    expected=np.einsum('k,skij,kji->',w,direction,r.J).real
    assert derivative==pytest.approx(expected,abs=2e-8)


@pytest.mark.parametrize('field,quantity',CAPS)
def test_global_storage_and_work_are_admitted_before_bad_inputs(field,quantity):
    a=global_case();source=manta.PeriodicGDFTiles(**a);x,_=inputs(source);source.forward(x,a['budget'])
    assert source.plan.global_action_bytes>0 and source.plan.global_action_scalar_terms>0
    b=a['budget'];p=source.plan_coupling(b)
    for name,value in CAPS:setattr(b,name,getattr(p,value))
    source.forward(x,b);setattr(b,field,getattr(p,quantity)-1);x[:]=np.nan
    with pytest.raises((ValueError,RuntimeError),match='budget'):source.forward(x,b)


@pytest.mark.parametrize('field,quantity',CAPS)
def test_parent_ri_composes_global_action_admission(field,quantity):
    a=global_case(True,fitted=True);source=manta.PeriodicGDFTiles(**a);o=manta.RIOptions();o.mode_block=6;b=a['budget']
    p=manta.plan_periodic_ri_q(source,2,o,b)
    for name,value in CAPS:setattr(b,name,getattr(p,value))
    owner=manta.PeriodicRIQBuilder(source,2,o,b);owner.build(densities(),np.array([.27,.73]),make_k=False)
    setattr(b,field,getattr(p,quantity)-1)
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.PeriodicRIQBuilder(source,2,o,b)


@pytest.mark.parametrize('inactive',['exact','leaf','unprepared','materialized','reciprocal'])
def test_inactive_routes_have_no_global_buffer(inactive):
    a=global_case()
    if inactive=='exact':a['options'].coupling_multipole_error=0.
    if inactive=='leaf':a['options'].coupling_multipole_hierarchy=False
    if inactive=='unprepared':a['options'].coupling_multipole_reuse=False
    if inactive=='materialized':a['options'].coupling_multipole_compress=False
    if inactive=='reciprocal':
        k=manta.PeriodicGDFKernelOptions();k.kernel='erf';k.omega=.3;a['physical_kernel']=k
    source=manta.PeriodicGDFTiles(**a)
    assert source.plan.global_action_bytes==source.plan.global_action_scalar_terms==source.plan.global_pair_slots==source.plan.global_auxiliary_slots==0


@pytest.mark.parametrize('ks',[False,True])
def test_public_global_pair_stage_preserves_shared_fock_receipts(tmp_path,ks):
    if ks:job,*_=hierarchy_ks_job(tmp_path,2)
    else:job,*_=prepared_job(tmp_path)
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],coupling_multipole_compress=True,coupling_multipole_global_actions=True)
    result=vq.run_periodic_job(**job);assert result.converged;r=result.native.receipt.fitted_fock
    assert r.coupling_multipole_global_pair_reuses>0 and r.coupling_multipole_global_auxiliary>0
    assert r.coupling_multipole_global_pairs==2*r.coupling_multipole_global_projections>0
    assert r.coupling_multipole_global_auxiliary==2*r.coupling_multipole_global_auxiliary_projections>0
    info=result.integral_provenance.get('coupling_approximation',result.integral_provenance.get('fitted_fock'))
    for field in ('pairs','pair_reuses','auxiliary','projections','auxiliary_projections'):
        name='coupling_multipole_global_'+field;assert info[name]==getattr(r,name)
    assert result.integral_provenance['fitting']['controls']['coupling_multipole_global_actions']


def test_typed_global_control():
    assert not manta.PeriodicFittingOptions(.2,8.).coupling_multipole_global_actions
    for value in (1,'true',None):
        with pytest.raises(TypeError,match='boolean'):manta.PeriodicFittingOptions(.2,8.,coupling_multipole_global_actions=value)



def test_global_source_reservation_precedes_invalid_geometry_and_input_never_leaks_state():
    a=global_case();p=source_plan(a);a['budget'].memory_bytes=p.modeled_peak_bytes-1;a['q'][:]=np.nan
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.PeriodicGDFTiles(**a)
    a=global_case();source=manta.PeriodicGDFTiles(**a);x,c=inputs(source);valid=x.copy();x[:]=np.nan
    with pytest.raises(ValueError,match='finite'):source.forward(x,a['budget'])
    assert source.preparation.moment_panels==source.partitions.entries==0
    result=source.forward(valid,a['budget']);check_global(result,source)
    np.testing.assert_array_equal(source.forward(valid,a['budget']).values,result.values)
