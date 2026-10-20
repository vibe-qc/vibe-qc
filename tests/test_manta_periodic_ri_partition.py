"""Retained exact-source partitions: density independence, bounded storage, reuse."""
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import threading

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from tests.test_manta_periodic_ri_compressed import compressed_case
from tests.test_manta_periodic_gdf_actions import inputs
from tests.test_manta_periodic_gdf_tiles import assemble, plan as source_plan
from tests.test_manta_periodic_ri_preparation import prepared_job
from tests.test_manta_periodic_ri_hierarchy import hierarchy_ks_job
from tests.test_manta_periodic_fitting import CAPS


@pytest.mark.parametrize('compressed',[False,True])
@pytest.mark.parametrize('gamma',[False,True])
@pytest.mark.parametrize('order',[0,2,4])
def test_cached_partition_is_bitwise_streaming_for_both_maps_and_dense_tiles(compressed,gamma,order):
    a=compressed_case(gamma,order);a['options'].coupling_multipole_compress=compressed
    source=manta.PeriodicGDFTiles(**a);initial=source.partitions
    assert initial.enabled and initial.entries==0 and initial.source_identity==source.three_center_source_identity
    assert initial.capacity_bytes==source.plan.partition_cache_bytes
    x,c=inputs(source);cold=source.forward(x,a['budget']);full=source.partitions
    assert cold.receipt.multipole_partition_builds==full.entries>0
    assert full.evictions==0 and full.modeled_live_bytes<=full.capacity_bytes
    a['options'].coupling_multipole_partition_bytes=0;stream=manta.PeriodicGDFTiles(**a)
    assert not stream.partitions.enabled and stream.plan.partition_cache_bytes==stream.plan.partition_workspace_bytes==0
    assert source.source_identity==stream.source_identity
    for name,arg in [('forward',x),('adjoint',c),('forward',0*x),('adjoint',(2-3j)*c)]:
        result=getattr(source,name)(arg,a['budget']);reference=getattr(stream,name)(arg,a['budget'])
        np.testing.assert_array_equal(result.values,reference.values)
        assert result.receipt.snapshot_identity==reference.receipt.snapshot_identity
        assert result.receipt.multipole_partition_builds==result.receipt.multipole_trials==0
        assert result.receipt.multipole_partition_reuses==full.entries
        for field in ('multipole_images','multipole_tree_nodes','multipole_far_blocks','coupling_error_bound','maximum_coupling'):
            assert getattr(result.receipt,field)==getattr(reference.receipt,field)
    np.testing.assert_array_equal(assemble(source),assemble(stream))
    assert source.partitions.entries==full.entries and initial.entries==0
    assert manta.PeriodicGDFTiles(**a).partitions.entries==0
    with pytest.raises(AttributeError):source.partitions.entries=0


@pytest.mark.parametrize('capacity',[1,128*1024,1024*1024])
def test_bypass_and_eviction_preserve_the_same_finite_map(capacity):
    a=compressed_case();a['options'].coupling_multipole_partition_bytes=capacity
    source=manta.PeriodicGDFTiles(**a);x,c=inputs(source)
    a['options'].coupling_multipole_partition_bytes=0;stream=manta.PeriodicGDFTiles(**a)
    for name,arg in [('forward',x),('adjoint',c),('forward',2*x)]:
        result=getattr(source,name)(arg,a['budget'])
        np.testing.assert_array_equal(result.values,getattr(stream,name)(arg,a['budget']).values)
        info=source.partitions
        assert info.modeled_live_bytes<=info.capacity_bytes==capacity+8192
        assert result.receipt.multipole_partition_builds>0
    if capacity==1:
        assert info.entries==info.evictions==0 and result.receipt.multipole_partition_bypasses>0
    if capacity==1024*1024:assert info.entries>0 and info.evictions>0


@pytest.mark.parametrize('compressed',[False,True])
def test_concurrent_cold_calls_publish_each_partition_once_without_density_state(compressed):
    a=compressed_case();a['options'].coupling_multipole_compress=compressed
    source=manta.PeriodicGDFTiles(**a);x,c=inputs(source);barrier=threading.Barrier(4)
    def run(i):
        barrier.wait();return source.forward((i+1)*x,a['budget']) if i%2==0 else source.adjoint((i+1)*c,a['budget'])
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(run,range(4)))
    assert sum(r.receipt.multipole_partition_builds for r in results)==source.partitions.entries>0
    assert source.partitions.evictions==0
    a['options'].coupling_multipole_partition_bytes=0;stream=manta.PeriodicGDFTiles(**a)
    for i,result in enumerate(results):
        expected=stream.forward((i+1)*x,a['budget']) if i%2==0 else stream.adjoint((i+1)*c,a['budget'])
        np.testing.assert_array_equal(result.values,expected.values)


@pytest.mark.parametrize('field,quantity',CAPS)
def test_full_cache_and_borrowed_entry_reservation_remain_required_when_warm(field,quantity):
    a=compressed_case();source=manta.PeriodicGDFTiles(**a);x,_=inputs(source)
    source.forward(x,a['budget']);before=source.partitions
    assert source.plan.partition_cache_bytes==source.plan.partition_workspace_bytes+8192
    b=a['budget'];p=source.plan_coupling(b)
    for name,value in CAPS:setattr(b,name,getattr(p,value))
    source.forward(x,b);setattr(b,field,getattr(p,quantity)-1);x[:]=np.nan
    with pytest.raises((ValueError,RuntimeError),match='budget'):source.forward(x,b)
    assert source.partitions.entries==before.entries and source.partitions.evictions==before.evictions


def test_invalid_input_never_populates_cache_and_memory_failure_precedes_source_copy():
    a=compressed_case();source=manta.PeriodicGDFTiles(**a);x,_=inputs(source);x[:]=np.nan
    with pytest.raises(ValueError,match='finite'):source.forward(x,a['budget'])
    assert source.partitions.entries==0
    p=source_plan(a);a['budget'].memory_bytes=p.modeled_peak_bytes-1;a['q'][:]=np.nan
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.PeriodicGDFTiles(**a)


@pytest.mark.parametrize('inactive',['exact','leaf','stream','reciprocal'])
def test_inactive_paths_reserve_no_partition_storage(inactive):
    a=compressed_case()
    if inactive=='exact':a['options'].coupling_multipole_error=0.
    if inactive=='leaf':a['options'].coupling_multipole_hierarchy=False
    if inactive=='stream':a['options'].coupling_multipole_reuse=False
    if inactive=='reciprocal':
        physical=manta.PeriodicGDFKernelOptions();physical.kernel='erf';physical.omega=.3;a['physical_kernel']=physical
    source=manta.PeriodicGDFTiles(**a)
    assert not source.partitions.enabled
    assert source.plan.partition_cache_bytes==source.plan.partition_workspace_bytes==source.plan.partition_scalar_terms==0


@pytest.mark.parametrize('ks',[False,True])
def test_complete_public_fock_reuses_partitions_with_receipted_controls(tmp_path,ks):
    if ks:job,*_=hierarchy_ks_job(tmp_path,2)
    else:job,*_=prepared_job(tmp_path)
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],coupling_multipole_compress=True)
    result=vq.run_periodic_job(**job);assert result.converged
    r=result.native.receipt.fitted_fock
    assert r.coupling_multipole_partition_reuses>0
    assert r.coupling_multipole_partition_builds==r.coupling_multipole_trials==r.coupling_multipole_partition_bypasses==0
    info=result.integral_provenance.get('coupling_approximation',result.integral_provenance.get('fitted_fock'))
    for field in ('builds','reuses','bypasses'):
        name='coupling_multipole_partition_'+field;assert info[name]==getattr(r,name)
    assert result.integral_provenance['fitting']['controls']['coupling_multipole_partition_bytes']==64*1024*1024


@pytest.mark.parametrize('value',[True,-1,1.5,'1000',None,2**64])
def test_partition_capacity_requires_a_nonnegative_uint64(value):
    with pytest.raises(ValueError,match='nonnegative uint64'):
        manta.PeriodicFittingOptions(.2,8.,coupling_multipole_partition_bytes=value)
