"""Global ordered pair-bubble actions over the producer-qualified finite GDF source."""
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import threading

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from vibeqc.manta_periodic_job import PeriodicFittingOptions
from tests.test_manta_periodic_ri_multipole import multipole_case
from tests.test_manta_periodic_ri_compressed import angular_case
from tests.test_manta_periodic_gdf import independent,quadrature_long_range
from tests.test_manta_periodic_fitting import CAPS,libint_reference
from tests.test_manta_periodic_gdf_actions import inputs
from tests.test_manta_periodic_gdf_tiles import assemble,plan as source_plan
from tests.test_manta_periodic_ri_q import builder,densities,reference
from tests.test_manta_periodic_ri_preparation import prepared_job
from tests.test_manta_periodic_ri_hierarchy import hierarchy_ks_job


def pair_actions_case(gamma=False,order=2,**kw):
    a=multipole_case(gamma,order=order,**kw)
    a['options'].coupling_multipole_pair_tree=True
    return a


def check_pair_actions(result,source):
    r=result.receipt
    assert r.algorithm=='ordered-periodic-gdf-global-pair-tree-v1'
    assert r.multipole_far_blocks==r.multipole_compressed_nodes>0
    assert 0<r.multipole_global_pairs<=(2 if source.receipt.zero_transfer else 1)*source.plan.global_pair_slots
    assert r.multipole_compressed_pairs==r.multipole_global_pairs
    assert r.multipole_tree_nodes>0 and r.multipole_action_products>0
    if r.direction=='forward':
        assert r.multipole_global_auxiliary==r.multipole_global_auxiliary_projections>0
    else:assert r.multipole_global_projections==r.multipole_global_pairs


@pytest.mark.parametrize('gamma',[False,True])
@pytest.mark.parametrize('order',[0,2,4])
@pytest.mark.parametrize('capacity',[0,64*1024*1024])
def test_global_tree_maps_dense_exports_and_independent_finite_errors(gamma,order,capacity):
    a=pair_actions_case(gamma,order);a['options'].coupling_multipole_partition_bytes=capacity
    source=manta.PeriodicGDFTiles(**a);x,c=inputs(source)
    assert source.preparation.pair_extents==0
    f=source.forward(x,a['budget']);rev=source.adjoint(c,a['budget']);t=assemble(source)
    dense=manta.compute_periodic_gdf(**a)
    assert dense.receipt.short_range.coupling_multipole_pair_tree
    np.testing.assert_array_equal(dense.three_center,t)
    sr_args={k:v for k,v in a.items() if k!='vectors'};sr=manta.compute_periodic_gdf_short_range(**sr_args)
    np.testing.assert_allclose(sr.three_center+quadrature_long_range(a)[1],t,atol=5e-12,rtol=3e-13)
    m,et=independent(a)
    if gamma:
        np.testing.assert_allclose(et,et.swapaxes(-1,-2).conj(),atol=5e-12,rtol=3e-13)
        np.testing.assert_allclose(t,t.swapaxes(-1,-2).conj(),atol=5e-12,rtol=3e-13)
    np.testing.assert_allclose(source.metric().values,m,atol=5e-12,rtol=3e-13)
    for result,expected in [(f,np.einsum('kpij,kij->p',t.conj(),x)),(rev,np.einsum('kpij,p->kij',t,c))]:
        check_pair_actions(result,source)
        np.testing.assert_allclose(result.values,expected,atol=5e-12,rtol=3e-13)
        assert result.receipt.maximum_coupling>=np.max(abs(t))
        assert 0<result.receipt.coupling_error_bound<=a['options'].coupling_multipole_error
    assert f.receipt.multipole_pair_groups>0 and f.receipt.multipole_translations<f.receipt.multipole_far_blocks
    assert abs(np.vdot(c,f.values)-np.vdot(rev.values,x))<5e-12
    assert f.receipt.coupling_error_bound==rev.receipt.coupling_error_bound==dense.receipt.evaluated_coupling_error_bound
    assert np.max(abs(t-et))<=f.receipt.coupling_error_bound+5e-12
    assert np.max(abs(f.values-np.einsum('kpij,kij->p',et.conj(),x)))<=f.receipt.approximation_bound+5e-12
    warm=source.forward(x,a['budget']);np.testing.assert_array_equal(warm.values,f.values)
    assert warm.receipt.multipole_moment_panels==warm.receipt.multipole_envelope_evaluations==0
    assert source.partitions.modeled_live_bytes<=source.partitions.capacity_bytes
    a['options'].coupling_multipole_pair_tree=False
    previous=manta.PeriodicGDFTiles(**a)
    assert previous.three_center_source_identity!=source.three_center_source_identity
    assert previous.metric_source_identity==source.metric_source_identity
    np.testing.assert_array_equal(previous.metric().values,source.metric().values)
    a['options'].coupling_multipole_error=0.;exact=manta.PeriodicGDFTiles(**a)
    assert source.receipt.reference_three_center_source_identity==exact.three_center_source_identity


@pytest.mark.parametrize('pure',[False,True])
def test_signed_angular_global_pair_actions(pure):
    a=angular_case(pure);a['options'].coupling_multipole_pair_tree=True
    s=manta.PeriodicGDFTiles(**a);x,c=inputs(s);t=assemble(s)
    f=s.forward(x,a['budget']);r=s.adjoint(c,a['budget'])
    for result in (f,r):check_pair_actions(result,s)
    np.testing.assert_allclose(f.values,np.einsum('kpij,kij->p',t.conj(),x),atol=6e-12,rtol=4e-13)
    np.testing.assert_allclose(r.values,np.einsum('kpij,p->kij',t,c),atol=6e-12,rtol=4e-13)
    et=libint_reference(a)[1]+quadrature_long_range(a)[1]
    assert np.max(abs(t-et))<=f.receipt.coupling_error_bound+6e-12


@pytest.mark.parametrize('gamma',[False,True])
def test_every_column_adjoint_linearity_and_zero_input(gamma):
    a=pair_actions_case(gamma);s=manta.PeriodicGDFTiles(**a);t=assemble(s)
    shape=(s.plan.momenta,s.plan.n_orb,s.plan.n_orb);m=np.prod(shape);na=s.plan.n_aux
    f=np.column_stack([s.forward(x.reshape(shape).astype(complex),a['budget']).values for x in np.eye(m)])
    r=np.column_stack([s.adjoint(c.astype(complex),a['budget']).values.ravel() for c in np.eye(na)])
    np.testing.assert_allclose(f,np.moveaxis(t,1,0).reshape(na,-1).conj(),atol=5e-12,rtol=3e-13)
    np.testing.assert_allclose(r,f.conj().T,atol=5e-12,rtol=3e-13)
    x,c=inputs(s)
    for method,v in [('forward',x),('adjoint',c)]:
        base=getattr(s,method)(v,a['budget']);scaled=getattr(s,method)((2-3j)*v,a['budget']);zero=getattr(s,method)(0*v,a['budget'])
        np.testing.assert_allclose(scaled.values,(2-3j)*base.values,atol=1e-11,rtol=3e-13)
        np.testing.assert_array_equal(zero.values,0*base.values)
        assert zero.receipt.multipole_far_blocks==base.receipt.multipole_far_blocks


@pytest.mark.parametrize('gamma',[False,True])
def test_canonical_pair_frames_preserve_image_and_reciprocal_gauges(gamma):
    a=pair_actions_case(gamma);moved=pair_actions_case(gamma,shift=True)
    source=manta.PeriodicGDFTiles(**a);other=manta.PeriodicGDFTiles(**moved);x,c=inputs(source)
    cell=a['system'].lattice[:,0];shift=np.array([0.,1.]);aux=np.exp(1j*shift*(a['q']@cell))
    phase=np.array([np.exp(1j*shift*((k-a['q'])@cell))[:,None]*np.exp(-1j*shift*(k@cell))[None,:] for k in a['ket_kpoints']])
    np.testing.assert_allclose(other.forward(x*phase,a['budget']).values,source.forward(x,a['budget']).values/aux,atol=8e-12,rtol=3e-13)
    np.testing.assert_allclose(other.adjoint(c/aux,a['budget']).values,source.adjoint(c,a['budget']).values*phase,atol=8e-12,rtol=3e-13)
    reciprocal=a['system'].reciprocal_lattice()@np.array([1,-1,2]);relabel=dict(a,q=a['q']+reciprocal,ket_kpoints=a['ket_kpoints']+reciprocal)
    np.testing.assert_allclose(manta.PeriodicGDFTiles(**relabel).forward(x,a['budget']).values,source.forward(x,a['budget']).values,atol=8e-12,rtol=3e-13)


def test_concurrent_inputs_reuse_only_immutable_source_data():
    a=pair_actions_case();s=manta.PeriodicGDFTiles(**a);x,c=inputs(s);barrier=threading.Barrier(4)
    def run(i):
        barrier.wait();return s.forward((i+1)*x,a['budget']) if i%2==0 else s.adjoint((i+1)*c,a['budget'])
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(run,range(4)))
    for i,result in enumerate(results):
        expected=s.forward(x,a['budget']) if i%2==0 else s.adjoint(c,a['budget'])
        np.testing.assert_allclose(result.values,(i+1)*expected.values,atol=8e-12,rtol=3e-13)
    assert sum(r.receipt.multipole_moment_panels for r in results)==s.preparation.moment_panels


@pytest.mark.parametrize('rank',[0,1,6])
def test_fixed_metric_rank_and_weighted_energy_derivative(rank):
    a=pair_actions_case(True,fitted=True);full=builder(a);e=np.linalg.eigvalsh(full.metric)
    cutoff=1e-12 if rank==6 else (2*e[-1] if rank==0 else .5*(e[-2]+e[-1]))
    b=builder(a,block=6,absolute_cutoff=cutoff);d=densities();w=np.array([.27,.73]);result=b.build(d,w,make_k=False)
    s=manta.PeriodicGDFTiles(**a);t=assemble(s);expected=reference(s.metric().values,t,d,w,absolute=cutoff)
    np.testing.assert_allclose(result.J,expected[0],atol=8e-12,rtol=3e-13)
    assert abs(result.receipt.coulomb_energy-expected[2])<8e-12
    assert b.metric_info.rank==rank
    trial=d[::-1].copy();step=2e-5
    plus=b.build(d+step*trial,w,make_k=False);minus=b.build(d-step*trial,w,make_k=False)
    derivative=(plus.receipt.coulomb_energy-minus.receipt.coulomb_energy)/(2*step)
    assert derivative==pytest.approx(np.einsum('k,skij,kji->',w,trial,result.J).real,abs=2e-10)


@pytest.mark.parametrize('field,quantity',CAPS)
def test_admission_precedes_preparation_and_input_copy(field,quantity):
    a=pair_actions_case();s=manta.PeriodicGDFTiles(**a);p=s.plan_coupling(a['budget']);x,_=inputs(s)
    setattr(a['budget'],field,getattr(p,quantity)-1);x[:]=np.nan
    with pytest.raises((ValueError,RuntimeError),match='budget'):s.forward(x,a['budget'])
    assert s.preparation.pair_extents==s.preparation.moment_panels==0


def test_always_near_exact_limit_and_rejected_routes():
    a=pair_actions_case();a['options'].coupling_multipole_error=1e-100
    s=manta.PeriodicGDFTiles(**a);x,_=inputs(s);r=s.forward(x,a['budget'])
    assert r.receipt.multipole_far_blocks==r.receipt.multipole_pair_groups==0
    assert r.receipt.coupling_error_bound==0
    a['options'].coupling_multipole_pair_tree=False;a['options'].coupling_multipole_error=0
    np.testing.assert_allclose(r.values,manta.PeriodicGDFTiles(**a).forward(x,a['budget']).values,atol=5e-12,rtol=3e-13)
    for field,value in [('coupling_multipole_error',0.),('coupling_multipole_reuse',False)]:
        a=pair_actions_case();setattr(a['options'],field,value)
        with pytest.raises(ValueError,match='requires'):manta.PeriodicGDFTiles(**a)
    a=pair_actions_case();kernel=manta.PeriodicGDFKernelOptions();kernel.kernel='erf';kernel.omega=.3
    with pytest.raises(ValueError,match='real short-range'):manta.PeriodicGDFTiles(**a,physical_kernel=kernel)


@pytest.mark.parametrize('method,species,limit',[('rhf','Be',100),('uhf','H',100),('uhf','Li',100),('uhf','Li',3),('rks','Be',100),('uks','Li',100),('uks','Li',3)])
def test_public_hf_ks_keep_shared_fock_and_output_contracts(tmp_path,method,species,limit):
    if method.endswith('ks'):
        job,args,*_=hierarchy_ks_job(tmp_path,1 if method=='rks' else 2,limit)
    else:job,args,_=prepared_job(tmp_path,method,species,limit)
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],coupling_multipole_pair_tree=True)
    result=vq.run_periodic_job(**job);r=result.native.receipt.fitted_fock
    assert result.converged==(limit==100)
    assert result.integral_provenance['fitting']['controls']['coupling_multipole_pair_tree']
    assert r.coupling_passes==2 and r.coupling_multipole_far_blocks>0
    assert r.coupling_multipole_global_pairs>0
    from vibeqc.guess_read import _spin_density_k_from_qvf_bloch_payload
    from vibeqc.output import validate_qvf
    saved=_spin_density_k_from_qvf_bloch_payload(job['output'].with_suffix('.qvf'))
    density=result.native.density
    expected=np.stack([.5*density]*2) if method=='rhf' else np.repeat(.5*density,2,axis=0) if method=='rks' else density
    np.testing.assert_array_equal(saved,expected)
    assert validate_qvf(job['output'].with_suffix('.qvf'))['valid']


def test_public_control_is_typed_and_requires_qualified_preparation():
    for kwargs in [dict(coupling_multipole_pair_tree=1),dict(coupling_multipole_pair_tree=True),
                   dict(coupling_multipole_pair_tree=True,coupling_multipole_error=1,coupling_multipole_reuse=False)]:
        with pytest.raises((TypeError,ValueError)):PeriodicFittingOptions(.2,8.,**kwargs)


def test_gamma_completion_refuses_unqualified_discrete_support_and_near_gamma():
    a=pair_actions_case(True,shift=True);a['options'].auxiliary_cutoff_bohr=7.052659073002182
    s=manta.PeriodicGDFTiles(**a);x,_=inputs(s)
    with pytest.raises(ValueError,match='conjugate-closed finite capsule'):s.forward(x,a['budget'])
    a=pair_actions_case(True);a['q'][0]=1e-13;a['vectors']+=a['q']
    s=manta.PeriodicGDFTiles(**a);x,_=inputs(s)
    with pytest.raises(ValueError,match='reciprocal Gamma'):s.forward(x,a['budget'])


@pytest.mark.parametrize('gamma',[False,True])
def test_physical_erfc_retains_split_zero_and_exact_metric_ownership(gamma):
    from tests.test_manta_periodic_range_fitting import physical,reference as range_reference
    a=pair_actions_case(gamma);a['physical_kernel']=physical('erfc',.43)
    source=manta.PeriodicGDFTiles(**a);x,c=inputs(source);t=assemble(source)
    dense=manta.compute_periodic_gdf(**a);m,exact=range_reference(a)
    np.testing.assert_array_equal(t,dense.three_center)
    np.testing.assert_allclose(source.metric().values,m,atol=5e-12,rtol=3e-13)
    f=source.forward(x,a['budget']);rev=source.adjoint(c,a['budget'])
    np.testing.assert_allclose(f.values,np.einsum('kpij,kij->p',t.conj(),x),atol=5e-12,rtol=3e-13)
    np.testing.assert_allclose(rev.values,np.einsum('kpij,p->kij',t,c),atol=5e-12,rtol=3e-13)
    assert np.max(abs(t-exact))<=f.receipt.coupling_error_bound+5e-12
    assert source.receipt.physical_zero_included==gamma


@pytest.mark.parametrize('route',['source','dense','short-range','ri'])
@pytest.mark.parametrize('field,quantity',CAPS)
def test_complete_parent_admission_at_exact_and_one_below_caps(route,field,quantity):
    from tests.test_manta_periodic_gdf import plan as dense_plan
    from tests.test_manta_periodic_fitting import plan as short_plan
    a=pair_actions_case(True);a['budget'].other_live_bytes=12345
    if route=='source':
        plan=lambda:source_plan(a)
        run=lambda:manta.PeriodicGDFTiles(**a).evaluate(0,0,0)
    elif route=='dense':
        plan=lambda:dense_plan(a);run=lambda:manta.compute_periodic_gdf(**a)
    elif route=='short-range':
        a.pop('vectors');plan=lambda:short_plan(a);run=lambda:manta.compute_periodic_gdf_short_range(**a)
    else:
        source=manta.PeriodicGDFTiles(**a);o=manta.RIOptions()
        plan=lambda:manta.plan_periodic_ri_q(source,2,o,a['budget'])
        run=lambda:manta.PeriodicRIQBuilder(source,2,o,a['budget']).build(densities(),np.array([.27,.73]),make_k=False)
    p=plan()
    for cap,value in CAPS:setattr(a['budget'],cap,getattr(p,value))
    run()
    setattr(a['budget'],field,getattr(p,quantity)-1)
    for key in ('q','ket_kpoints','vectors'):
        if key in a:a[key][:]=np.nan
    with pytest.raises((ValueError,RuntimeError),match='budget'):plan()
    with pytest.raises((ValueError,RuntimeError),match='budget'):run()


def test_uncached_partition_workspace_is_admitted_at_high_order():
    a=pair_actions_case(order=4);a['options'].coupling_multipole_partition_bytes=0
    s=manta.PeriodicGDFTiles(**a);gp=manta.plan_periodic_gdf_pair_tree(s,a['budget'])
    h=35
    assert s.plan.pair_tree_action_bytes>=4096+gp.node_capacity*(256+8*h*h)
    assert s.plan.partition_cache_bytes==0
    a['budget'].other_live_bytes+=123456
    assert source_plan(a).modeled_peak_bytes==s.plan.modeled_peak_bytes+123456


def test_complete_action_succeeds_at_all_reported_limits():
    a=pair_actions_case(True);s=manta.PeriodicGDFTiles(**a);x,c=inputs(s)
    p=s.plan_coupling(a['budget'])
    for cap,value in CAPS:setattr(a['budget'],cap,getattr(p,value))
    f=s.forward(x,a['budget']);rev=s.adjoint(c,a['budget'])
    assert abs(np.vdot(c,f.values)-np.vdot(rev.values,x))<5e-12



@pytest.mark.parametrize('gamma',[False,True])
@pytest.mark.parametrize('order',[0,2,4])
@pytest.mark.parametrize('sparse',[False,True])
def test_block_errors_bound_complex_actions_in_both_norms(gamma,order,sparse):
    a=pair_actions_case(gamma,order);s=manta.PeriodicGDFTiles(**a);x,c=inputs(s)
    if sparse:
        x[:,:1,:]=0.;c[:1]=0.
    exact=independent(a)[1]
    for method,value,reference in [
            ('forward',x,np.einsum('kpij,kij->p',exact.conj(),x)),
            ('adjoint',c,np.einsum('kpij,p->kij',exact,c))]:
        result=getattr(s,method)(value,a['budget']);r=result.receipt
        error=result.values-reference
        assert np.max(abs(error))<=r.approximation_bound+5e-12
        assert np.linalg.norm(error.ravel())<=r.approximation_l2_bound+1e-11
        assert 0<r.approximation_bound<=r.coupling_error_bound*r.input_l1_bound*(1+1e-12)
        if not sparse:assert r.approximation_bound<r.coupling_error_bound*r.input_l1_bound
        assert 0<r.approximation_l2_bound<=np.sqrt(result.values.size)*r.approximation_bound*(1+1e-12)
        phase=getattr(s,method)(1j*value,a['budget']).receipt
        assert phase.approximation_bound==r.approximation_bound
        assert phase.approximation_l2_bound==r.approximation_l2_bound
        scaled=getattr(s,method)(3*value,a['budget']).receipt
        assert scaled.approximation_bound==pytest.approx(3*r.approximation_bound,rel=2e-12)
        assert scaled.approximation_l2_bound==pytest.approx(3*r.approximation_l2_bound,rel=2e-12)
        zero=getattr(s,method)(0*value,a['budget']).receipt
        assert zero.approximation_bound==zero.approximation_l2_bound==0.
        assert zero.coupling_error_bound==r.coupling_error_bound
        assert zero.maximum_coupling==r.maximum_coupling
        assert zero.multipole_far_blocks==r.multipole_far_blocks


@pytest.mark.parametrize('gamma',[False,True])
def test_duplicate_auxiliary_outputs_do_not_inflate_component_envelopes(gamma):
    a=pair_actions_case(gamma);s=manta.PeriodicGDFTiles(**a);x,c=inputs(s)
    f=s.forward(x,a['budget']);rev=s.adjoint(c,a['budget'])
    shells=a['auxiliary'].shells()+a['auxiliary'].shells()
    a['auxiliary']=vq.BasisSet(a['system'].unit_cell_molecule(),shells,'duplicate-auxiliary',True)
    doubled=manta.PeriodicGDFTiles(**a)
    df=doubled.forward(x,a['budget']);dr=doubled.adjoint(np.tile(c/2,2),a['budget'])
    np.testing.assert_array_equal(df.values,np.tile(f.values,2))
    np.testing.assert_allclose(dr.values,rev.values,atol=5e-12,rtol=3e-13)
    for first,second in [(f,df),(rev,dr)]:
        assert second.receipt.coupling_error_bound==first.receipt.coupling_error_bound
        assert second.receipt.maximum_coupling==first.receipt.maximum_coupling
        assert second.receipt.approximation_bound==pytest.approx(first.receipt.approximation_bound,rel=2e-12)
    assert df.receipt.approximation_l2_bound==pytest.approx(np.sqrt(2)*f.receipt.approximation_l2_bound,rel=2e-12)
    assert dr.receipt.approximation_l2_bound==pytest.approx(rev.receipt.approximation_l2_bound,rel=2e-12)


@pytest.mark.parametrize('rank',[0,1,6])
def test_ri_propagates_block_l2_error_with_the_same_retained_metric(rank):
    a=pair_actions_case(True,fitted=True);full=builder(a);e=np.linalg.eigvalsh(full.metric)
    cutoff=1e-12 if rank==6 else (2*e[-1] if rank==0 else .5*(e[-2]+e[-1]))
    b=builder(a,block=6,absolute_cutoff=cutoff);d=densities();w=np.array([.27,.73])
    result=b.build(d,w,make_k=False)
    source=manta.PeriodicGDFTiles(**a);f=source.forward(w[:,None,None]*d.sum(axis=0),a['budget'])
    assert result.receipt.forward_error_l2_bound==f.receipt.approximation_l2_bound
    a['options'].coupling_multipole_pair_tree=False;a['options'].coupling_multipole_error=0.
    exact=builder(a,block=6,absolute_cutoff=cutoff);reference=exact.build(d,w,make_k=False)
    np.testing.assert_array_equal(b.factor,exact.factor)
    assert np.max(abs(result.J-reference.J))<=result.receipt.approximation_j_bound+8e-12
    assert abs(result.receipt.coulomb_energy-reference.receipt.coulomb_energy)<=result.receipt.approximation_energy_bound+8e-12



@pytest.mark.parametrize('gamma',[False,True])
def test_block_euclidean_bound_does_not_square_large_finite_errors(gamma):
    a=pair_actions_case(gamma);s=manta.PeriodicGDFTiles(**a);x,c=inputs(s)
    scale=1e180
    for method,value in [('forward',x),('adjoint',c)]:
        base=getattr(s,method)(value,a['budget']);large=getattr(s,method)(scale*value,a['budget'])
        np.testing.assert_allclose(large.values/scale,base.values,atol=5e-12,rtol=3e-13)
        assert np.isfinite(large.receipt.approximation_l2_bound)
        assert large.receipt.approximation_l2_bound/scale==pytest.approx(base.receipt.approximation_l2_bound,rel=2e-12)



def accuracy_job(directory,order=2,error=1e-8):
    job,args,k=prepared_job(directory,'rhf','Be',100)
    options=args['options'].fitting
    options.coupling_multipole_error=error;options.coupling_multipole_order=order
    options.coupling_multipole_pair_tree=error>0
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],coupling_multipole_error=error,
        coupling_multipole_order=order,coupling_multipole_pair_tree=error>0)
    return job,args,k


@pytest.mark.parametrize('order',[2,4])
def test_tight_coupling_tolerance_retains_far_terms_in_complete_scf(tmp_path,order):
    from tests.test_manta_periodic_ri_multipole import multipole_hf_reference
    from tests.test_manta_periodic_ri_domains import explicit
    from tests.test_manta_periodic_rhf import _nuclear_reference
    job,args,k=accuracy_job(tmp_path,order)
    a=explicit(args,manta.build_periodic_ri_rhf_domains(**args))
    _,_,build,exact,_=multipole_hf_reference(a,k,'rhf')
    result=vq.run_periodic_job(**job);n=result.native;r=n.receipt.fitted_fock
    f,e=build(n.density);ef,ee=exact(n.density)
    assert result.converged and r.coupling_multipole_far_blocks>0
    assert 0<r.coupling_error_bound<=1e-8
    np.testing.assert_allclose(n.fock,f,atol=8e-11,rtol=0)
    assert abs(result.energy-sum(_nuclear_reference(a)[0])-sum(e))<8e-11
    assert np.max(abs(n.fock-ef))<=r.approximation_j_bound+8e-11
    assert abs(sum(e)-sum(ee))<=r.approximation_energy_bound+8e-11


@pytest.mark.parametrize('order',[2,4,6])
@pytest.mark.parametrize('gamma',[False,True])
def test_compact_high_order_partitions_reuse_actual_storage(order,gamma):
    a=pair_actions_case(gamma,order);a['options'].coupling_multipole_error=1e-8
    source=manta.PeriodicGDFTiles(**a);x,c=inputs(source)
    first=source.forward(x,a['budget']);info=source.partitions
    assert info.entries>0 and info.evictions==0
    assert info.modeled_live_bytes<=info.capacity_bytes
    a['options'].coupling_multipole_partition_bytes=0;stream=manta.PeriodicGDFTiles(**a)
    assert source.three_center_source_identity==stream.three_center_source_identity
    for method,arg in [('forward',x),('adjoint',c),('forward',(2-3j)*x)]:
        result=getattr(source,method)(arg,a['budget'])
        reference=getattr(stream,method)(arg,a['budget'])
        np.testing.assert_array_equal(result.values,reference.values)
        assert result.receipt.snapshot_identity==reference.receipt.snapshot_identity
        assert result.receipt.multipole_partition_builds==result.receipt.multipole_partition_bypasses==0
        assert result.receipt.multipole_partition_reuses>0
        for name in ('coupling_error_bound','approximation_bound','approximation_l2_bound','maximum_coupling'):
            assert getattr(result.receipt,name)==getattr(reference.receipt,name)
    assert first.receipt.multipole_far_blocks>0


@pytest.mark.parametrize('capacity',[1,128*1024,1024*1024])
def test_compact_partition_bypass_and_eviction_do_not_rebuild_twice(capacity):
    a=pair_actions_case(False,6);a['options'].coupling_multipole_partition_bytes=capacity
    source=manta.PeriodicGDFTiles(**a);x,c=inputs(source)
    a['options'].coupling_multipole_partition_bytes=0;stream=manta.PeriodicGDFTiles(**a)
    for method,arg in [('forward',x),('adjoint',c),('forward',x)]:
        result=getattr(source,method)(arg,a['budget']);reference=getattr(stream,method)(arg,a['budget'])
        np.testing.assert_array_equal(result.values,reference.values)
        assert result.receipt.multipole_partition_builds+result.receipt.multipole_partition_reuses==reference.receipt.multipole_partition_builds
        assert source.partitions.modeled_live_bytes<=source.partitions.capacity_bytes
    if capacity==1:assert source.partitions.entries==source.partitions.evictions==0
    else:assert source.partitions.entries>0 and source.partitions.evictions>0


def test_rank_six_ri_retains_order_six_partition_working_set():
    a=pair_actions_case(True,6,fitted=True);a['options'].coupling_multipole_error=1e-8
    b=builder(a);d=densities();w=np.array([.27,.73]);cold=b.build(d,w,make_k=False)
    warm=b.build(d,w,make_k=False)
    np.testing.assert_array_equal(cold.J,warm.J)
    assert warm.receipt.coupling_multipole_partition_reuses>0
    assert warm.receipt.coupling_multipole_partition_builds==warm.receipt.coupling_multipole_partition_bypasses==0
    assert warm.receipt.coupling_multipole_trials==0
    a['options'].coupling_multipole_partition_bytes=0
    reference=builder(a).build(d,w,make_k=False)
    np.testing.assert_array_equal(warm.J,reference.J)
    for name in ('coulomb_energy','source_identity','projector_identity','approximation_j_bound','approximation_energy_bound'):
        assert getattr(warm.receipt,name)==getattr(reference.receipt,name)


def test_complete_order_six_rhf_uses_retained_partitions(tmp_path):
    job,_,_=accuracy_job(tmp_path,6)
    result=vq.run_periodic_job(**job);assert result.converged
    r=result.native.receipt.fitted_fock
    assert r.coupling_multipole_far_blocks>0 and r.coupling_multipole_partition_reuses>0
    assert r.coupling_multipole_partition_builds==r.coupling_multipole_partition_bypasses==0


@pytest.mark.parametrize('order',[8,12])
def test_compact_layout_at_high_orders_with_bounded_finite_support(order):
    a=pair_actions_case(False,order);a['options'].pair_cutoff_bohr=.5
    # The retained dense-antichain admission remains conservative at p=12;
    # this tiny actual pair domain needs an explicit larger modeled budget.
    a['budget'].memory_bytes=8*1024**3
    source=manta.PeriodicGDFTiles(**a);x,c=inputs(source)
    f=source.forward(x,a['budget']);adj=source.adjoint(c,a['budget'])
    assert f.receipt.multipole_far_blocks>0
    assert adj.receipt.multipole_partition_builds==0 and adj.receipt.multipole_partition_reuses>0
    assert source.partitions.modeled_live_bytes<=source.partitions.capacity_bytes
    a['options'].coupling_multipole_partition_bytes=0;stream=manta.PeriodicGDFTiles(**a)
    np.testing.assert_array_equal(f.values,stream.forward(x,a['budget']).values)
    np.testing.assert_array_equal(adj.values,stream.adjoint(c,a['budget']).values)
    assert abs(np.vdot(c,f.values)-np.vdot(adj.values,x))<1e-11
    _,exact=independent(a)
    assert np.max(abs(f.values-np.einsum('kpij,kij->p',exact.conj(),x)))<=f.receipt.approximation_bound+1e-11
    assert np.max(abs(adj.values-np.einsum('kpij,p->kij',exact,c)))<=adj.receipt.approximation_bound+1e-11



@pytest.mark.parametrize('gamma',[False,True])
@pytest.mark.parametrize('route',['screen','materialized','compressed','global','pair_tree'])
def test_component_envelopes_cover_independent_maps(gamma,route):
    from tests.test_manta_periodic_gdf import case
    a=case(gamma) if route=='screen' else multipole_case(gamma,order=2)
    o=a['options']
    if route=='screen':o.coupling_screen_error=.1
    else:
        o.coupling_multipole_hierarchy=route in ('compressed','global','pair_tree')
        o.coupling_multipole_compress=route in ('compressed','global','pair_tree')
        o.coupling_multipole_global_actions=route in ('global','pair_tree')
        o.coupling_multipole_pair_tree=route=='pair_tree'
    s=manta.PeriodicGDFTiles(**a);x,c=inputs(s)
    exact=independent(a)[1];column_max=np.max(abs(exact),axis=(0,2,3))
    for method,value,reference in [
            ('forward',x,np.einsum('kpij,kij->p',exact.conj(),x)),
            ('adjoint',c,np.einsum('kpij,p->kij',exact,c))]:
        r=getattr(s,method)(value,a['budget'])
        e=r.approximation_bounds;maximum=r.reference_magnitude_bounds
        assert e.shape==r.values.shape and maximum.shape==(s.plan.n_aux,)
        assert e.flags.owndata and maximum.flags.owndata
        assert not e.flags.writeable and not maximum.flags.writeable
        assert np.all(abs(r.values-reference)<=e+5e-12)
        assert np.all(column_max<=maximum+5e-12)
        assert np.all(e>=0.) and np.all(e<=r.receipt.approximation_bound)
        assert np.all(maximum<=
            (r.receipt.maximum_coupling+r.receipt.coupling_error_bound)*(1+1e-12))
        zero=getattr(s,method)(0*value,a['budget'])
        np.testing.assert_array_equal(zero.approximation_bounds,np.zeros_like(e))
        np.testing.assert_array_equal(zero.reference_magnitude_bounds,maximum)
        rotated=getattr(s,method)(1j*value,a['budget'])
        np.testing.assert_array_equal(rotated.approximation_bounds,e)
        np.testing.assert_array_equal(rotated.reference_magnitude_bounds,maximum)
        large=getattr(s,method)(1e180*value,a['budget'])
        np.testing.assert_allclose(large.approximation_bounds/1e180,e,rtol=2e-12,atol=0.)
        # An exported copy cannot change the retained producer or result.
        e.setflags(write=True);e[:]=0.
        np.testing.assert_array_equal(r.approximation_bounds,rotated.approximation_bounds)
    o.coupling_screen_error=o.coupling_multipole_error=0.
    o.coupling_multipole_pair_tree=False
    exact_source=manta.PeriodicGDFTiles(**a)
    for method,value in [('forward',x),('adjoint',c)]:
        r=getattr(exact_source,method)(value,a['budget'])
        np.testing.assert_array_equal(r.approximation_bounds,np.zeros_like(r.values.real))
        assert r.reference_magnitude_bounds.size==0 and not r.receipt.approximation_evaluated


@pytest.mark.parametrize('rank',[0,1,6])
def test_component_ri_bounds_cover_adversarial_forward_errors(rank):
    a=pair_actions_case(True,fitted=True);full=builder(a);eigenvalues=np.linalg.eigvalsh(full.metric)
    cutoff=1e-12 if rank==6 else (2*eigenvalues[-1] if rank==0 else .5*(eigenvalues[-2]+eigenvalues[-1]))
    b=builder(a,block=6,absolute_cutoff=cutoff);source=manta.PeriodicGDFTiles(**a)
    d=densities();w=np.array([.27,.73]);result=b.build(d,w,make_k=False)
    f=source.forward(w[:,None,None]*d.sum(axis=0),a['budget'])
    factor=b.factor;projected=factor.T@f.values;coefficients=factor.conj()@projected
    adjoint=source.adjoint(coefficients,a['budget'])
    exact=independent(a)[1]
    components=f.approximation_bounds;magnitudes=f.reference_magnitude_bounds
    u=abs(factor).T@components;g=abs(factor)@u
    new_energy=min(abs(coefficients)@components,abs(projected)@u)+.5*(u@u)
    new_j=adjoint.receipt.approximation_bound+magnitudes@g
    old_energy=np.linalg.norm(coefficients)*f.receipt.approximation_l2_bound+.5*result.receipt.metric_inverse_norm_bound*f.receipt.approximation_l2_bound**2
    old_j=adjoint.receipt.approximation_bound+np.sqrt(len(components))*(f.receipt.maximum_coupling+f.receipt.coupling_error_bound)*result.receipt.metric_inverse_norm_bound*f.receipt.approximation_l2_bound
    assert result.receipt.approximation_energy_bound==pytest.approx(min(new_energy,old_energy),rel=2e-11,abs=0.)
    assert result.receipt.approximation_j_bound==pytest.approx(min(new_j,old_j),rel=2e-11,abs=0.)
    if rank==6:
        assert result.receipt.approximation_energy_bound<.9*old_energy
        assert result.receipt.approximation_j_bound<.9*old_j
    rng=np.random.default_rng(811)
    for _ in range(16):
        delta=components*np.exp(1j*rng.uniform(-np.pi,np.pi,len(components)))
        if np.linalg.norm(delta)>f.receipt.approximation_l2_bound:
            delta*=f.receipt.approximation_l2_bound/np.linalg.norm(delta)
        changed_projected=factor.T@(f.values-delta)
        changed_j=np.einsum('kpij,p->kij',exact,factor.conj()@changed_projected)
        changed_energy=.5*np.linalg.norm(changed_projected)**2
        assert np.max(abs(result.J-changed_j))<=result.receipt.approximation_j_bound+8e-12
        assert abs(result.receipt.coulomb_energy-changed_energy)<=result.receipt.approximation_energy_bound+8e-12
