"""Complete weighted periodic RI Fock, independent sources and variational checks."""
import numpy as np
import pytest

from vibeqc import manta
from tests.test_manta_periodic_gdf import case as source_case,independent,legacy
from tests.test_manta_periodic_ri_q import densities,reference
from tests.test_manta_periodic_fitting import CAPS


def case(nk=2,shift=False,block=1,cutoff=1e-12):
    a=source_case(True,shift);k=np.array([[.23,.09,-.13],[-.15,.2,.31],[.03,-.23,.16],[.32,-.11,.27]])[:nk].copy()
    w=np.arange(1,nk+1,dtype=float);w/=w.sum();pair=[];offsets=[0]
    for i in range(nk):
        for j in range(nk):
            v=a['vectors']+k[j]-k[i] if i!=j else np.empty((0,3))
            pair.extend(v);offsets.append(len(pair))
    ri=manta.RIOptions();ri.mode_block=block;ri.absolute_cutoff=cutoff
    return dict(orbital=a['orbital'],auxiliary=a['auxiliary'],system=a['system'],momenta=k,weights=w,
        gamma_vectors=a['vectors'],pair_vectors=np.asarray(pair,dtype=float).reshape(-1,3),
        transfer_offsets=np.asarray(offsets,np.int64),options=a['options'],ri_options=ri,budget=a['budget'])


def plan(a):
    return manta.plan_periodic_ri_fock(a['orbital'],a['auxiliary'],a['system'],len(a['gamma_vectors']),
        int(max(np.diff(a['transfer_offsets']))),len(a['momenta']),a['options'],a['ri_options'],a['budget'])


def oracle(a,h,d,scale=1.,source=independent):
    k=a['momenta'];w=a['weights'];ri=a['ri_options'];nk=len(k)
    args=dict(orbital=a['orbital'],auxiliary=a['auxiliary'],system=a['system'],q=np.zeros(3),
        ket_kpoints=k,vectors=a['gamma_vectors'],options=a['options'],budget=a['budget'])
    m,t=source(args)[:2];j,K,ej,_,_=reference(m,t,d,w,ri.absolute_cutoff,ri.relative_cutoff)
    for i in range(nk):
        for q in range(nk):
            if i==q:continue
            lo,hi=a['transfer_offsets'][i*nk+q:i*nk+q+2]
            args.update(q=k[q]-k[i],ket_kpoints=k[q:q+1],vectors=a['pair_vectors'][lo:hi])
            m,t=source(args)[:2];K[:,i]+=reference(m,t,d[:,q:q+1],w[q:q+1],ri.absolute_cutoff,ri.relative_cutoff)[1][:,0]
    e1=np.einsum('k,skij,kji->',w,d,h).real;ek=-.5*scale*np.einsum('k,skij,skji->',w,d,K).real
    return j,K,h[None]+j[None]-scale*K,np.array([e1,ej,ek,e1+ej+ek])


@pytest.mark.parametrize('nk',[1,2,4])
@pytest.mark.parametrize('scale',[-.3,0.,1.])
def test_independent_whole_mesh_fock_and_every_energy_component(nk,scale):
    a=case(nk);b=manta.PeriodicRIFockBuilder(**a);d=densities(nk);h=densities(nk,spins=1,seed=311)[0]
    r=b.build(h,d,scale)
    for source in (independent,legacy):
        j,k,f,e=oracle(a,h,d,scale,source)
        np.testing.assert_allclose(r.J,j,atol=4e-12,rtol=2e-12)
        np.testing.assert_allclose(r.K,k,atol=4e-12,rtol=2e-12)
        np.testing.assert_allclose(r.values,f,atol=4e-12,rtol=2e-12)
        np.testing.assert_allclose([r.receipt.one_electron_energy,r.receipt.coulomb_energy,
            r.receipt.exchange_energy,r.receipt.electronic_energy],e,atol=5e-12,rtol=2e-12)
    assert b.plan.channels==1+nk*(nk-1)==len(r.components)==len(b.info.metrics)
    assert r.receipt.shell_calls<=b.plan.build_shell_calls
    assert r.receipt.primitive_products<=b.plan.build_primitive_products
    assert b.info.metric_shell_calls<=b.plan.metric_shell_calls
    assert b.info.metric_primitive_products<=b.plan.metric_primitive_products
    assert max(b.info.reverse_metric_residual,b.info.reverse_projector_residual,b.info.reverse_inverse_residual)<2e-12


@pytest.mark.parametrize('cutoff',[1e-12,10.,1e9])
def test_energy_derivative_weighted_reciprocity_and_fixed_projectors(cutoff):
    a=case(4,cutoff=cutoff);b=manta.PeriodicRIFockBuilder(**a);d=densities(4);delta=densities(4,seed=312);h=densities(4,spins=1,seed=132)[0];step=1e-5
    r=b.build(h,d,.72);rp=b.build(h,d+step*delta,.72);rm=b.build(h,d-step*delta,.72)
    derivative=(rp.receipt.electronic_energy-rm.receipt.electronic_energy)/(2*step)
    expected=np.einsum('k,skij,skji->',a['weights'],delta,r.values).real
    assert derivative==pytest.approx(expected,abs=3e-8)
    response=b.build(np.zeros_like(h),delta,.72)
    lhs=np.einsum('k,skij,skji->',a['weights'],delta,r.values-h).real
    rhs=np.einsum('k,skij,skji->',a['weights'],d,response.values).real
    assert lhs==pytest.approx(rhs,abs=5e-12)
    for c,p,m in zip(r.components,rp.components,rm.components):
        assert c.source_identity==p.source_identity==m.source_identity
        assert c.projector_identity==p.projector_identity==m.projector_identity
    assert r.receipt.source_identity==rp.receipt.source_identity!=b.build(h,d,1.).receipt.source_identity
    assert r.receipt.density_identity==b.build(2*h,d,.72).receipt.density_identity
    assert r.receipt.density_identity!=rp.receipt.density_identity


@pytest.mark.parametrize('cutoff',[1e-12,10.])
def test_image_gauge_preserves_whole_mesh_operator_energy_and_projector_rank(cutoff):
    from vibeqc.symmetry_shared import BlockSpaceAction, Budget, SpaceIdentity
    a=case(2,cutoff=cutoff);other=case(2,shift=True,cutoff=cutoff);d=densities();h=densities(spins=1,seed=612)[0]
    L=a['system'].lattice[:,0];phase=np.exp(1j*np.stack([np.zeros(2),a['momenta']@L],axis=1));factor=phase[:,:,None]*phase.conj()[:,None,:]
    b=manta.PeriodicRIFockBuilder(**a);shifted=manta.PeriodicRIFockBuilder(**other)
    r=b.build(h,d);s=shifted.build(h*factor,d*factor)
    for name in ('J','K','values'):np.testing.assert_allclose(getattr(s,name),getattr(r,name)*factor,atol=4e-12,rtol=0)
    # Use the shared density-push/operator-pull contract as well as the
    # independent diagonal phase formula. A gauge audit grants no reduction.
    for ik in range(2):
        source=SpaceIdentity('finite-ri-cell',b.info.frame_identity,'AO',f'input-k{ik}')
        target=SpaceIdentity('finite-ri-cell',shifted.info.frame_identity,'AO',f'image-k{ik}')
        action=BlockSpaceAction(source,target,np.array([0,1,2],np.int64),np.array([0,1],np.int64),
            np.ascontiguousarray(phase[ik]),antiunitary=False,budget=Budget(2**20,10**6))
        for spin in range(2):
            np.testing.assert_allclose(action.push_density(d[spin,ik]),d[spin,ik]*factor[ik],atol=2e-14,rtol=0)
            np.testing.assert_allclose(action.pull_operator(s.values[spin,ik]),r.values[spin,ik],atol=4e-12,rtol=0)
    assert s.receipt.electronic_energy==pytest.approx(r.receipt.electronic_energy,abs=4e-12)
    assert [x.rank for x in b.info.metrics]==[x.rank for x in shifted.info.metrics]
    assert b.info.frame_identity!=shifted.info.frame_identity


def test_owned_source_snapshots_outputs_and_mode_block_invariance():
    a=case();b=manta.PeriodicRIFockBuilder(**a);h=densities(spins=1)[0];d=densities();r=b.build(h,d)
    two=manta.PeriodicRIFockBuilder(**case(block=2));s=two.build(h,d)
    assert s.receipt.source_identity==r.receipt.source_identity
    np.testing.assert_allclose(s.values,r.values,atol=1e-14,rtol=0)
    for name in ('momenta','weights','gamma_vectors','pair_vectors'):a[name][:]=np.nan
    a['system'].lattice=np.eye(3)*99.;a['options'].omega=0.;a['ri_options'].absolute_cutoff=1e9
    for value in (r.J,r.K,r.values):value[:]=np.nan
    np.testing.assert_array_equal(b.build(h,d).values,r.values)
    b.info.metrics.clear();r.components.clear()
    assert len(b.info.metrics)==len(r.components)==3
    np.testing.assert_array_equal(b.build(np.asfortranarray(h),np.asfortranarray(d)).values,r.values)


def test_reordering_the_explicit_mesh_transports_targets_sources_and_weights():
    a=case(4);a['weights']=np.array([.2,0.,.3,.5]);b=manta.PeriodicRIFockBuilder(**a)
    h=densities(4,spins=1)[0];d=densities(4);r=b.build(h,d)
    order=np.array([2,0,3,1]);pieces=[];offsets=[0]
    for i in order:
        for j in order:
            lo,hi=a['transfer_offsets'][i*4+j:i*4+j+2]
            pieces.append(a['pair_vectors'][lo:hi]);offsets.append(offsets[-1]+hi-lo)
    permuted=dict(a,momenta=a['momenta'][order],weights=a['weights'][order],
        pair_vectors=np.concatenate(pieces),transfer_offsets=np.asarray(offsets,np.int64))
    other=manta.PeriodicRIFockBuilder(**permuted);s=other.build(h[order],np.ascontiguousarray(d[:,order]))
    np.testing.assert_allclose(s.J,r.J[order],atol=3e-12,rtol=0)
    for name in ('K','values'):
        np.testing.assert_allclose(getattr(s,name),getattr(r,name)[:,order],atol=3e-12,rtol=0)
    assert s.receipt.electronic_energy==pytest.approx(r.receipt.electronic_energy,abs=4e-12)
    assert s.receipt.frame_identity!=r.receipt.frame_identity
    assert s.receipt.weight_identity!=r.receipt.weight_identity


@pytest.mark.parametrize('field,quantity',CAPS)
def test_joint_all_channel_admission_before_numerical_copies(field,quantity):
    a=case();p=plan(a)
    for cap,value in CAPS:setattr(a['budget'],cap,getattr(p,value))
    b=manta.PeriodicRIFockBuilder(**a);b.build(densities(spins=1)[0],densities())
    setattr(a['budget'],field,getattr(p,quantity)-1)
    for name in ('momenta','weights','gamma_vectors','pair_vectors'):a[name][:]=np.nan
    with pytest.raises((ValueError,RuntimeError),match='budget'):plan(a)
    with pytest.raises((ValueError,RuntimeError),match='budget'):manta.PeriodicRIFockBuilder(**a)


@pytest.mark.parametrize('fault',['gamma','reverse','duplicate_g','duplicate_k','diagonal','empty','weights','nonfinite','domain','metric'])
def test_source_refuses_inconsistent_reciprocal_or_mesh_inputs(fault):
    a=case()
    if fault=='gamma':a['gamma_vectors']=a['gamma_vectors'][1:].copy()
    elif fault=='reverse':a['pair_vectors'][0]+=a['system'].reciprocal_lattice()[:,0]
    elif fault=='duplicate_g':a['gamma_vectors'][0]=a['gamma_vectors'][1]
    elif fault=='duplicate_k':a['momenta'][1]=a['momenta'][0]
    elif fault=='diagonal':a['transfer_offsets'][1]=1
    elif fault=='empty':a['transfer_offsets'][2]=0
    elif fault=='weights':a['weights'][:]=.2
    elif fault=='nonfinite':a['pair_vectors'][0,0]=np.nan
    elif fault=='domain':a['pair_vectors'][0,0]+=.01
    elif fault=='metric':a['options'].compute_metric=False
    with pytest.raises((ValueError,RuntimeError)):manta.PeriodicRIFockBuilder(**a)
