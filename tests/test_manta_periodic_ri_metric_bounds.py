"""Nested finite metric-factor errors, separately from fitting/eigensolver errors."""
import dataclasses
from decimal import Decimal, localcontext
import numpy as np
import pytest
import vibeqc as vq
from vibeqc import manta
from tests.test_manta_periodic_ri_pair_tree import pair_actions_case, automatic_accuracy
from tests.test_manta_periodic_ri_q import densities, reference
from tests.test_manta_periodic_gdf import independent
from tests.test_manta_periodic_fitting import CAPS


def metric_case(rank=2,component=True,reference_rank=6):
    a=pair_actions_case(True,fitted=True);source=manta.PeriodicGDFTiles(**a)
    values=np.linalg.eigvalsh(source.metric().values);assert len(values)==6 and values[0]>1e-12
    ri=manta.RIOptions();ri.relative_cutoff=0.
    def cutoff(rank):
        return 1e-12 if rank==6 else (2*values[-1] if rank==0 else .5*(values[-rank-1]+values[-rank]))
    ri.absolute_cutoff=cutoff(rank)
    accuracy=automatic_accuracy(1e12,1e12)
    accuracy.metric_reference_cutoff=cutoff(reference_rank)
    if component:
        limits=np.ones((source.plan.momenta,source.plan.n_orb,source.plan.n_orb))
        limits[:,0,0]=.02;limits[:,1,1]=4.
        accuracy.density_entry_limits=limits.ravel().tolist()
    return a,source,ri,accuracy


@pytest.mark.parametrize('rank',[0,2,6])
@pytest.mark.parametrize('component',[False,True])
def test_nested_metric_modes_cover_independent_finer_ri_states(rank,component):
    a,source,ri,accuracy=metric_case(rank,component)
    b=manta.PeriodicRIQBuilder(source,2,ri,a['budget'],accuracy);c=b.metric_info.accuracy
    assert c.qualified and c.metric_truncation_certified
    assert c.metric_reference_rank==6 and c.metric_discarded_rank==6-rank
    h=b.metric_discarded_factor
    assert h.shape==(6,6-rank) and len(c.metric_reference_identity)==64
    values,u=np.linalg.eigh(b.metric)
    low=u[:,values<=ri.absolute_cutoff]/np.sqrt(values[values<=ri.absolute_cutoff])
    np.testing.assert_allclose(h@h.conj().T,low@low.conj().T,atol=3e-12,rtol=3e-12)
    assert c.j_bound>=c.metric_j_bound+c.coupling_j_bound
    assert c.energy_bound>=c.metric_energy_bound+c.coupling_energy_bound
    assert (c.metric_j_bound==c.metric_energy_bound==0.)==(rank==6)
    # Independent high-precision positive contraction of the exported proof
    # inputs checks the enclosure arithmetic, including the energy alternative.
    a['options'].coupling_multipole_error=c.selected_coupling_error
    selected=manta.PeriodicGDFTiles(**a)
    shape=(source.plan.momenta,source.plan.n_orb,source.plan.n_orb)
    envelope=np.array(accuracy.density_entry_limits).reshape(shape) if component else np.zeros(shape)
    probe=selected.forward(envelope.astype(complex),a['budget'])
    with localcontext() as ctx:
        ctx.prec=90;D=lambda x:Decimal.from_float(float(x));L=D(accuracy.density_l1_limit)
        C=[D(v) for v in probe.reference_magnitude_bounds]
        R=[min(C[i]*L,D(probe.reference_action_bounds[i])) if component else C[i]*L for i in range(6)]
        j=Decimal(0);e=Decimal(0)
        for mode in range(h.shape[1]):
            H=[(D(h[i,mode].real)**2+D(h[i,mode].imag)**2).sqrt() for i in range(6)]
            z=sum(x*y for x,y in zip(H,R));column=sum(x*y for x,y in zip(H,C))
            j+=column*z;e+=z*z/2
        assert D(c.metric_j_bound)>=j
        assert D(c.metric_energy_bound)>=min(e,L*j/2)
    exact_m,exact_t=independent(a);w=np.array([.27,.73])
    # Arbitrary Hermitian indefinite spin densities exercise the entire
    # declared family rather than a single occupied SCF state.
    accuracy.metric_reference_cutoff=ri.absolute_cutoff
    exported=b.metric_discarded_factor;exported[:]=0
    np.testing.assert_array_equal(b.metric_discarded_factor,h)
    for seed in range(5):
        d=densities(seed=seed)
        if component:
            d*=np.minimum(1.,.8*envelope/np.maximum(1e-30,w[:,None,None]*abs(d.sum(axis=0))))[None]
        else:d*=31./np.einsum('k,kij->',w,abs(d.sum(axis=0)))
        actual=b.build(d,w,make_k=False)
        fine=reference(exact_m,exact_t,d,w,absolute=1e-12,relative=0.)
        coarse=reference(exact_m,exact_t,d,w,absolute=ri.absolute_cutoff,relative=0.)
        assert np.max(abs(actual.J-fine[0]))<=c.j_bound+8e-11
        assert abs(actual.receipt.coulomb_energy-fine[2])<=c.energy_bound+8e-11
        assert np.max(abs(coarse[0]-fine[0]))<=c.metric_j_bound+8e-11
        assert -8e-11<=fine[2]-coarse[2]<=c.metric_energy_bound+8e-11
        assert actual.receipt.approximation_j_bound<=c.coupling_j_bound
        assert actual.receipt.approximation_energy_bound<=c.coupling_energy_bound
        assert actual.receipt.accuracy_certified


def test_nested_reference_can_omit_its_own_small_positive_modes():
    a,s,ri,accuracy=metric_case(1,reference_rank=3)
    b=manta.PeriodicRIQBuilder(s,2,ri,a['budget'],accuracy);c=b.metric_info.accuracy
    assert c.metric_reference_rank==3 and c.metric_discarded_rank==2
    m,t=independent(a);d=densities();w=np.array([.27,.73]);d*=.01
    actual=b.build(d,w,make_k=False);fine=reference(m,t,d,w,absolute=accuracy.metric_reference_cutoff,relative=0.)
    assert np.max(abs(actual.J-fine[0]))<=c.j_bound+8e-11
    assert abs(actual.receipt.coulomb_energy-fine[2])<=c.energy_bound+8e-11


def test_empty_removed_space_preserves_selected_operator_and_old_default_certificate():
    a,s,ri,accuracy=metric_case(2,reference_rank=2)
    b=manta.PeriodicRIQBuilder(s,2,ri,a['budget'],accuracy);c=b.metric_info.accuracy
    assert c.metric_discarded_rank==0 and c.metric_j_bound==c.metric_energy_bound==0.
    accuracy.metric_reference_cutoff=None
    old=manta.PeriodicRIQBuilder(s,2,ri,a['budget'],accuracy)
    assert not old.metric_info.accuracy.metric_truncation_certified
    assert old.metric_discarded_factor.shape==(0,0)
    assert old.metric_info.source_identity==b.metric_info.source_identity
    d=densities()*.01;w=np.array([.27,.73])
    np.testing.assert_array_equal(b.build(d,w,make_k=False).J,old.build(d,w,make_k=False).J)
    assert c.certificate_identity!=old.metric_info.accuracy.certificate_identity


@pytest.mark.parametrize('bad',[-1.,float('nan'),float('inf')])
def test_invalid_metric_reference_is_refused_before_preparation(bad):
    a,s,ri,accuracy=metric_case();accuracy.metric_reference_cutoff=bad
    with pytest.raises(ValueError,match='reference cutoff'):manta.PeriodicRIQBuilder(s,2,ri,a['budget'],accuracy)
    assert s.preparation.pair_extents==s.preparation.moment_panels==0


def test_finer_reference_and_combined_targets_are_enforced():
    a,s,ri,accuracy=metric_case(0);accuracy.maximum_attempts=1
    accuracy.metric_reference_cutoff=ri.absolute_cutoff+1
    with pytest.raises(ValueError,match='exceeds'):manta.PeriodicRIQBuilder(s,2,ri,a['budget'],accuracy)
    accuracy.metric_reference_cutoff=1e-12;accuracy.j_error=accuracy.energy_error=1e-12
    with pytest.raises(RuntimeError,match='targets unmet'):manta.PeriodicRIQBuilder(s,2,ri,a['budget'],accuracy)
    # A zero-entry density family has exactly zero removed-mode error even
    # when the selected metric has zero rank and the reference has full rank.
    accuracy.density_entry_limits=[0.]*(s.plan.momenta*s.plan.n_orb**2)
    b=manta.PeriodicRIQBuilder(s,2,ri,a['budget'],accuracy)
    assert b.metric_info.accuracy.metric_energy_bound==b.metric_info.accuracy.metric_j_bound==0.


@pytest.mark.parametrize('field,quantity',CAPS)
def test_optional_factor_storage_and_work_obey_all_caps(field,quantity):
    a,s,ri,accuracy=metric_case();budget=a['budget']
    p=manta.plan_periodic_ri_q(s,2,ri,budget,accuracy)
    accuracy.metric_reference_cutoff=None;old=manta.plan_periodic_ri_q(s,2,ri,budget,accuracy)
    accuracy.metric_reference_cutoff=1e-12
    assert p.modeled_peak_bytes>old.modeled_peak_bytes and p.scalar_terms>old.scalar_terms
    setattr(budget,field,getattr(p,quantity)-1)
    with pytest.raises(ValueError,match='budget'):manta.PeriodicRIQBuilder(s,2,ri,budget,accuracy)
    assert s.preparation.pair_extents==s.preparation.moment_panels==0
    setattr(budget,field,getattr(p,quantity));manta.PeriodicRIQBuilder(s,2,ri,budget,accuracy)


def metric_job(directory,method='rhf',truncate=True):
    from tests.test_manta_periodic_ri_preparation import prepared_job
    from tests.test_manta_periodic_ri_hierarchy import hierarchy_ks_job
    if method in ('rhf','uhf'):
        job,a,k=prepared_job(directory,method,'Be' if method=='rhf' else 'Li',100 if method=='rhf' else 3)
        domains=getattr(manta,'build_periodic_ri_'+method+'_domains')(**a)
        source=manta.PeriodicGDFTiles(orbital=a['basis'],auxiliary=a['auxiliary'],system=a['system'],q=np.zeros(3),
            ket_kpoints=k,vectors=domains.reciprocal_vectors,options=a['options'].fitting,budget=a['budget'])
        eigenvalues=np.linalg.eigvalsh(source.metric().values)
        cutoff=float(.5*(eigenvalues[0]+eigenvalues[1])) if truncate else 1e-12
    else:
        job,a,_,k,_=hierarchy_ks_job(directory,1,100);cutoff=1e-12
        if method!='rks':job.update(functional=method,exchange=manta.ExchangeBuildOptions(algorithm='exact'),exchange_exxdiv='ewald')
    ri=job['coulomb'].metric;ri.absolute_cutoff=cutoff;ri.relative_cutoff=0.
    accuracy=automatic_accuracy(1e8 if truncate else 1e-6,1e8 if truncate else 1e-6,0.)
    accuracy.metric_reference_cutoff=1e-12
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],coupling_multipole_pair_tree=True,accuracy=accuracy)
    job['integral_engine'].budget.memory_bytes=4*2**30
    return job,a,k


@pytest.mark.parametrize('method',['rhf','uhf','rks','pbe0','hse06'])
def test_complete_jobs_keep_fixed_source_and_shared_outputs(tmp_path,method):
    from tests.test_manta_periodic_ri_job import _as_fock_build
    from vibeqc.output import validate_qvf
    job,a,k=metric_job(tmp_path,method,truncate=method in ('rhf','uhf'))
    result=vq.run_periodic_job(**job);c=result.integral_provenance['fitting']['accuracy'];m=c['metric_truncation']
    assert c['qualified'] and c['density_limit_origin']=='canonical-overlap'
    assert m['metric_discarded_rank']==(1 if method in ('rhf','uhf') else 0)
    assert c['j_bound']>=m['metric_j_bound'] and c['energy_bound']>=m['metric_energy_bound']
    assert result.converged==(method!='uhf')
    assert validate_qvf(str(job['output'].with_suffix('.qvf')))['valid']
    if method in ('rhf','uhf'):
        state=_as_fock_build(result)
    else:
        from tests.test_manta_periodic_ri_ks import as_fock_build
        state=as_fock_build(result.native,k)
    assert abs(state.energy.total-result.energy)<2e-12
    assert not result.integral_provenance['symmetry_reduced']
    manual=dict(job,output=tmp_path/'manual'/'job')
    manual['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],accuracy=None,coupling_multipole_error=c['selected_coupling_error'])
    control=vq.run_periodic_job(**manual)
    for name in ('density','fock','overlap','hcore'):np.testing.assert_array_equal(getattr(result.native,name),getattr(control.native,name))
    assert result.energy==control.energy
    assert result.native.receipt.source_identity==control.native.receipt.source_identity
