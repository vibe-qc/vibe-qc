"""Independent reference contributions and component-directed bounded search."""
import dataclasses
import numpy as np
import pytest
import vibeqc as vq
from vibeqc import manta
from tests.test_manta_periodic_bupo_domain import bounded
from tests.test_manta_periodic_bupo import density, options, K, WEIGHTS, cube
from tests.test_manta_periodic_ewald import reference
from tests.test_manta_periodic_ri_fitting_bounds import case
from tests.test_manta_periodic_ri_auxiliary_selection import energy_selection_job,selected_job

COMPONENTS=('approximation','real_domain','reciprocal_domain','pair_image')


def scatter(s,potential,n):
    out=np.zeros((len(K),n,n),complex);offset=0
    for e in s.entries:
        i,j=e.offsets;ni,nj=e.sizes;value=potential[offset:offset+ni*nj].reshape(ni,nj);offset+=ni*nj
        ph=np.exp(1j*K@e.translation)
        out[:,i:i+ni,j:j+nj]+=ph[:,None,None]*value
        if e.paired:out[:,j:j+nj,i:i+ni]+=ph.conj()[:,None,None]*value.T
    return out


@pytest.mark.parametrize('angular',[False,True])
def test_real_and_reciprocal_envelopes_cover_independent_domain_increments(angular):
    b,s,t,e,c,cell=bounded(0,real_radius=0,reciprocal_radius=0,eta=.4,tolerance=0.,
        angular=angular,images=False,exponent=.7)
    d=density(b.nbasis)*.01;actual=c.build(d)
    small=reference(b,s,d,cell,.4,cube(0),cube(0,reciprocal=True))
    large=reference(b,s,d,cell,.4,cube(3),cube(5,reciprocal=True))
    np.testing.assert_array_equal(actual.finite_j_error_bound,0.)
    for index,name in enumerate(('real','reciprocal')):
        matrix=large['component_matrices'][index]-small['component_matrices'][index]
        j=scatter(s,matrix@small['coefficients'],b.nbasis)
        energy=float(.5*small['coefficients']@matrix@small['coefficients'])
        bounds=getattr(actual,'domain_'+name+'_j_error_bound')
        assert np.max(abs(j))>1e-9
        assert np.all(abs(j)<=bounds[None]+3e-11)
        assert abs(energy)<=getattr(actual.receipt.domain,name+'_energy_error_bound')+3e-11
        assert np.max(bounds)==getattr(actual.receipt.domain,name+'_maximum_j_error_bound')
    batch=c.build_batch(np.stack([d,-2*d]))
    for dd,value in zip((d,-2*d),batch.results):
        scalar=c.build(dd)
        for name in ('values','j_error_bound','finite_j_error_bound','domain_real_j_error_bound','domain_reciprocal_j_error_bound'):
            np.testing.assert_array_equal(getattr(value,name),getattr(scalar,name))


def test_finite_approximation_envelope_is_retained_before_domain_addition():
    b,s,t,e,c,cell=bounded(0,tolerance=10.,images=False)
    assert c.partition.approximate_blocks>0
    d=density(b.nbasis);actual=c.build(d);direct=e.build(d)
    assert np.max(actual.finite_j_error_bound)>0
    assert np.all(abs(actual.values-direct.values)<=actual.finite_j_error_bound[None]+3e-11)
    assert np.max(actual.finite_j_error_bound)==actual.receipt.finite_maximum_j_error_bound
    assert abs(actual.receipt.energy-direct.receipt.energy)<=actual.receipt.finite_energy_error_bound+3e-11
    changed=actual.finite_j_error_bound;changed[:]=0
    assert np.max(actual.finite_j_error_bound)>0


@pytest.mark.parametrize('entries',[False,True])
@pytest.mark.parametrize('weights',[(.27,.73),(0.,1.)])
def test_fixed_family_components_preserve_batch_order_and_quadratic_energy_scaling(entries,weights):
    a,ri,ref,o,w=case(component=entries,weights=weights);o.probe_batch_size=1
    first=manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget'])
    o.probe_batch_size=3;batch=manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget'])
    assert first.receipt.certificate_identity==batch.receipt.certificate_identity
    fields=[f'reference_{c}_{q}_bound' for c in COMPONENTS for q in ('j','energy')]
    for name in fields:assert getattr(first.receipt,name)==getattr(batch.receipt,name)
    assert first.receipt.reference_approximation_j_bound==first.receipt.reference_approximation_energy_bound==0.
    o.density_l1_limit*=2
    if entries:o.density_entry_limits=(2*np.asarray(o.density_entry_limits)).tolist()
    larger=manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget'])
    for c in COMPONENTS:
        j,e=(getattr(first.receipt,f'reference_{c}_{q}_bound') for q in ('j','energy'))
        assert getattr(larger.receipt,f'reference_{c}_j_bound')==pytest.approx(2*j,rel=3e-12)
        assert getattr(larger.receipt,f'reference_{c}_energy_bound')==pytest.approx(4*e,rel=3e-12)
        assert e<=.5*first.receipt.density_l1_limit*j*(1+2e-12)


@pytest.mark.parametrize('method',['rhf','uhf'])
def test_selective_tight_reference_keeps_adequate_domains_and_fixed_control(tmp_path,method):
    job,args,accuracy=energy_selection_job(tmp_path,method,tight=True)
    assert accuracy.reference_selection.selective_refinement
    result=vq.run_periodic_job(**job);assert result.converged
    full=result.integral_provenance['fitting']['full_coulomb_accuracy'];trials=full['reference_selection']['trials']
    assert len(trials)>1 and trials[0]['refinement']=='initial' and trials[-1]['ready']
    assert any(t['refinement']=='pair-images' for t in trials[1:])
    for previous,current in zip(trials,trials[1:]):
        p,c=previous['preparation'],current['preparation']
        assert not previous['ready']
        if current['refinement']=='pair-images':
            assert c['real_cube_radius']==p['real_cube_radius'] and c['reciprocal_cube_radius']==p['reciprocal_cube_radius']
            assert c['integral_tolerance']==p['integral_tolerance']
            assert c['pair_cube_radius']>p['pair_cube_radius'] or c['pair_cutoff_bohr']>p['pair_cutoff_bohr']
    assert full['j_error_bound']<=accuracy.j_error and full['energy_error_bound']<=accuracy.energy_error
    control=vq.run_periodic_job(**selected_job(job,args,accuracy,result,tmp_path/'fixed'))
    assert control.energy==result.energy and control.native.receipt.source_identity==result.native.receipt.source_identity
    assert control.integral_provenance['fitting']['full_coulomb_accuracy']['certificate_identity']==full['certificate_identity']
    for name in ('density','fock','hcore','overlap'):np.testing.assert_array_equal(getattr(result.native,name),getattr(control.native,name))


def directed_reference_job(directory,which):
    from tests.test_manta_periodic_ri_fitting_accuracy import reference_selection_job
    job,args,accuracy=reference_selection_job(directory,compact=which!='default',metric=True)
    policy=accuracy.reference_selection;source=policy.initial
    if which=='real':source.real_cube_radius=0
    if which=='reciprocal':source.reciprocal_cube_radius=0
    policy.initial=source;accuracy.reference_selection=policy
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=accuracy)
    return job,args,accuracy


@pytest.mark.parametrize('which',['default','real','reciprocal'])
def test_default_and_single_inadequate_domains_follow_component_guidance(tmp_path,which):
    job,args,accuracy=directed_reference_job(tmp_path,which)
    result=vq.run_periodic_job(**job);assert result.converged
    full=result.integral_provenance['fitting']['full_coulomb_accuracy'];trials=full['reference_selection']['trials']
    assert 1<len(trials)<=accuracy.reference_selection.maximum_refinements+1
    if which=='default':assert {t['refinement'] for t in trials[1:]}=={'pair-images','real-domain','reciprocal-domain'}
    else:assert {t['refinement'] for t in trials[1:]}=={which+'-domain'}
    for previous,current in zip(trials,trials[1:]):
        before,after=previous['preparation'],current['preparation'];action=current['refinement']
        for name in ('pair','real','reciprocal'):
            field=name+'_cube_radius'
            if action!=(name+'-images' if name=='pair' else name+'-domain'):assert after[field]==before[field]
        assert after['integral_tolerance']==before['integral_tolerance']
    assert full['j_error_bound']<=accuracy.j_error and full['energy_error_bound']<=accuracy.energy_error
    assert len(full['metric_selection']['trials'])==2


@pytest.mark.parametrize('saturated',[False,True])
def test_component_selection_limits_still_refuse_before_scf(tmp_path,saturated):
    job,args,accuracy=directed_reference_job(tmp_path,'real')
    policy=accuracy.reference_selection
    if saturated:
        source=policy.initial;source.pair_cube_radius=source.real_cube_radius=source.reciprocal_cube_radius=0
        source.pairs.pair_cutoff_bohr=policy.maximum_pair_cutoff_bohr=3.
        policy.initial=source;policy.maximum_cube_radius=0
    else:policy.maximum_refinements=1
    accuracy.reference_selection=policy
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=accuracy)
    with pytest.raises(RuntimeError,match='reference targets unmet'):
        vq.run_periodic_job(**job)
    assert not job['output'].with_suffix('.qvf').exists()
