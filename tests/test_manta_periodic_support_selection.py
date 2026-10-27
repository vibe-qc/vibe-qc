"""Automatic physical supports with shared component certificates and fixed SCF."""
import dataclasses
import numpy as np
import pytest
import vibeqc as vq
from vibeqc import manta


def support_job(directory,profile='compact',target=1e-8):
    from tests.test_manta_periodic_exchange_bounds import compact_ri_exchange_accuracy_job,ri_exchange_accuracy_job
    from tests.test_manta_periodic_nuclear import nuclear_accuracy_targets
    from tests.test_manta_periodic_one_electron_images import image_accuracy_targets,ri_image_job
    if profile=='compact':
        job,a,k,scf=compact_ri_exchange_accuracy_job(directory,cutoff=32.,target=target)
    elif profile in ('ordinary','open','global'):
        method,species,limit={'ordinary':('rhf','Be',100),'open':('uhf','Li',3),'global':('pbe0','Be',100)}[profile]
        job,a,k,scf=ri_exchange_accuracy_job(directory,method,species,limit,automatic=True)
        job['periodic_integrals'].exchange_accuracy.matrix_error=target
        job['periodic_integrals'].exchange_accuracy.energy_error=target
        # The old 12-bohr capsule lands exactly on translated 12-bohr
        # endpoints when pair supports grow. Keep the producer's conjugacy
        # refusal; select an explicit auxiliary source away from that tie.
        job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],auxiliary_cutoff_bohr=12.1)
        a['options'].fitting.auxiliary_cutoff_bohr=12.1
    else:
        method,species,limit={'pure':('rks','Be',100),'range':('hse06','Li',100),'range-ri-k':('hse06-ri-k','Li',3)}[profile]
        job,a,k,scf=ri_image_job(directory,method,species,limit,automatic=profile!='range-ri-k')
    policy=manta.PeriodicSupportSelectionOptions()
    job['periodic_integrals']=dataclasses.replace(job['periodic_integrals'],
        one_electron_accuracy=image_accuracy_targets(target),nuclear_accuracy=nuclear_accuracy_targets(target),support_selection=policy)
    job['integral_engine'].budget.memory_bytes=4*2**30
    return job,a,k,scf


def manual_job(job,result,directory):
    r=result.native.receipt;e=result.integral_provenance
    control=dict(job,output=directory/'job')
    control['periodic_integrals']=dataclasses.replace(job['periodic_integrals'],support_selection=None,
        reciprocal_energy_cutoff=e['reciprocal_energy_cutoff'],pair_cutoff_bohr=e['pair_cutoff_bohr'],
        nuclear_cutoff_bohr=e['nuclear_cutoff_bohr'],one_electron_accuracy=r.one_electron_accuracy.targets,
        nuclear_accuracy=r.nuclear_accuracy.targets,exchange_accuracy=r.exchange_accuracy.targets if r.exchange_accuracy else None)
    return control


def native_request(job,a,scf):
    d=job['periodic_integrals'];q=manta.PeriodicSupportRequest()
    for name in ('reciprocal_energy_cutoff','pair_cutoff_bohr','nuclear_cutoff_bohr','alpha','madelung_alpha','madelung_tolerance',
                 'one_electron_accuracy','nuclear_accuracy'):
        setattr(q,name,getattr(d,name))
    q.scf=scf
    if d.exchange_accuracy:
        q.exchange_accuracy=d.exchange_accuracy;q.exchange_scale=.5 if a['system'].multiplicity==1 else 1.
        q.exchange_q0='BvK-ewald'
    return dict(basis=a['basis'],system=a['system'],mesh=a['mesh'],request=q,
                policy=d.support_selection,budget=job['integral_engine'].budget)


def test_automatic_compact_supports_meet_targets_and_match_explicit_job(tmp_path):
    job,a,k,scf=support_job(tmp_path)
    original=(job['periodic_integrals'].reciprocal_energy_cutoff,job['periodic_integrals'].nuclear_cutoff_bohr)
    result=vq.run_periodic_job(**job);e=result.integral_provenance;c=e['support_selection'];trials=c['trials']
    assert result.converged and c['qualified'] and c['termination']=='targets-passed'
    assert len(trials)>1 and any(not t['qualified'] for t in trials[:-1]) and trials[-1]['qualified']
    assert e['reciprocal_energy_cutoff']>original[0] and e['nuclear_cutoff_bohr']>original[1]
    assert (job['periodic_integrals'].reciprocal_energy_cutoff,job['periodic_integrals'].nuclear_cutoff_bohr)==original
    for name in ('overlap_error','kinetic_error','kinetic_energy_error','nuclear_matrix_error','attraction_energy_error',
                 'nuclear_energy_error','exchange_matrix_error','exchange_energy_error'):
        assert trials[-1][name]<=1e-8
    for name in ('pair_cutoff_bohr','reciprocal_energy_cutoff','nuclear_cutoff_bohr'):
        assert np.all(np.diff([t[name] for t in trials])>=0)
    assert c['scalar_terms']==c['preparation_scalar_terms']+sum(t['scalar_terms'] for t in trials)
    assert c['kernel_terms']==c['preparation_kernel_terms']+sum(t['kernel_terms'] for t in trials)
    assert c['shell_calls']==sum(t['shell_calls'] for t in trials)
    assert not e['symmetry_reduced'] and 'density-relaxation' in c['excludes']
    assert e['scalar_terms']==result.native.plan.scalar_terms+c['scalar_terms']
    assert e['job_shell_calls_upper_bound']==result.native.plan.shell_calls+c['shell_calls']
    control=vq.run_periodic_job(**manual_job(job,result,tmp_path/'manual'))
    for name in ('density','fock','overlap','hcore'):np.testing.assert_array_equal(getattr(result.native,name),getattr(control.native,name))
    assert result.energy==control.energy
    for name in ('source_identity','nuclear_identity','coulomb_identity','exchange_identity'):
        assert getattr(result.native.receipt,name)==getattr(control.native.receipt,name)


@pytest.mark.parametrize('profile',['ordinary','open','global','pure','range','range-ri-k'])
def test_automatic_supports_preserve_complete_ri_routes(tmp_path,profile):
    job,a,k,scf=support_job(tmp_path,profile,target=1e6)
    result=vq.run_periodic_job(**job);e=result.integral_provenance;c=e['support_selection']
    assert c['qualified'] and len(c['trials'])==1
    assert result.converged==(profile not in ('open','range-ri-k'))
    assert c['exchange_certified']==(profile in ('ordinary','open','global'))
    assert e['pair_cutoff_bohr']==job['periodic_integrals'].pair_cutoff_bohr
    control=vq.run_periodic_job(**manual_job(job,result,tmp_path/'manual'))
    assert result.energy==control.energy
    for name in ('density','fock','overlap','hcore'):np.testing.assert_array_equal(getattr(result.native,name),getattr(control.native,name))


@pytest.mark.parametrize('field,quantity',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
    ('primitive_quartets','primitive_pairs'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_selection_charges_all_caps_and_keeps_owned_controls(tmp_path,field,quantity):
    job,a,k,scf=support_job(tmp_path,'ordinary',target=1e6);args=native_request(job,a,scf)
    result=manta.select_periodic_supports(**args);assert result.qualified
    setattr(args['budget'],field,getattr(result,quantity))
    other=manta.select_periodic_supports(**args);assert other.certificate_identity==result.certificate_identity
    setattr(args['budget'],field,getattr(result,quantity)-1)
    with pytest.raises(ValueError,match='budget'):manta.select_periodic_supports(**args)
    selected=result.selected;selected.nuclear_cutoff_bohr=100.;assert result.selected.nuclear_cutoff_bohr!=100.
    policy=result.policy;policy.maximum_attempts=1;assert result.policy.maximum_attempts==16


@pytest.mark.parametrize('limit',['attempt','physical','cube'])
def test_unsatisfied_support_limits_refuse_before_scf(tmp_path,limit,monkeypatch):
    from vibeqc import manta_periodic
    job,a,k,scf=support_job(tmp_path)
    policy=job['periodic_integrals'].support_selection
    if limit=='attempt':policy.maximum_attempts=1
    elif limit=='cube':
        policy.maximum_completion_cube_radius=40
        policy.maximum_reciprocal_energy_cutoff=2048
        job['periodic_integrals'].exchange_accuracy.matrix_error=1e-30
        job['periodic_integrals'].exchange_accuracy.energy_error=1e-30
    else:
        policy.maximum_pair_cutoff_bohr=job['periodic_integrals'].pair_cutoff_bohr
        policy.maximum_nuclear_cutoff_bohr=job['periodic_integrals'].nuclear_cutoff_bohr
        policy.maximum_reciprocal_energy_cutoff=job['periodic_integrals'].reciprocal_energy_cutoff
    def forbidden(*args,**kwargs):raise AssertionError('SCF must not start for an unqualified selection')
    monkeypatch.setattr(manta_periodic,'run_periodic_ri_rhf_from_cutoffs',forbidden)
    with pytest.raises(ValueError,match='did not meet component targets'):vq.run_periodic_job(**job)


def test_support_dry_run_does_not_select_or_run(tmp_path,monkeypatch):
    from vibeqc import _vibeqc_core as core
    job,_,_,_=support_job(tmp_path)
    def forbidden(*args,**kwargs):raise AssertionError('Dry run must not select physical supports')
    monkeypatch.setattr(core._manta,'select_periodic_supports',forbidden)
    job['dry_run']=True;assert vq.run_periodic_job(**job) is None


def test_support_policy_controls_and_route_guards(tmp_path):
    job,a,k,scf=support_job(tmp_path,'ordinary',target=1e6);d=job['periodic_integrals']
    for field in ('one_electron_accuracy','nuclear_accuracy'):
        with pytest.raises(ValueError,match='require both'):dataclasses.replace(d,**{field:None})
    with pytest.raises(TypeError,match='PeriodicSupportSelectionOptions'):dataclasses.replace(d,support_selection=object())
    for field,value in [('maximum_attempts',0),('maximum_completion_cube_radius',0),('growth_factor',1.),('growth_factor',float('nan')),
                         ('maximum_pair_cutoff_bohr',float('inf')),('maximum_nuclear_cutoff_bohr',-1.),('maximum_reciprocal_energy_cutoff',0.)]:
        policy=manta.PeriodicSupportSelectionOptions();setattr(policy,field,value)
        with pytest.raises(ValueError):dataclasses.replace(d,support_selection=policy)
        args=native_request(job,a,scf);args['policy']=policy
        with pytest.raises(ValueError):manta.select_periodic_supports(**args)
    job['coulomb']=manta.CoulombBuildOptions(algorithm='exact');job['periodic_fitting']=None
    with pytest.raises(NotImplementedError,match='Automatic physical supports'):vq.run_periodic_job(**job)


@pytest.mark.parametrize('profile',['ordinary','open','global'])
def test_diffuse_sources_select_useful_component_targets(tmp_path,profile):
    job,a,k,scf=support_job(tmp_path,profile,target=1e-8)
    args=native_request(job,a,scf)
    if profile=='global':args['request'].exchange_scale=.125
    result=manta.select_periodic_supports(**args)
    assert result.qualified and len(result.trials)>1
    last=result.trials[-1]
    for name in ('overlap_error','kinetic_error','kinetic_energy_error','nuclear_matrix_error',
                 'attraction_energy_error','nuclear_energy_error','exchange_matrix_error','exchange_energy_error'):
        assert getattr(last,name)<=1e-8
    for name in ('pair_cutoff_bohr','reciprocal_energy_cutoff','nuclear_cutoff_bohr'):
        assert getattr(result.selected,name)>getattr(job['periodic_integrals'],name)
    # Complete tight diffuse jobs are calculation benchmarks: their retained
    # exact exchange has nearly one million vectors. Compact complete state
    # equivalence and loose diffuse route coverage above keep this lane small.


def test_physical_selection_preserves_fitting_conjugacy_refusal(tmp_path):
    job,_,_,_=support_job(tmp_path,'ordinary',target=1e-8)
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],auxiliary_cutoff_bohr=12.)
    with pytest.raises(ValueError,match='conjugate-closed finite capsule'):
        vq.run_periodic_job(**job)



def test_overlap_image_rank_drives_support_growth(tmp_path):
    job,a,k,scf=support_job(tmp_path,'ordinary',target=1e6)
    job['periodic_integrals'].one_electron_accuracy.overlap_image_rank=True
    # Exercise an explicit near-cutoff request, preserving the production
    # cutoff defaults and the actual physical integral source.
    scf.overlap_rank_absolute_floor=.0089
    args=native_request(job,a,scf)
    result=manta.select_periodic_supports(**args)
    assert result.qualified and len(result.trials)>1
    assert all(t.overlap_image_rank_requested for t in result.trials)
    assert not result.trials[0].overlap_image_rank_qualified
    assert result.trials[-1].overlap_image_rank_qualified
    assert result.selected.pair_cutoff_bohr>job['periodic_integrals'].pair_cutoff_bohr
    for t in result.trials:
        if not t.overlap_image_rank_qualified:assert t.grow_pair and not t.qualified
    args['policy'].maximum_attempts=1
    refused=manta.select_periodic_supports(**args)
    assert not refused.qualified and not refused.trials[-1].overlap_image_rank_qualified


def test_overlap_image_rank_support_complete_state(tmp_path):
    job,_,_,_=support_job(tmp_path,'ordinary',target=1e6)
    job['periodic_integrals'].one_electron_accuracy.overlap_image_rank=True
    selected=vq.run_periodic_job(**job)
    assert selected.converged
    e=selected.integral_provenance
    assert e['one_electron_accuracy']['overlap_image_rank']['qualified']
    assert e['support_selection']['trials'][-1]['overlap_image_rank_qualified']
    control=vq.run_periodic_job(**manual_job(job,selected,tmp_path/'manual'))
    assert control.converged and selected.energy==control.energy
    for name in ('density','fock','hcore','overlap'):
        np.testing.assert_array_equal(getattr(selected.native,name),getattr(control.native,name))
