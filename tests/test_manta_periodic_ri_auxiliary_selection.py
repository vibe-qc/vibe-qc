"""Actual Gamma auxiliary selection under a fixed full-Coulomb certificate."""
import dataclasses
from pathlib import Path
import numpy as np
import pytest
import vibeqc as vq
from vibeqc import manta
from vibeqc.auxiliary_generation import ProductAuxiliaryOptions,prepare_product_auxiliary_basis
from vibeqc.output import validate_qvf
from tests.test_manta_periodic_ri_domain_selection import domain_job,source_inputs,enlarged_inputs
from tests.test_manta_periodic_ri_fitting_accuracy import selected_reference


def selection_job(directory,method='rhf',*,metric=False,explicit=False):
    job,args,accuracy=domain_job(directory,method,metric=metric,explicit=explicit)
    policy=manta.PeriodicRIFittingAuxiliaryOptions();policy.maximum_refinements=1
    accuracy.auxiliary_selection=policy;accuracy.j_error=.005;accuracy.energy_error=.7
    args['options'].fitting_accuracy=accuracy
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=accuracy)
    job['citations']=True
    return job,args,accuracy


def native_selection(args,accuracy):
    policy=manta.prepare_periodic_ri_auxiliary_selection(args['basis'],args['auxiliary'],args['system'],accuracy.auxiliary_selection,args['budget'])
    accuracy.auxiliary_selection=policy;args['options'].fitting_accuracy=accuracy
    return policy.prepared


def selected_basis(args,accuracy,result):
    trial=result.native.receipt.fitting_info.auxiliary_trials[-1]
    if not trial.candidate:return args['auxiliary']
    o=ProductAuxiliaryOptions();initial=accuracy.auxiliary_selection.initial
    o.beta=initial.beta;o.refinement=initial.refinement+trial.candidate-1;o.maximum_angular_momentum=initial.maximum_angular_momentum
    generated=prepare_product_auxiliary_basis(args['system'].unit_cell_molecule(),args['basis'],args['auxiliary'],o,accuracy.auxiliary_selection.generation_limits)
    assert generated.receipt.identity==trial.generation_identity
    return generated.basis


def selected_job(job,args,accuracy,result,directory):
    trial=result.native.receipt.fitting_info.auxiliary_trials[-1].source
    fixed=manta.PeriodicRIFittingAccuracyOptions();fixed.reference=accuracy.reference or selected_reference(args,accuracy,result)
    fixed.j_error=accuracy.j_error;fixed.energy_error=accuracy.energy_error
    domain=manta.PeriodicRIFittingDomainOptions();domain.maximum_refinements=0
    domain.initial_pair_cutoff_bohr=trial.pair_cutoff_bohr;domain.initial_auxiliary_cutoff_bohr=trial.auxiliary_cutoff_bohr
    domain.initial_reciprocal_cube_radius=trial.reciprocal_cube_radius;fixed.domain_selection=domain
    metric=job['coulomb'].metric
    if trial.metric_refinement:
        metric=manta.RIOptions();metric.absolute_cutoff=trial.metric_cutoff;metric.relative_cutoff=0.
    return dict(job,output=directory/'job',
        coulomb=dataclasses.replace(job['coulomb'],auxiliary_basis=selected_basis(args,accuracy,result),metric=metric),
        periodic_fitting=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=fixed))


@pytest.mark.parametrize('method',['rhf','uhf','rks','uks','pbe0','hse06'])
def test_candidate_selection_retains_certified_owner_and_fixed_state(tmp_path,method):
    job,args,accuracy=selection_job(tmp_path,method)
    result=vq.run_periodic_job(**job);assert result.converged
    assert accuracy.auxiliary_selection.prepared is None  # caller controls captured
    full=result.integral_provenance['fitting']['full_coulomb_accuracy'];selection=full['auxiliary_selection'];trials=selection['trials']
    assert trials[-1]['candidate']==1 and {t['candidate'] for t in trials}=={0,1}
    assert len({t['bound']['reference_source_identity'] for t in trials})==1
    assert all(t['bound']['j_error_bound']>accuracy.j_error or t['bound']['energy_error_bound']>accuracy.energy_error for t in trials[:-1])
    assert full['j_error_bound']<=accuracy.j_error and full['energy_error_bound']<=accuracy.energy_error
    assert selection['selected_generation']['identity']==trials[-1]['generation_identity']
    assert result.integral_provenance['auxiliary_basis_functions']==trials[-1]['functions']
    assert result.integral_provenance['auxiliary_basis_functions_requested']==args['auxiliary'].nbasis
    assert result.run_fields['auxiliary_basis_functions']==trials[-1]['functions']
    assert result.run_fields['auxiliary_basis_functions_requested']==args['auxiliary'].nbasis
    control=vq.run_periodic_job(**selected_job(job,args,accuracy,result,tmp_path/'fixed'))
    assert control.converged and result.energy==control.energy
    assert result.native.receipt.source_identity==control.native.receipt.source_identity
    assert full['certificate_identity']==control.integral_provenance['fitting']['full_coulomb_accuracy']['certificate_identity']
    for name in ('density','fock','hcore','overlap'):
        np.testing.assert_array_equal(getattr(result.native,name),getattr(control.native,name))
    assert validate_qvf(str(job['output'].with_suffix('.qvf')))['valid']
    assert 'stoychev_autoaux_2017' in job['output'].with_suffix('.bibtex').read_text()
    assert not result.integral_provenance['symmetry_reduced'] and not full['roundoff_included']
    capacity=result.integral_provenance['fitting']['plan'];actual=result.integral_provenance['fitting']['selected_plan']
    assert capacity['n_auxiliary']>=actual['n_auxiliary']==trials[-1]['functions']


def test_metric_controls_reset_at_each_candidate_and_domain(tmp_path):
    job,args,accuracy=selection_job(tmp_path,metric=True,explicit=True)
    result=vq.run_periodic_job(**job);trials=result.native.receipt.fitting_info.auxiliary_trials
    assert len(trials)==12
    assert [t.source.rank for t in trials]==[0,3,0,3,0,3,0,8,0,8,0,8]
    assert [t.source.metric_refinement for t in trials]==[0,1]*6
    control=vq.run_periodic_job(**selected_job(job,args,accuracy,result,tmp_path/'fixed'))
    assert result.energy==control.energy


@pytest.mark.parametrize('field,quantity',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
    ('primitive_quartets','primitive_pairs'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_complete_capacity_counts_every_auxiliary_source(tmp_path,field,quantity):
    _,args,accuracy=selection_job(tmp_path);sequence=native_selection(args,accuracy)
    plan=manta.plan_periodic_ri_rhf_from_cutoffs(**args)
    assert plan.rhf.fock.n_auxiliary==sequence.candidate(1).basis.nbasis
    required=getattr(plan,quantity);setattr(args['budget'],field,required-1)
    with pytest.raises((ValueError,OverflowError),match='budget'):manta.plan_periodic_ri_rhf_from_cutoffs(**args)
    setattr(args['budget'],field,required);manta.plan_periodic_ri_rhf_from_cutoffs(**args)


@pytest.mark.parametrize('field,quantity',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
    ('primitive_quartets','primitive_products'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_joint_search_budget_charges_rejected_candidates(tmp_path,field,quantity):
    job,_,accuracy=selection_job(tmp_path,explicit=True);result=vq.run_periodic_job(**job)
    charged=result.integral_provenance['fitting']['full_coulomb_accuracy']['auxiliary_selection']['charged'][quantity]
    policy=accuracy.auxiliary_selection;budget=policy.budget;setattr(budget,field,charged-1);policy.budget=budget;accuracy.auxiliary_selection=policy
    with pytest.raises((ValueError,OverflowError),match='budget'):
        vq.run_periodic_job(**dict(job,output=tmp_path/'refused'/'job',
            periodic_fitting=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=accuracy)))
    assert not (tmp_path/'refused/job.qvf').exists()


def test_seed_pass_retains_original_owner_and_records_no_selected_generation(tmp_path):
    job,args,accuracy=selection_job(tmp_path);accuracy.j_error=accuracy.energy_error=1e6
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=accuracy)
    result=vq.run_periodic_job(**job);s=result.integral_provenance['fitting']['full_coulomb_accuracy']['auxiliary_selection']
    assert len(s['trials'])==1 and s['selected']['candidate']==0 and s['selected_generation'] is None
    control=vq.run_periodic_job(**selected_job(job,args,accuracy,result,tmp_path/'fixed'))
    assert control.energy==result.energy and control.native.receipt.source_identity==result.native.receipt.source_identity


def test_unmet_basis_target_refuses_before_scf_and_dry_run_has_no_certificate(tmp_path):
    job,_,accuracy=selection_job(tmp_path);accuracy.j_error=accuracy.energy_error=1e-12
    policy=accuracy.auxiliary_selection;policy.maximum_refinements=0;accuracy.auxiliary_selection=policy
    # Keep a fixed qualified reference: failure belongs to the basis/domain target.
    from tests.test_manta_periodic_ri_fitting_bounds import prepared_fitting_reference
    fixed_job,args,_=selection_job(tmp_path/'reference',explicit=True)
    accuracy.reference=fixed_job['periodic_fitting'].full_coulomb_accuracy.reference;accuracy.reference_selection=None
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=accuracy)
    assert vq.run_periodic_job(**dict(job,dry_run=True)) is None
    with pytest.raises(RuntimeError,match='auxiliary/domain/metric limit'):vq.run_periodic_job(**job)
    assert not job['output'].with_suffix('.qvf').exists()


def test_prepared_sequence_mismatch_and_invalid_controls(tmp_path):
    job,args,accuracy=selection_job(tmp_path);sequence=native_selection(args,accuracy)
    policy=accuracy.auxiliary_selection
    with pytest.raises(ValueError,match='seed source mismatch'):
        manta.prepare_periodic_ri_auxiliary_selection(args['basis'],args['basis'],args['system'],policy,args['budget'])
    initial=policy.initial;initial.beta=3.;policy.initial=initial
    with pytest.raises(ValueError,match='prepared controls mismatch'):manta.validate_periodic_ri_auxiliary_options(policy)
    policy.prepared=None;policy.maximum_refinements=9
    with pytest.raises(ValueError,match='auxiliary selection'):manta.validate_periodic_ri_auxiliary_options(policy)


@pytest.mark.parametrize('component',[False,True])
def test_auxiliary_selection_preserves_separate_component_contracts(tmp_path,component):
    _,args,accuracy=selection_job(tmp_path);native_selection(args,accuracy)
    if component:
        from tests.test_manta_periodic_ri_pair_tree import automatic_accuracy
        args['options'].ri_accuracy=automatic_accuracy(1e8,1e8,1e6)
    else:args['options'].exchange_algorithm='ri'
    with pytest.raises(ValueError,match='J-only fitting without a separate component'):manta.plan_periodic_ri_rhf_from_cutoffs(**args)



@pytest.mark.parametrize('method',['rhf','uhf'])
def test_finer_generated_candidate_is_selected_by_actual_bound(tmp_path,method):
    job,args,accuracy=selection_job(tmp_path,method);accuracy.energy_error=.48182
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=accuracy)
    result=vq.run_periodic_job(**job);full=result.integral_provenance['fitting']['full_coulomb_accuracy']
    trials=full['auxiliary_selection']['trials']
    assert trials[-1]['candidate']==2 and trials[-1]['functions']==12 and trials[-1]['rank']==10
    assert {t['candidate'] for t in trials}=={0,1,2}
    assert all(t['bound']['j_error_bound']>accuracy.j_error or t['bound']['energy_error_bound']>accuracy.energy_error for t in trials[:-1])
    assert full['energy_error_bound']<accuracy.energy_error
    control=vq.run_periodic_job(**selected_job(job,args,accuracy,result,tmp_path/'fixed'))
    assert control.converged and result.energy==control.energy


def test_shared_auxiliary_factory_matches_independent_integrals(tmp_path):
    from tests.test_manta_periodic_gdf import legacy
    from tests.test_manta_periodic_gdf_tiles import assemble
    _,args,accuracy=selection_job(tmp_path);sequence=native_selection(args,accuracy)
    a=source_inputs(args);base=manta.PeriodicGDFTiles(**a);auxiliary=sequence.candidate(0).basis
    from types import SimpleNamespace
    trial=SimpleNamespace(pair_cutoff_bohr=6.75,auxiliary_cutoff_bohr=9.,reciprocal_cube_radius=2)
    candidate=base.with_auxiliary_domains(auxiliary,6.75,9.,2,args['budget'])
    independent=enlarged_inputs(dict(a,auxiliary=auxiliary),trial)
    metric,coupling,_=legacy(independent)
    np.testing.assert_allclose(candidate.metric().values,metric,atol=3e-12,rtol=3e-12)
    np.testing.assert_allclose(assemble(candidate),coupling,atol=3e-12,rtol=3e-12)
    assert candidate.orbital_basis_identity==base.orbital_basis_identity
    assert candidate.auxiliary_basis_identity!=base.auxiliary_basis_identity


@pytest.mark.parametrize('field,quantity',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
    ('primitive_quartets','primitive_products'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_shared_auxiliary_factory_admits_each_cap(tmp_path,field,quantity):
    _,args,accuracy=selection_job(tmp_path);sequence=native_selection(args,accuracy)
    a=source_inputs(args);base=manta.PeriodicGDFTiles(**a);auxiliary=sequence.candidate(0).basis
    plan=base.plan_auxiliary_domains(auxiliary,6.75,9.,2,args['budget'])
    required=getattr(plan.source,quantity) if field in ('shell_calls','primitive_quartets') else getattr(plan,quantity)
    setattr(args['budget'],field,required-1)
    with pytest.raises((ValueError,OverflowError),match='budget'):base.with_auxiliary_domains(auxiliary,6.75,9.,2,args['budget'])
    setattr(args['budget'],field,required);base.with_auxiliary_domains(auxiliary,6.75,9.,2,args['budget'])


def test_auxiliary_selection_without_domain_search_preserves_fixed_supports(tmp_path):
    from tests.test_manta_periodic_ri_fitting_accuracy import reference_selection_job
    job,args,accuracy=reference_selection_job(tmp_path,compact=True)
    policy=manta.PeriodicRIFittingAuxiliaryOptions();policy.maximum_refinements=1
    accuracy.auxiliary_selection=policy;accuracy.j_error=.005;accuracy.energy_error=.7
    args['options'].fitting_accuracy=accuracy
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=accuracy)
    result=vq.run_periodic_job(**job);assert result.converged
    full=result.integral_provenance['fitting']['full_coulomb_accuracy']
    trials=full['auxiliary_selection']['trials']
    assert [t['candidate'] for t in trials]==[0,1]
    assert all(t['domain_attempt']==t['reciprocal_cube_radius']==0 for t in trials)
    assert all(t['pair_cutoff_bohr']==3. and t['auxiliary_cutoff_bohr']==12. for t in trials)
    control=vq.run_periodic_job(**selected_job(job,args,accuracy,result,tmp_path/'fixed'))
    assert control.converged and result.energy==control.energy
    assert result.native.receipt.source_identity==control.native.receipt.source_identity
    for name in ('density','fock','hcore','overlap'):
        np.testing.assert_array_equal(getattr(result.native,name),getattr(control.native,name))


def energy_selection_job(directory,method='rhf',*,tight=False):
    job,args,accuracy=selection_job(directory,method)
    domain=accuracy.domain_selection;domain.maximum_refinements=4
    domain.maximum_pair_cutoff_bohr=6.75;domain.maximum_auxiliary_cutoff_bohr=9.
    domain.maximum_reciprocal_cube_radius=4;accuracy.domain_selection=domain
    policy=accuracy.auxiliary_selection;policy.maximum_refinements=0;accuracy.auxiliary_selection=policy
    reference=accuracy.reference_selection;reference.maximum_refinements=3 if tight else 0
    accuracy.reference_selection=reference
    accuracy.j_error=1e-6 if tight else .001;accuracy.energy_error=1e-4 if tight else .05
    args['options'].fitting_accuracy=accuracy
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=accuracy)
    return job,args,accuracy


@pytest.mark.parametrize('method',['rhf','uhf','rks','uks','pbe0','hse06'])
def test_reference_energy_envelope_avoids_unnecessary_refinement(tmp_path,method):
    job,args,accuracy=energy_selection_job(tmp_path,method)
    result=vq.run_periodic_job(**job);assert result.converged
    full=result.integral_provenance['fitting']['full_coulomb_accuracy']
    refs=full['reference_selection']['trials'];assert len(refs)==1 and refs[0]['ready']
    bound=refs[0]['bound'];fraction=accuracy.reference_selection.reference_fraction
    assert bound['reference_remainder_energy_bound']<=fraction*accuracy.energy_error
    assert .5*full['density_l1_limit']*bound['reference_remainder_j_bound']>fraction*accuracy.energy_error
    assert full['j_error_bound']<=accuracy.j_error and full['energy_error_bound']<=accuracy.energy_error
    assert full['reference_remainder_energy_bound']<=full['energy_error_bound']
    control=vq.run_periodic_job(**selected_job(job,args,accuracy,result,tmp_path/'fixed'))
    assert control.converged and result.energy==control.energy
    assert full['certificate_identity']==control.integral_provenance['fitting']['full_coulomb_accuracy']['certificate_identity']
    for name in ('density','fock','hcore','overlap'):
        np.testing.assert_array_equal(getattr(result.native,name),getattr(control.native,name))


@pytest.mark.parametrize('method',['rhf','uhf'])
def test_tighter_automatic_fitting_accuracy_uses_shared_domain_limits(tmp_path,method):
    job,args,accuracy=energy_selection_job(tmp_path,method,tight=True)
    result=vq.run_periodic_job(**job);assert result.converged
    full=result.integral_provenance['fitting']['full_coulomb_accuracy']
    assert full['j_error_bound']<=1e-6 and full['energy_error_bound']<=1e-4
    assert len(full['reference_selection']['trials'])>1
    selected=full['auxiliary_selection']['selected']
    assert selected['candidate']==1 and selected['reciprocal_cube_radius']==4
    assert selected['pair_cutoff_bohr']==6.75 and selected['auxiliary_cutoff_bohr']==9.
    control=vq.run_periodic_job(**selected_job(job,args,accuracy,result,tmp_path/'fixed'))
    assert control.converged and result.energy==control.energy
    for name in ('density','fock','hcore','overlap'):
        np.testing.assert_array_equal(getattr(result.native,name),getattr(control.native,name))
