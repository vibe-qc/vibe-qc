"""Rank-margin admission uses the selected shared Gamma owner before SCF."""
import dataclasses
import numpy as np
import pytest
import vibeqc as vq
from vibeqc import manta
from vibeqc.output import validate_qvf
from tests.test_manta_periodic_ri_fitting_accuracy import accuracy_job,reference_selection_job,selected_reference
from tests.test_manta_periodic_ri_domain_selection import source_inputs,domain_job,fixed_job
from tests.test_manta_periodic_ri_auxiliary_selection import selected_job


def rank_job(directory,method='rhf',kind='fixed'):
    if kind=='fixed':job,args,_,accuracy=accuracy_job(directory,method)
    elif kind=='domain':job,args,accuracy=domain_job(directory,method)
    else:job,args,accuracy=reference_selection_job(directory,method,compact=True)
    accuracy.j_error=accuracy.energy_error=1e8
    if accuracy.reference_selection is not None:
        policy=accuracy.reference_selection;policy.maximum_refinements=0;policy.reference_fraction=1e-16
        accuracy.reference_selection=policy
    if kind=='auxiliary':
        policy=manta.PeriodicRIFittingAuxiliaryOptions();policy.maximum_refinements=0
        accuracy.auxiliary_selection=policy
    # The same shared producer/eigensolver fixes the initial cutoff exactly
    # at its lowest eigenvalue. The requested rank margin alone rejects it.
    source=manta.PeriodicGDFTiles(**source_inputs(args,method))
    initial=manta.PeriodicRIQBuilder(source,2,args['options'].ri,args['budget'])
    metric=manta.RIOptions();metric.relative_cutoff=0.
    metric.absolute_cutoff=initial.metric_info.minimum_eigenvalue
    assert metric.absolute_cutoff>0.
    accuracy.metric_rank_perturbation=1e-10
    accuracy.maximum_metric_refinements=1 if kind in ('fixed','reference') else 0
    accuracy.metric_cutoff_reduction=.1
    args['options'].ri=metric;args['options'].fitting_accuracy=accuracy
    job['coulomb']=dataclasses.replace(job['coulomb'],metric=metric)
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=accuracy)
    return job,args,accuracy


def rank_control(job,args,accuracy,result,directory,kind):
    if kind=='auxiliary':fixed=selected_job(job,args,accuracy,result,directory)
    elif kind=='domain':fixed=fixed_job(job,args,accuracy,result,directory)
    else:
        selected=result.integral_provenance['fitting']['full_coulomb_accuracy']['metric_selection']['trials'][-1]
        metric=manta.RIOptions();metric.relative_cutoff=0.;metric.absolute_cutoff=selected['cutoff']
        policy=manta.PeriodicRIFittingAccuracyOptions()
        policy.reference=accuracy.reference or selected_reference(args,accuracy,result)
        policy.j_error=accuracy.j_error;policy.energy_error=accuracy.energy_error
        fixed=dict(job,output=directory/'job',coulomb=dataclasses.replace(job['coulomb'],metric=metric),
            periodic_fitting=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=policy))
    policy=fixed['periodic_fitting'].full_coulomb_accuracy
    policy.metric_rank_perturbation=accuracy.metric_rank_perturbation
    fixed['periodic_fitting']=dataclasses.replace(fixed['periodic_fitting'],full_coulomb_accuracy=policy)
    return fixed


@pytest.mark.parametrize('rank',[0,2,6])
def test_raw_source_strict_margin_boundary_and_optional_contract(rank):
    from tests.test_manta_periodic_ri_fitting_bounds import case
    a,ri,ref,o,w=case(rank=rank);base=manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget'])
    assert base.receipt.metric_rank_perturbation is None and not base.receipt.metric_rank_qualified
    radius=ri.metric_info.spectrum.rank_stability_radius;assert radius>0
    for margin in (0.,np.nextafter(radius,0.)):
        o.metric_rank_perturbation=margin
        checked=manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget'])
        assert checked.receipt.metric_rank_qualified and checked.receipt.metric_spectrum.rank==rank
        assert checked.receipt.certificate_identity==base.receipt.certificate_identity
        np.testing.assert_array_equal(checked.j_error_bounds,base.j_error_bounds)
    for margin in (radius,np.nextafter(radius,np.inf)):
        o.metric_rank_perturbation=margin
        with pytest.raises(RuntimeError,match='metric rank target unmet'):
            manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget'])


@pytest.mark.parametrize('value',[-1.,float('nan'),float('inf')])
def test_invalid_margin_rejected_in_public_and_native_planning(tmp_path,value):
    job,args,_,accuracy=accuracy_job(tmp_path);accuracy.metric_rank_perturbation=value
    with pytest.raises(ValueError,match='metric rank perturbation'):
        dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=accuracy)
    args['options'].fitting_accuracy=accuracy
    with pytest.raises(ValueError,match='metric rank perturbation'):
        manta.plan_periodic_ri_rhf_from_cutoffs(**args)


@pytest.mark.parametrize('method,kind',[('rhf','fixed'),('uhf','fixed'),('rhf','reference'),('rhf','domain'),('rhf','auxiliary')])
def test_rank_requirement_drives_search_and_retains_actual_passing_owner(tmp_path,method,kind):
    job,args,accuracy=rank_job(tmp_path,method,kind);result=vq.run_periodic_job(**job)
    assert result.converged;full=result.integral_provenance['fitting']['full_coulomb_accuracy']
    key={'fixed':'metric_selection','reference':'metric_selection','domain':'domain_selection','auxiliary':'auxiliary_selection'}[kind]
    trials=full[key]['trials'];assert len(trials)>=2
    first=trials[0] if kind in ('fixed','reference') else trials[0]['bound']
    assert first['j_error_bound']<=accuracy.j_error and first['energy_error_bound']<=accuracy.energy_error
    assert not first['metric_rank']['qualified'] and full['metric_rank']['qualified']
    assert full['metric_rank']['requested_perturbation']==accuracy.metric_rank_perturbation
    assert full['metric_rank']['spectrum']==result.integral_provenance['fitting']['metrics'][0]['spectrum']
    if kind!='fixed':
        references=full['reference_selection']['trials'];assert len(references)==1 and references[0]['ready']
        assert not references[0]['bound']['metric_rank']['qualified']
        assert references[0]['bound']['reference_remainder_j_bound']>accuracy.reference_selection.reference_fraction*accuracy.j_error
    control=vq.run_periodic_job(**rank_control(job,args,accuracy,result,tmp_path/'control',kind))
    assert control.converged and result.energy==control.energy
    assert result.native.receipt.source_identity==control.native.receipt.source_identity
    for name in ('density','fock','hcore','overlap'):np.testing.assert_array_equal(getattr(result.native,name),getattr(control.native,name))
    assert full['certificate_identity']==control.integral_provenance['fitting']['full_coulomb_accuracy']['certificate_identity']
    assert validate_qvf(str(job['output'].with_suffix('.qvf')))['valid']
    assert not full['roundoff_included'] and not result.integral_provenance['symmetry_reduced']


@pytest.mark.parametrize('kind',['fixed','reference','domain','auxiliary'])
def test_exhausted_rank_margin_is_refused_before_scf(tmp_path,kind):
    job,args,accuracy=rank_job(tmp_path,kind=kind);accuracy.metric_rank_perturbation=1e8
    if kind=='fixed':accuracy.maximum_metric_refinements=0
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=accuracy)
    assert vq.run_periodic_job(**dict(job,dry_run=True)) is None
    with pytest.raises(RuntimeError,match='metric rank target unmet'):
        vq.run_periodic_job(**job)
    assert not job['output'].with_suffix('.qvf').exists()


def test_zero_margin_requires_certified_rank_and_unset_preserves_prior_policy(tmp_path):
    job,args,accuracy=rank_job(tmp_path);accuracy.maximum_metric_refinements=0;accuracy.metric_rank_perturbation=0.
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=accuracy)
    with pytest.raises(RuntimeError,match='metric rank target unmet'):vq.run_periodic_job(**job)
    accuracy.metric_rank_perturbation=None
    result=vq.run_periodic_job(**dict(job,periodic_fitting=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=accuracy)))
    assert result.converged
    admission=result.integral_provenance['fitting']['full_coulomb_accuracy']['metric_rank']
    assert admission['requested_perturbation'] is None and not admission['qualified']
    assert not admission['spectrum']['rank_certified']
