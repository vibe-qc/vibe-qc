"""Pre-SCF full-Coulomb fitting certificates on complete fixed RI methods."""
import dataclasses
from types import SimpleNamespace
import numpy as np
import pytest
import vibeqc as vq
from vibeqc import manta
from vibeqc.output import validate_qvf
from tests.test_manta_periodic_ri_fitting_bounds import fitting_job,prepared_fitting_reference
from tests.test_manta_periodic_ri_hierarchy import hierarchy_ks_job


def accuracy_job(directory,method='rhf'):
    if method in ('rhf','uhf'):
        job,args,*_=fitting_job(directory,method)
    else:
        job,args,_,_,_=hierarchy_ks_job(directory,2 if method=='uks' else 1,100)
        if method in ('pbe0','hse06'):
            job.update(functional=method,exchange=manta.ExchangeBuildOptions(algorithm='exact'),exchange_exxdiv='ewald')
    args['budget'].memory_bytes=4*2**30
    job['integral_engine'].budget.memory_bytes=4*2**30
    physical=args['options'].ks if hasattr(args['options'],'ks') else args['options']
    reference=prepared_fitting_reference(dict(args,options=SimpleNamespace(pair_cutoff_bohr=physical.pair_cutoff_bohr)))
    accuracy=manta.PeriodicRIFittingAccuracyOptions();accuracy.reference=reference
    accuracy.j_error=accuracy.energy_error=1e8
    args['options'].fitting_accuracy=accuracy
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=accuracy)
    return job,args,reference,accuracy


@pytest.mark.parametrize('method',['rhf','uhf','rks','uks','pbe0','hse06'])
def test_complete_fixed_source_and_canonical_family_provenance(tmp_path,method):
    job,args,prepared,accuracy=accuracy_job(tmp_path,method)
    result=vq.run_periodic_job(**job)
    assert result.converged
    info=result.integral_provenance['fitting'];c=info['full_coulomb_accuracy']
    assert c['ri_source_identity']==info['metrics'][0]['source_identity']
    assert c['reference_source_identity']==prepared.source.source_identity
    assert c['component_density_envelope'] and c['targets_requested'] and not c['roundoff_included']
    assert c['density_limit_origin']=='canonical-overlap' and not c['component_bounds_additive']
    assert c['j_error_bound']<=accuracy.j_error and c['energy_error_bound']<=accuracy.energy_error
    assert c['density_l1_limit']==result.native.receipt.accuracy_density.density_l1_limit
    assert c['plan']['modeled_peak_bytes']>=prepared.plan.source_live_bytes
    assert validate_qvf(str(job['output'].with_suffix('.qvf')))['valid']
    assert not result.integral_provenance['symmetry_reduced']
    assert 'manta_periodic_ri_fitting_accuracy' in result.integral_provenance['citation_numerics']
    control=vq.run_periodic_job(**dict(job,output=tmp_path/'control'/'job',
        periodic_fitting=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=None)))
    for name in ('density','fock','hcore','overlap'):
        np.testing.assert_array_equal(getattr(result.native,name),getattr(control.native,name))
    assert result.energy==control.energy and result.native.receipt.source_identity==control.native.receipt.source_identity


def test_native_certificate_matches_shared_selected_ri_and_canonical_bounds(tmp_path):
    _,args,reference,accuracy=accuracy_job(tmp_path)
    result=manta.run_periodic_ri_rhf_from_cutoffs(**{k:v for k,v in args.items() if k!='budget'},
        integral_engine=manta.IntegralEngineOptions(backend='manta',budget=args['budget']))
    domains=manta.build_periodic_ri_rhf_domains(**args)
    k=np.array([args['system'].reciprocal_lattice()@args['mesh'].fractional(args['mesh'].address(i)) for i in range(len(args['mesh']))])
    ri=manta.PeriodicRIQBuilder(manta.PeriodicGDFTiles(args['basis'],args['auxiliary'],args['system'],
        np.zeros(3),k,domains.reciprocal_vectors,args['options'].fitting,args['budget']),2,args['options'].ri,args['budget'])
    envelope=result.receipt.accuracy_density
    family=manta.PeriodicRIFittingBoundOptions();family.density_l1_limit=envelope.density_l1_limit
    family.density_entry_limits=np.asarray(envelope.weighted_entry_bounds).ravel().tolist()
    family.j_error=accuracy.j_error;family.energy_error=accuracy.energy_error
    control=manta.PeriodicRIFittingBoundSource(ri,reference.source,family,args['budget'])
    assert control.receipt.certificate_identity==result.receipt.fitting_info.fitting_bound.certificate_identity


@pytest.mark.parametrize('field,quantity',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
    ('primitive_quartets','primitive_pairs'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_complete_count_admission_includes_reference_and_all_probes(tmp_path,field,quantity):
    _,args,reference,_=accuracy_job(tmp_path)
    plan=manta.plan_periodic_ri_rhf_from_cutoffs(**args)
    assert plan.rhf.fitting_accuracy.maximum_probes==len(args['mesh'])*args['basis'].nbasis**2
    assert plan.rhf.fitting_accuracy.shell_calls>=reference.plan.shell_calls
    limit=getattr(plan,quantity);setattr(args['budget'],field,limit-1)
    with pytest.raises((ValueError,OverflowError),match='budget'):
        manta.plan_periodic_ri_rhf_from_cutoffs(**args)
    setattr(args['budget'],field,limit)
    manta.plan_periodic_ri_rhf_from_cutoffs(**args)


def test_unmet_target_refuses_before_scf_and_dry_run_only_plans(tmp_path):
    job,args,_,accuracy=accuracy_job(tmp_path)
    accuracy.j_error=accuracy.energy_error=1e-20
    args['options'].fitting_accuracy=accuracy
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=accuracy)
    assert vq.run_periodic_job(**dict(job,dry_run=True)) is None
    with pytest.raises(RuntimeError,match='fitting error targets unmet'):
        vq.run_periodic_job(**job)
    assert not job['output'].with_suffix('.qvf').exists()


@pytest.mark.parametrize('field,value',[('reference',None),('j_error',0.),('energy_error',float('nan')),('maximum_probes',1)])
def test_invalid_or_inadequate_policy_refuses(tmp_path,field,value):
    job,args,_,accuracy=accuracy_job(tmp_path)
    setattr(accuracy,field,value);args['options'].fitting_accuracy=accuracy
    if field!='maximum_probes':
        with pytest.raises(ValueError,match='reference|targets'):
            dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=accuracy)
    with pytest.raises(ValueError,match='reference|targets|probe limit'):
        manta.plan_periodic_ri_rhf_from_cutoffs(**args)


def test_full_coulomb_certificate_keeps_fitted_exchange_owner(tmp_path):
    job,_,_,_=accuracy_job(tmp_path)
    job['exchange']=manta.ExchangeBuildOptions(algorithm='ri')
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],accuracy=None,
        coupling_multipole_pair_tree=False,coupling_multipole_error=0.)
    result=vq.run_periodic_job(**job)
    assert result.integral_provenance['fitting']['has_k']
    assert result.integral_provenance['fitting']['full_coulomb_accuracy']['targets_requested']
    control=vq.run_periodic_job(**dict(job,output=tmp_path/'control'/'job',
        periodic_fitting=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=None)))
    np.testing.assert_array_equal(result.native.fock,control.native.fock)
    assert result.energy==control.energy


def test_combined_component_policy_keeps_its_explicit_family(tmp_path):
    from tests.test_manta_periodic_ri_pair_tree import automatic_accuracy
    job,_,_,_=accuracy_job(tmp_path)
    component=automatic_accuracy(1e8,1e8,1e6);component.metric_reference_cutoff=1e-12
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],accuracy=component)
    result=vq.run_periodic_job(**job);info=result.integral_provenance['fitting']
    assert info['accuracy']['density_limit_origin']=='explicit'
    assert info['accuracy']['density_l1_limit']==1e6 and 'density_envelope' not in info['accuracy']
    assert info['full_coulomb_accuracy']['density_limit_origin']=='canonical-overlap'
    assert info['full_coulomb_accuracy']['density_l1_limit']<1e6
    assert info['accuracy']['metric_truncation']['metric_discarded_rank']==0


def test_reference_frame_mismatch_is_rejected_before_scf(tmp_path):
    _,args,_,_=accuracy_job(tmp_path)
    args['system'].lattice=np.asarray(args['system'].lattice)*1.01
    with pytest.raises(ValueError,match='frame mismatch'):
        manta.run_periodic_ri_rhf_from_cutoffs(**{k:v for k,v in args.items() if k!='budget'},
            integral_engine=manta.IntegralEngineOptions(backend='manta',budget=args['budget']))


def test_no_fitted_coulomb_cannot_claim_full_fitting_accuracy(tmp_path):
    _,args,_,_=accuracy_job(tmp_path)
    args['options'].coulomb_algorithm='exact';args['options'].exchange_algorithm='ri'
    args['options'].fitting.coupling_multipole_pair_tree=False
    args['options'].fitting.coupling_multipole_error=0.
    with pytest.raises(ValueError,match='requires fitted Coulomb'):
        manta.plan_periodic_ri_rhf_from_cutoffs(**args)


@pytest.mark.parametrize('method',['rhf','uhf'])
def test_full_s_reference_reaches_tighter_fitting_targets_without_new_support(tmp_path,method):
    job,args,reference,accuracy=accuracy_job(tmp_path,method)
    accuracy.j_error=.02;accuracy.energy_error=1.
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=accuracy)
    result=vq.run_periodic_job(**job);proof=result.integral_provenance['fitting']['full_coulomb_accuracy']
    assert result.converged and reference.receipt.pairs==6
    assert proof['j_error_bound']<.02 and proof['energy_error_bound']<1.
    assert proof['reference_remainder_j_bound']<5e-4
    # These same supports and canonical families previously needed J>18 and
    # energy>1000 because constant pair polynomials lost half their exponent.
    assert proof['finite_difference_j_bound']>1e-2



def metric_selection_job(directory, method='rhf', *, relative=False):
    job, args, reference, accuracy = accuracy_job(directory, method)
    metric = manta.RIOptions()
    metric.absolute_cutoff = 1e-12 if relative else 1e6
    metric.relative_cutoff = 2. if relative else 0.
    job['coulomb'] = dataclasses.replace(job['coulomb'], metric=metric)
    args['options'].ri = metric
    accuracy.maximum_metric_refinements = 1
    accuracy.metric_cutoff_reduction = 1e-18
    accuracy.minimum_metric_cutoff = 1e-12
    accuracy.j_error = .02 if method in ('rhf', 'uhf') else .01
    accuracy.energy_error = 1. if method in ('rhf', 'uhf') else 2.
    args['options'].fitting_accuracy = accuracy
    job['periodic_fitting'] = dataclasses.replace(job['periodic_fitting'], full_coulomb_accuracy=accuracy)
    return job, args, reference, accuracy


@pytest.mark.parametrize('method', ['rhf', 'uhf', 'rks', 'uks', 'pbe0', 'hse06'])
def test_metric_selection_keeps_passing_owner_and_matches_explicit_fixed_method(tmp_path, method):
    job, args, reference, accuracy = metric_selection_job(tmp_path, method)
    result = vq.run_periodic_job(**job)
    full = result.integral_provenance['fitting']['full_coulomb_accuracy']
    trials = full['metric_selection']['trials']
    assert result.converged and len(trials) == 2
    assert trials[0]['rank'] == 0 and trials[1]['rank'] > 0
    assert trials[0]['j_error_bound'] > accuracy.j_error
    assert trials[1]['certificate_identity'] == full['certificate_identity']
    assert trials[1]['ri_source_identity'] == result.integral_provenance['fitting']['metrics'][0]['source_identity']
    assert all(t['reference_source_identity'] == reference.source.source_identity for t in trials)
    assert full['j_error_bound'] <= accuracy.j_error and full['energy_error_bound'] <= accuracy.energy_error
    assert full['plan']['maximum_trials'] == 2
    metric = manta.RIOptions(); metric.absolute_cutoff = trials[-1]['cutoff']; metric.relative_cutoff = 0.
    accuracy.maximum_metric_refinements = 0
    control = vq.run_periodic_job(**dict(job, output=tmp_path/'fixed'/'job',
        coulomb=dataclasses.replace(job['coulomb'], metric=metric),
        periodic_fitting=dataclasses.replace(job['periodic_fitting'], full_coulomb_accuracy=accuracy)))
    assert control.energy == result.energy
    assert control.integral_provenance['fitting']['full_coulomb_accuracy']['certificate_identity'] == full['certificate_identity']
    for name in ('density', 'fock', 'hcore', 'overlap'):
        np.testing.assert_array_equal(getattr(result.native, name), getattr(control.native, name))
    assert result.native.receipt.source_identity == control.native.receipt.source_identity
    projection = result.integral_provenance['fitting']['projection']
    assert projection['requested_absolute_cutoff'] == 1e6 and projection['absolute_cutoff'] == trials[-1]['cutoff']
    assert validate_qvf(str(job['output'].with_suffix('.qvf')))['valid']
    assert not result.integral_provenance['symmetry_reduced']


def test_metric_selection_resolves_relative_cutoff_and_stops_on_first_pass(tmp_path):
    job, _, _, accuracy = metric_selection_job(tmp_path, relative=True)
    result = vq.run_periodic_job(**job)
    trials = result.integral_provenance['fitting']['full_coulomb_accuracy']['metric_selection']['trials']
    assert len(trials) == 2 and trials[0]['cutoff'] > 1e-12
    assert result.integral_provenance['fitting']['projection']['relative_cutoff'] == 0.
    # The initial full-rank source already passes: no refactorization or
    # rank increase is justified merely because further trials are allowed.
    job, _, _, accuracy = accuracy_job(tmp_path/'first')
    accuracy.maximum_metric_refinements = 3; accuracy.j_error = .02; accuracy.energy_error = 1.
    job['periodic_fitting'] = dataclasses.replace(job['periodic_fitting'], full_coulomb_accuracy=accuracy)
    result = vq.run_periodic_job(**job)
    assert len(result.integral_provenance['fitting']['full_coulomb_accuracy']['metric_selection']['trials']) == 1


@pytest.mark.parametrize('field,quantity', [('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
    ('primitive_quartets','primitive_pairs'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_metric_search_reserves_rejected_probes_and_replacements(tmp_path, field, quantity):
    _, args, _, accuracy = metric_selection_job(tmp_path)
    search = manta.plan_periodic_ri_rhf_from_cutoffs(**args)
    accuracy.maximum_metric_refinements = 0; args['options'].fitting_accuracy = accuracy
    fixed = manta.plan_periodic_ri_rhf_from_cutoffs(**args)
    assert getattr(search, quantity) > getattr(fixed, quantity)
    accuracy.maximum_metric_refinements = 1; args['options'].fitting_accuracy = accuracy
    required = getattr(search, quantity); setattr(args['budget'], field, required-1)
    with pytest.raises((ValueError, OverflowError), match='budget'):
        manta.plan_periodic_ri_rhf_from_cutoffs(**args)
    setattr(args['budget'], field, required)
    manta.plan_periodic_ri_rhf_from_cutoffs(**args)


@pytest.mark.parametrize('floor,reduction', [(1e6,1e-18),(1e-12,.9)])
def test_metric_selection_failure_and_dry_run_do_not_enter_scf(tmp_path, floor, reduction):
    job, _, _, accuracy = metric_selection_job(tmp_path)
    accuracy.minimum_metric_cutoff = floor; accuracy.metric_cutoff_reduction = reduction
    job['periodic_fitting'] = dataclasses.replace(job['periodic_fitting'], full_coulomb_accuracy=accuracy)
    assert vq.run_periodic_job(**dict(job, dry_run=True)) is None
    with pytest.raises(RuntimeError, match='targets unmet after bounded metric refinement'):
        vq.run_periodic_job(**job)
    assert not job['output'].with_suffix('.qvf').exists()


@pytest.mark.parametrize('field,value', [('maximum_metric_refinements',32),('metric_cutoff_reduction',1.),
    ('metric_cutoff_reduction',0.),('metric_cutoff_reduction',float('nan')),('minimum_metric_cutoff',-1.)])
def test_invalid_metric_search_policy_is_rejected_before_values(tmp_path, field, value):
    job, args, _, accuracy = metric_selection_job(tmp_path)
    setattr(accuracy, field, value); args['options'].fitting_accuracy = accuracy
    with pytest.raises(ValueError, match='metric refinement'):
        dataclasses.replace(job['periodic_fitting'], full_coulomb_accuracy=accuracy)
    with pytest.raises(ValueError, match='metric refinement'):
        manta.plan_periodic_ri_rhf_from_cutoffs(**args)


@pytest.mark.parametrize('component', [False, True])
def test_metric_search_does_not_silently_change_exchange_or_component_contract(tmp_path, component):
    _, args, _, _ = metric_selection_job(tmp_path)
    if component:
        from tests.test_manta_periodic_ri_pair_tree import automatic_accuracy
        args['options'].ri_accuracy = automatic_accuracy(1e8,1e8,1e6)
    else:
        args['options'].exchange_algorithm = 'ri'
        args['options'].fitting.coupling_multipole_pair_tree = False
        args['options'].fitting.coupling_multipole_error = 0.
    with pytest.raises(ValueError, match='J-only fitting without a separate component'):
        manta.plan_periodic_ri_rhf_from_cutoffs(**args)



def test_default_gradual_metric_search_keeps_first_passing_cutoff(tmp_path):
    job, _, _, accuracy = metric_selection_job(tmp_path, relative=True)
    accuracy.maximum_metric_refinements = 8; accuracy.metric_cutoff_reduction = .1
    job['periodic_fitting'] = dataclasses.replace(job['periodic_fitting'], full_coulomb_accuracy=accuracy)
    result = vq.run_periodic_job(**job)
    trials = result.integral_provenance['fitting']['full_coulomb_accuracy']['metric_selection']['trials']
    assert result.converged and 2 < len(trials) <= 9
    assert trials[-1]['cutoff'] > accuracy.minimum_metric_cutoff
    for previous, current in zip(trials, trials[1:]):
        assert current['cutoff'] == .1*previous['cutoff']
        assert previous['j_error_bound'] > accuracy.j_error or previous['energy_error_bound'] > accuracy.energy_error
    selected = manta.RIOptions(); selected.absolute_cutoff = trials[-1]['cutoff']; selected.relative_cutoff = 0.
    control = vq.run_periodic_job(**dict(job, output=tmp_path/'fixed'/'job',
        coulomb=dataclasses.replace(job['coulomb'], metric=selected),
        periodic_fitting=dataclasses.replace(job['periodic_fitting'], full_coulomb_accuracy=None)))
    assert control.energy == result.energy and control.native.receipt.source_identity == result.native.receipt.source_identity
    np.testing.assert_array_equal(control.native.fock, result.native.fock)
    np.testing.assert_array_equal(control.native.density, result.native.density)



def reference_selection_job(directory, method='rhf', *, compact=False, metric=False):
    """No prepared BUPO reference is supplied to the automatic consumer."""
    from tests.test_manta_periodic_ri_metric_bounds import metric_job
    if method in ('rhf', 'uhf'):
        job, args, _ = metric_job(directory, method, truncate=False)
        job['max_iter'] = args['options'].scf.maximum_iterations = 100
        args['options'].fitting.coupling_multipole_pair_tree = True
        args['options'].fitting.coupling_multipole_error = 1e-12
        args['options'].ri = job['coulomb'].metric
        job['periodic_fitting'] = dataclasses.replace(job['periodic_fitting'], accuracy=None, coupling_multipole_error=1e-12)
    else:
        job, args, _, _, _ = hierarchy_ks_job(directory, 2 if method == 'uks' else 1, 100)
        if method in ('pbe0', 'hse06'):
            job.update(functional=method, exchange=manta.ExchangeBuildOptions(algorithm='exact'), exchange_exxdiv='ewald')
    args['budget'].memory_bytes = job['integral_engine'].budget.memory_bytes = 4*2**30
    policy = manta.PeriodicRIFittingReferenceOptions()
    if compact:
        from tests.test_manta_periodic_pairs import options as pair_options
        from tests.test_manta_periodic_bubbles import bubble_options
        initial = policy.initial
        initial.pairs = pair_options(0, None, .2)
        initial.bubbles = bubble_options()
        initial.pair_cube_radius = 1; initial.real_cube_radius = 3; initial.reciprocal_cube_radius = 5
        policy.initial = initial
    accuracy = manta.PeriodicRIFittingAccuracyOptions(); accuracy.reference_selection = policy
    accuracy.j_error = .02 if method in ('rhf', 'uhf') else .01
    accuracy.energy_error = 1. if method in ('rhf', 'uhf') else 2.
    if metric:
        ri = manta.RIOptions(); ri.absolute_cutoff = 1e6; ri.relative_cutoff = 0.
        args['options'].ri = ri; job['coulomb'] = dataclasses.replace(job['coulomb'], metric=ri)
        accuracy.maximum_metric_refinements = 1; accuracy.metric_cutoff_reduction = 1e-18
    args['options'].fitting_accuracy = accuracy
    job['periodic_fitting'] = dataclasses.replace(job['periodic_fitting'], full_coulomb_accuracy=accuracy)
    return job, args, accuracy


def reference_budget_job(directory, profile, *, supported=True):
    """Explicit reservations for the two bounded reference-search regressions."""
    if profile not in ('joint-metric', 'component'):
        raise ValueError('Unknown bounded reference budget profile')
    job, args, accuracy = reference_selection_job(directory, compact=profile=='component', metric=profile=='joint-metric')
    policy = accuracy.reference_selection
    if profile == 'joint-metric':
        policy.selective_refinement = False
    if supported:
        budget = policy.budget
        # Conservative producer plans charge rejected references and reserve
        # retained sources. These are profile caps, not larger accuracy targets.
        if profile == 'joint-metric':
            budget.scalar_terms = 2*10**17
            budget.kernel_terms = 15*10**17
        else:
            budget.memory_bytes = 1100*2**20
        policy.budget = budget
    accuracy.reference_selection = policy
    args['options'].fitting_accuracy = accuracy
    fitting = dataclasses.replace(job['periodic_fitting'], full_coulomb_accuracy=accuracy)
    if profile == 'component':
        from tests.test_manta_periodic_ri_pair_tree import automatic_accuracy
        component = automatic_accuracy(1e8, 1e8, 1e6); component.metric_reference_cutoff = 1e-12
        fitting = dataclasses.replace(fitting, accuracy=component)
    job['periodic_fitting'] = fitting
    return job, args, accuracy


@pytest.mark.parametrize('profile', ['joint-metric', 'component'])
def test_reference_profile_caps_preserve_original_refusals_and_exact_admission(tmp_path, profile):
    job, _, _ = reference_budget_job(tmp_path/'original', profile, supported=False)
    message = 'scalar-work budget' if profile == 'joint-metric' else 'memory budget'
    with pytest.raises(ValueError, match=message):
        vq.run_periodic_job(**job)
    assert not job['output'].with_suffix('.qvf').exists()
    job, _, accuracy = reference_budget_job(tmp_path/'supported', profile)
    baseline = vq.run_periodic_job(**job)
    assert baseline.converged
    charged = baseline.integral_provenance['fitting']['full_coulomb_accuracy']['reference_selection']['charged']
    fields = [('memory_bytes','modeled_peak_bytes'), ('shell_calls','shell_calls'),
        ('primitive_quartets','primitive_products'), ('scalar_terms','scalar_terms'), ('kernel_terms','kernel_terms')]
    policy = accuracy.reference_selection; budget = policy.budget
    for field, quantity in fields:
        setattr(budget, field, charged[quantity])
    policy.budget = budget; accuracy.reference_selection = policy
    exact = vq.run_periodic_job(**dict(job, output=tmp_path/'exact'/'job',
        periodic_fitting=dataclasses.replace(job['periodic_fitting'], full_coulomb_accuracy=accuracy)))
    assert exact.converged and exact.energy == baseline.energy
    for name in ('density','fock','hcore','overlap'):
        np.testing.assert_array_equal(getattr(exact.native,name),getattr(baseline.native,name))
    assert exact.integral_provenance['fitting']['full_coulomb_accuracy']['certificate_identity'] == baseline.integral_provenance['fitting']['full_coulomb_accuracy']['certificate_identity']
    for field, quantity in fields:
        policy = accuracy.reference_selection; budget = policy.budget
        setattr(budget, field, charged[quantity]-1)
        policy.budget = budget; accuracy.reference_selection = policy
        refused = dict(job, output=tmp_path/field/'job',
            periodic_fitting=dataclasses.replace(job['periodic_fitting'], full_coulomb_accuracy=accuracy))
        with pytest.raises((ValueError, OverflowError), match='budget'):
            vq.run_periodic_job(**refused)
        assert not refused['output'].with_suffix('.qvf').exists()
        setattr(budget, field, charged[quantity])
        policy.budget = budget; accuracy.reference_selection = policy


def selected_reference(args, accuracy, result):
    receipt = result.native.receipt.fitting_info.fitting_reference
    initial = accuracy.reference_selection.initial
    for field in ('pair_cube_radius', 'real_cube_radius', 'reciprocal_cube_radius', 'near_cube_radius', 'eta'):
        setattr(initial, field, getattr(receipt, field))
    pairs = initial.pairs; pairs.pair_cutoff_bohr = receipt.pair_cutoff_bohr; initial.pairs = pairs
    bupo = initial.bupo
    bupo.integral_tolerance = receipt.integral_tolerance
    initial.bupo = bupo
    return manta.prepare_periodic_bupo_source(args['basis'], args['system'], args['mesh'], initial, 64, args['budget'])


@pytest.mark.parametrize('method', ['rhf', 'uhf', 'rks', 'uks', 'pbe0', 'hse06'])
def test_automatic_reference_matches_explicit_selected_source(tmp_path, method):
    job, args, accuracy = reference_selection_job(tmp_path, method, compact=True)
    result = vq.run_periodic_job(**job); full = result.integral_provenance['fitting']['full_coulomb_accuracy']
    selection = full['reference_selection']; assert result.converged and len(selection['trials']) == 1
    assert selection['trials'][-1]['ready'] and accuracy.reference is None
    assert full['j_error_bound'] <= accuracy.j_error and full['energy_error_bound'] <= accuracy.energy_error
    reference = selected_reference(args, accuracy, result)
    fixed = manta.PeriodicRIFittingAccuracyOptions(); fixed.reference = reference
    fixed.j_error = accuracy.j_error; fixed.energy_error = accuracy.energy_error
    control = vq.run_periodic_job(**dict(job, output=tmp_path/'fixed'/'job', periodic_fitting=dataclasses.replace(
        job['periodic_fitting'], full_coulomb_accuracy=fixed)))
    assert control.energy == result.energy
    assert control.native.receipt.source_identity == result.native.receipt.source_identity
    assert full['certificate_identity'] == control.integral_provenance['fitting']['full_coulomb_accuracy']['certificate_identity']
    for name in ('density', 'fock', 'hcore', 'overlap'):
        np.testing.assert_array_equal(getattr(result.native, name), getattr(control.native, name))
    assert validate_qvf(str(job['output'].with_suffix('.qvf')))['valid']
    assert not result.integral_provenance['symmetry_reduced'] and not full['roundoff_included']
    assert full['ri_source_identity'] == result.integral_provenance['fitting']['metrics'][0]['source_identity']


def test_joint_reference_refinement_then_metric_reuses_actual_owner(tmp_path):
    job, _, accuracy = reference_budget_job(tmp_path, 'joint-metric')
    result = vq.run_periodic_job(**job); full = result.integral_provenance['fitting']['full_coulomb_accuracy']
    refs = full['reference_selection']['trials']; metrics = full['metric_selection']['trials']
    assert result.converged and 1 < len(refs) <= 5 and len(metrics) == 2
    assert all(not t['ready'] for t in refs[:-1]) and refs[-1]['ready']
    assert len({t['bound']['ri_source_identity'] for t in refs}) == 1
    assert len({t['reference_source_identity'] for t in metrics}) == 1
    assert metrics[0]['rank'] == 0 and metrics[1]['rank'] > 0
    assert metrics[-1]['certificate_identity'] == full['certificate_identity']
    assert refs[-1]['bound']['reference_remainder_j_bound'] <= accuracy.reference_selection.reference_fraction*accuracy.j_error
    for before, after in zip(refs, refs[1:]):
        assert after['preparation']['real_cube_radius'] == before['preparation']['real_cube_radius']+1
        assert after['preparation']['pair_cutoff_bohr'] > before['preparation']['pair_cutoff_bohr']
    assert full['j_error_bound'] <= accuracy.j_error and full['energy_error_bound'] <= accuracy.energy_error
    for field, quantity in [('memory_bytes','modeled_peak_bytes'), ('shell_calls','shell_calls'),
        ('primitive_quartets','primitive_products'), ('scalar_terms','scalar_terms'), ('kernel_terms','kernel_terms')]:
        assert full['reference_selection']['charged'][quantity] <= full['reference_selection']['reservation'][field]


@pytest.mark.parametrize('field,quantity', [('memory_bytes','modeled_peak_bytes'), ('shell_calls','shell_calls'),
    ('primitive_quartets','primitive_pairs'), ('scalar_terms','scalar_terms'), ('kernel_terms','kernel_terms')])
def test_automatic_reference_reservation_is_admitted_by_complete_count_plan(tmp_path, field, quantity):
    _, args, _ = reference_selection_job(tmp_path, compact=True)
    plan = manta.plan_periodic_ri_rhf_from_cutoffs(**args); required = getattr(plan, quantity)
    setattr(args['budget'], field, required-1)
    with pytest.raises((ValueError, OverflowError), match='budget'):
        manta.plan_periodic_ri_rhf_from_cutoffs(**args)
    setattr(args['budget'], field, required); manta.plan_periodic_ri_rhf_from_cutoffs(**args)


@pytest.mark.parametrize('field,quantity', [('memory_bytes','modeled_peak_bytes'), ('shell_calls','shell_calls'),
    ('primitive_quartets','primitive_products'), ('scalar_terms','scalar_terms'), ('kernel_terms','kernel_terms')])
def test_automatic_reference_cumulative_internal_budget_refuses(tmp_path, field, quantity):
    job, _, accuracy = reference_selection_job(tmp_path, compact=True, metric=True)
    result = vq.run_periodic_job(**job)
    charged = result.integral_provenance['fitting']['full_coulomb_accuracy']['reference_selection']['charged'][quantity]
    policy = accuracy.reference_selection; budget = policy.budget; setattr(budget, field, charged-1)
    policy.budget = budget; accuracy.reference_selection = policy
    job.update(output=tmp_path/'refused'/'job', periodic_fitting=dataclasses.replace(job['periodic_fitting'], full_coulomb_accuracy=accuracy))
    with pytest.raises((ValueError, OverflowError), match='budget'):
        vq.run_periodic_job(**job)
    assert not job['output'].with_suffix('.qvf').exists()


def test_reference_limits_and_dry_run_do_not_claim_accuracy(tmp_path):
    job, _, accuracy = reference_selection_job(tmp_path)
    policy = accuracy.reference_selection; policy.maximum_refinements = 0; accuracy.reference_selection = policy
    job['periodic_fitting'] = dataclasses.replace(job['periodic_fitting'], full_coulomb_accuracy=accuracy)
    assert vq.run_periodic_job(**dict(job, dry_run=True)) is None
    with pytest.raises(RuntimeError, match='reference targets unmet'):
        vq.run_periodic_job(**job)
    assert not job['output'].with_suffix('.qvf').exists()


@pytest.mark.parametrize('field,value', [('maximum_refinements',32), ('maximum_cube_radius',0),
    ('growth_factor',1.), ('reference_fraction',0.), ('maximum_pair_cutoff_bohr',0.)])
def test_invalid_reference_selection_policy_refuses_before_values(tmp_path, field, value):
    job, args, accuracy = reference_selection_job(tmp_path)
    policy = accuracy.reference_selection; setattr(policy, field, value); accuracy.reference_selection = policy
    args['options'].fitting_accuracy = accuracy
    with pytest.raises(ValueError, match='reference selection'):
        dataclasses.replace(job['periodic_fitting'], full_coulomb_accuracy=accuracy)
    with pytest.raises(ValueError, match='reference selection'):
        manta.plan_periodic_ri_rhf_from_cutoffs(**args)


def test_automatic_reference_keeps_fitted_exchange_and_component_contracts(tmp_path):
    job, _, accuracy = reference_selection_job(tmp_path, compact=True)
    job['exchange'] = manta.ExchangeBuildOptions(algorithm='ri')
    job['periodic_fitting'] = dataclasses.replace(job['periodic_fitting'], coupling_multipole_pair_tree=False, coupling_multipole_error=0.)
    result = vq.run_periodic_job(**job)
    assert result.integral_provenance['fitting']['has_k'] and result.converged
    job, _, accuracy = reference_budget_job(tmp_path/'component', 'component')
    result = vq.run_periodic_job(**job)
    assert result.integral_provenance['fitting']['accuracy']['density_limit_origin'] == 'explicit'
    assert result.integral_provenance['fitting']['full_coulomb_accuracy']['density_limit_origin'] == 'canonical-overlap'


@pytest.mark.parametrize('invalid', ['external-live', 'missing-domain-bound', 'missing-image-bound', 'both-references'])
def test_automatic_reference_requires_closed_full_reference_policy(tmp_path, invalid):
    job, args, accuracy = reference_selection_job(tmp_path, compact=True)
    policy = accuracy.reference_selection
    if invalid == 'external-live':
        budget = policy.budget; budget.other_live_bytes = 1; policy.budget = budget
    elif invalid == 'both-references':
        accuracy.reference = prepared_fitting_reference(args)
    else:
        initial = policy.initial; bupo = initial.bupo
        setattr(bupo, 'bound_infinite_domain' if invalid == 'missing-domain-bound' else 'bound_pair_images', False)
        initial.bupo = bupo; policy.initial = initial
    accuracy.reference_selection = policy; args['options'].fitting_accuracy = accuracy
    with pytest.raises(ValueError, match='reference'):
        dataclasses.replace(job['periodic_fitting'], full_coulomb_accuracy=accuracy)
    with pytest.raises(ValueError, match='reference'):
        manta.plan_periodic_ri_rhf_from_cutoffs(**args)


@pytest.mark.parametrize('field', ['memory_bytes', 'scalar_terms', 'kernel_terms'])
def test_automatic_reference_count_plan_rejects_insufficient_setup(tmp_path, field):
    _, args, accuracy = reference_selection_job(tmp_path, compact=True)
    policy = accuracy.reference_selection; budget = policy.budget
    setattr(budget, field, budget.backend_margin_bytes+1 if field == 'memory_bytes' else 1)
    policy.budget = budget; accuracy.reference_selection = policy; args['options'].fitting_accuracy = accuracy
    with pytest.raises(ValueError, match='reference search setup.*budget'):
        manta.plan_periodic_ri_rhf_from_cutoffs(**args)


def test_automatic_reference_larger_search_margin_keeps_ri_leaf_plan(tmp_path):
    job, _, accuracy = reference_selection_job(tmp_path, compact=True, metric=True)
    baseline = vq.run_periodic_job(**job)
    policy = accuracy.reference_selection; budget = policy.budget
    budget.backend_margin_bytes = job['integral_engine'].budget.backend_margin_bytes + 2**20
    policy.budget = budget; accuracy.reference_selection = policy
    result = vq.run_periodic_job(**dict(job, output=tmp_path/'larger-margin'/'job',
        periodic_fitting=dataclasses.replace(job['periodic_fitting'], full_coulomb_accuracy=accuracy)))
    a = baseline.integral_provenance['fitting']['full_coulomb_accuracy']
    b = result.integral_provenance['fitting']['full_coulomb_accuracy']
    assert result.energy == baseline.energy and b['certificate_identity'] == a['certificate_identity']
    np.testing.assert_array_equal(result.native.fock, baseline.native.fock)
    assert b['reference_selection']['charged']['modeled_peak_bytes'] > a['reference_selection']['charged']['modeled_peak_bytes']
    assert result.native.plan.fock.gamma.modeled_peak_bytes == baseline.native.plan.fock.gamma.modeled_peak_bytes
    assert b['reference_selection']['reservation']['backend_margin_bytes'] == budget.backend_margin_bytes
    assert b['reference_preparation'] == b['reference_selection']['trials'][-1]['preparation']
    for name in ('pairs', 'nodes', 'order', 'opening', 'integral_tolerance', 'pointwise_tolerance', 'maximum_radius'):
        assert b['reference_preparation'][name] == getattr(result.native.receipt.fitting_info.fitting_reference, name)


def full_coulomb_ledger_job(directory):
    """Reuse the admitted component/reference profile and existing physical controls."""
    from benchmarks.manta_release_controls import make_job
    job, args, accuracy = reference_budget_job(directory, 'component')
    physical = make_job(directory/'physical-template', 'ordinary-rhf')['periodic_integrals']
    job['periodic_integrals'] = dataclasses.replace(job['periodic_integrals'],
        one_electron_accuracy=physical.one_electron_accuracy,
        nuclear_accuracy=physical.nuclear_accuracy, exchange_accuracy=physical.exchange_accuracy)
    return job, args, accuracy


def test_full_coulomb_ledger_reuses_native_family_and_replaces_coupling(tmp_path):
    from fractions import Fraction
    job, _, _ = full_coulomb_ledger_job(tmp_path)
    result = vq.run_periodic_job(**job); assert result.converged
    e = result.integral_provenance; full = e['fitting']['full_coulomb_accuracy']; ledger = e['component_ledger']
    assert full['qualified'] and ledger['qualified']
    assert full['density_envelope_identity'] == result.native.receipt.fitting_info.fitting_bound.density_envelope_identity
    assert full['density_envelope_identity'] == e['one_electron_accuracy']['density_envelope_identity'] == e['nuclear_accuracy']['density_envelope_identity']
    assert full['ri_source_identity'] == e['fitting']['metrics'][0]['source_identity']
    coulomb = [c for c in ledger['components'] if c['component'] == 'coulomb']
    assert len(coulomb) == 1 and coulomb[0]['supersedes'] == ['fitting_accuracy']
    assert not any(c['component'] == 'coulomb-fitting' for c in ledger['components'])
    assert coulomb[0]['energy_error_bound'] == full['energy_error_bound']
    exact = sum((Fraction.from_float(c['energy_error_bound']) for c in ledger['components'] if c['requested']), Fraction())
    assert Fraction.from_float(ledger['fixed_density_energy_error_bound']) >= exact
    assert ledger['fixed_density_energy_error_bound'] < float(exact) + e['fitting']['accuracy']['energy_bound']
    assert 'hartree-omitted-domains' not in ledger['excludes']
    assert {'integral-roundoff', 'rank-stability', 'density-relaxation', 'overlap-rank-stability'} <= set(ledger['excludes'])
    assert ledger['relaxation_delimiter']['fock_entry_error_bounds']['coulomb'] == full['j_error_bound']
    assert not ledger['relaxation_delimiter']['is_certificate']
    control = vq.run_periodic_job(**dict(job, output=tmp_path/'control'/'job',
        periodic_fitting=dataclasses.replace(job['periodic_fitting'], full_coulomb_accuracy=None)))
    assert control.converged and result.energy == control.energy
    for name in ('density', 'fock', 'hcore', 'overlap'):
        np.testing.assert_array_equal(getattr(result.native, name), getattr(control.native, name))
    assert 'hartree-omitted-domains' in control.integral_provenance['component_ledger']['excludes']
    assert validate_qvf(str(job['output'].with_suffix('.qvf')))['valid']


def ledger_full_evidence():
    from tests.test_manta_periodic_ri_rhf import _ledger_evidence
    e = _ledger_evidence()
    e['fitting'] = dict(metrics=[dict(source_identity='gamma')], accuracy=dict(enabled=True, qualified=True, energy_bound=20.),
        full_coulomb_accuracy=dict(qualified=True, targets_requested=True,
            qualification='full-zero-average-coulomb-on-fixed-weighted-hermitian-density-family;stored-integrals',
            density_limit_origin='canonical-overlap', component_density_envelope=True, density_envelope_identity='env-a',
            ri_source_identity='gamma', reference_source_identity='reference', certificate_identity='full',
            energy_error_bound=.5, j_error_bound=.1, excludes=['integral-contraction-arithmetic', 'rank-stability', 'density-relaxation']))
    return e


@pytest.mark.parametrize('change', ['source', 'missing-source', 'family', 'missing-family', 'noncanonical',
    'scalar-family', 'unqualified', 'unsupported-reference', 'invalid-energy', 'invalid-matrix', 'coupling-unqualified'])
def test_full_coulomb_ledger_refuses_incompatible_receipts(change):
    import json
    e = ledger_full_evidence(); full = e['fitting']['full_coulomb_accuracy']
    if change == 'source': full['ri_source_identity'] = 'other'
    elif change == 'missing-source': e['fitting']['metrics'] = []
    elif change == 'family': full['density_envelope_identity'] = 'other'
    elif change == 'missing-family': full.pop('density_envelope_identity')
    elif change == 'noncanonical': full['density_limit_origin'] = 'explicit'
    elif change == 'scalar-family': full['component_density_envelope'] = False
    elif change == 'unqualified': full['qualified'] = False
    elif change == 'unsupported-reference': full['qualification'] = 'finite-reference-only'
    elif change == 'invalid-energy': full['energy_error_bound'] = float('nan')
    elif change == 'invalid-matrix': full['j_error_bound'] = -1.
    elif change == 'coupling-unqualified': e['fitting']['accuracy']['qualified'] = False
    ledger = manta.compose_fixed_density_ledger(e, method='uhf')
    assert not ledger['qualified'] and ledger['fixed_density_energy_error_bound'] is None
    assert ledger['reasons'] and 'hartree-omitted-domains' in ledger['excludes']
    json.dumps(ledger, allow_nan=False)


def test_full_coulomb_ledger_requires_requested_targets_and_keeps_single_contribution():
    e = ledger_full_evidence()
    ledger = manta.compose_fixed_density_ledger(e, method='uhf')
    assert ledger['qualified'] and ledger['fixed_density_energy_error_bound'] < 1.
    assert ledger['components'][-1]['certificate'] == 'full_coulomb_accuracy'
    e['fitting']['full_coulomb_accuracy']['targets_requested'] = False
    ledger = manta.compose_fixed_density_ledger(e, method='uhf')
    assert ledger['qualified'] and ledger['fixed_density_energy_error_bound'] >= 20.
    assert ledger['components'][-1]['certificate'] == 'fitting_accuracy'
    assert 'hartree-omitted-domains' in ledger['excludes']
