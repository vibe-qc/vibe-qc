"""Bounded physical RI J domains, shared producers and complete-method owners."""
import dataclasses
import itertools
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import vibeqc as vq
from vibeqc import manta
from vibeqc.output import validate_qvf
from tests.test_manta_periodic_ri_fitting_accuracy import reference_selection_job, selected_reference


def domain_job(directory, method='rhf', *, metric=False, explicit=False):
    job, args, accuracy = reference_selection_job(directory, method, compact=True, metric=metric)
    job['periodic_integrals'] = dataclasses.replace(job['periodic_integrals'], reciprocal_energy_cutoff=1.)
    args['reciprocal_energy_cutoff'] = 1.
    args['options'].fitting.auxiliary_cutoff_bohr = 4.
    args['options'].fitting.coupling_multipole_pair_tree = False
    args['options'].fitting.coupling_multipole_error = 0.
    job['periodic_fitting'] = dataclasses.replace(job['periodic_fitting'], auxiliary_cutoff_bohr=4.,
        coupling_multipole_pair_tree=False, coupling_multipole_error=0.)
    policy = manta.PeriodicRIFittingDomainOptions(); policy.maximum_refinements = 2
    accuracy.domain_selection = policy; accuracy.j_error = .02; accuracy.energy_error = 2.
    if explicit:
        from tests.test_manta_periodic_ri_fitting_bounds import prepared_fitting_reference
        physical = args['options'].ks if hasattr(args['options'], 'ks') else args['options']
        accuracy.reference = prepared_fitting_reference(dict(args, options=SimpleNamespace(pair_cutoff_bohr=physical.pair_cutoff_bohr)))
        accuracy.reference_selection = None
    args['options'].fitting_accuracy = accuracy
    job['periodic_fitting'] = dataclasses.replace(job['periodic_fitting'], full_coulomb_accuracy=accuracy)
    return job, args, accuracy


def fixed_job(job, args, accuracy, result, directory):
    selected = result.native.receipt.fitting_info.domain_trials[-1]
    fixed = manta.PeriodicRIFittingAccuracyOptions()
    fixed.reference = accuracy.reference or selected_reference(args, accuracy, result)
    fixed.j_error = accuracy.j_error; fixed.energy_error = accuracy.energy_error
    domain = accuracy.domain_selection; domain.maximum_refinements = 0
    domain.initial_pair_cutoff_bohr = selected.pair_cutoff_bohr
    domain.initial_auxiliary_cutoff_bohr = selected.auxiliary_cutoff_bohr
    domain.initial_reciprocal_cube_radius = selected.reciprocal_cube_radius
    fixed.domain_selection = domain
    ri = job['coulomb'].metric
    if selected.metric_refinement:
        ri = manta.RIOptions(); ri.absolute_cutoff = selected.metric_cutoff; ri.relative_cutoff = 0.
    return dict(job, output=directory/'job', coulomb=dataclasses.replace(job['coulomb'], metric=ri),
        periodic_fitting=dataclasses.replace(job['periodic_fitting'], full_coulomb_accuracy=fixed))


def source_inputs(args, method='rhf'):
    domains = getattr(manta, 'build_periodic_ri_'+method+'_domains')(**args)
    k = np.array([args['system'].reciprocal_lattice()@args['mesh'].fractional(args['mesh'].address(i)) for i in range(len(args['mesh']))])
    return dict(orbital=args['basis'], auxiliary=args['auxiliary'], system=args['system'], q=np.zeros(3),
        ket_kpoints=k, vectors=domains.reciprocal_vectors, options=args['options'].fitting, budget=args['budget'])


def enlarged_inputs(a, trial):
    # Independent integer union, keeping all original numerical vectors.
    inverse = a['system'].lattice.T/(2*np.pi)
    labels = {tuple(np.rint(inverse@g).astype(int)) for g in a['vectors']}
    r = trial.reciprocal_cube_radius
    extra = [a['system'].reciprocal_lattice()@np.array(n) for n in itertools.product(range(-r,r+1), repeat=3)
             if any(n) and n not in labels]
    vectors = np.ascontiguousarray(np.concatenate([a['vectors'], np.array(extra).reshape(-1,3)]))
    options = manta.PeriodicGDFShortRangeOptions()
    for field in ('omega', 'pair_cutoff_bohr', 'auxiliary_cutoff_bohr', 'integral_screen_error', 'coupling_screen_error',
                  'coupling_multipole_error', 'coupling_multipole_order', 'coupling_multipole_hierarchy',
                  'coupling_multipole_reuse', 'coupling_multipole_compress', 'coupling_multipole_global_actions',
                  'coupling_multipole_partition_bytes', 'coupling_multipole_extents', 'coupling_multipole_pair_tree',
                  'image_candidate_cap', 'compute_metric'):
        setattr(options, field, getattr(a['options'], field))
    options.pair_cutoff_bohr = trial.pair_cutoff_bohr; options.auxiliary_cutoff_bohr = trial.auxiliary_cutoff_bohr
    return dict(a, vectors=vectors, options=options)


@pytest.mark.parametrize('method', ['rhf', 'uhf', 'rks', 'uks', 'pbe0', 'hse06'])
def test_domains_select_actual_owner_and_preserve_other_components(tmp_path, method):
    job, args, accuracy = domain_job(tmp_path, method)
    result = vq.run_periodic_job(**job)
    full = result.integral_provenance['fitting']['full_coulomb_accuracy']; trials = full['domain_selection']['trials']
    assert result.converged and 1 < len(trials) <= 3
    assert all(t['bound']['j_error_bound'] > accuracy.j_error or t['bound']['energy_error_bound'] > accuracy.energy_error for t in trials[:-1])
    assert full['j_error_bound'] <= accuracy.j_error and full['energy_error_bound'] <= accuracy.energy_error
    assert len({t['bound']['reference_source_identity'] for t in trials}) == 1
    assert len({t['bound']['ri_source_identity'] for t in trials}) == len(trials)
    assert full['ri_source_identity'] == result.native.receipt.fitting_info.metrics[0].source_identity
    control = vq.run_periodic_job(**fixed_job(job,args,accuracy,result,tmp_path/'fixed'))
    assert result.energy == control.energy and result.native.receipt.source_identity == control.native.receipt.source_identity
    assert full['certificate_identity'] == control.integral_provenance['fitting']['full_coulomb_accuracy']['certificate_identity']
    for name in ('density','fock','hcore','overlap'):
        np.testing.assert_array_equal(getattr(result.native,name),getattr(control.native,name))
    assert validate_qvf(str(job['output'].with_suffix('.qvf')))['valid']
    assert not result.integral_provenance['symmetry_reduced'] and not full['roundoff_included']
    selected = result.integral_provenance['fitting']['selected_plan']; capacity = result.integral_provenance['fitting']['plan']
    assert selected['modeled_peak_bytes'] <= capacity['modeled_peak_bytes']
    assert result.integral_provenance['fitting']['controls']['pair_cutoff_bohr_requested'] == job['periodic_integrals'].pair_cutoff_bohr


def test_nested_metric_trials_reset_controls_and_keep_selected_certificate(tmp_path):
    job, args, accuracy = domain_job(tmp_path, metric=True, explicit=True)
    result = vq.run_periodic_job(**job)
    full = result.integral_provenance['fitting']['full_coulomb_accuracy']; trials = full['domain_selection']['trials']
    assert len(trials) == 6 and [t['rank'] for t in trials] == [0,3,0,3,0,3]
    assert [t['domain_attempt'] for t in trials] == [0,0,1,1,2,2]
    assert [t['metric_refinement'] for t in trials] == [0,1,0,1,0,1]
    control = vq.run_periodic_job(**fixed_job(job,args,accuracy,result,tmp_path/'fixed'))
    assert result.energy == control.energy and result.native.receipt.source_identity == control.native.receipt.source_identity
    for field,quantity in [('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
        ('primitive_quartets','primitive_products'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')]:
        assert full['domain_selection']['charged'][quantity] <= full['domain_selection']['effective_reservation'][field]


def test_shared_domain_source_matches_independent_union_integrals_and_actions(tmp_path):
    from tests.test_manta_periodic_gdf import legacy
    from tests.test_manta_periodic_gdf_tiles import assemble
    _, args, _ = domain_job(tmp_path)
    a = source_inputs(args); original = manta.PeriodicGDFTiles(**a)
    trial = SimpleNamespace(pair_cutoff_bohr=6.75, auxiliary_cutoff_bohr=9., reciprocal_cube_radius=2)
    source = original.with_domains(6.75,9.,2,a['budget']); independent = enlarged_inputs(a,trial)
    np.testing.assert_array_equal(source.vectors[:len(a['vectors'])],a['vectors'])
    np.testing.assert_allclose(source.vectors,independent['vectors'],atol=2e-15,rtol=2e-15)
    m,t,_ = legacy(independent); actual_m,actual_t = source.metric().values,assemble(source)
    np.testing.assert_allclose(actual_m,m,atol=3e-12,rtol=3e-12)
    np.testing.assert_allclose(actual_t,t,atol=3e-12,rtol=3e-12)
    rng=np.random.default_rng(781);x=(rng.normal(size=t[:,0].shape)+1j*rng.normal(size=t[:,0].shape))
    out=source.forward(np.ascontiguousarray(x),a['budget'])
    np.testing.assert_allclose(out.values,np.einsum('kpij,kij->p',t.conj(),x).ravel(),atol=3e-12,rtol=3e-12)
    assert source.orbital_basis_identity == original.orbital_basis_identity and source.metric_source_identity != original.metric_source_identity
    snapshot=source.vectors;snapshot[:]=0;assert np.any(source.vectors)


@pytest.mark.parametrize('field,quantity', [('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
    ('primitive_quartets','primitive_pairs'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_complete_capacity_and_search_caps_precede_values(tmp_path, field, quantity):
    _,args,a=domain_job(tmp_path);p=manta.plan_periodic_ri_rhf_from_cutoffs(**args)
    required=getattr(p,quantity);setattr(args['budget'],field,required-1)
    with pytest.raises((ValueError,OverflowError),match='budget'):manta.plan_periodic_ri_rhf_from_cutoffs(**args)
    setattr(args['budget'],field,required);manta.plan_periodic_ri_rhf_from_cutoffs(**args)
    a.domain_selection=None;args['options'].fitting_accuracy=a
    fixed=manta.plan_periodic_ri_rhf_from_cutoffs(**args)
    assert getattr(p.rhf.fock.gamma,'modeled_peak_bytes') > getattr(fixed.rhf.fock.gamma,'modeled_peak_bytes')
    assert getattr(p,quantity) > getattr(fixed,quantity)


@pytest.mark.parametrize('field,quantity', [('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
    ('primitive_quartets','primitive_products'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_joint_search_caps_include_discarded_domains(tmp_path, field, quantity):
    job,_,a=domain_job(tmp_path,explicit=True)
    result=vq.run_periodic_job(**job)
    charged=result.integral_provenance['fitting']['full_coulomb_accuracy']['domain_selection']['charged'][quantity]
    d=a.domain_selection;b=d.budget;setattr(b,field,charged-1);d.budget=b;a.domain_selection=d
    job.update(output=tmp_path/'refused'/'job',periodic_fitting=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=a))
    with pytest.raises((ValueError,OverflowError),match='budget'):vq.run_periodic_job(**job)
    assert not job['output'].with_suffix('.qvf').exists()


def test_domain_limit_refuses_and_dry_run_claims_no_certificate(tmp_path):
    job,_,a=domain_job(tmp_path);d=a.domain_selection;d.maximum_refinements=1;a.domain_selection=d
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=a)
    assert vq.run_periodic_job(**dict(job,dry_run=True)) is None
    with pytest.raises(RuntimeError,match='physical fitting domain/metric limit'):vq.run_periodic_job(**job)
    assert not job['output'].with_suffix('.qvf').exists()


@pytest.mark.parametrize('field,value',[('maximum_refinements',32),('growth_factor',1.),('initial_pair_cutoff_bohr',float('nan')),
    ('maximum_auxiliary_cutoff_bohr',0.),('initial_reciprocal_cube_radius',129)])
def test_invalid_domain_controls_refuse_before_values(tmp_path,field,value):
    job,args,a=domain_job(tmp_path);d=a.domain_selection;setattr(d,field,value);a.domain_selection=d
    args['options'].fitting_accuracy=a
    with pytest.raises(ValueError,match='domain selection'):manta.plan_periodic_ri_rhf_from_cutoffs(**args)
    with pytest.raises(ValueError,match='domain selection'):dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=a)


@pytest.mark.parametrize('component',[False,True])
def test_domain_search_preserves_exchange_and_component_contracts(tmp_path,component):
    _,args,_=domain_job(tmp_path)
    if component:
        from tests.test_manta_periodic_ri_pair_tree import automatic_accuracy
        args['options'].ri_accuracy=automatic_accuracy(1e8,1e8,1e6)
    else:args['options'].exchange_algorithm='ri'
    with pytest.raises(ValueError,match='J-only fitting without a separate component'):manta.plan_periodic_ri_rhf_from_cutoffs(**args)


@pytest.mark.parametrize('field,quantity',[('memory_bytes','modeled_peak_bytes'),('shell_calls','shell_calls'),
    ('primitive_quartets','primitive_products'),('scalar_terms','scalar_terms'),('kernel_terms','kernel_terms')])
def test_shared_domain_generation_admits_all_five_caps(tmp_path,field,quantity):
    _,args,_=domain_job(tmp_path);a=source_inputs(args);s=manta.PeriodicGDFTiles(**a)
    p=s.plan_domains(4.5,6.,1,a['budget'])
    required=getattr(p.source,quantity) if field in ('shell_calls','primitive_quartets') else getattr(p,quantity)
    setattr(a['budget'],field,required-1)
    with pytest.raises((ValueError,OverflowError),match='budget'):s.with_domains(4.5,6.,1,a['budget'])
    setattr(a['budget'],field,required);s.with_domains(4.5,6.,1,a['budget'])


def test_nested_domain_controls_and_joint_margin_are_explicit(tmp_path):
    job,args,a=domain_job(tmp_path)
    d=a.domain_selection;d.initial_pair_cutoff_bohr=2.;a.domain_selection=d;args['options'].fitting_accuracy=a
    with pytest.raises(ValueError,match='initial radii'):manta.plan_periodic_ri_rhf_from_cutoffs(**args)
    d.initial_pair_cutoff_bohr=None;b=d.budget
    b.backend_margin_bytes=job['integral_engine'].budget.backend_margin_bytes+2**20
    b.shell_calls//=2;d.budget=b;a.domain_selection=d
    result=vq.run_periodic_job(**dict(job,periodic_fitting=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=a)))
    receipt=result.integral_provenance['fitting']['full_coulomb_accuracy']['domain_selection']
    assert receipt['effective_reservation']['shell_calls']==b.shell_calls
    assert receipt['effective_reservation']['backend_margin_bytes']==b.backend_margin_bytes
    control=vq.run_periodic_job(**fixed_job(job,args,a,result,tmp_path/'fixed'))
    assert result.energy==control.energy


def test_first_passing_domain_keeps_original_source_and_does_not_refine(tmp_path):
    job,args,a=domain_job(tmp_path);a.j_error=a.energy_error=1e6
    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=a)
    result=vq.run_periodic_job(**job)
    trials=result.native.receipt.fitting_info.domain_trials
    assert len(trials)==1 and trials[0].domain_attempt==0 and trials[0].vectors==8
    control=vq.run_periodic_job(**dict(job,output=tmp_path/'plain'/'job',periodic_fitting=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=None)))
    assert result.energy==control.energy and result.native.receipt.source_identity==control.native.receipt.source_identity
    for name in ('density','fock','hcore','overlap'):
        np.testing.assert_array_equal(getattr(result.native,name),getattr(control.native,name))
