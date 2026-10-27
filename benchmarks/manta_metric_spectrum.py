"""Counted spectral-certificate cost; synthetic Hermitian matrices, no QC reference."""
import argparse,json,tempfile,time
from pathlib import Path
import numpy as np
from vibeqc import manta


def run(dimensions=(2,8,32),repetitions=3):
    rows=[];rng=np.random.default_rng(7412)
    for n in dimensions:
        z=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
        a=z+z.conj().T;a+=n*np.eye(n)
        w,q=np.linalg.eigh(a);o=manta.RIOptions();o.absolute_cutoff=.01;o.relative_cutoff=0.
        b=manta.ResourceBudget();b.memory_bytes=2**30;b.scalar_terms=10**15;b.kernel_terms=10**17
        plan=manta.plan_hermitian_spectrum(n,o,b);times=[]
        for _ in range(repetitions):
            start=time.perf_counter();c=manta.certify_hermitian_spectrum(a,w,q,o,b);times.append(time.perf_counter()-start)
            assert c.spectrum_certified and c.rank_certified
        rows.append(dict(dimension=n,wall_seconds=times,rank=c.rank,
            eigenvalue_error_bound=c.eigenvalue_error_bound,rank_stability_radius=c.rank_stability_radius,
            plan={k:getattr(plan,k) for k in ('scalar_terms','kernel_terms','modeled_peak_bytes')}))
    return dict(scope='synthetic stored Hermitian metrics; certificate cost only',rows=rows)


def run_inverse(dimensions=(2,8,32),repetitions=3):
    rows=[];rng=np.random.default_rng(7412)
    for n in dimensions:
        z=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
        a=z+z.conj().T;a+=n*np.eye(n)
        w,q=np.linalg.eigh(a);o=manta.RIOptions();o.absolute_cutoff=.01;o.relative_cutoff=0.
        f=np.ascontiguousarray(q[:,w>.01]/np.sqrt(w[w>.01]))
        b=manta.ResourceBudget();b.memory_bytes=2**30;b.scalar_terms=10**15;b.kernel_terms=10**17
        plan=manta.plan_hermitian_inverse(n,o,b);times=[]
        for _ in range(repetitions):
            start=time.perf_counter();c=manta.certify_hermitian_inverse(a,w,q,f,o,b);times.append(time.perf_counter()-start)
            assert c.qualified
        rows.append(dict(dimension=n,wall_seconds=times,rank=c.spectrum.rank,
            inverse_error_bound=c.inverse_error_bound,factor_product_bound=c.factor_product_bound,
            plan={k:getattr(plan,k) for k in ('scalar_terms','kernel_terms','modeled_peak_bytes')}))
    return dict(scope='stored factor versus exact retained inverse; combined spectrum and factor audit cost',rows=rows)


def rank_case(directory,method='rhf',kind='fixed'):
    """One complete rank-driven search and the actual selected-source control."""
    import vibeqc as vq
    from vibeqc.output import validate_qvf
    from tests.test_manta_periodic_rank_admission import rank_job,rank_control
    job,args,accuracy=rank_job(directory,method,kind)
    start=time.perf_counter();result=vq.run_periodic_job(**job);elapsed=time.perf_counter()-start
    assert result.converged and validate_qvf(str(job['output'].with_suffix('.qvf')))['valid']
    full=result.integral_provenance['fitting']['full_coulomb_accuracy']
    key={'fixed':'metric_selection','reference':'metric_selection','domain':'domain_selection','auxiliary':'auxiliary_selection'}[kind]
    trials=full[key]['trials'];first=trials[0] if kind in ('fixed','reference') else trials[0]['bound']
    assert len(trials)>=2 and not first['metric_rank']['qualified'] and full['metric_rank']['qualified']
    assert first['j_error_bound']<=accuracy.j_error and first['energy_error_bound']<=accuracy.energy_error
    control=vq.run_periodic_job(**rank_control(job,args,accuracy,result,directory/'control',kind))
    assert control.converged and result.energy==control.energy
    assert result.native.receipt.source_identity==control.native.receipt.source_identity
    assert full['certificate_identity']==control.integral_provenance['fitting']['full_coulomb_accuracy']['certificate_identity']
    for name in ('density','fock','hcore','overlap'):np.testing.assert_array_equal(getattr(result.native,name),getattr(control.native,name))
    return dict(method=method,search=kind,wall_seconds=elapsed,energy=result.energy,iterations=result.iterations,
        trial_count=len(trials),first=first,selected=full['metric_rank'],certificate=full,
        control='converged; bitwise physical state, source and full-Coulomb certificate')


def run_rank_admission(repetitions=1):
    rows=[]
    with tempfile.TemporaryDirectory(prefix='manta-rank-benchmark-') as directory:
        for repetition in range(repetitions):
            for method,kind in [('rhf','fixed'),('uhf','fixed'),('rhf','reference'),('rhf','domain'),('rhf','auxiliary')]:
                rows.append(rank_case(Path(directory)/str(repetition)/(method+'-'+kind),method,kind))
    return dict(scope='complete periodic fitting rank-driven selection and fixed-owner controls',rows=rows)


def action_case(directory,method='rhf',kind='fixed',multipole=False):
    """Metric-action admission with the actual retained-source control."""
    import dataclasses
    import vibeqc as vq
    from tests.test_manta_inverse_action import action_job
    from tests.test_manta_periodic_rank_admission import rank_control
    job,args,accuracy=action_job(directory,method,kind)
    if multipole:
        job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],
            coupling_multipole_error=1.,coupling_multipole_order=0,
            coupling_multipole_pair_tree=False,coupling_multipole_compress=False,
            coupling_multipole_global_actions=False,coupling_multipole_hierarchy=False)
    start=time.perf_counter();result=vq.run_periodic_job(**job);elapsed=time.perf_counter()-start
    assert result.converged
    full=result.integral_provenance['fitting']['full_coulomb_accuracy'];bound=full['metric_action']
    assert bound['requested'] and bound['qualified'] and bound['j_bound']<=accuracy.metric_action_j_error
    assert bound['energy_bound']<=accuracy.metric_action_energy_error
    key={'fixed':'metric_selection','reference':'metric_selection','domain':'domain_selection','auxiliary':'auxiliary_selection'}[kind]
    trials=full[key]['trials'];first=trials[0] if kind in ('fixed','reference') else trials[0]['bound']
    assert len(trials)>=2 and not first['metric_action']['qualified']
    fixed=rank_control(job,args,accuracy,result,directory/'control',kind)
    policy=fixed['periodic_fitting'].full_coulomb_accuracy
    policy.metric_action_j_error=accuracy.metric_action_j_error;policy.metric_action_energy_error=accuracy.metric_action_energy_error
    fixed['periodic_fitting']=dataclasses.replace(fixed['periodic_fitting'],full_coulomb_accuracy=policy)
    control=vq.run_periodic_job(**fixed)
    assert control.converged and result.energy==control.energy
    assert result.native.receipt.source_identity==control.native.receipt.source_identity
    assert full['certificate_identity']==control.integral_provenance['fitting']['full_coulomb_accuracy']['certificate_identity']
    for name in ('density','fock','hcore','overlap'):np.testing.assert_array_equal(getattr(result.native,name),getattr(control.native,name))
    return dict(method=method,search=kind,explicit_multipole=multipole,wall_seconds=elapsed,energy=result.energy,
        iterations=result.iterations,trial_count=len(trials),first=first,selected=bound,
        control='converged; bitwise physical state, source and full-Coulomb certificate')


def run_action_admission(repetitions=1):
    rows=[]
    with tempfile.TemporaryDirectory(prefix='manta-action-benchmark-') as directory:
        for repetition in range(repetitions):
            for method,kind in [('rhf','fixed'),('uhf','fixed'),('rhf','reference'),('rhf','domain'),('rhf','auxiliary')]:
                rows.append(action_case(Path(directory)/str(repetition)/(method+'-'+kind),method,kind))
    return dict(scope='complete periodic metric-action numerical admission and fixed-source controls',rows=rows)


def run_density_family(repetitions=1):
    """Synthetic mixed density constraints; no production scaling claim."""
    a=np.diag([.4,2.7]).astype(complex);w,q=np.linalg.eigh(a)
    f=np.ascontiguousarray(q/np.sqrt(w));f[0,0]*=1.001-.0005j
    o=manta.RIOptions();o.absolute_cutoff=0.;o.relative_cutoff=0.
    owner=manta.HermitianInverseAction(a,w,q,f,o)
    h=[.0001]+[1.]*16;limit=1.
    columns=[owner.column(np.array([x,0.],complex)) for x in [1000.]+[1.]*16]
    old={name:min(limit*max(getattr(c,name) for c in columns),sum(x*getattr(c,name) for x,c in zip(h,columns)))
         for name in ('factor_score','residual_norm','coupling_norm')}
    c=owner.certificate
    old_j=[min(col.factor_score*old['factor_score']+c.inverse_norm_bound*col.residual_norm*old['residual_norm'],
               c.inverse_error_bound*col.coupling_norm*old['coupling_norm']) for col in columns]
    old_e=.5*min(limit*max(old_j),sum(x*y for x,y in zip(h,old_j)))
    times=[]
    for _ in range(repetitions):
        start=time.perf_counter();bound=owner.family(columns,limit,h);times.append(time.perf_counter()-start)
    assert bound.qualified and bound.j_error_bound<max(old_j)/10
    plan=manta.plan_hermitian_inverse_family(2,len(columns))
    return dict(scope='synthetic fixed-family metric bound; former expression comparison is float64 diagnostic',
        old_j_bound=max(old_j),old_energy_bound=old_e,j_bound=bound.j_error_bound,energy_bound=bound.energy_error_bound,
        j_reduction=max(old_j)/bound.j_error_bound,energy_reduction=old_e/bound.energy_error_bound,
        wall_seconds=times,plan={k:getattr(plan,k) for k in ('scalar_terms','kernel_terms','modeled_peak_bytes')})



def compressed_action_case(directory,method='rhf',global_actions=False):
    """Complete compressed admission, fixed-source and materialized controls."""
    import dataclasses
    import vibeqc as vq
    from vibeqc.output import validate_qvf
    from tests.test_manta_inverse_action import compressed_metric_job,retained_metric_job,assert_retained_assembly
    job,policy=compressed_metric_job(directory,method,global_actions)
    start=time.perf_counter();result=vq.run_periodic_job(**job);elapsed=time.perf_counter()-start
    assert result.converged and validate_qvf(str(job['output'].with_suffix('.qvf')))['valid']
    full=result.integral_provenance['fitting']['full_coulomb_accuracy'];bound=full['metric_action']
    assert bound['requested'] and bound['qualified'] and 'compressed-moment' in bound['coupling']
    assert bound['j_bound']<=policy.metric_action_j_error and bound['energy_bound']<=policy.metric_action_energy_error
    assert full['j_error_bound']<=policy.j_error and full['energy_error_bound']<=policy.energy_error
    assert not full['roundoff_included'] and not result.integral_provenance['symmetry_reduced']
    receipt=result.native.receipt.fitted_fock
    assert receipt.coupling_multipole_compressed_nodes>0
    if global_actions:assert receipt.coupling_multipole_global_pair_reuses>0
    policy.metric_action_j_error=policy.metric_action_energy_error=0.
    fixed=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=policy)
    control=vq.run_periodic_job(**dict(job,output=directory/'control'/'job',periodic_fitting=fixed))
    assert control.converged and result.energy==control.energy
    assert result.native.receipt.source_identity==control.native.receipt.source_identity
    for name in ('density','fock','overlap','hcore'):
        np.testing.assert_array_equal(getattr(result.native,name),getattr(control.native,name))
    materialized=vq.run_periodic_job(**dict(job,output=directory/'materialized'/'job',
        periodic_fitting=dataclasses.replace(fixed,coupling_multipole_compress=False,coupling_multipole_global_actions=False)))
    assert materialized.converged and abs(result.energy-materialized.energy)<2e-9
    differences={name:float(np.max(abs(getattr(result.native,name)-getattr(materialized.native,name))))
                 for name in ('density','fock','overlap','hcore')}
    assert max(differences.values())<2e-9
    policy.metric_action_j_error=policy.metric_action_energy_error=1e-100
    try:
        vq.run_periodic_job(**dict(job,output=directory/'refused'/'job',
            periodic_fitting=dataclasses.replace(fixed,full_coulomb_accuracy=policy)))
    except RuntimeError as error:
        assert 'metric action targets unmet' in str(error)
    else:raise AssertionError('strict compressed metric target did not refuse')
    assert not (directory/'refused'/'job.qvf').exists()
    return dict(method=method,global_actions=global_actions,energy=result.energy,iterations=result.iterations,
        wall_seconds=elapsed,metric_action=bound,full_coulomb_j_bound=full['j_error_bound'],
        full_coulomb_energy_bound=full['energy_error_bound'],plan=full['plan'],
        source_identity=result.native.receipt.source_identity,
        compressed_nodes=receipt.coupling_multipole_compressed_nodes,
        global_pair_reuses=receipt.coupling_multipole_global_pair_reuses,
        fixed_source_control='bitwise energy/state/source',strict_target='refused before SCF',
        materialized_energy_difference=result.energy-materialized.energy,materialized_state_differences=differences)


def run_compressed_action(repetitions=1):
    rows=[]
    with tempfile.TemporaryDirectory(prefix='manta-compressed-metric-benchmark-') as directory:
        for repetition in range(repetitions):
            for method in ('rhf','uhf','uks'):
                for global_actions in (False,True):
                    rows.append(compressed_action_case(Path(directory)/str(repetition)/(method+str(int(global_actions))),method,global_actions))
    return dict(scope='bounded compressed metric admission; loose transition targets, no production accuracy claim',rows=rows)


def pair_tree_action_case(directory,method='rhf',retained=False):
    """Complete paired/retained admission and independent materialized Fock witness."""
    import dataclasses
    import vibeqc as vq
    from vibeqc.output import validate_qvf
    from tests.test_manta_inverse_action import compressed_metric_job,retained_metric_job,assert_retained_assembly
    from tests.test_manta_periodic_ri_fitting_accuracy import accuracy_job
    from tests.test_manta_periodic_ri_hierarchy import hierarchy_ks_job
    from tests.test_manta_periodic_ri_multipole import multipole_hf_reference,multipole_ks_reference
    from tests.test_manta_periodic_rhf import _nuclear_reference
    job,policy=retained_metric_job(directory,method) if retained else compressed_metric_job(directory,method,True,True)
    start=time.perf_counter();result=vq.run_periodic_job(**job);elapsed=time.perf_counter()-start
    assert result.converged and validate_qvf(str(job['output'].with_suffix('.qvf')))['valid']
    full=result.integral_provenance['fitting']['full_coulomb_accuracy'];bound=full['metric_action']
    assert bound['qualified'] and bound['j_bound']<=policy.metric_action_j_error
    assert bound['energy_bound']<=policy.metric_action_energy_error and not full['roundoff_included']
    assert not result.integral_provenance['symmetry_reduced']
    receipt=result.native.receipt.fitted_fock
    assembly=result.integral_provenance['assembly_roundoff']
    if retained:
        assert_retained_assembly(result)
    else:
        assert receipt.coupling_multipole_compressed_nodes>0
        assert receipt.metric_roundoff_evaluated and receipt.metric_j_roundoff_evaluated and receipt.coupling_roundoff_evaluated
        assert receipt.forward_metric_roundoff_evaluated and receipt.paired_roundoff_evaluated
        assert receipt.weighted_input_roundoff_evaluated and receipt.coupling_forward_input_error_included and receipt.paired_input_roundoff_included
        assert assembly['evaluated'] and assembly['paired_coulomb_included']
    assert not assembly['source_construction_included'] and not assembly['relaxed_scf_included']
    assert assembly['exchange_correction_roundoff_included']==(result.native.receipt.exchange_q0=='BvK-ewald')
    if assembly['exchange_correction_roundoff_included']:
        assert assembly['exchange_correction_fock_bound']>0. and assembly['exchange_correction_energy_bound']>0.
    exchange=getattr(result.native.receipt,'exact_exchange',None)
    exchange_arithmetic=None if exchange is None else {name:getattr(exchange,name) for name in (
        'roundoff_evaluated','roundoff_scope','roundoff_bound','energy_roundoff_bounds')}
    assert assembly['direct_exchange_roundoff_included']==(exchange is not None)
    if exchange is not None:
        assert exchange.roundoff_evaluated and exchange.roundoff_bound>0.
        assert assembly['direct_exchange_fock_bound']>0. and assembly['direct_exchange_energy_bound']>0.
    arithmetic={name:getattr(receipt,name) for name in (
        'metric_projection_roundoff_bound','metric_coefficient_roundoff_bound','metric_energy_roundoff_bound','metric_j_roundoff_bound','coupling_forward_roundoff_bound','coupling_adjoint_roundoff_bound',
        'forward_metric_coefficient_roundoff_bound','forward_metric_energy_roundoff_bound','paired_j_roundoff_bound','paired_energy_roundoff_bound','weighted_input_roundoff_bound')}
    if retained:arithmetic={name:getattr(receipt,name) for name in ('retained_j_roundoff_bound','retained_energy_roundoff_bound')}
    assert all(0.<value<1e-8 for value in arithmetic.values())
    policy.metric_action_j_error=policy.metric_action_energy_error=0.
    fixed=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=policy)
    control_start=time.perf_counter()
    control=vq.run_periodic_job(**dict(job,output=directory/'control'/'job',periodic_fitting=fixed))
    control_elapsed=time.perf_counter()-control_start
    assert not control.native.receipt.fitted_fock.metric_roundoff_evaluated
    assert not control.native.receipt.fitted_fock.metric_j_roundoff_evaluated
    assert not control.native.receipt.assembly_roundoff.evaluated
    assert not control.native.receipt.fitted_fock.paired_roundoff_evaluated
    assert control.converged and result.energy==control.energy
    assert result.native.receipt.source_identity==control.native.receipt.source_identity
    for name in ('density','fock','overlap','hcore'):
        np.testing.assert_array_equal(getattr(result.native,name),getattr(control.native,name))
    # Reference-only dense export: separate NumPy metric solve/contractions,
    # independent finite-source one-electron/exchange and shared libxc for KS.
    if method in ('rhf','uhf'):
        _,args,_,_=accuracy_job(directory/'reference',method)
    else:
        _,args,a,k,old=hierarchy_ks_job(directory/'reference',2,100)
    fitting=args['options'].fitting
    for name in ('coupling_multipole_error','coupling_multipole_order','coupling_multipole_hierarchy',
                 'coupling_multipole_reuse','coupling_multipole_pair_tree','coupling_multipole_compress','coupling_multipole_global_actions'):
        setattr(fitting,name,getattr(job['periodic_fitting'],name))
    args['options'].fitting=fitting
    if method in ('rhf','uhf'):
        from tests.test_manta_periodic_ri_domains import explicit
        domains=getattr(manta,'build_periodic_ri_'+method+'_domains')(**args)
        a=explicit(args,domains);k=np.array([args['system'].reciprocal_lattice()@args['mesh'].fractional(args['mesh'].address(i)) for i in range(len(args['mesh']))])
        s,h,build,_,_=multipole_hf_reference(a,k,method)
        f,energies=build(result.native.density);nuclear=sum(_nuclear_reference(a)[0])
        from tests.test_manta_periodic_ri_job import _as_fock_build
        state=_as_fock_build(result)
    else:
        from tests.test_manta_periodic_ri_ks_domains import explicit
        from tests.test_manta_periodic_ri_ks import as_fock_build
        a['options']=args['options'];explicit(args,a,old)
        s,h,build,_,_=multipole_ks_reference(a,k,old)
        f,energies,_=build(result.native.density);nuclear=sum(_nuclear_reference(old)[0])
        state=as_fock_build(result.native,k)
    errors=dict(energy=abs(result.energy-nuclear-sum(energies)),fock=float(np.max(abs(result.native.fock-f))),
                overlap=float(np.max(abs(result.native.overlap-s))),hcore=float(np.max(abs(result.native.hcore-h))))
    assert max(errors.values())<8e-11,errors
    assert abs(state.energy.total-result.energy)<1e-12
    policy.metric_action_j_error=policy.metric_action_energy_error=1e-100
    try:
        vq.run_periodic_job(**dict(job,output=directory/'refused'/'job',periodic_fitting=dataclasses.replace(fixed,full_coulomb_accuracy=policy)))
    except RuntimeError as error:
        assert 'metric action targets unmet' in str(error)
    else:raise AssertionError('strict pair-tree metric target did not refuse')
    assert not (directory/'refused'/'job.qvf').exists()
    return dict(method=method,energy=result.energy,iterations=result.iterations,wall_seconds=elapsed,
                metric_action=bound,metric_arithmetic=arithmetic,assembly_roundoff=assembly,exchange_arithmetic=exchange_arithmetic,control_wall_seconds=control_elapsed,plan=full['plan'],compressed_nodes=receipt.coupling_multipole_compressed_nodes,
                source_identity=result.native.receipt.source_identity,fixed_source_control='bitwise energy/state/source',
                independent_errors=errors,strict_target='refused before SCF',unified_fock='same evaluated energy')


def run_pair_tree_action(repetitions=1):
    rows=[]
    with tempfile.TemporaryDirectory(prefix='manta-pair-metric-benchmark-') as directory:
        for repetition in range(repetitions):
            for method in ('rhf','uhf','uks'):
                rows.append(pair_tree_action_case(Path(directory)/str(repetition)/method,method))
    return dict(scope='pair-tree stored-graph metric admission and executed metric step; no complete arithmetic or whole-method accuracy claim',rows=rows)


def run_retained_action(repetitions=1):
    rows=[]
    with tempfile.TemporaryDirectory(prefix='manta-retained-benchmark-') as directory:
        for repetition in range(repetitions):
            for method in ('rhf','uhf','uks'):
                rows.append(pair_tree_action_case(Path(directory)/str(repetition)/method,method,retained=True))
    return dict(scope='retained-mode stored-factor/tile arithmetic; complete fitted-K aggregation and input construction excluded',rows=rows)


def fitted_exchange_reference(directory,profile,job,result,fraction=None):
    """Independent libint/Gaussian sources, NumPy contraction and shared libxc."""
    import vibeqc as vq
    from tests.test_manta_periodic_ri_fitting_accuracy import accuracy_job
    from tests.test_manta_periodic_ri_hierarchy import hierarchy_ks_job
    from tests.test_manta_periodic_rhf import _nuclear_reference
    hf=profile in ('rhf','uhf')
    if hf:
        _,args,_,_=accuracy_job(directory,profile)
    else:
        ns=2 if profile.endswith('uks') else 1
        _,args,a,k,old=hierarchy_ks_job(directory,ns,100)
        previous=args['options']
        range_separated=not profile.startswith('pbe0')
        o=manta.PeriodicRIRangeSeparatedKSOptions() if range_separated else manta.PeriodicRIGlobalHybridKSOptions()
        o.ks=previous.ks;o.fitting=previous.fitting;o.ri=job['coulomb'].metric
        o.coulomb_algorithm=o.exchange_algorithm='ri';o.exchange_q0='BvK-ewald'
        args.update(options=o,functional=vq.Functional(job['functional'],spin=ns))
        a.update(options=o,functional=args['functional'])
    fitting=args['options'].fitting
    for name in ('coupling_multipole_error','coupling_multipole_order','coupling_multipole_hierarchy',
                 'coupling_multipole_reuse','coupling_multipole_pair_tree','coupling_multipole_compress','coupling_multipole_global_actions'):
        setattr(fitting,name,getattr(job['periodic_fitting'],name))
    args['options'].fitting=fitting;args['options'].ri=job['coulomb'].metric
    args['options'].exchange_q0='BvK-ewald'
    k=np.array([args['system'].reciprocal_lattice()@args['mesh'].fractional(args['mesh'].address(i)) for i in range(len(args['mesh']))])
    if hf:
        from tests.test_manta_periodic_ri_domains import explicit
        from tests.test_manta_periodic_ri_uhf import reference
        domains=getattr(manta,'build_periodic_ri_'+profile+'_domains')(**args);a=explicit(args,domains)
        s,h,build,_=reference(a,k)
        d=result.native.density;spin=np.stack([.5*d,.5*d]) if profile=='rhf' else d
        f,energies=build(spin);f=f[0] if profile=='rhf' else f;nuclear=sum(_nuclear_reference(a)[0])
        from tests.test_manta_periodic_ri_job import _as_fock_build
        state=_as_fock_build(result)
    else:
        da=dict(args)
        grid=manta.build_periodic_quadrature(args['system'],da.pop('grid_options'),args['budget'])
        da['grid_points']=grid.plan.grid_points;a.update(points=grid.points,quadrature_weights=grid.weights)
        kind='range_separated' if range_separated else 'global_hybrid'
        domains=getattr(manta,'build_periodic_ri_'+kind+'_ks_domains')(**da)
        for name in ('translations','reciprocal_vectors','xc_images','exchange_vectors','transfer_offsets'):
            a[name]=getattr(domains,name)
        old.update(reciprocal_vectors=a['reciprocal_vectors'],translations=a['translations'])
        from tests.test_manta_periodic_ri_range_ks import reference as range_reference,as_fock_build
        from tests.test_manta_periodic_ri_ks import reference as global_reference
        s,h,build=range_reference(a,k,old) if range_separated else global_reference(a,k,old,fraction=fraction)
        f,energies,_=build(result.native.density);nuclear=sum(_nuclear_reference(old)[0])
        state=as_fock_build(result.native,k)
    errors=dict(energy=abs(result.energy-nuclear-sum(energies)),fock=float(np.max(abs(result.native.fock-f))),
                overlap=float(np.max(abs(result.native.overlap-s))),hcore=float(np.max(abs(result.native.hcore-h))))
    assert max(errors.values())<8e-11,errors
    assert abs(state.energy.total-result.energy)<1e-12
    return errors


def fitted_exchange_case(directory,profile='rhf',capped=False):
    import dataclasses
    import vibeqc as vq
    from vibeqc.output import validate_qvf
    from tests.test_manta_inverse_action import fitted_exchange_job,assert_fitted_exchange_assembly
    job,policy=fitted_exchange_job(directory,profile)
    if capped:job['max_iter']=1
    start=time.perf_counter();result=vq.run_periodic_job(**job);elapsed=time.perf_counter()-start
    assert result.converged!=capped;assert_fitted_exchange_assembly(result)
    assert validate_qvf(str(job['output'].with_suffix('.qvf')))['valid']
    errors=fitted_exchange_reference(directory/'reference',profile,job,result)
    policy.metric_action_j_error=policy.metric_action_energy_error=0.
    control=vq.run_periodic_job(**dict(job,output=directory/'control'/'job',
        periodic_fitting=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=policy)))
    assert control.converged==result.converged and not control.native.receipt.assembly_roundoff.evaluated
    assert result.energy==control.energy and result.native.receipt.source_identity==control.native.receipt.source_identity
    for name in ('density','fock','hcore','overlap'):
        np.testing.assert_array_equal(getattr(result.native,name),getattr(control.native,name))
    return dict(profile=profile,capped=capped,converged=result.converged,energy=result.energy,iterations=result.iterations,
        wall_seconds=elapsed,assembly_roundoff=result.integral_provenance['assembly_roundoff'],
        source_identity=result.native.receipt.source_identity,independent_errors=errors,
        control='bitwise energy, evaluated state and source',unified_fock='same evaluated energy')


def run_fitted_exchange(repetitions=1):
    rows=[]
    with tempfile.TemporaryDirectory(prefix='manta-fitted-exchange-benchmark-') as directory:
        for repetition in range(repetitions):
            for profile in ('rhf','uhf','pbe0-rks','pbe0-uks','hse06-rks','hse06-uks','cam-b3lyp-rks','cam-b3lyp-uks'):
                rows.append(fitted_exchange_case(Path(directory)/str(repetition)/profile,profile))
    return dict(scope='stored-factor whole-mesh fitted K arithmetic and evaluated Fock/energy composition; source construction and relaxed SCF excluded',rows=rows)


def exchange_fitting_case(directory,method='rhf',admission=False,joint=True):
    """Full-Coulomb K component certificate on an actual completed HF state."""
    import vibeqc as vq
    from tests.test_manta_inverse_action import fitted_exchange_job
    from tests.test_manta_periodic_ri_fitting_accuracy import accuracy_job
    from tests.test_manta_periodic_pairs import options as pair_options
    from tests.test_manta_periodic_ewald import cube
    from tests.test_manta_periodic_exchange_bounds import transfers,reference
    from vibeqc.output import validate_qvf
    if admission:
        from tests.test_manta_periodic_exchange_bounds import exchange_admission_job
        job,_=exchange_admission_job(directory,method,joint)
    else:job,_=fitted_exchange_job(directory,method)
    result=vq.run_periodic_job(**job)
    if admission:
        from dataclasses import replace
        control=vq.run_periodic_job(**dict(job,output=directory/'control'/'job',
            periodic_fitting=replace(job['periodic_fitting'],full_coulomb_exchange_accuracy=None)))
        assert result.energy==control.energy and result.iterations==control.iterations
        for field in ('density','fock','coefficients','orbital_energies'):
            np.testing.assert_array_equal(getattr(result.native,field),getattr(control.native,field))
    assert result.converged and validate_qvf(str(job['output'].with_suffix('.qvf')))['valid']
    _,args,_,_=accuracy_job(directory/'reference',method)
    fitting=args['options'].fitting
    for name in ('coupling_multipole_error','coupling_multipole_order','coupling_multipole_hierarchy',
                 'coupling_multipole_reuse','coupling_multipole_pair_tree','coupling_multipole_compress','coupling_multipole_global_actions'):
        setattr(fitting,name,getattr(job['periodic_fitting'],name))
    args['options'].fitting=fitting;args['options'].ri=job['coulomb'].metric
    args['options'].exchange_algorithm='ri'
    domains=getattr(manta,'build_periodic_ri_'+method+'_domains')(**args)
    cell=args['system'].lattice
    k=np.array([args['system'].reciprocal_lattice()@args['mesh'].fractional(args['mesh'].address(i)) for i in range(len(args['mesh']))])
    w=np.full(len(k),1./len(k))
    q,offsets=transfers(cell,k,8)
    # Derive an explicit family from the already admitted canonical envelope;
    # convert its weighted spin-absolute bounds to this component's unweighted
    # convention, with outward products. No observed-density fitted envelope.
    envelope=result.native.receipt.accuracy_density
    o=manta.PeriodicRIExchangeFittingOptions()
    o.density_l1_limit=float(np.nextafter(envelope.density_l1_limit*len(k),np.inf))
    entries=np.nextafter(np.array(envelope.weighted_entry_bounds)*len(k),np.inf)
    o.density_entry_limits=np.maximum(entries,entries.swapaxes(-1,-2)).ravel().tolist()
    o.ao_completion_cube_radius=2;o.reciprocal_completion_cube_radius=8
    ri_plan=manta.plan_periodic_ri_fock(args['basis'],args['auxiliary'],args['system'],
        len(domains.reciprocal_vectors),int(max(np.diff(domains.transfer_offsets))),len(k),
        fitting,args['options'].ri,args['budget'])
    capacity=manta.plan_periodic_ri_exchange_fitting_capacity(args['basis'],ri_plan,27,len(q),
        pair_options(0,None,.2),o,budget=args['budget'])
    ri=manta.PeriodicRIFockBuilder(args['basis'],args['auxiliary'],args['system'],k,w,
        domains.reciprocal_vectors,domains.pair_vectors,domains.transfer_offsets,fitting,args['options'].ri,args['budget'])
    assert ri.info.source_identity==result.native.receipt.fitting_info.source_identity
    pairs=manta.PeriodicPairSource(args['basis'],cell,cube(1),k,w,pair_options(0,None,.2),args['budget'])
    start=time.perf_counter();bound=manta.certify_periodic_ri_exchange_fitting(ri,pairs,q,offsets,o,args['budget']);elapsed=time.perf_counter()-start
    for name in ('modeled_peak_bytes','scalar_terms','kernel_terms','shell_calls','primitive_products'):
        assert getattr(capacity,name)>=getattr(bound.plan,name)
    d=result.native.density;spin=np.stack([.5*d,.5*d]) if method=='rhf' else d
    bound.validate_density(np.ascontiguousarray(spin))
    fitted=ri.build_exchange(spin,True)
    context=dict(basis=args['basis'],cell=cell,k=k,weights=w)
    finite=reference(context,spin,q,offsets,cube(1))
    full_q,full_offsets=transfers(cell,k,10)
    full=reference(context,spin,full_q,full_offsets,cube(2))
    discrepancy=float(np.max(abs(fitted.K-full)))
    energy_error=.5*abs(np.einsum('k,skij,skji->',w,spin,fitted.K-full).real)
    assert np.all(abs(fitted.K-finite)<=bound.finite_difference_bounds+fitted.K_roundoff_bounds+8e-11)
    assert np.all(abs(finite-full)<=bound.reference_remainder_bounds+8e-11)
    assert discrepancy<=bound.maximum_error+np.max(fitted.K_roundoff_bounds)+8e-11
    assert energy_error<=bound.energy_error_bound+8e-11
    attached=None
    if admission:
        c=result.native.receipt.exchange_fitting_accuracy
        assert c.ri_source_identity==ri.info.source_identity
        assert np.all(abs(fitted.K-full)<=c.error_bounds+fitted.K_roundoff_bounds+8e-11)
        assert energy_error<=c.energy_error_bound+8e-11
        attached=dict(certificate_identity=c.certificate_identity,matrix_bound=c.maximum_error,
            energy_bound=c.energy_error_bound,qualified=c.qualified,joint_j=joint,
            targets=dict(matrix=job['periodic_fitting'].full_coulomb_exchange_accuracy.matrix_error,
                energy=job['periodic_fitting'].full_coulomb_exchange_accuracy.energy_error),
            response_roundoff=result.native.receipt.fitted_fock.fitted_exchange_roundoff_bound,
            control_bitwise_equal=True)
    independent=fitted_exchange_reference(directory/'method-reference',method,job,result)
    return dict(method=method,energy=result.energy,iterations=result.iterations,source_identity=ri.info.source_identity,
        certificate_identity=bound.certificate_identity,probes=bound.probes,wall_seconds=elapsed,
        matrix_bound=bound.maximum_error,energy_bound=bound.energy_error_bound,
        finite_matrix_bound=bound.finite_difference_maximum,reference_matrix_bound=bound.reference_remainder_maximum,
        measured_matrix_error=discrepancy,measured_energy_error=energy_error,independent_method_errors=independent,
        plan={name:getattr(bound.plan,name) for name in ('modeled_peak_bytes','scalar_terms','kernel_terms','shell_calls','primitive_products')},
        preparation_capacity={name:getattr(capacity,name) for name in ('modeled_peak_bytes','scalar_terms','kernel_terms','shell_calls','primitive_products')},
        scope=bound.qualification,source_construction_included=bound.source_construction_included,
        automatic_method_attachment=admission,attached=attached,reference='independent Gaussian contour quadrature and enlarged full-Coulomb support')


def run_exchange_fitting(repetitions=1):
    rows=[]
    with tempfile.TemporaryDirectory(prefix='manta-exchange-fitting-benchmark-') as directory:
        for repetition in range(repetitions):
            for method in ('rhf','uhf'):rows.append(exchange_fitting_case(Path(directory)/str(repetition)/method,method))
    return dict(scope='full-Coulomb fitted K component certificate; stored integrals, fixed Hermitian density family, no automatic method attachment',rows=rows)


def run_exchange_admission(repetitions=1):
    rows=[]
    with tempfile.TemporaryDirectory(prefix='manta-exchange-admission-benchmark-') as directory:
        for repetition in range(repetitions):
            for method in ('rhf','uhf'):
                for joint in (False,True):
                    rows.append(exchange_fitting_case(Path(directory)/str(repetition)/method/str(joint),method,True,joint))
    return dict(scope='automatic pre-SCF full-Coulomb K component attachment; separate response arithmetic; no relaxed-SCF guarantee',rows=rows)


def exchange_ks_admission_case(directory,profile='pbe0-rks',joint=True,capped=False,fraction=None):
    """KS attachment using the established independent complete-method oracle."""
    from dataclasses import replace
    import vibeqc as vq
    from vibeqc.output import validate_qvf
    from tests.test_manta_periodic_exchange_bounds import exchange_admission_job
    job,options=exchange_admission_job(directory,profile,joint)
    if fraction is not None:
        name='manta-exchange-admission-signed-benchmark'
        vq.define_functional(name,[('GGA_X_PBE',.75),('GGA_C_PBE',1.)],hf_exchange_fraction=fraction)
        job['functional']=name
    if capped:job['max_iter']=1
    start=time.perf_counter();result=vq.run_periodic_job(**job);elapsed=time.perf_counter()-start
    assert result.converged== (not capped)
    control=vq.run_periodic_job(**dict(job,output=directory/'control'/'job',
        periodic_fitting=replace(job['periodic_fitting'],full_coulomb_exchange_accuracy=None)))
    assert result.energy==control.energy and result.iterations==control.iterations
    assert result.native.receipt.source_identity==control.native.receipt.source_identity
    for name in ('density','fock','coefficients','orbital_energies','hcore','overlap'):
        np.testing.assert_array_equal(getattr(result.native,name),getattr(control.native,name))
    certificate=result.native.receipt.exchange_fitting_accuracy
    spin=np.repeat(.5*result.native.density,2,axis=0) if profile.endswith('-rks') else result.native.density
    certificate.validate_density(np.ascontiguousarray(spin))
    assert certificate.qualified and certificate.ri_source_identity==result.native.receipt.fitting_info.source_identity
    assert certificate.maximum_error<=options.matrix_error and certificate.energy_error_bound<=options.energy_error
    evidence=result.integral_provenance['exchange_fitting_accuracy']
    scale=.25 if fraction is None else fraction
    assert evidence['exchange_fraction']==scale and evidence['response_roundoff_evaluated']
    assert evidence['scaled_component_matrix_bound']>=abs(scale)*certificate.maximum_error
    assert evidence['scaled_component_energy_bound']>=abs(scale)*certificate.energy_error_bound
    assert validate_qvf(str(job['output'].with_suffix('.qvf')))['valid']
    errors=fitted_exchange_reference(directory/'reference',profile,job,result,fraction=fraction)
    return dict(profile=profile,joint_j=joint,capped=capped,converged=result.converged,
        energy=result.energy,iterations=result.iterations,wall_seconds=elapsed,
        source_identity=result.native.receipt.source_identity,certificate=evidence,
        independent_errors=errors,control_bitwise_equal=True,
        reference='independent libint/Gaussian Fourier sources, NumPy contractions and shared libxc',
        limitation='raw stored-integral full-Coulomb K bound only; no total Fock or relaxed SCF guarantee')


def run_exchange_ks_admission(repetitions=1):
    rows=[]
    with tempfile.TemporaryDirectory(prefix='manta-exchange-ks-benchmark-') as directory:
        for repetition in range(repetitions):
            for profile in ('pbe0-rks','pbe0-uks'):
                for joint in (False,True):
                    rows.append(exchange_ks_admission_case(Path(directory)/str(repetition)/profile/str(joint),profile,joint))
    return dict(scope='global-hybrid RI pre-SCF raw fitted K certificate with separate coefficient scaling and response arithmetic',rows=rows)


def range_exchange_method_case(name='hse06',spins=1):
    """Complete range KS plus certificates on matching reconstructed RI owners."""
    from tests.test_manta_periodic_ri_range_ks import case,run,reference as method_reference,as_fock_build
    from tests.test_manta_periodic_rhf import _nuclear_reference
    from tests.test_manta_periodic_exchange_bounds import transfers,reference,cube
    from tests.test_manta_periodic_pairs import options as pair_options
    from tests.test_manta_periodic_range_fitting import physical
    a,k,old=case('JK',spins=spins,name=name,order=2);start=time.perf_counter();r=run(a)
    k=np.array([a['system'].reciprocal_lattice()@a['mesh'].fractional(a['mesh'].address(i)) for i in range(len(a['mesh']))])
    assert r.converged and r.receipt.confirmed
    s,h,build=method_reference(a,k,old);f,energies,_=build(r.density)
    errors=dict(energy=abs(r.receipt.energy-sum(_nuclear_reference(old)[0])-sum(energies)),fock=float(np.max(abs(r.fock-f))))
    assert max(errors.values())<8e-11,errors
    assert abs(as_fock_build(r,k).energy.total-r.receipt.energy)<1e-12
    spin=np.ascontiguousarray(np.repeat(.5*r.density,2,axis=0) if spins==1 else r.density)
    nk=len(k);weights=np.full(nk,1./nk);o=a['options'];images=cube(0)
    pair=manta.PeriodicPairSource(a['basis'],a['system'].lattice,images,k,weights,pair_options(0,None,.2),a['budget'])
    family=manta.PeriodicRIExchangeFittingOptions();family.density_l1_limit=float(np.sum(abs(spin))*2)
    family.reciprocal_completion_cube_radius=2;family.ao_completion_cube_radius=1
    context=dict(basis=a['basis'],cell=a['system'].lattice,k=k,weights=weights);components=[]
    for part in r.receipt.exchange_components:
        owner=manta.PeriodicRIFockBuilder(orbital=a['basis'],auxiliary=a['auxiliary'],system=a['system'],
            momenta=k,weights=weights,gamma_vectors=a['reciprocal_vectors'],pair_vectors=a['exchange_vectors'],
            transfer_offsets=a['transfer_offsets'],options=o.fitting,ri_options=o.ri,budget=a['budget'],
            make_j=False,make_k=True,exchange_kernel=physical(part.kernel,part.omega))
        assert owner.info.source_identity==part.fitting_info.source_identity
        zero=part.kernel=='erfc' and part.omega>0
        q,offsets=transfers(context['cell'],k,2,screened_zero=zero)
        bound=manta.certify_periodic_ri_exchange_fitting(owner,pair,q,offsets,family,a['budget']);bound.validate_density(spin)
        fitted=owner.build_exchange(spin,True)
        fq,fo=transfers(context['cell'],k,5,screened_zero=zero)
        finite=reference(context,spin,q,offsets,images,kernel=part.kernel,omega=part.omega)
        full=reference(context,spin,fq,fo,cube(1),kernel=part.kernel,omega=part.omega)
        matrix_error=float(np.max(abs(fitted.K-full)))
        energy_error=.5*abs(np.einsum('k,skij,skji->',weights,spin,fitted.K-full).real)
        assert np.all(abs(fitted.K-finite)<=bound.finite_difference_bounds+fitted.K_roundoff_bounds+8e-11)
        assert np.all(abs(finite-full)<=bound.reference_remainder_bounds+8e-11)
        assert np.all(abs(fitted.K-full)<=bound.error_bounds+fitted.K_roundoff_bounds+8e-11)
        assert energy_error<=bound.energy_error_bound+8e-11
        components.append(dict(kernel=part.kernel,omega=part.omega,coefficient=part.coefficient,
            source=owner.info.source_identity,certificate=bound.certificate_identity,probes=bound.probes,
            bound=bound.maximum_error,energy_bound=bound.energy_error_bound,
            measured_matrix_error=matrix_error,measured_energy_error=energy_error,
            qualification=bound.qualification,raw_response_roundoff=fitted.receipt.fitted_exchange_roundoff_bound))
    control=run(a);assert control.receipt.energy==r.receipt.energy and control.receipt.source_identity==r.receipt.source_identity
    for field in ('density','fock','coefficients','orbital_energies'):np.testing.assert_array_equal(getattr(r,field),getattr(control,field))
    return dict(functional=name,spins=spins,energy=r.receipt.energy,converged=r.converged,iterations=r.solver_snapshot.evaluated_iteration,
        source=r.receipt.source_identity,independent_errors=errors,components=components,control_bitwise_equal=True,
        density=r.density.real.tolist(),fock_real=r.fock.real.tolist(),fock_imag=r.fock.imag.tolist(),
        wall_seconds=time.perf_counter()-start,
        limitation='post-SCF fixed-family component diagnostics, not automatic range KS admission; no XC or relaxed-SCF bound')


def run_range_exchange_bounds(repetitions=1):
    return dict(scope='raw physical range exchange bounds using actual method-source identities',
        rows=[range_exchange_method_case(name,spins) for _ in range(repetitions)
            for name in ('hse06','cam-b3lyp') for spins in (1,2)])


def range_exchange_admission_case(directory,profile='hse06-rks',joint=True,capped=False):
    from dataclasses import replace
    import vibeqc as vq
    from vibeqc.output import validate_qvf
    from tests.test_manta_periodic_exchange_bounds import range_admission_job
    job,options=range_admission_job(directory,profile,joint)
    if capped:job['max_iter']=1
    start=time.perf_counter();r=vq.run_periodic_job(**job)
    assert r.converged==(not capped)
    control=vq.run_periodic_job(**dict(job,output=directory/'control'/'job',periodic_fitting=replace(job['periodic_fitting'],range_exchange_accuracy=None)))
    assert control.energy==r.energy and control.iterations==r.iterations
    assert r.native.receipt.source_identity==control.native.receipt.source_identity
    for name in ('density','fock','coefficients','orbital_energies'):np.testing.assert_array_equal(getattr(r.native,name),getattr(control.native,name))
    bound=r.native.receipt.range_exchange_accuracy;assert bound.qualified
    assert bound.maximum_error<=options.matrix_error and bound.energy_error_bound<=options.energy_error
    spin=np.ascontiguousarray(np.repeat(.5*r.native.density,2,axis=0) if profile.endswith('-rks') else r.native.density)
    combined=np.zeros_like(bound.error_bounds);energy=0.
    for c,p,a in zip(bound.components,r.native.receipt.exchange_components,bound.coefficients):
        c.validate_density(spin);assert c.ri_source_identity==p.fitting_info.source_identity and c.kernel==p.kernel and c.physical_omega==p.omega
        assert p.fitted_fock.fitted_exchange_roundoff_evaluated and not c.targets_requested
        combined+=abs(a)*c.error_bounds;energy+=abs(a)*c.energy_error_bound
    assert np.all(bound.error_bounds>=combined) and bound.energy_error_bound>=energy
    assert validate_qvf(str(job['output'].with_suffix('.qvf')))['valid']
    errors=fitted_exchange_reference(directory/'reference',profile,job,r)
    return dict(profile=profile,joint_j=joint,capped=capped,converged=r.converged,energy=r.energy,iterations=r.iterations,
        source_identity=r.native.receipt.source_identity,certificate=r.integral_provenance['range_exchange_accuracy'],
        independent_errors=errors,control_bitwise_equal=True,wall_seconds=time.perf_counter()-start,
        limitation='canonical fixed-family exchange component sum; no correction/arithmetic/XC or relaxed-SCF guarantee')


def run_range_exchange_admission(repetitions=1):
    rows=[]
    with tempfile.TemporaryDirectory(prefix='manta-range-admission-benchmark-') as directory:
        for repetition in range(repetitions):
            for profile in ('hse06-rks','hse06-uks','cam-b3lyp-rks','cam-b3lyp-uks'):
                for joint in (False,True):
                    rows.append(range_exchange_admission_case(Path(directory)/str(repetition)/profile/str(joint),profile,joint))
    return dict(scope='pre-SCF canonical family admission of signed physical RI exchange components',rows=rows)


def decay_group_admission_case(directory,profile='rhf',joint=True,capped=False,bins=4):
    """Tighter certificates for unchanged complete physical states."""
    import vibeqc as vq
    from vibeqc.output import validate_qvf
    from tests.test_manta_periodic_exchange_bounds import exchange_admission_job,range_admission_job
    range_method=profile.startswith(('hse06','cam-b3lyp'))
    factory=range_admission_job if range_method else exchange_admission_job
    job,policy=factory(directory,profile,joint);policy.fourier_decay_bins=bins
    if capped:job['max_iter']=1
    start=time.perf_counter();result=vq.run_periodic_job(**job)
    policy.fourier_decay_bins=1
    control=vq.run_periodic_job(**dict(job,output=directory/'single-group'/'job'))
    assert result.energy==control.energy and result.iterations==control.iterations and result.converged==control.converged==(not capped)
    assert result.native.receipt.source_identity==control.native.receipt.source_identity
    for name in ('density','fock','coefficients','orbital_energies'):np.testing.assert_array_equal(getattr(result.native,name),getattr(control.native,name))
    key='range_exchange_accuracy' if range_method else 'exchange_fitting_accuracy'
    new=getattr(result.native.receipt,key);old=getattr(control.native.receipt,key)
    assert new.qualified and new.certificate_identity!=old.certificate_identity
    assert new.maximum_error<=old.maximum_error and new.energy_error_bound<=old.energy_error_bound
    parts=new.components if range_method else [new];previous=old.components if range_method else [old]
    for c,b in zip(parts,previous):
        assert c.ri_source_identity==b.ri_source_identity and c.finite_reference_source_identity==b.finite_reference_source_identity
        np.testing.assert_array_equal(c.finite_difference_bounds,b.finite_difference_bounds)
        assert np.all(c.reference_remainder_bounds<=b.reference_remainder_bounds)
    evidence=result.integral_provenance[key]
    assert evidence['targets']['fourier_decay_bins']==bins
    assert validate_qvf(str(job['output'].with_suffix('.qvf')))['valid']
    errors=fitted_exchange_reference(directory/'reference',profile,job,result)
    return dict(profile=profile,joint_j=joint,capped=capped,converged=result.converged,energy=result.energy,iterations=result.iterations,
        source_identity=result.native.receipt.source_identity,bins=bins,certificate=evidence,
        legacy_bounds=dict(matrix=old.maximum_error,energy=old.energy_error_bound,certificate_identity=old.certificate_identity),
        independent_errors=errors,control_bitwise_equal=True,wall_seconds=time.perf_counter()-start,
        limitation='fixed-family component enclosure; source construction, correction, response arithmetic, XC and relaxation remain separate')


def run_decay_group_admission(repetitions=1):
    rows=[]
    with tempfile.TemporaryDirectory(prefix='manta-decay-group-benchmark-') as directory:
        for repetition in range(repetitions):
            for profile in ('rhf','uhf','pbe0-rks','pbe0-uks','hse06-rks','hse06-uks','cam-b3lyp-rks','cam-b3lyp-uks'):
                rows.append(decay_group_admission_case(Path(directory)/str(repetition)/profile,profile))
    return dict(scope='positive Fourier decay groups in shared exchange reference bounds',rows=rows)



def run_metric_assembly(repetitions=1):
    """Complete finite-source controls; timings do not establish a speedup."""
    from dataclasses import replace
    import numpy as np
    import vibeqc as vq
    from tests.test_manta_periodic_one_electron_images import overlap_rank_job
    rows=[]
    with tempfile.TemporaryDirectory(prefix='manta-metric-assembly-') as directory:
        for repetition in range(repetitions):
            for method,species in [('rhf','Be'),('uhf','H'),('rks','Be'),('pbe0','Be'),('hse06','Li')]:
                path=Path(directory)/str(repetition)/method
                job,_,_,_=overlap_rank_job(path,method,species)
                control=vq.run_periodic_job(**job)
                job['periodic_fitting']=replace(job['periodic_fitting'],metric_assembly_rank=True)
                job['output']=path/'audited/job'
                start=time.perf_counter();result=vq.run_periodic_job(**job);wall=time.perf_counter()-start
                assert result.converged and control.converged and result.energy==control.energy
                for name in ('density','fock','overlap','hcore'):
                    np.testing.assert_array_equal(getattr(result.native,name),getattr(control.native,name))
                metrics=result.integral_provenance['fitting']['metrics']
                assert all(m['assembly_rank']['qualified'] for m in metrics)
                assert all(m['assembly_response']['qualified'] for m in metrics)
                rows.append(dict(method=method,species=species,energy=result.energy,iterations=result.iterations,
                    wall_seconds=wall,metrics=metrics,control_bitwise_equal=True))
    return dict(scope='conditional stored-operand metric assembly rank; no energy-accuracy or speedup claim',rows=rows)


def source_action_admission_case(directory,method='rhf',compressed=None):
    """Complete optional source-action target and identical physical control."""
    import dataclasses
    import numpy as np
    import vibeqc as vq
    from tests.test_manta_inverse_action import source_action_job
    from vibeqc.output import validate_qvf
    job,policy=source_action_job(directory,method,compressed)
    tick=time.perf_counter();result=vq.run_periodic_job(**job);wall=time.perf_counter()-tick
    assert result.converged
    evidence=result.integral_provenance['fitting']['full_coulomb_accuracy']['metric_action']
    assert evidence['qualified'] and evidence['assembly_requested'] and evidence['assembly_qualified']
    assert evidence['assembly_identity']
    policy.metric_action_include_assembly=False
    control_job=dict(job,output=directory/'control/job',periodic_fitting=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=policy))
    control=vq.run_periodic_job(**control_job);assert control.converged and control.energy==result.energy
    assert result.native.receipt.source_identity==control.native.receipt.source_identity
    for name in ('density','fock','hcore','overlap','coefficients','orbital_energies'):
        np.testing.assert_array_equal(getattr(result.native,name),getattr(control.native,name))
    for j in (job,control_job):assert validate_qvf(str(j['output'].with_suffix('.qvf')))['valid']
    return dict(method=method,compressed=compressed,converged=True,energy=result.energy,iterations=result.iterations,
        wall_seconds=wall,certificate=evidence,control_bitwise_equal=True,provenance=result.integral_provenance)


def run_source_action(repetitions=1):
    rows=[]
    with tempfile.TemporaryDirectory(prefix='manta-source-action-') as directory:
        for repetition in range(repetitions):
            for method,compressed in [('rhf',None),('uhf',(False,False)),('rhf',(True,False)),('uhf',(True,True)),('rks',None)]:
                rows.append(source_action_admission_case(Path(directory)/str(repetition)/(method+str(compressed)),method,compressed))
    return dict(scope='conditional metric-source contribution to fixed-coupling J/energy; no relaxed-energy or speedup claim',rows=rows)


def metric_exchange_admission_case(directory,profile='rhf'):
    import numpy as np
    import vibeqc as vq
    from tests.test_manta_periodic_exchange_bounds import metric_exchange_job
    from benchmarks.manta_release_controls import configuration
    job,policy=metric_exchange_job(directory,profile)
    tick=time.perf_counter();result=vq.run_periodic_job(**job);wall=time.perf_counter()-tick
    assert result.converged
    evidence=result.integral_provenance
    parts=evidence['range_exchange_accuracy']['components'] if profile.startswith(('hse','cam')) else [evidence['exchange_fitting_accuracy']]
    assert all(p['metric_assembly']['qualified'] for p in parts)
    inputs=configuration(job)
    policy.metric_assembly_matrix_error=policy.metric_assembly_energy_error=0.
    control=vq.run_periodic_job(**dict(job,output=directory/'control/job'))
    assert control.converged and control.energy==result.energy
    for name in ('density','fock','hcore','overlap','coefficients','orbital_energies'):
        np.testing.assert_array_equal(getattr(result.native,name),getattr(control.native,name))
    assert evidence['source_identity']==control.integral_provenance['source_identity']
    assert evidence['component_ledger']==control.integral_provenance['component_ledger']
    return dict(profile=profile,configuration=inputs,converged=True,energy=result.energy,iterations=result.iterations,
        wall_seconds=wall,components=[p['metric_assembly'] for p in parts],control_bitwise_equal=True,provenance=evidence)


def run_metric_exchange(repetitions=1):
    rows=[]
    with tempfile.TemporaryDirectory(prefix='manta-metric-exchange-') as directory:
        for repetition in range(repetitions):
            for profile in ('rhf','uhf','pbe0-rks','pbe0-uks','hse06-uks','cam-b3lyp-rks'):
                rows.append(metric_exchange_admission_case(Path(directory)/str(repetition)/profile,profile))
    return dict(scope='conditional metric-source K/energy component; no total relaxed-energy or speedup claim',rows=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--rank-admission',action='store_true')
    parser.add_argument('--metric-source-action',action='store_true')
    parser.add_argument('--metric-source-exchange',action='store_true')
    parser.add_argument('--metric-assembly',action='store_true')
    parser.add_argument('--inverse',action='store_true')
    parser.add_argument('--metric-action',action='store_true')
    parser.add_argument('--density-family',action='store_true')
    parser.add_argument('--compressed-action',action='store_true')
    parser.add_argument('--pair-tree-action',action='store_true')
    parser.add_argument('--retained-action',action='store_true')
    parser.add_argument('--fitted-exchange',action='store_true')
    parser.add_argument('--exchange-fitting',action='store_true')
    parser.add_argument('--exchange-admission',action='store_true')
    parser.add_argument('--exchange-ks-admission',action='store_true')
    parser.add_argument('--range-exchange-bounds',action='store_true')
    parser.add_argument('--range-exchange-admission',action='store_true')
    parser.add_argument('--decay-group-admission',action='store_true')
    args=parser.parse_args();runner=run_density_family if args.density_family else (run_action_admission if args.metric_action else (run_inverse if args.inverse else (run_rank_admission if args.rank_admission else run)))
    if args.compressed_action:runner=run_compressed_action
    if args.pair_tree_action:runner=run_pair_tree_action
    if args.retained_action:runner=run_retained_action
    if args.fitted_exchange:runner=run_fitted_exchange
    if args.exchange_fitting:runner=run_exchange_fitting
    if args.exchange_admission:runner=run_exchange_admission
    if args.exchange_ks_admission:runner=run_exchange_ks_admission
    if args.range_exchange_bounds:runner=run_range_exchange_bounds
    if args.range_exchange_admission:runner=run_range_exchange_admission
    if args.decay_group_admission:runner=run_decay_group_admission
    if args.metric_assembly:runner=run_metric_assembly
    if args.metric_source_action:runner=run_source_action
    if args.metric_source_exchange:runner=run_metric_exchange
    print(json.dumps(runner(repetitions=args.repetitions),indent=2))
