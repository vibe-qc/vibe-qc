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


def fitted_exchange_reference(directory,profile,job,result):
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
        s,h,build=(range_reference if range_separated else global_reference)(a,k,old)
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


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--rank-admission',action='store_true')
    parser.add_argument('--inverse',action='store_true')
    parser.add_argument('--metric-action',action='store_true')
    parser.add_argument('--density-family',action='store_true')
    parser.add_argument('--compressed-action',action='store_true')
    parser.add_argument('--pair-tree-action',action='store_true')
    parser.add_argument('--retained-action',action='store_true')
    parser.add_argument('--fitted-exchange',action='store_true')
    args=parser.parse_args();runner=run_density_family if args.density_family else (run_action_admission if args.metric_action else (run_inverse if args.inverse else (run_rank_admission if args.rank_admission else run)))
    if args.compressed_action:runner=run_compressed_action
    if args.pair_tree_action:runner=run_pair_tree_action
    if args.retained_action:runner=run_retained_action
    if args.fitted_exchange:runner=run_fitted_exchange
    print(json.dumps(runner(repetitions=args.repetitions),indent=2))
