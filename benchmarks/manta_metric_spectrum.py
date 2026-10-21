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


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--rank-admission',action='store_true')
    parser.add_argument('--inverse',action='store_true')
    parser.add_argument('--metric-action',action='store_true')
    args=parser.parse_args();runner=run_action_admission if args.metric_action else (run_inverse if args.inverse else (run_rank_admission if args.rank_admission else run))
    print(json.dumps(runner(repetitions=args.repetitions),indent=2))
