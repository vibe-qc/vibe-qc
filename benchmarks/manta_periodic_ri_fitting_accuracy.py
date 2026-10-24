"""Complete periodic RI job cost with fixed-family full-Coulomb admission."""
import argparse,dataclasses,hashlib,json,os,resource,subprocess,sys,tempfile,time
from pathlib import Path
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
import numpy as np
import vibeqc as vq
from vibeqc import _vibeqc_core as core
from tests.test_manta_periodic_ri_fitting_accuracy import accuracy_job,metric_selection_job

def run(metric_selection=False):
    rows=[]
    with tempfile.TemporaryDirectory(prefix='manta-fitting-admission-benchmark-') as temporary:
        for method in ('rhf','uhf'):
            job,_,reference,accuracy=(metric_selection_job if metric_selection else accuracy_job)(Path(temporary)/method,method)
            for repeat in range(3):
                pair={};states={}
                for enabled in ([False,True] if repeat%2 else [True,False]):
                    fields=dict(job,output=Path(temporary)/method/str(repeat)/str(enabled)/'job')
                    if not enabled:
                        if metric_selection:
                            from vibeqc import manta
                            fixed=manta.PeriodicRIFittingAccuracyOptions();fixed.reference=reference
                            fixed.j_error=accuracy.j_error;fixed.energy_error=accuracy.energy_error
                            metric=manta.RIOptions();metric.absolute_cutoff=accuracy.minimum_metric_cutoff;metric.relative_cutoff=0.
                            fields['coulomb']=dataclasses.replace(job['coulomb'],metric=metric)
                            fields['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=fixed)
                        else:fields['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=None)
                    t=time.perf_counter();c=time.process_time();out=vq.run_periodic_job(**fields)
                    pair[('automatic' if enabled else 'fixed') if metric_selection else ('enabled' if enabled else 'plain')]=dict(wall=time.perf_counter()-t,cpu=time.process_time()-c)
                    assert out.converged;states[enabled]=out
                for name in ('density','fock','hcore','overlap'):
                    np.testing.assert_array_equal(getattr(states[True].native,name),getattr(states[False].native,name))
                assert states[True].energy==states[False].energy
                proof=states[True].integral_provenance['fitting']['full_coulomb_accuracy']
                rows.append(dict(method=method,repeat=repeat,**pair,j_bound=proof['j_error_bound'],
                    energy_bound=proof['energy_error_bound'],density_l1_limit=proof['density_l1_limit'],
                    metric_trials=proof.get('metric_selection',{}).get('trials',[]),probes=proof['probes'],certificate_plan=proof['plan'],reference_live_bytes=reference.plan.source_live_bytes))
    return dict(commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        comparison='automatic-versus-fixed-certified' if metric_selection else 'certified-versus-plain',native_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),rows=rows,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))

def run_reference_selection(repetitions=3):
    from vibeqc import manta
    from tests.test_manta_periodic_ri_fitting_accuracy import reference_selection_job, selected_reference
    rows=[]
    with tempfile.TemporaryDirectory(prefix='manta-reference-selection-benchmark-') as temporary:
        for method in ('rhf','uhf'):
            for repeat in range(repetitions):
                job,args,accuracy=reference_selection_job(Path(temporary)/method/str(repeat),method)
                t=time.perf_counter();c=time.process_time();automatic=vq.run_periodic_job(**job)
                auto=dict(wall=time.perf_counter()-t,cpu=time.process_time()-c)
                t=time.perf_counter();reference=selected_reference(args,accuracy,automatic);preparation=time.perf_counter()-t
                fixed=manta.PeriodicRIFittingAccuracyOptions();fixed.reference=reference
                fixed.j_error=accuracy.j_error;fixed.energy_error=accuracy.energy_error
                t=time.perf_counter();c=time.process_time()
                control=vq.run_periodic_job(**dict(job,output=Path(temporary)/method/str(repeat)/'fixed'/'job',
                    periodic_fitting=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=fixed)))
                timing=dict(wall=time.perf_counter()-t,cpu=time.process_time()-c)
                assert automatic.converged and control.converged and automatic.energy==control.energy
                for name in ('density','fock','hcore','overlap'):
                    np.testing.assert_array_equal(getattr(automatic.native,name),getattr(control.native,name))
                proof=automatic.integral_provenance['fitting']['full_coulomb_accuracy']
                assert proof['certificate_identity']==control.integral_provenance['fitting']['full_coulomb_accuracy']['certificate_identity']
                rows.append(dict(method=method,repeat=repeat,automatic=auto,fixed=timing,
                    fixed_reference_preparation_seconds=preparation,energy=automatic.energy,
                    j_bound=proof['j_error_bound'],energy_bound=proof['energy_error_bound'],
                    reference_selection=proof['reference_selection'],certificate_plan=proof['plan']))
    return dict(commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        comparison='automatic-reference-versus-explicit-selected-reference',order='automatic-then-fixed',
        native_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),rows=rows,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))

def run_domain_selection(repetitions=3):
    from tests.test_manta_periodic_ri_domain_selection import domain_job,fixed_job
    rows=[]
    with tempfile.TemporaryDirectory(prefix='manta-domain-selection-benchmark-') as temporary:
        for method in ('rhf','uhf'):
            for repeat in range(repetitions):
                job,args,accuracy=domain_job(Path(temporary)/method/str(repeat),method)
                t=time.perf_counter();c=time.process_time();automatic=vq.run_periodic_job(**job)
                auto=dict(wall=time.perf_counter()-t,cpu=time.process_time()-c)
                t=time.perf_counter();fixed=fixed_job(job,args,accuracy,automatic,Path(temporary)/method/str(repeat)/'fixed')
                preparation=time.perf_counter()-t
                t=time.perf_counter();c=time.process_time();control=vq.run_periodic_job(**fixed)
                timing=dict(wall=time.perf_counter()-t,cpu=time.process_time()-c)
                assert automatic.converged and control.converged and automatic.energy==control.energy
                for name in ('density','fock','hcore','overlap'):
                    np.testing.assert_array_equal(getattr(automatic.native,name),getattr(control.native,name))
                proof=automatic.integral_provenance['fitting']['full_coulomb_accuracy']
                assert proof['certificate_identity']==control.integral_provenance['fitting']['full_coulomb_accuracy']['certificate_identity']
                rows.append(dict(method=method,repeat=repeat,automatic=auto,fixed=timing,
                    fixed_reference_preparation_seconds=preparation,energy=automatic.energy,
                    j_bound=proof['j_error_bound'],energy_bound=proof['energy_error_bound'],
                    domain_selection=proof['domain_selection'],certificate_plan=proof['plan']))
    return dict(commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        comparison='automatic-fitting-domains-versus-fixed-selected-source',order='automatic-then-fixed',
        native_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),rows=rows,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))

def run_auxiliary_generation(repetitions=3):
    from vibeqc.auxiliary_generation import ProductAuxiliaryOptions, prepare_product_auxiliary_basis
    from tests.test_manta_periodic_ri_domain_selection import domain_job
    rows=[]
    with tempfile.TemporaryDirectory(prefix='manta-auxiliary-benchmark-') as temporary:
        for method in ('rhf','uhf'):
            for refinement in (0,1):
                for repeat in range(repetitions):
                    job,args,_=domain_job(Path(temporary)/method/str(refinement)/str(repeat),method)
                    options=ProductAuxiliaryOptions();options.refinement=refinement
                    t=time.perf_counter();c=time.process_time()
                    generated=prepare_product_auxiliary_basis(args['system'].unit_cell_molecule(),args['basis'],args['auxiliary'],options)
                    generation=dict(wall=time.perf_counter()-t,cpu=time.process_time()-c)
                    states={};timing={}
                    for typed in ([False,True] if repeat%2 else [True,False]):
                        fields=dict(job,output=Path(temporary)/method/str(refinement)/str(repeat)/str(typed)/'job',
                            coulomb=dataclasses.replace(job['coulomb'],auxiliary_basis=generated if typed else generated.basis))
                        t=time.perf_counter();c=time.process_time();states[typed]=vq.run_periodic_job(**fields)
                        timing['prepared' if typed else 'plain']=dict(wall=time.perf_counter()-t,cpu=time.process_time()-c)
                        assert states[typed].converged
                    assert states[True].energy==states[False].energy
                    for name in ('density','fock','hcore','overlap'):
                        np.testing.assert_array_equal(getattr(states[True].native,name),getattr(states[False].native,name))
                    proof=states[True].integral_provenance['fitting']['full_coulomb_accuracy']
                    rows.append(dict(method=method,refinement=refinement,repeat=repeat,generation=generation,**timing,
                        functions=generated.basis.nbasis,energy=states[True].energy,j_bound=proof['j_error_bound'],
                        energy_bound=proof['energy_error_bound'],generator=states[True].integral_provenance['auxiliary_generation']))
    return dict(commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        comparison='prepared-generated-versus-plain-same-auxiliary',order='alternating-prepared-first',
        native_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),rows=rows,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))

def run_auxiliary_selection(repetitions=3):
    from tests.test_manta_periodic_ri_auxiliary_selection import selection_job,selected_job
    rows=[]
    with tempfile.TemporaryDirectory(prefix='manta-auxiliary-selection-benchmark-') as temporary:
        for method in ('rhf','uhf'):
            for refinement in (0,1):
                for repeat in range(repetitions):
                    directory=Path(temporary)/method/str(refinement)/str(repeat)
                    job,args,accuracy=selection_job(directory,method)
                    if refinement:
                        accuracy.energy_error=.48182
                        job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=accuracy)
                    t=time.perf_counter();c=time.process_time();automatic=vq.run_periodic_job(**job)
                    auto=dict(wall=time.perf_counter()-t,cpu=time.process_time()-c)
                    t=time.perf_counter();fixed=selected_job(job,args,accuracy,automatic,directory/'fixed')
                    preparation=time.perf_counter()-t
                    t=time.perf_counter();c=time.process_time();control=vq.run_periodic_job(**fixed)
                    timing=dict(wall=time.perf_counter()-t,cpu=time.process_time()-c)
                    assert automatic.converged and control.converged and automatic.energy==control.energy
                    for name in ('density','fock','hcore','overlap'):
                        np.testing.assert_array_equal(getattr(automatic.native,name),getattr(control.native,name))
                    proof=automatic.integral_provenance['fitting']['full_coulomb_accuracy']
                    assert proof['certificate_identity']==control.integral_provenance['fitting']['full_coulomb_accuracy']['certificate_identity']
                    rows.append(dict(method=method,refinement=refinement,repeat=repeat,automatic=auto,fixed=timing,
                        fixed_source_and_reference_preparation_seconds=preparation,energy=automatic.energy,
                        j_bound=proof['j_error_bound'],energy_bound=proof['energy_error_bound'],
                        auxiliary_selection=proof['auxiliary_selection'],certificate_plan=proof['plan']))
    return dict(commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        comparison='automatic-auxiliary-versus-fixed-selected-source',order='automatic-then-fixed',
        native_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),rows=rows,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))

def run_reference_energy(repetitions=3, tight_profiles=False):
    from tests.test_manta_periodic_ri_auxiliary_selection import energy_selection_job,selected_job
    rows=[]
    with tempfile.TemporaryDirectory(prefix='manta-reference-energy-benchmark-') as temporary:
        for method in ('rhf','uhf'):
            for tight in ((False,True) if tight_profiles else (False,)):
                for repeat in range(repetitions):
                    directory=Path(temporary)/method/str(tight)/str(repeat)
                    job,args,accuracy=energy_selection_job(directory,method,tight=tight)
                    t=time.perf_counter();c=time.process_time();automatic=vq.run_periodic_job(**job)
                    auto=dict(wall=time.perf_counter()-t,cpu=time.process_time()-c)
                    t=time.perf_counter();fixed=selected_job(job,args,accuracy,automatic,directory/'fixed')
                    preparation=time.perf_counter()-t
                    t=time.perf_counter();c=time.process_time();control=vq.run_periodic_job(**fixed)
                    timing=dict(wall=time.perf_counter()-t,cpu=time.process_time()-c)
                    assert automatic.converged and control.converged and automatic.energy==control.energy
                    for name in ('density','fock','hcore','overlap'):
                        np.testing.assert_array_equal(getattr(automatic.native,name),getattr(control.native,name))
                    proof=automatic.integral_provenance['fitting']['full_coulomb_accuracy']
                    assert proof['certificate_identity']==control.integral_provenance['fitting']['full_coulomb_accuracy']['certificate_identity']
                    rows.append(dict(method=method,profile='tight' if tight else 'reference-energy',repeat=repeat,automatic=auto,fixed=timing,
                        fixed_source_and_reference_preparation_seconds=preparation,energy=automatic.energy,
                        j_bound=proof['j_error_bound'],energy_bound=proof['energy_error_bound'],
                        reference_selection=proof['reference_selection'],auxiliary_selection=proof['auxiliary_selection'],
                        finite_energy_bound=proof['finite_difference_energy_bound'],reference_energy_bound=proof['reference_remainder_energy_bound'],certificate_plan=proof['plan']))
    return dict(commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        comparison='automatic-reference-energy-versus-fixed-selected-source',order='automatic-then-fixed',
        native_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),rows=rows,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))

def run_reference_batches(repetitions=3,tight_profiles=False):
    from tests.test_manta_periodic_ri_auxiliary_selection import energy_selection_job
    rows=[]
    with tempfile.TemporaryDirectory(prefix='manta-reference-batch-benchmark-') as temporary:
        for method in ('rhf','uhf'):
            for tight in ((False,True) if tight_profiles else (False,)):
                for repeat in range(repetitions):
                    pair={};states={}
                    for batch in ((8,1) if repeat%2==0 else (1,8)):
                        job,_,accuracy=energy_selection_job(Path(temporary)/method/str(tight)/str(repeat)/str(batch),method,tight=tight)
                        accuracy.probe_batch_size=batch
                        job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=accuracy)
                        t=time.perf_counter();c=time.process_time();out=vq.run_periodic_job(**job)
                        pair[str(batch)]=dict(wall=time.perf_counter()-t,cpu=time.process_time()-c)
                        assert out.converged;states[batch]=out
                    assert states[1].energy==states[8].energy
                    for name in ('density','fock','hcore','overlap'):
                        np.testing.assert_array_equal(getattr(states[1].native,name),getattr(states[8].native,name))
                    certificates={str(b):states[b].integral_provenance['fitting']['full_coulomb_accuracy'] for b in (1,8)}
                    assert certificates['1']['certificate_identity']==certificates['8']['certificate_identity']
                    rows.append(dict(method=method,profile='tight' if tight else 'reference-energy',repeat=repeat,timing=pair,certificates=certificates))
    return dict(commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        comparison='same-automatic-search-scalar-versus-batch-eight',order='alternating-first-batch-eight',rows=rows,
        native_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))

def run_reference_components(repetitions=3,tight_profiles=False):
    from tests.test_manta_periodic_ri_auxiliary_selection import energy_selection_job
    rows=[]
    with tempfile.TemporaryDirectory(prefix='manta-reference-component-benchmark-') as temporary:
        for method in ('rhf','uhf'):
            for tight in ((False,True) if tight_profiles else (False,)):
                for repeat in range(repetitions):
                    timing={};states={}
                    for selective in ((True,False) if repeat%2==0 else (False,True)):
                        label='selective' if selective else 'joint'
                        job,_,accuracy=energy_selection_job(Path(temporary)/method/str(tight)/str(repeat)/label,method,tight=tight)
                        policy=accuracy.reference_selection;policy.selective_refinement=selective;accuracy.reference_selection=policy
                        job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],full_coulomb_accuracy=accuracy)
                        t=time.perf_counter();c=time.process_time();out=vq.run_periodic_job(**job)
                        timing[label]=dict(wall=time.perf_counter()-t,cpu=time.process_time()-c)
                        assert out.converged;states[label]=out
                    assert states['selective'].energy==states['joint'].energy
                    for name in ('density','fock','hcore','overlap'):
                        np.testing.assert_array_equal(getattr(states['selective'].native,name),getattr(states['joint'].native,name))
                    certificates={label:out.integral_provenance['fitting']['full_coulomb_accuracy'] for label,out in states.items()}
                    for full in certificates.values():
                        assert full['j_error_bound']<=accuracy.j_error and full['energy_error_bound']<=accuracy.energy_error
                    rows.append(dict(method=method,profile='tight' if tight else 'reference-energy',repeat=repeat,timing=timing,certificates=certificates))
    return dict(commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        comparison='same-RI-state-selective-versus-joint-reference',order='alternating-first-selective',rows=rows,
        native_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))

if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group()
    group.add_argument('--reference-components',action='store_true')
    group.add_argument('--reference-batches',action='store_true')
    group.add_argument('--reference-energy',action='store_true')
    group.add_argument('--auxiliary-selection',action='store_true')
    group.add_argument('--auxiliary-generation',action='store_true')
    group.add_argument('--domain-selection',action='store_true')
    group.add_argument('--metric-selection',action='store_true');group.add_argument('--reference-selection',action='store_true')
    parser.add_argument('--tight-profiles',action='store_true',help='Include expensive tight reference-energy profiles')
    parser.add_argument('--repetitions',type=int,default=3);args=parser.parse_args()
    if args.tight_profiles and not (args.reference_energy or args.reference_batches or args.reference_components):parser.error('--tight-profiles requires a reference benchmark mode')
    if args.repetitions<1:parser.error('--repetitions must be positive')
    print(json.dumps(run_reference_components(args.repetitions,args.tight_profiles) if args.reference_components else run_reference_batches(args.repetitions,args.tight_profiles) if args.reference_batches else run_reference_energy(args.repetitions,args.tight_profiles) if args.reference_energy else run_auxiliary_selection(args.repetitions) if args.auxiliary_selection else run_auxiliary_generation(args.repetitions) if args.auxiliary_generation else run_domain_selection(args.repetitions) if args.domain_selection else run_reference_selection(args.repetitions) if args.reference_selection else run(args.metric_selection),indent=2,allow_nan=False))
