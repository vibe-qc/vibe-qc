"""Optional nested metric certificate overhead at an unchanged periodic RI source."""
import argparse,dataclasses,hashlib,json,os,resource,subprocess,sys,tempfile,time
from pathlib import Path


def run(repeats=2):
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    import numpy as np
    import vibeqc as vq
    from vibeqc import _vibeqc_core as core
    from tests.test_manta_periodic_ri_metric_bounds import metric_job
    native=Path(core.__file__).resolve();assert native.is_relative_to(root)
    rows=[]
    with tempfile.TemporaryDirectory(prefix='manta-metric-bounds-') as directory:
        for truncate in (False,True):
            samples=[];baseline=None
            for repeat in range(repeats+1):
                sample={}
                for enabled in ((True,False) if repeat%2==0 else (False,True)):
                    path=Path(directory)/str(truncate)/str(repeat)/str(enabled)
                    job,_,_=metric_job(path,truncate=truncate)
                    if not enabled:job['periodic_fitting'].accuracy.metric_reference_cutoff=None
                    start=time.perf_counter();cpu=time.process_time();result=vq.run_periodic_job(**job)
                    sample['enabled' if enabled else 'coupling_only']=dict(wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu)
                    assert result.converged
                    if baseline is None:baseline=result
                    else:
                        assert result.energy==baseline.energy
                        for name in ('density','fock','overlap','hcore'):np.testing.assert_array_equal(getattr(result.native,name),getattr(baseline.native,name))
                    if enabled:certificate=result.integral_provenance['fitting']['accuracy'];peak=result.native.plan.modeled_peak_bytes
                if repeat==0:cold=sample
                else:samples.append(sample)
            rows.append(dict(truncated=truncate,energy=baseline.energy,iterations=baseline.iterations,certificate=certificate,modeled_peak_bytes=peak,cold=cold,samples=samples))
    return dict(commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        dirty=bool(subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True)),
        native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),rows=rows,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repeats',type=int,default=2);args=parser.parse_args()
    if args.repeats<1:parser.error('--repeats must be positive')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    print(json.dumps(run(args.repeats),indent=2,allow_nan=False))
