"""Complete RI nuclear accuracy admission with fixed physical sources."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time


def accuracy_job(directory,profile='ordinary-rhf',enabled=True):
    from benchmarks.manta_periodic_nuclear_bounds import job_context
    from tests.test_manta_periodic_nuclear import nuclear_accuracy_targets
    job,c,scf=job_context(directory,profile)
    if enabled:
        target=nuclear_accuracy_targets(1e-8 if profile=='compact-tight' else 1e6,profile.startswith('compact'))
        job['periodic_integrals']=dataclasses.replace(job['periodic_integrals'],nuclear_accuracy=target)
    return job,c,scf


def run(repeats=3):
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    import numpy as np
    import vibeqc as vq
    from vibeqc import _vibeqc_core as core
    native=Path(core.__file__).resolve()
    if not native.is_relative_to(root):raise RuntimeError('Require this checkout native runtime')
    rows=[]
    with tempfile.TemporaryDirectory(prefix='manta-ri-nuclear-accuracy-') as directory:
        for profile in ('ordinary-rhf','global-pbe0','compact-tight'):
            samples=[];baseline=None
            for repeat in range(repeats+1):
                sample={}
                # Alternate order after the cold pair to reduce order bias.
                for enabled in ((False,True) if repeat%2==0 else (True,False)):
                    job,c,scf=accuracy_job(Path(directory)/profile/str(repeat)/str(enabled),profile,enabled)
                    wall=time.perf_counter();cpu=time.process_time();result=vq.run_periodic_job(**job)
                    sample['enabled' if enabled else 'disabled']=dict(wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu)
                    assert result.converged
                    if baseline is None:baseline=result
                    else:
                        np.testing.assert_array_equal(result.native.density,baseline.native.density)
                        assert result.energy==baseline.energy
                    if enabled:selected=result
                if repeat==0:cold=sample
                else:samples.append(sample)
            certificate=selected.integral_provenance['nuclear_accuracy']
            rows.append(dict(profile=profile,energy=selected.energy,iterations=selected.iterations,
                certificate=certificate,method_peak_bytes=selected.native.plan.modeled_peak_bytes,
                cold=cold,samples=samples))
    return dict(commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        dirty=bool(subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True)),
        native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),rows=rows,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    if args.repeats<1:parser.error('--repeats must be positive')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    print(json.dumps(run(args.repeats),indent=2,allow_nan=False))
