"""Automatic physical support search versus the same explicit RI source."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time


def run(profile='compact',repeats=3):
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    import numpy as np
    import vibeqc as vq
    from vibeqc import _vibeqc_core as core
    from tests.test_manta_periodic_support_selection import support_job,manual_job
    native=Path(core.__file__).resolve()
    if not native.is_relative_to(root):raise RuntimeError('Require this checkout native runtime')
    samples=[];baseline=None
    with tempfile.TemporaryDirectory(prefix='manta-support-selection-') as directory:
        directory=Path(directory)
        for repeat in range(repeats+1):
            job,_,_,_=support_job(directory/str(repeat)/'automatic',profile,target=1e-8)
            sample={}
            # The first selection supplies the explicit control. Alternate
            # subsequent order to expose preparation and cache variability.
            for automatic in ((True,False) if repeat%2==0 else (False,True)):
                request=job if automatic else manual_job(job,baseline,directory/str(repeat)/'explicit')
                wall=time.perf_counter();cpu=time.process_time();result=vq.run_periodic_job(**request)
                sample['automatic' if automatic else 'explicit']=dict(wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu)
                assert result.converged==(profile!='open')
                if baseline is None:baseline=result
                else:
                    assert result.energy==baseline.energy
                    for name in ('density','fock','overlap','hcore'):
                        np.testing.assert_array_equal(getattr(result.native,name),getattr(baseline.native,name))
            if repeat==0:cold=sample
            else:samples.append(sample)
    return dict(commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        dirty=bool(subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True)),
        native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),profile=profile,
        energy=baseline.energy,iterations=baseline.iterations,converged=baseline.converged,
        selection=baseline.integral_provenance['support_selection'],method_peak_bytes=baseline.native.plan.modeled_peak_bytes,
        cold=cold,samples=samples,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=('compact','ordinary','open','global'),default='compact')
    parser.add_argument('--repeats',type=int,default=3);args=parser.parse_args()
    if args.repeats<1:parser.error('--repeats must be positive')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    print(json.dumps(run(args.profile,args.repeats),indent=2,allow_nan=False))
