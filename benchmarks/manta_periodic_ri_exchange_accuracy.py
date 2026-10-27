"""Whole periodic RI jobs with optional uniform exchange accuracy admission."""
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


def run(repeats=3):
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    import numpy as np
    import vibeqc as vq
    from vibeqc import _vibeqc_core as core
    from tests.test_manta_periodic_exchange_bounds import ri_exchange_accuracy_job,compact_ri_exchange_accuracy_job
    native=Path(core.__file__).resolve()
    if not native.is_relative_to(root):raise RuntimeError('Require this checkout native runtime')
    rows=[]
    with tempfile.TemporaryDirectory(prefix='manta-ri-exchange-accuracy-') as directory:
        for profile in ('ordinary-rhf','global-pbe0','compact-rhf'):
            control=None
            for enabled in (False,True):
                samples=[];previous=None
                for repeat in range(repeats+1):
                    where=Path(directory)/profile/str(enabled)/str(repeat)
                    job,a,k,scf=compact_ri_exchange_accuracy_job(where) if profile=='compact-rhf' else ri_exchange_accuracy_job(
                        where,method='pbe0' if profile=='global-pbe0' else 'rhf',automatic=True)
                    if not enabled:job['periodic_integrals']=dataclasses.replace(job['periodic_integrals'],exchange_accuracy=None)
                    job['integral_engine'].budget.memory_bytes=4*2**30
                    start=time.perf_counter();cpu=time.process_time();result=vq.run_periodic_job(**job)
                    sample=dict(wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu)
                    assert result.converged
                    if previous is None:cold=sample
                    else:
                        np.testing.assert_array_equal(result.native.density,previous.native.density)
                        assert result.energy==previous.energy;samples.append(sample)
                    previous=result
                if not enabled:control=result
                else:
                    for name in ('density','fock','hcore','overlap'):
                        np.testing.assert_array_equal(getattr(result.native,name),getattr(control.native,name))
                    assert result.energy==control.energy
                    assert result.native.receipt.source_identity==control.native.receipt.source_identity
                rows.append(dict(profile=profile,enabled=enabled,energy=result.energy,iterations=result.iterations,
                    peak_bytes=result.native.plan.modeled_peak_bytes,exchange_accuracy=result.integral_provenance.get('exchange_accuracy'),
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
