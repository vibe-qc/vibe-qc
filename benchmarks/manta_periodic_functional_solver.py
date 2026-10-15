"""Bounded periodic joint functional numerics with an independent nonlinear oracle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time


def worker(profile, repeats):
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root))
    import numpy as np
    from vibeqc import _vibeqc_core as core
    from tests.test_bounded_periodic_functional_solver import case, run, initial, occupy, functional, energy, _options
    native=Path(core.__file__).resolve()
    if not native.is_relative_to(root) or native.stat().st_mtime <= max(
        p.stat().st_mtime for p in (root/'cpp').rglob('*') if p.suffix in ('.cpp','.hpp','.h')):
        raise RuntimeError('Require a fresh own-checkout editable native build')
    c=case(profile!='restricted',k=3,n=3,alpha=1 if profile=='empty-beta' else 2,beta=0 if profile=='empty-beta' else 1)
    o=_options()
    start=time.perf_counter();run(c,o);cold=time.perf_counter()-start
    walls=[];cpus=[]
    for _ in range(repeats):
        r=None  # Release the previous result before the next admitted call.
        start=time.perf_counter();cpu=time.process_time();r=run(c,o)
        walls.append(time.perf_counter()-start);cpus.append(time.process_time()-cpu)
        if not r.converged:raise RuntimeError('Nonlinear native model failed to converge')
    expected=initial(c,o)
    for _ in range(150):
        updated,_,_=occupy(c,c['h']+functional(c,expected)[1],o)
        if np.max(abs(updated-expected))<1e-13:expected=updated;break
        expected=updated
    else:raise RuntimeError('Independent nonlinear SCF failed to converge')
    d,f=r.density_copy(),r.fock_copy()
    errors=dict(density=float(np.max(abs(d-expected))),fock=float(np.max(abs(f-c['h']-functional(c,d)[1]))),
        energy=abs(r.final_snapshot.energy_per_cell-energy(c,expected)))
    if errors['density']>4e-9 or errors['fock']>4e-12 or errors['energy']>4e-12:
        raise RuntimeError('Independent functional acceptance failed')
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    return dict(profile=profile,repeats=repeats,cold_s=cold,median_s=statistics.median(walls),
        min_s=min(walls),max_s=max(walls),cpu_median_s=statistics.median(cpus),peak_rss_bytes=rss,
        owned_bytes=r.memory.peak_owned_numerical_bytes,total_bytes=r.memory.total_numerical_bytes,
        provider_retained_bytes=r.memory.provider_retained_bytes,provider_workspace_bytes=r.memory.provider_workspace_bytes,
        iterations=r.final_snapshot.evaluated_iteration,charged_work=r.final_snapshot.charged_work_units,
        errors=errors,core_sha256=hashlib.sha256(native.read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',choices=('restricted','unequal-spin','empty-beta'))
    p.add_argument('--repeats',type=int,default=5)
    args=p.parse_args()
    if args.repeats<3:p.error('At least three repeats required')
    if args.worker:print(json.dumps(worker(args.worker,args.repeats)));return
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    rows=[]
    for profile in ('restricted','unequal-spin','empty-beta'):
        out=subprocess.run([sys.executable,__file__,'--worker',profile,'--repeats',str(args.repeats)],
            env=env,check=True,capture_output=True,text=True)
        rows.append(json.loads(out.stdout))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
