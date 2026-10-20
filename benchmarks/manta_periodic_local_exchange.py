"""Finite complex Bloch local exchange versus full-grid polynomial/libxc oracle."""
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


def worker(profile,repeats):
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    import numpy as np
    from vibeqc import manta,_vibeqc_core as core
    from tests.test_manta_periodic_local_exchange import case,reference
    native=Path(core.__file__).resolve()
    if not native.is_relative_to(root) or native.stat().st_mtime<=max(
        p.stat().st_mtime for p in (root/'cpp').rglob('*') if p.suffix in ('.cpp','.hpp','.h')):
        raise RuntimeError('Require a fresh own-checkout editable native build')
    spins,pure=(1,True) if profile=='restricted' else (2,profile!='cartesian')
    c=case(spins,pure,order=10);c['grid_batch']=64
    start=time.perf_counter();manta.compute_periodic_local_exchange(**c);cold=time.perf_counter()-start
    wall=[];cpu=[]
    for _ in range(repeats):
        r=None
        start=time.perf_counter();t=time.process_time();r=manta.compute_periodic_local_exchange(**c)
        wall.append(time.perf_counter()-start);cpu.append(time.process_time()-t)
    start=time.perf_counter();e,v,ne=reference(c);oracle=time.perf_counter()-start
    errors=dict(energy=abs(r.receipt.energy-e),potential=float(np.max(abs(r.values-v))),
                population=float(np.max(abs(np.array(r.receipt.electrons_grid)-ne))))
    if max(errors.values())>2e-10:raise RuntimeError('Independent physical local exchange acceptance failed')
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    return dict(profile=profile,repeats=repeats,cold_s=cold,median_s=statistics.median(wall),min_s=min(wall),max_s=max(wall),
        cpu_median_s=statistics.median(cpu),oracle_s=oracle,peak_rss_bytes=rss,modeled_peak_bytes=r.plan.modeled_peak_bytes,
        panel_bytes=r.plan.panel_bytes,grid_points=r.plan.grid_points,n_basis=r.plan.n_basis,images=r.plan.images,
        momenta=r.plan.momenta,shell_calls=r.receipt.shell_calls,primitive_points=r.receipt.primitive_points,
        energy=r.receipt.energy,errors=errors,core_sha256=hashlib.sha256(native.read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',choices=('restricted','spin','cartesian'));p.add_argument('--repeats',type=int,default=5)
    a=p.parse_args()
    if a.repeats<3:p.error('At least three repeats required')
    if a.worker:print(json.dumps(worker(a.worker,a.repeats)));return
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    rows=[]
    for profile in ('restricted','spin','cartesian'):
        out=subprocess.run([sys.executable,__file__,'--worker',profile,'--repeats',str(a.repeats)],env=env,
            check=True,capture_output=True,text=True)
        rows.append(json.loads(out.stdout))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
