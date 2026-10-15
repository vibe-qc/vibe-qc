#!/usr/bin/env python3
"""Cost and contraction of whole-source infinite-Ewald bounds."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time


def worker(radius,angular,repeats):
    began=time.perf_counter()
    import numpy as np
    from vibeqc import manta,_vibeqc_core as core
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    from tests.test_manta_periodic_bupo_domain import bounded,options,cube,density
    native=Path(core.__file__).resolve()
    if not native.is_relative_to(root):raise RuntimeError('Use the native runtime belonging to this checkout')
    b,s,t,e,c,cell=bounded(6,angular=angular,real_radius=radius,reciprocal_radius=radius)
    o=options();finite=manta.PeriodicBupoSource(t,e,cube(0),o);o.bound_infinite_domain=True
    d=density(b.nbasis);c.build(d);finite.build(d)
    wall={k:[] for k in ('setup','bounded_build','finite_build')};cpu={k:[] for k in wall}
    for _ in range(repeats):
        c=a=f=None
        start,clock=time.perf_counter(),time.process_time();c=manta.PeriodicBupoSource(t,e,cube(0),o)
        wall['setup'].append(time.perf_counter()-start);cpu['setup'].append(time.process_time()-clock)
        start,clock=time.perf_counter(),time.process_time();a=c.build(d)
        wall['bounded_build'].append(time.perf_counter()-start);cpu['bounded_build'].append(time.process_time()-clock)
        start,clock=time.perf_counter(),time.process_time();f=finite.build(d)
        wall['finite_build'].append(time.perf_counter()-start);cpu['finite_build'].append(time.process_time()-clock)
    np.testing.assert_array_equal(a.values,f.values);assert a.receipt.energy==f.receipt.energy
    r=a.receipt;domain=r.domain
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024)
    return dict(radius=radius,angular=angular,repeats=repeats,pairs=c.plan.pairs,domain_pair_interactions=domain.pair_interactions,
        majorant_terms=domain.majorant_terms,majorant_pairs_evaluated=domain.majorant_pairs_evaluated,
        far_pairs=c.partition.approximate_pairs,direct_pairs=c.partition.direct_pairs,
        median_wall_s={k:statistics.median(v) for k,v in wall.items()},median_cpu_s={k:statistics.median(v) for k,v in cpu.items()},
        worker_wall_s=time.perf_counter()-began,peak_process_rss_bytes=rss,modeled_peak_bytes=c.plan.modeled_peak_bytes,
        finite_modeled_peak_bytes=finite.plan.modeled_peak_bytes,energy=r.energy,finite_energy_bound=r.finite_energy_error_bound,
        real_domain_energy_bound=domain.real_energy_error_bound,reciprocal_domain_energy_bound=domain.reciprocal_energy_error_bound,
        total_energy_bound=r.energy_error_bound,maximum_j_bound=r.maximum_j_error_bound,source_identity=c.source_identity,
        domain_source_identity=domain.source_identity,core_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        platform=platform.system(),architecture=platform.machine(),python=platform.python_version(),numpy=np.__version__,threads=1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true')
    p.add_argument('--angular',action='store_true');p.add_argument('--radius',type=int,choices=[1,2,3],default=1)
    p.add_argument('--repeats',type=int,default=5);args=p.parse_args()
    if not 3<=args.repeats<=100:p.error('Use three to one hundred repeats')
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
    if args.worker:print(json.dumps(worker(args.radius,args.angular,args.repeats)));return
    rows=[]
    for angular in (False,True):
        for radius in (1,2,3):
            cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--radius',str(radius),'--repeats',str(args.repeats)]
            if angular:cmd.append('--angular')
            r=subprocess.run(cmd,text=True,capture_output=True,check=True);rows.append(json.loads(r.stdout))
    print(json.dumps(dict(benchmark='manta-periodic-bupo-infinite-domain-bound',results=rows),indent=2))


if __name__=='__main__':main()
