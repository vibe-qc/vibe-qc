#!/usr/bin/env python3
"""Fixed finite periodic BUPO partitions, whole-mesh J and conservative bounds."""
from __future__ import annotations
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


def worker(order, angular, repeats):
    began=time.perf_counter()
    import numpy as np
    from vibeqc import manta, _vibeqc_core as core
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    from tests.test_manta_periodic_bupo import fixture, options, density, cube
    native=Path(core.__file__).resolve()
    if not native.is_relative_to(root):raise RuntimeError('Use the native runtime belonging to this checkout')
    tolerance={4:.01,6:1e-5,8:1e-7}[order]
    b,s,t,e,c,cell=fixture(order,angular=angular,tolerance=tolerance)
    d=density(b.nbasis);c.build(d);e.build(d)
    wall={key:[] for key in ('setup','build','direct')};cpu={key:[] for key in wall}
    for _ in range(repeats):
        c=out=direct=None
        start,clock=time.perf_counter(),time.process_time()
        c=manta.PeriodicBupoSource(t,e,cube(0),options(tolerance))
        wall['setup'].append(time.perf_counter()-start);cpu['setup'].append(time.process_time()-clock)
        start,clock=time.perf_counter(),time.process_time();out=c.build(d)
        wall['build'].append(time.perf_counter()-start);cpu['build'].append(time.process_time()-clock)
        start,clock=time.perf_counter(),time.process_time();direct=e.build(d)
        wall['direct'].append(time.perf_counter()-start);cpu['direct'].append(time.process_time()-clock)
    if c.partition.approximate_blocks==0:raise RuntimeError('Benchmark did not exercise accepted far fields')
    matrix_error=float(np.max(abs(out.values-direct.values)));energy_error=abs(out.receipt.energy-direct.receipt.energy)
    assert np.all(abs(out.values-direct.values)<=out.j_error_bound[None,:,:]+2e-11)
    assert energy_error<=out.receipt.energy_error_bound+2e-11
    assert abs(out.receipt.energy-out.receipt.trace_energy)<3e-11
    plan={k:getattr(c.plan,k) for k in ('pairs','nodes','moments','maximum_queries','shell_calls','primitive_products','scalar_terms','kernel_terms','modeled_peak_bytes')}
    partition={k:getattr(c.partition,k) for k in ('approximate_blocks','direct_blocks','approximate_pairs','direct_pairs','maximum_integral_error','identity')}
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024)
    return dict(order=order,angular=angular,repeats=repeats,integral_tolerance=tolerance,plan=plan,partition=partition,
        median_wall_s={k:statistics.median(v) for k,v in wall.items()},median_cpu_s={k:statistics.median(v) for k,v in cpu.items()},
        worker_wall_s=time.perf_counter()-began,peak_process_rss_bytes=rss,energy=out.receipt.energy,reference_energy=direct.receipt.energy,
        energy_error=energy_error,matrix_error=matrix_error,energy_bound=out.receipt.energy_error_bound,
        maximum_j_bound=out.receipt.maximum_j_error_bound,half_trace_error=abs(out.receipt.energy-out.receipt.trace_energy),
        real_calls=out.receipt.direct.real_shell_calls,direct_real_calls=direct.receipt.real_shell_calls,
        fourier_calls=out.receipt.direct.fourier_shell_calls,direct_fourier_calls=direct.receipt.fourier_shell_calls,
        source_identity=c.source_identity,core_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        platform=platform.system(),architecture=platform.machine(),python=platform.python_version(),numpy=np.__version__,threads=1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true');p.add_argument('--angular',action='store_true')
    p.add_argument('--order',type=int,choices=[4,6,8],default=6);p.add_argument('--repeats',type=int,default=5);args=p.parse_args()
    if not 3<=args.repeats<=100:p.error('Use three to one hundred repeats')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    if args.worker:print(json.dumps(worker(args.order,args.angular,args.repeats)));return
    rows=[]
    for angular in (False,True):
        for order in (4,6,8):
            command=[sys.executable,str(Path(__file__).resolve()),'--worker','--order',str(order),'--repeats',str(args.repeats)]
            if angular:command.append('--angular')
            run=subprocess.run(command,text=True,capture_output=True,check=True);rows.append(json.loads(run.stdout))
    print(json.dumps(dict(benchmark='manta-finite-periodic-bupo',results=rows),indent=2))


if __name__=='__main__':main()
