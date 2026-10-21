#!/usr/bin/env python3
"""Matched electronic/point electrostatics setup, response and error bounds."""
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

CASES=('signed-s','signed-sp','neutral-s')


def worker(case,repeats):
    import numpy as np
    from vibeqc import manta,_vibeqc_core as core
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    from tests.test_manta_periodic_electrostatic import fixture,options,POSITIONS,CHARGES,WEIGHTS,cube,density
    native=Path(core.__file__).resolve()
    if not native.is_relative_to(root):raise RuntimeError('Use the native runtime belonging to this checkout')
    args=dict(angular=1 if case=='signed-sp' else 0)
    if case=='neutral-s':args.update(two_shells=False,exponent=4.,skew=False,cell_size=6.,images=cube(1),
                                    real_radius=4,reciprocal_radius=8,eta=.4)
    b,p,t,e,c,cell=fixture(**args);d=density(b.nbasis);positions=POSITIONS;charges=CHARGES;o=options()
    if case=='neutral-s':positions=np.array([b.shells()[0].origin]);charges=np.array([1.]);o=options('neutral')
    s=manta.PeriodicElectrostaticSource(c,positions,charges,o)
    if case=='neutral-s':d/=np.einsum('k,kij,kji->',WEIGHTS,d,s.overlap).real
    s.build(d);wall={k:[] for k in ('setup','build','electronic_build')};cpu={k:[] for k in wall}
    for _ in range(repeats):
        for key,call in [('setup',lambda:manta.PeriodicElectrostaticSource(c,positions,charges,o)),
                         ('build',lambda:s.build(d)),('electronic_build',lambda:c.build(d))]:
            start,clock=time.perf_counter(),time.process_time();out=call()
            wall[key].append(time.perf_counter()-start);cpu[key].append(time.process_time()-clock)
            if key=='setup':s=out
            if key=='build':a=out
            if key=='electronic_build':ee=out
    np.testing.assert_array_equal(a.electronic.values,ee.values)
    assert a.electronic.receipt.energy==ee.receipt.energy
    r=a.receipt
    return dict(case=case,repeats=repeats,n_basis=b.nbasis,pairs=p.receipt.pairs,points=a.plan.points,
        median_wall_s={k:statistics.median(v) for k,v in wall.items()},median_cpu_s={k:statistics.median(v) for k,v in cpu.items()},
        modeled_peak_bytes=a.plan.modeled_peak_bytes,electronic_modeled_peak_bytes=ee.plan.modeled_peak_bytes,
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024),
        energy=r.energy,energy_bound=r.energy_error_bound,j_bound=r.maximum_j_error_bound,
        charge_uncertainty=r.charge_uncertainty_bound,net_charge=r.retained_net_charge,
        nuclear_shell_calls=r.nuclear_shell_calls,overlap_shell_calls=r.overlap_shell_calls,
        fourier_shell_calls=r.fourier_shell_calls,point_pair_terms=r.point_pair_terms,
        source_identity=r.source_identity,core_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        platform=platform.system(),architecture=platform.machine(),python=platform.python_version(),numpy=np.__version__,threads=1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',choices=CASES)
    p.add_argument('--repeats',type=int,default=5);args=p.parse_args()
    if not 3<=args.repeats<=100:p.error('Use three to one hundred repeats')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    if args.worker:print(json.dumps(worker(args.worker,args.repeats)));return
    rows=[]
    for case in CASES:
        r=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',case,'--repeats',str(args.repeats)],capture_output=True,text=True,check=True)
        rows.append(json.loads(r.stdout))
    print(json.dumps(dict(benchmark='manta-periodic-electrostatic',results=rows),indent=2))


if __name__=='__main__':main()
