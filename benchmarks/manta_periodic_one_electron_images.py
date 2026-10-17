#!/usr/bin/env python3
"""Full AO-image overlap/kinetic bound construction and admitted HF."""
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

CASES=('sparse-sp','central-cutoff-p','tight-hf')


def worker(case_name,repeats):
    import numpy as np
    from vibeqc import manta,_vibeqc_core as core
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    from tests.test_manta_periodic_one_electron_images import fixture
    from tests.test_manta_periodic_bupo_hf import case
    native=Path(core.__file__).resolve()
    if not native.is_relative_to(root):raise RuntimeError('Use the native runtime belonging to this checkout')
    if case_name=='tight-hf':
        args,c=case();source=c['pairs']
    else:
        c=fixture(angular=1,cutoff=.3 if case_name=='central-cutoff-p' else None);source=c['source']
    manta.bound_periodic_one_electron_images(source)
    wall=[];cpu=[];scf=[]
    for _ in range(repeats):
        start,clock=time.perf_counter(),time.process_time();out=manta.bound_periodic_one_electron_images(source)
        wall.append(time.perf_counter()-start);cpu.append(time.process_time()-clock)
        if case_name=='tight-hf':
            start=time.perf_counter();hf=core._manta.run_periodic_bupo_hf(**args);scf.append(time.perf_counter()-start)
            assert hf.converged and hf.receipt.one_electron_accuracy_passed
    r=out.receipt;p=out.plan
    return dict(case=case_name,repeats=repeats,n_basis=p.n_basis,nk=p.momenta,
        median_wall_s=statistics.median(wall),median_cpu_s=statistics.median(cpu),
        complete_scf_wall_s=statistics.median(scf) if scf else None,
        modeled_peak_bytes=p.modeled_peak_bytes,hf_modeled_peak_bytes=hf.plan.modeled_peak_bytes if scf else None,
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024),
        overlap_bound=r.maximum_overlap_error,kinetic_bound=r.maximum_kinetic_error,
        kinetic_energy_bound=hf.receipt.maximum_observed_kinetic_energy_error if scf else None,
        cube_radius=r.cube_radius,retained_ordered_pairs=r.retained_ordered_pairs,omitted_ordered_pairs=r.omitted_ordered_pairs_in_cube,
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
    print(json.dumps(dict(benchmark='manta-periodic-one-electron-images',results=rows),indent=2))


if __name__=='__main__':main()
