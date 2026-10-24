#!/usr/bin/env python3
"""Native AO pair-image envelope overhead and measured bound contraction."""
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

CASES=('sparse-s','sparse-sp','cutoff-sp','complete-s')


def worker(case,repeats):
    import numpy as np
    from vibeqc import manta,_vibeqc_core as core
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    from tests.test_manta_periodic_image_bounds import fixture,options,cube,density
    native=Path(core.__file__).resolve()
    if not native.is_relative_to(root):raise RuntimeError('Use the native runtime belonging to this checkout')
    args=dict(angular=1 if case.endswith('sp') else 0)
    if case=='cutoff-sp':args['cutoff']=1.
    if case=='complete-s':args.update(two_shells=False,images=cube(1),exponent=4.,cell_size=6.,skew=False,
                                      eta=.4,real_radius=4,reciprocal_radius=8)
    b,s,t,e,c,cell=fixture(**args);d=density(b.nbasis)
    o=options(0.);o.bound_infinite_domain=True;baseline=manta.PeriodicBupoSource(t,e,cube(0),o)
    o.bound_pair_images=True;c.build(d);baseline.build(d)
    wall={k:[] for k in ('image_setup','image_build','retained_setup','retained_build')};cpu={k:[] for k in wall}
    for _ in range(repeats):
        for enabled,label in ((True,'image'),(False,'retained')):
            o.bound_pair_images=enabled
            start,clock=time.perf_counter(),time.process_time();source=manta.PeriodicBupoSource(t,e,cube(0),o)
            wall[label+'_setup'].append(time.perf_counter()-start);cpu[label+'_setup'].append(time.process_time()-clock)
            start,clock=time.perf_counter(),time.process_time();result=source.build(d)
            wall[label+'_build'].append(time.perf_counter()-start);cpu[label+'_build'].append(time.process_time()-clock)
            if enabled:a=result
            else:f=result
    np.testing.assert_array_equal(a.values,f.values);assert a.receipt.energy==f.receipt.energy
    r=a.receipt;i=r.pair_images
    return dict(case=case,repeats=repeats,n_basis=b.nbasis,pairs=s.receipt.pairs,image_cells=i.cube_cells,
        image_shell_pairs=i.shell_pairs_evaluated,image_primitive_pairs=i.primitive_pairs_evaluated,
        omitted_ordered_pairs_in_cube=i.omitted_ordered_pairs_in_cube,
        median_wall_s={k:statistics.median(v) for k,v in wall.items()},median_cpu_s={k:statistics.median(v) for k,v in cpu.items()},
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024),
        modeled_peak_bytes=a.plan.modeled_peak_bytes,retained_modeled_peak_bytes=f.plan.modeled_peak_bytes,
        energy=r.energy,image_energy_bound=i.energy_error_bound,image_j_bound=i.maximum_j_error_bound,
        energy_bound=r.energy_error_bound,j_bound=r.maximum_j_error_bound,qr=i.retained_density_mass_bound,qm=i.omitted_density_mass_bound,
        maximum_gaussian_exponent=i.maximum_gaussian_exponent,maximum_omitted_pair_mass=i.maximum_omitted_pair_mass,
        kernel_bound=i.kernel_bound,source_identity=r.source_identity,core_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        platform=platform.system(),architecture=platform.machine(),python=platform.python_version(),numpy=np.__version__,threads=1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',choices=CASES)
    p.add_argument('--repeats',type=int,default=5);args=p.parse_args()
    if not 3<=args.repeats<=100:p.error('Use three to one hundred repeats')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    if args.worker:print(json.dumps(worker(args.worker,args.repeats)));return
    rows=[]
    for case in CASES:
        r=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',case,'--repeats',str(args.repeats)],
                         capture_output=True,text=True,check=True)
        rows.append(json.loads(r.stdout))
    print(json.dumps(dict(benchmark='manta-periodic-pair-image-bounds',results=rows),indent=2))


if __name__=='__main__':main()
