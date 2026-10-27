#!/usr/bin/env python3
"""Complete periodic BUPO RHF/UHF setup, SCF and admitted error bounds."""
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

CASES=('rhf-gamma','uhf-open-shifted','mixed-bvk-shifted')


def worker(case_name,repeats):
    import numpy as np
    from vibeqc import _vibeqc_core as core
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    from tests.test_manta_periodic_bupo_hf import case,mixed_case
    native=Path(core.__file__).resolve()
    if not native.is_relative_to(root):raise RuntimeError('Use the native runtime belonging to this checkout')
    def setup():
        if case_name=='mixed-bvk-shifted':return mixed_case(shifted=True,q0='BvK-ewald')
        return case(spin='unrestricted',z=3,shifted=True) if case_name=='uhf-open-shifted' else case()
    args,context=setup();core._manta.run_periodic_bupo_hf(**args)
    wall={k:[] for k in ('source_setup','complete_scf')};cpu={k:[] for k in wall}
    for _ in range(repeats):
        start,clock=time.perf_counter(),time.process_time();args,context=setup()
        wall['source_setup'].append(time.perf_counter()-start);cpu['source_setup'].append(time.process_time()-clock)
        start,clock=time.perf_counter(),time.process_time();out=core._manta.run_periodic_bupo_hf(**args)
        wall['complete_scf'].append(time.perf_counter()-start);cpu['complete_scf'].append(time.process_time()-clock)
        assert out.converged and out.receipt.electrostatic_accuracy_passed
    r=out.receipt;p=out.plan
    return dict(case=case_name,repeats=repeats,n_basis=p.n_basis,nk=p.momenta,spins=p.spins,
        pairs=context['pairs'].receipt.pairs,iterations=out.solver_snapshot.evaluated_iteration,fock_calls=r.fock_calls,
        median_wall_s={k:statistics.median(v) for k,v in wall.items()},median_cpu_s={k:statistics.median(v) for k,v in cpu.items()},
        modeled_peak_bytes=p.modeled_peak_bytes,
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024),
        energy=r.energy,energy_bound=r.maximum_observed_electrostatic_energy_error,
        matrix_bound=r.maximum_observed_electrostatic_matrix_error,
        far_pairs=r.coulomb.partition.approximate_pairs,direct_pairs=r.coulomb.partition.direct_pairs,
        source_identity=r.source_identity,partition_identity=r.coulomb.partition.identity,
        core_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
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
    print(json.dumps(dict(benchmark='manta-periodic-bupo-hf',results=rows),indent=2))


if __name__=='__main__':main()
