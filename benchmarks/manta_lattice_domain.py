#!/usr/bin/env python3
"""Bound construction cost and independent full Ewald ionic energies."""
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


def worker(cutoff, skew, repeats):
    began = time.perf_counter()
    import numpy as np
    from vibeqc import manta, _vibeqc_core as core
    root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
    from tests.test_manta_lattice_domain import fixture, ionic_calculation
    native = Path(core.__file__).resolve()
    if not native.is_relative_to(root): raise RuntimeError('Use this checkout\'s native runtime')
    args = fixture(skew=skew, real=cutoff, reciprocal=cutoff)
    result = manta.lattice_domain_remainder(**args)
    wall = []; cpu = []
    for _ in range(repeats):
        result = None
        start, clock = time.perf_counter(), time.process_time()
        result = manta.lattice_domain_remainder(**args)
        wall.append(time.perf_counter()-start); cpu.append(time.process_time()-clock)
    calculation = ionic_calculation(skew=skew, cutoff=cutoff)
    error = abs(calculation['energy']-calculation['reference_energy'])
    if error > calculation['bound']+5e-14: raise RuntimeError('Independent energy bound comparison failed')
    plan = {k: getattr(result.plan, k) for k in ('real_cells','reciprocal_vectors','near_cells',
            'scalar_terms','kernel_terms','domain_bytes','workspace_bytes','output_bytes','modeled_peak_bytes')}
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system() == 'Darwin' else 1024)
    return dict(cutoff=cutoff, cell='skew' if skew else 'cubic', repeats=repeats,
                median_wall_s=statistics.median(wall), median_cpu_s=statistics.median(cpu),
                worker_wall_s=time.perf_counter()-began, peak_process_rss_bytes=rss, plan=plan,
                real_bound=result.real_remainder, reciprocal_bound=result.reciprocal_remainder,
                energy=calculation['energy'], reference_energy=calculation['reference_energy'],
                energy_error=error, energy_bound=calculation['bound'],
                core_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
                platform=platform.system(), architecture=platform.machine(), python=platform.python_version(),
                numpy=np.__version__, threads=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', action='store_true'); p.add_argument('--skew', action='store_true')
    p.add_argument('--cutoff', type=int, choices=[1, 3, 6], default=3); p.add_argument('--repeats', type=int, default=5)
    args = p.parse_args()
    if not 3 <= args.repeats <= 100: p.error('Use three to one hundred repeats')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'): os.environ[key] = '1'
    if args.worker:
        print(json.dumps(worker(args.cutoff, args.skew, args.repeats))); return
    rows = []
    for skew in (False, True):
        for cutoff in (1, 3, 6):
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--cutoff', str(cutoff), '--repeats', str(args.repeats)]
            if skew: command.append('--skew')
            run = subprocess.run(command, text=True, capture_output=True, check=True)
            rows.append(json.loads(run.stdout))
    print(json.dumps(dict(benchmark='manta-lattice-domain-remainder', results=rows), indent=2))


if __name__ == '__main__': main()
