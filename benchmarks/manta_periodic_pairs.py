#!/usr/bin/env python3
"""Periodic pair-source setup, density contraction and adjoint measurements."""
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


def worker(order, skew, repeats):
    began = time.perf_counter()
    import numpy as np
    from vibeqc import _vibeqc_core as core
    root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
    from tests.test_manta_periodic_pairs import _basis, source, density, CELL, finite_fourier_calculation
    native = Path(core.__file__).resolve()
    if not native.is_relative_to(root): raise RuntimeError('Use this checkout\'s containing native runtime')
    basis = _basis((0, 1)); cell = CELL if skew else np.eye(3)*3.9; d = density(basis.nbasis)
    s = source(basis, order, lattice=cell); forward = s.contract(d)
    local = np.ones_like(forward.moments); s.adjoint(local)
    times = {key: [] for key in ('setup', 'forward', 'adjoint')}
    cpus = {key: [] for key in times}
    for _ in range(repeats):
        s = forward = reverse = None
        start, cpu = time.perf_counter(), time.process_time()
        s = source(basis, order, lattice=cell)
        times['setup'].append(time.perf_counter()-start); cpus['setup'].append(time.process_time()-cpu)
        start, cpu = time.perf_counter(), time.process_time()
        forward = s.contract(d)
        times['forward'].append(time.perf_counter()-start); cpus['forward'].append(time.process_time()-cpu)
        start, cpu = time.perf_counter(), time.process_time()
        reverse = s.adjoint(local)
        times['adjoint'].append(time.perf_counter()-start); cpus['adjoint'].append(time.process_time()-cpu)
    row = finite_fourier_calculation(order, skew=skew)
    if row['absolute_error'] > row['energy_bound']+5e-12 or row['same_order_error'] > 5e-12:
        raise RuntimeError('Independent finite Coulomb comparison failed')
    if abs(row['energy']-row['trace_energy']) > 5e-12:
        raise RuntimeError('Paired forward/adjoint energy identity failed')
    fields = ('maximum_pairs', 'moments', 'shell_calls', 'primitive_pairs', 'scalar_terms', 'kernel_terms',
              'retained_bytes', 'driver_live_bytes', 'modeled_peak_bytes')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system() == 'Darwin' else 1024)
    return dict(order=order, cell='skew' if skew else 'cubic', repeats=repeats,
                median_wall_s={k: statistics.median(v) for k, v in times.items()},
                median_cpu_s={k: statistics.median(v) for k, v in cpus.items()},
                worker_wall_s=time.perf_counter()-began, peak_process_rss_bytes=rss,
                plan={k: getattr(s.plan, k) for k in fields}, pairs=s.receipt.pairs,
                ordered_pairs=s.receipt.ordered_pairs, shell_calls_per_pass=forward.receipt.shell_calls,
                energy=row['energy'], reference_energy=row['reference_energy'], energy_error=row['absolute_error'],
                energy_bound=row['energy_bound'], same_order_fourier_error=row['same_order_error'],
                trace_identity_error=abs(row['energy']-row['trace_energy']),
                source_identity=s.receipt.source_identity, core_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
                platform=platform.system(), architecture=platform.machine(), python=platform.python_version(),
                numpy=np.__version__, threads=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', action='store_true'); p.add_argument('--skew', action='store_true')
    p.add_argument('--order', choices=[0, 6, 12], type=int, default=6)
    p.add_argument('--repeats', type=int, default=5)
    args = p.parse_args()
    if not 3 <= args.repeats <= 100: p.error('Use three to one hundred repeats')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    if args.worker:
        print(json.dumps(worker(args.order, args.skew, args.repeats))); return
    rows = []
    for skew in (False, True):
        for order in (0, 6, 12):
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--order', str(order), '--repeats', str(args.repeats)]
            if skew: command.append('--skew')
            run = subprocess.run(command, text=True, capture_output=True, check=True)
            rows.append(json.loads(run.stdout))
    print(json.dumps(dict(benchmark='manta-periodic-pair-source', results=rows), indent=2))


if __name__ == '__main__': main()
