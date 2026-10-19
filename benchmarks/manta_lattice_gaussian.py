#!/usr/bin/env python3
"""Joint finite Gaussian lattice interaction cost and independent charge energies."""
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
    from tests.test_manta_lattice_gaussian import source, fixture, interaction, moments, scalar_reference, gaussian_calculation
    native = Path(core.__file__).resolve()
    if not native.is_relative_to(root):
        raise RuntimeError('Use the containing editable native runtime from this checkout')
    args = fixture(order, skew=skew)
    a = source('pair', [.37, -.23, .19], exponent=1.7, radius=.5)
    b = source('aux', [0., 0., 0.], exponent=2.1, radius=.5)
    interaction(args, a, b); wall, cpu = [], []
    for _ in range(repeats):
        result = None
        start, process = time.perf_counter(), time.process_time()
        result = interaction(args, a, b)
        wall.append(time.perf_counter()-start); cpu.append(time.process_time()-process)
    value = float((moments(a, order)@result.matrix.values@moments(b, order).T)[0, 0])
    exact = scalar_reference(args, a, b)
    if abs(value-exact) > result.error.total+2e-13:
        raise RuntimeError('Independent Gaussian convolution gate failed')
    row = gaussian_calculation(order, skew=skew)
    if row['absolute_error'] > row['bound']+3e-14 or np.any(abs(row['pair_values']-row['reference_values']) > row['pair_bounds']+3e-14):
        raise RuntimeError('Complete finite Gaussian charge energy gate failed')
    fields = ('scalar_terms', 'kernel_terms', 'envelope_bytes', 'workspace_bytes', 'modeled_peak_bytes')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system() == 'Darwin' else 1024)
    return dict(order=order, cell='skew' if skew else 'cubic', repeats=repeats,
                median_wall_s=statistics.median(wall), minimum_wall_s=min(wall), maximum_wall_s=max(wall),
                median_cpu_s=statistics.median(cpu), worker_wall_s=time.perf_counter()-began,
                peak_process_rss_bytes=rss, plan={k: getattr(result.plan, k) for k in fields},
                moment_products_evaluated=result.error.moment_products_evaluated,
                reciprocal_terms_evaluated=result.error.reciprocal_terms_evaluated,
                ledger={k: getattr(result.error, k) for k in ('compact_remainder', 'tail_penetration', 'full_moment_tail', 'total')},
                pair_error=abs(value-exact), energy=row['energy'], reference_energy=row['reference_energy'],
                energy_error=row['absolute_error'], energy_bound=row['bound'], source_identity=result.error.source_identity,
                core_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
                platform=platform.system(), architecture=platform.machine(), python=platform.python_version(),
                numpy=np.__version__, threads=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--order', type=int, choices=[0, 6, 12], default=6)
    parser.add_argument('--skew', action='store_true')
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if not 3 <= args.repeats <= 100:
        parser.error('repeats must be between three and one hundred')
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
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
    print(json.dumps(dict(benchmark='manta-finite-lattice-gaussian', results=rows), indent=2))


if __name__ == '__main__':
    main()
