#!/usr/bin/env python3
"""Isolated finite lattice-coefficient timings and independent numerical gates."""
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
    from vibeqc import manta, _vibeqc_core as core

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from tests.test_manta_lattice_tail import fixture, reference_coefficients, reference_matrix, gaussian_calculation

    native = Path(core.__file__).resolve()
    if not native.is_relative_to(root):
        raise RuntimeError('Use the rebuilt editable runtime from this checkout')
    newest = max(p.stat().st_mtime for p in (root/'cpp').rglob('*') if p.suffix in ('.cpp', '.hpp', '.h'))
    if native.stat().st_mtime <= newest:
        raise RuntimeError('Rebuild the native extension before benchmarking')
    args = fixture(order, skew=skew)
    def build():
        result = manta.multipole_lattice_tail(**args)
        return result, result.values
    build()
    wall, cpu = [], []
    for _ in range(repeats):
        result = matrix = None  # do not retain an earlier export during the next admission
        start, process = time.perf_counter(), time.process_time()
        result, matrix = build()
        wall.append(time.perf_counter()-start); cpu.append(time.process_time()-process)
    start = time.perf_counter()
    expected = reference_matrix(reference_coefficients(args, result.indices), result.indices, order)
    error = float(np.max(abs(matrix-expected)))
    if error > 3e-12:
        raise RuntimeError('Independent lattice coefficient gate failed')
    gaussian = None
    if order == 12:
        row = gaussian_calculation(skew=skew, order=12)
        if row['maximum_pair_error'] > 5e-13:
            raise RuntimeError('Independent periodic Gaussian energy gate failed')
        gaussian = {k: row[k] for k in ('energy', 'reference_energy', 'maximum_pair_error')}
    fields = ('moments', 'real_cells', 'reciprocal_vectors', 'near_cells', 'scalar_terms',
              'kernel_terms', 'output_bytes', 'domain_bytes', 'workspace_bytes', 'modeled_peak_bytes')
    work = ('radial_derivatives_evaluated', 'reciprocal_coefficients_evaluated', 'matrix_coefficients_evaluated')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system() == 'Darwin' else 1024)
    return dict(order=order, cell='skew' if skew else 'cubic', repeats=repeats,
                median_wall_s=statistics.median(wall), minimum_wall_s=min(wall), maximum_wall_s=max(wall),
                median_cpu_s=statistics.median(cpu), validation_wall_s=time.perf_counter()-start,
                worker_wall_s=time.perf_counter()-began, peak_process_rss_bytes=rss,
                maximum_coefficient_error=error, gaussian=gaussian,
                plan={k: getattr(result.plan, k) for k in fields}, work={k: getattr(result, k) for k in work},
                source_identity=result.source_identity, core_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
                platform=platform.system(), architecture=platform.machine(), python=platform.python_version(),
                numpy=np.__version__, threads=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--order', type=int, choices=[2, 6, 12], default=6)
    parser.add_argument('--skew', action='store_true')
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if not 3 <= args.repeats <= 100:
        parser.error('repeats must be between three and one hundred')
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    if args.worker:
        print(json.dumps(worker(args.order, args.skew, args.repeats)))
        return
    rows = []
    for skew in (False, True):
        for order in (2, 6, 12):
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--order', str(order), '--repeats', str(args.repeats)]
            if skew:
                command.append('--skew')
            run = subprocess.run(command, text=True, capture_output=True, check=True)
            rows.append(json.loads(run.stdout))
    print(json.dumps(dict(benchmark='manta-finite-lattice-tail', results=rows), indent=2))


if __name__ == '__main__':
    main()
