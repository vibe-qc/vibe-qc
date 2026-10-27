#!/usr/bin/env python3
"""Native finite Ewald source setup and complete whole-mesh Coulomb response."""
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


def worker(angular, skew, batch, repeats):
    began = time.perf_counter()
    import numpy as np
    from vibeqc import manta, _vibeqc_core as core
    root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
    from tests.test_manta_periodic_ewald import make_basis, source, density, CELL, IMAGES, cube, reference
    native = Path(core.__file__).resolve()
    if not native.is_relative_to(root): raise RuntimeError('Use this checkout\'s containing native runtime')
    basis = make_basis((0, 1) if angular else (0, 0), contracted=not angular)
    cell = CELL if skew else np.eye(3)*3.9; images = IMAGES[:3].copy() if angular else IMAGES
    real = IMAGES[:3].copy() if angular else cube(1); reciprocal = cube(2, reciprocal=True)
    pairs = source(basis, order=0, lattice=cell, images=images); d = density(basis.nbasis)
    engine = manta.PeriodicEwaldSource(pairs, .63, real, reciprocal, batch); engine.build(d)
    times = {key: [] for key in ('setup', 'build')}; cpus = {key: [] for key in times}
    for _ in range(repeats):
        engine = result = None
        start, cpu = time.perf_counter(), time.process_time()
        engine = manta.PeriodicEwaldSource(pairs, .63, real, reciprocal, batch)
        times['setup'].append(time.perf_counter()-start); cpus['setup'].append(time.process_time()-cpu)
        start, cpu = time.perf_counter(), time.process_time()
        result = engine.build(d)
        times['build'].append(time.perf_counter()-start); cpus['build'].append(time.process_time()-cpu)
    ref = reference(basis, pairs, d, cell, .63, real, reciprocal)
    matrix_error = float(np.max(abs(result.values-ref['values'])))
    energy_error = abs(result.receipt.energy-ref['energy'])
    trace_error = abs(result.receipt.energy-result.receipt.trace_energy)
    if max(matrix_error, energy_error, trace_error) > 3e-12: raise RuntimeError('Independent complete Ewald comparison failed')
    fields = ('pairs', 'pair_interactions', 'real_cells', 'reciprocal_vectors', 'reciprocal_batch',
              'shell_calls', 'primitive_products', 'scalar_terms', 'kernel_terms', 'modeled_peak_bytes')
    r = result.receipt
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system() == 'Darwin' else 1024)
    return dict(basis='s-p' if angular else 's-s', cell='skew' if skew else 'cubic', batch=batch, repeats=repeats,
                median_wall_s={k: statistics.median(v) for k, v in times.items()},
                median_cpu_s={k: statistics.median(v) for k, v in cpus.items()},
                worker_wall_s=time.perf_counter()-began, peak_process_rss_bytes=rss,
                plan={k: getattr(engine.plan, k) for k in fields}, real_shell_calls=r.real_shell_calls,
                fourier_shell_calls=r.fourier_shell_calls, overlap_shell_calls=r.overlap_shell_calls,
                primitive_products=r.primitive_products, vector_primitive_products=r.vector_primitive_products,
                energy=r.energy, reference_energy=ref['energy'], matrix_error=matrix_error,
                energy_error=energy_error, trace_error=trace_error,
                source_identity=engine.source_identity, core_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
                platform=platform.system(), architecture=platform.machine(), python=platform.python_version(),
                numpy=np.__version__, threads=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', action='store_true'); p.add_argument('--skew', action='store_true')
    p.add_argument('--angular', action='store_true'); p.add_argument('--batch', type=int, choices=[1, 32], default=32)
    p.add_argument('--repeats', type=int, default=5); args = p.parse_args()
    if not 3 <= args.repeats <= 100: p.error('Use three to one hundred repeats')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    if args.worker:
        print(json.dumps(worker(args.angular, args.skew, args.batch, args.repeats))); return
    rows = []
    for skew in (False, True):
        for angular in (False, True):
            for batch in (1, 32):
                command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--batch', str(batch), '--repeats', str(args.repeats)]
                if skew: command.append('--skew')
                if angular: command.append('--angular')
                run = subprocess.run(command, text=True, capture_output=True, check=True)
                rows.append(json.loads(run.stdout))
    print(json.dumps(dict(benchmark='manta-periodic-direct-ewald', results=rows), indent=2))


if __name__ == '__main__': main()
