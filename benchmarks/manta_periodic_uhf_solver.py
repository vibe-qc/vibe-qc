"""Bounded UHF numerical-leaf evidence; no physical MANTA source or speedup claim.

Run from the repository root. JSON on stdout is operational evidence: redirect
it outside the checkout. Each worker has its own RSS high-water measurement.
"""
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


def worker(k, repeats):
    import numpy as np
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from tests.test_bounded_periodic_uhf_solver import case, run, diagonalize, fock, energy
    from vibeqc import _vibeqc_core as core
    native = Path(core.__file__).resolve()
    newest = max(p.stat().st_mtime for p in (root / 'cpp').rglob('*') if p.suffix in ('.cpp', '.hpp', '.h'))
    if not native.is_relative_to(root) or native.stat().st_mtime <= newest:
        raise RuntimeError('Use a freshly rebuilt editable core from this checkout')
    c = case(k=k)
    start = time.perf_counter()
    run(c)
    cold = time.perf_counter() - start
    walls, cpus = [], []
    for _ in range(repeats):
        start, cpu = time.perf_counter(), time.process_time()
        result = run(c)
        cpus.append(time.process_time() - cpu)
        walls.append(time.perf_counter() - start)
        if not result.converged:
            raise RuntimeError('UHF fixture did not converge')
    reference, _, _ = diagonalize(c, np.stack([c['h']] * 2))
    for _ in range(100):
        updated, _, _ = diagonalize(c, fock(c, reference))
        delta = np.linalg.norm(updated - reference)
        reference = updated
        if delta < 1e-14:
            break
    else:
        raise RuntimeError('Independent reference did not converge')
    d, f = result.density_copy(), result.fock_copy()
    errors = dict(density_max=float(np.max(abs(d - reference))),
                  fock_max=float(np.max(abs(f - fock(c, d)))),
                  energy_abs=abs(result.final_snapshot.energy_per_cell - energy(c, reference, fock(c, reference))))
    if errors['density_max'] > 4e-9 or errors['fock_max'] > 3e-12 or errors['energy_abs'] > 3e-12:
        raise RuntimeError('Independent numerical acceptance failed')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != 'darwin':
        rss *= 1024
    p, row = result.memory, result.final_snapshot
    return dict(k=k, basis=3, alpha=2, beta=1, repeats=repeats,
                cold_s=cold, median_s=statistics.median(walls), min_s=min(walls), max_s=max(walls),
                cpu_median_s=statistics.median(cpus), peak_rss_bytes=rss,
                owned_peak_bytes=p.peak_owned_numerical_bytes, total_numerical_bytes=p.total_numerical_bytes,
                iterations=row.evaluated_iteration, fock_calls=row.fock_calls,
                charged_work_units=row.charged_work_units,
                core_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
                python=platform.python_version(), numpy=np.__version__, platform=sys.platform,
                architecture=platform.machine(), errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, choices=(1, 3, 8))
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error('at least three repeats required')
    if args.worker:
        print(json.dumps(worker(args.worker, args.repeats)))
        return
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    rows = []
    for k in (1, 3, 8):
        result = subprocess.run([sys.executable, __file__, '--worker', str(k), '--repeats', str(args.repeats)],
                                env=env, check=True, capture_output=True, text=True)
        rows.append(json.loads(result.stdout))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
