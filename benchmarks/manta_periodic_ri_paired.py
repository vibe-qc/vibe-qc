"""Warm paired periodic RI J cost; source construction is measured separately."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time


def worker(repeats):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import numpy as np
    from vibeqc import _vibeqc_core as core
    from tests.test_manta_periodic_ri_paired import paired_case
    from tests.test_manta_periodic_ri_q import builder, densities
    native = Path(core.__file__).resolve()
    if not native.is_relative_to(root):
        raise RuntimeError('Require the native runtime from this checkout')
    a = paired_case(); d = densities(); w = np.array([.27, .73])
    start = time.perf_counter(); b = builder(a)
    preparation = time.perf_counter()-start
    start = time.perf_counter(); result = b.build(d, w, make_k=False)
    cold = time.perf_counter()-start
    samples = []
    for _ in range(repeats):
        start = time.perf_counter(); cpu = time.process_time()
        result = b.build(d, w, make_k=False)
        samples.append(dict(wall_seconds=time.perf_counter()-start, cpu_seconds=time.process_time()-cpu))
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        route=getattr(result.receipt, 'coulomb_contraction', 'retained-mode'), samples=samples,
        preparation_seconds=preparation, cold_seconds=cold, rank=b.metric_info.rank,
        block=b.plan.mode_block, source_identity=b.metric_info.source_identity,
        factor_identity=b.metric_info.factor_identity, projector_identity=b.metric_info.projector_identity,
        passes=result.receipt.coupling_passes, shell_calls=result.receipt.shell_calls,
        primitive_products=result.receipt.primitive_products, modeled_peak_bytes=b.plan.modeled_peak_bytes,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024),
        J_real=result.J.real.tolist(), J_imag=result.J.imag.tolist(), energy=result.receipt.coulomb_energy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.repeats < 1:
        parser.error('workers and repeats must be positive')
    if args.worker:
        print(json.dumps(worker(args.repeats), allow_nan=False))
        return
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', MKL_NUM_THREADS='1')
    rows = [json.loads(subprocess.check_output([sys.executable, __file__, '--worker', '--repeats', str(args.repeats)],
        env=env, text=True)) for _ in range(args.workers)]
    samples = [s['wall_seconds'] for r in rows for s in r['samples']]
    print(json.dumps(dict(workers=rows, median_seconds=statistics.median(samples),
        min_seconds=min(samples), max_seconds=max(samples)), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
