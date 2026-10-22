"""Warm unweighted GDF actions and the consuming paired RI-J build."""
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
    from vibeqc import manta, _vibeqc_core as core
    from tests.test_manta_periodic_ri_paired import paired_case
    from tests.test_manta_periodic_ri_q import builder, densities
    native = Path(core.__file__).resolve()
    if not native.is_relative_to(root):
        raise RuntimeError('Require the native runtime from this checkout')
    a = paired_case(); s = manta.PeriodicGDFTiles(**a); b = builder(a)
    d = densities(); w = np.array([.27, .73]); x = w[:, None, None]*d.sum(axis=0)
    f = s.forward(x, a['budget']); c = b.factor.conj()@(b.factor.T@f.values)
    r = s.adjoint(c, a['budget']); j = b.build(d, w, make_k=False)
    np.testing.assert_allclose(r.values, j.J, atol=2e-13, rtol=3e-13)
    samples = {}
    for name, action in [('forward', lambda: s.forward(x, a['budget'])), ('adjoint', lambda: s.adjoint(c, a['budget'])),
                         ('paired_ri_j', lambda: b.build(d, w, make_k=False))]:
        action(); rows = []
        for _ in range(repeats):
            start = time.perf_counter(); cpu = time.process_time(); action()
            rows.append(dict(wall_seconds=time.perf_counter()-start, cpu_seconds=time.process_time()-cpu))
        samples[name] = rows
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(), samples=samples,
        source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
            ('cpp/include/vibeqc/manta_periodic_fitting.hpp', 'cpp/include/vibeqc/manta_periodic_ri.hpp',
             'cpp/src/periodic_gdf_short_range.cpp', 'cpp/src/manta_periodic_ri.cpp',
             'cpp/src/manta_periodic_fitting_bindings.cpp', 'python/vibeqc/manta.py')},
        working_tree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True)),
        rank=b.metric_info.rank, source_identity=b.metric_info.source_identity,
        factor_identity=b.metric_info.factor_identity, projector_identity=b.metric_info.projector_identity,
        coupling_identity=f.receipt.coupling_identity, action_shell_calls=f.receipt.shell_calls,
        action_primitive_products=f.receipt.primitive_products, action_elements=f.receipt.integral_elements_consumed,
        action_peak_bytes=f.plan.modeled_peak_bytes, ri_peak_bytes=b.plan.modeled_peak_bytes,
        passes=j.receipt.coupling_passes, ri_shell_calls=j.receipt.shell_calls,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024),
        J_real=j.J.real.tolist(), J_imag=j.J.imag.tolist(), energy=j.receipt.coulomb_energy)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers', type=int, default=3); p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--worker', action='store_true'); a = p.parse_args()
    if min(a.workers, a.repeats) < 1:
        p.error('workers and repeats must be positive')
    if a.worker:
        print(json.dumps(worker(a.repeats), allow_nan=False)); return
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', MKL_NUM_THREADS='1')
    rows = [json.loads(subprocess.check_output([sys.executable, __file__, '--worker', '--repeats', str(a.repeats)],
        env=env, text=True)) for _ in range(a.workers)]
    medians = {name: statistics.median(v['wall_seconds'] for r in rows for v in r['samples'][name])
               for name in rows[0]['samples']}
    print(json.dumps(dict(workers=rows, median_seconds=medians), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
