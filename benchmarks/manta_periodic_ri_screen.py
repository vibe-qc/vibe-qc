"""Measured screened RI-J cost/error against the same finite unscreened metric."""
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
    root=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(root))
    import numpy as np
    from vibeqc import _vibeqc_core as core
    from tests.test_manta_periodic_ri_paired import paired_case
    from tests.test_manta_periodic_ri_q import builder, densities
    native=Path(core.__file__).resolve()
    if not native.is_relative_to(root): raise RuntimeError('Require this checkout native runtime')
    a=paired_case(); d=densities(); w=np.array([.27,.73]); exact=builder(a)
    ref=exact.build(d,w,make_k=False); rows=[]
    for epsilon in (0.,1e-6,1e-3,.1):
        a['options'].coupling_screen_error=epsilon; b=builder(a)
        r=b.build(d,w,make_k=False); receipt=r.receipt
        np.testing.assert_array_equal(b.factor,exact.factor)
        error=float(np.max(abs(r.J-ref.J))); energy_error=abs(receipt.coulomb_energy-ref.receipt.coulomb_energy)
        assert error<=receipt.approximation_j_bound+5e-12
        assert energy_error<=receipt.approximation_energy_bound+5e-12
        samples=[]
        for _ in range(repeats):
            start=time.perf_counter(); cpu=time.process_time(); b.build(d,w,make_k=False)
            samples.append(dict(wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu))
        rows.append(dict(epsilon=epsilon,samples=samples,J_error=error,energy_error=energy_error,
            J_bound=receipt.approximation_j_bound,energy_bound=receipt.approximation_energy_bound,
            shell_calls=receipt.shell_calls,primitive_products=receipt.primitive_products,
            passes=receipt.coupling_passes,peak_bytes=b.plan.modeled_peak_bytes,
            source_identity=receipt.source_identity,reference_coupling_identity=receipt.reference_coupling_identity,
            factor_identity=b.metric_info.factor_identity))
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),rows=rows,
        working_tree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)),
        source_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in
            ('cpp/src/manta_periodic_ri.cpp','cpp/src/periodic_gdf_short_range.cpp',
             'cpp/include/vibeqc/manta_periodic_error.hpp','tests/test_manta_periodic_ri_screen.py')},
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=3);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--worker',action='store_true');a=p.parse_args()
    if min(a.workers,a.repeats)<1:p.error('workers and repeats must be positive')
    if a.worker:print(json.dumps(worker(a.repeats),allow_nan=False));return
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1')
    rows=[json.loads(subprocess.check_output([sys.executable,__file__,'--worker','--repeats',str(a.repeats)],env=env,text=True)) for _ in range(a.workers)]
    medians={str(e):statistics.median(s['wall_seconds'] for worker in rows for r in worker['rows'] if r['epsilon']==e for s in r['samples']) for e in (0.,1e-6,1e-3,.1)}
    print(json.dumps(dict(workers=rows,median_seconds=medians),indent=2,allow_nan=False))


if __name__=='__main__':main()
