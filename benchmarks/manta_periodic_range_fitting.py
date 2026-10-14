"""Finite physical range sources and complete Gamma SCF diagnostics."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def worker(kernel, gamma):
    root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
    import numpy as np
    from vibeqc import manta, _vibeqc_core as core
    from benchmarks.manta_periodic_ri_ks import native_binary
    from tests.test_manta_periodic_range_fitting import case, physical, reference, assemble, scf_calculation
    binary = native_binary(root, core)
    a = case(gamma); a['physical_kernel'] = physical(kernel, .43); expected_m, expected_t = reference(a)
    times = []; dense = []; source = None
    for _ in range(5):
        start = time.perf_counter(); r = manta.compute_periodic_gdf(**a); dense.append(time.perf_counter()-start)
        start = time.perf_counter(); source = manta.PeriodicGDFTiles(**a); m = source.metric(); t = assemble(source)
        times.append(time.perf_counter()-start)
        np.testing.assert_array_equal(m.values, r.metric); np.testing.assert_array_equal(t, r.three_center)
    errors = dict(metric=float(np.max(abs(m.values-expected_m))), coupling=float(np.max(abs(t-expected_t))))
    if max(errors.values()) > 4e-12: raise RuntimeError(errors)
    return dict(kernel=kernel, gamma=gamma, physical_omega=.43, numerical_omega=a['options'].omega,
        errors=errors, dense_median_s=float(np.median(dense)), tiled_median_s=float(np.median(times)),
        dense_peak_bytes=r.plan.modeled_peak_bytes, tile_peak_bytes=source.plan.modeled_peak_bytes,
        retained_bytes=source.plan.retained_bytes, actual_dense_shell_calls=r.receipt.shell_calls,
        raw_sr_shell_calls=r.receipt.short_range.metric_shell_calls+r.receipt.short_range.three_center_shell_calls,
        shell_bound=r.plan.shell_calls, source_identity=r.receipt.source_identity,
        core_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        scf=[scf_calculation(kernel, spins) for spins in (1, 2)] if gamma else [])


if __name__ == '__main__':
    if len(sys.argv) > 1:
        print(json.dumps(worker(sys.argv[1], sys.argv[2]=='gamma')))
    else:
        env=dict(os.environ, OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
        rows=[]
        for kernel in ('erfc','erf'):
            for q in ('general','gamma'):
                p=subprocess.run([sys.executable,__file__,kernel,q],env=env,capture_output=True,text=True,check=True)
                rows.append(json.loads(p.stdout))
        print(json.dumps(rows,indent=2))
