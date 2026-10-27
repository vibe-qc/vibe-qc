#!/usr/bin/env python3
"""Finite lattice compact-charge remainder cost and independent energy checks."""
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
    from tests.test_manta_lattice_tail import fixture
    from tests.test_manta_lattice_remainder import controls, reference_components, compact_calculation
    native = Path(core.__file__).resolve()
    if not native.is_relative_to(root):
        raise RuntimeError('Use the containing editable native runtime from this checkout')
    args = fixture(order, skew=skew); envelopes = controls()
    def build():
        return manta.lattice_tail_remainder(**args, **envelopes)
    build(); wall, cpu = [], []
    for _ in range(repeats):
        result = None
        start, process = time.perf_counter(), time.process_time()
        result = build()
        wall.append(time.perf_counter()-start); cpu.append(time.process_time()-process)
    components = np.array([result.near_remainder, result.real_remainder, result.reciprocal_remainder])
    expected = reference_components(args, envelopes)
    if np.any(components < expected*(1-2e-13)):
        raise RuntimeError('Independent incomplete-gamma remainder gate failed')
    row = compact_calculation(order, skew=skew)
    if row['absolute_error'] > row['bound'].total+2e-14:
        raise RuntimeError('Complete signed compact-charge energy gate failed')
    fields = ('real_cells', 'reciprocal_vectors', 'near_cells', 'radial_integrals', 'scalar_terms',
              'kernel_terms', 'output_bytes', 'domain_bytes', 'workspace_bytes', 'modeled_peak_bytes')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system() == 'Darwin' else 1024)
    return dict(order=order, cell='skew' if skew else 'cubic', repeats=repeats,
                median_wall_s=statistics.median(wall), minimum_wall_s=min(wall), maximum_wall_s=max(wall),
                median_cpu_s=statistics.median(cpu), worker_wall_s=time.perf_counter()-began,
                peak_process_rss_bytes=rss, plan={k: getattr(result.plan, k) for k in fields},
                radial_integrals_evaluated=result.radial_integrals_evaluated,
                reciprocal_terms_evaluated=result.reciprocal_terms_evaluated,
                remainder_components=components.tolist(), reference_components=expected.tolist(),
                energy=row['energy'], reference_energy=row['reference_energy'], energy_error=row['absolute_error'],
                energy_remainder=row['bound'].total, source_identity=result.source_identity,
                core_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
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
    print(json.dumps(dict(benchmark='manta-finite-lattice-remainder', results=rows), indent=2))


if __name__ == '__main__':
    main()
