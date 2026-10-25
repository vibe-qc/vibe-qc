#!/usr/bin/env python3
"""Complete public BUPO jobs, managed artifacts and count-only output discovery."""
import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CASES = ('rhf-gamma', 'uhf-shifted-empty-beta', 'uhf-shifted-capped')


def worker(name, repeats):
    import vibeqc as vq
    from vibeqc import _vibeqc_core as core
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from tests.test_manta_periodic_bupo_job import job_case
    native = Path(core.__file__).resolve()
    if not native.is_relative_to(root):
        raise RuntimeError('Use this checkout native runtime')
    capped = name.endswith('capped')
    timings, discovery = [], []
    with tempfile.TemporaryDirectory(prefix='manta-bupo-job-benchmark-') as temporary:
        job, _, _ = job_case(Path(temporary), spin='restricted' if name == 'rhf-gamma' else 'unrestricted',
            z=2 if name == 'rhf-gamma' else 3 if capped else 1, shifted=name != 'rhf-gamma',
            q0='BvK-ewald', iterations=1 if capped else 40)
        for repeat in range(repeats):
            job['output'] = Path(temporary)/str(repeat)/'job'
            start = time.perf_counter()
            result = vq.run_periodic_job(**job)
            timings.append(time.perf_counter()-start)
            assert result.converged == (not capped)
            assert result.integral_provenance['exchange_accuracy_passed']
            row = dict(case=name, repeats=repeats, converged=result.converged, iterations=result.iterations,
                energy=result.energy, source_identity=result.native.receipt.source_identity,
                preparation_identity=result.native.receipt.preparation.identity,
                modeled_peak_bytes=result.native.plan.modeled_peak_bytes,
                preparation_admission=result.integral_provenance['preparation_admission'],
                job_reserved_bytes=result.integral_provenance['job_reserved_bytes'],
                job_scalar_terms=result.integral_provenance['job_scalar_terms'],
                job_kernel_terms=result.integral_provenance['job_kernel_terms'],
                artifacts=len(list(job['output'].parent.iterdir())),
                artifact_bytes=sum(p.stat().st_size for p in job['output'].parent.iterdir()))
            job['output'] = Path(temporary)/('discovery-'+str(repeat))/'job'
            # Dry-run summaries are user-facing output; keep worker JSON clean.
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                start = time.perf_counter()
                assert vq.run_periodic_job(**dict(job, dry_run=True)) is None
                discovery.append(time.perf_counter()-start)
            assert [p.suffix for p in job['output'].parent.iterdir()] == ['.system']
            del result
    row.update(complete_job_s=statistics.median(timings), dry_run_s=statistics.median(discovery),
        native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(), threads=1,
        python=platform.python_version(), platform=platform.system(), architecture=platform.machine(),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system() == 'Darwin' else 1024))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=CASES)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if not 3 <= args.repeats <= 100:
        parser.error('Use three to one hundred repeats')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    if args.worker:
        print(json.dumps(worker(args.worker, args.repeats)))
        return
    rows = []
    for name in CASES:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', name,
            '--repeats', str(args.repeats)], check=True, text=True, capture_output=True)
        rows.append(json.loads(result.stdout))
    print(json.dumps(dict(benchmark='manta-periodic-bupo-job', results=rows), indent=2))


if __name__ == '__main__':
    main()
