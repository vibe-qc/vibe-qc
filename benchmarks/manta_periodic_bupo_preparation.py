#!/usr/bin/env python3
"""Complete staged BUPO source preparation and independent HF run cost."""
import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

CASES = ('rhf-gamma', 'uhf-shifted-empty-beta', 'uhf-shifted-capped')


def worker(name, repeats):
    from vibeqc import manta, _vibeqc_core as core
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from tests.test_manta_periodic_bupo_domains import fixture
    from tests.test_manta_periodic_bupo_preparation import preparation_args
    native = Path(core.__file__).resolve()
    if not native.is_relative_to(root):
        raise RuntimeError('Use this checkout native runtime')
    capped = name.endswith('capped')
    a, c = fixture(spin='restricted' if name == 'rhf-gamma' else 'unrestricted',
                   z=2 if name == 'rhf-gamma' else 3 if capped else 1,
                   shifted=name != 'rhf-gamma', q0='BvK-ewald', iterations=1 if capped else 40)
    args = preparation_args(a, c, 900. if capped else 500.)
    construction, scf, complete = [], [], []
    for _ in range(repeats):
        start = time.perf_counter()
        prepared = manta.prepare_periodic_bupo_hf(**args)
        built = time.perf_counter()
        result = prepared.run()
        done = time.perf_counter()
        construction.append(built-start)
        scf.append(done-built)
        complete.append(done-start)
        assert result.converged == (not capped)
        assert result.receipt.exchange_accuracy_passed and result.receipt.electrostatic_accuracy_passed
        plan = prepared.plan
        row = dict(case=name, repeats=repeats, reciprocal_energy_cutoff=args['reciprocal_energy_cutoff'],
            converged=result.converged, iterations=result.solver_snapshot.evaluated_iteration,
            n_basis=result.plan.n_basis, nk=result.plan.momenta, spins=result.plan.spins,
            source_identity=prepared.receipt.identity, electronic_source_identity=prepared.receipt.electronic_source_identity,
            source_live_bytes=plan.source_live_bytes, source_scalar_terms=plan.source_scalar_terms,
            source_kernel_terms=plan.source_kernel_terms, source_shell_calls=plan.source_shell_calls,
            source_primitive_pairs=plan.source_primitive_pairs, planned_peak_bytes=plan.modeled_peak_bytes,
            actual_support_modeled_peak_bytes=result.plan.modeled_peak_bytes,
            vectors=result.receipt.domain_exchange_vectors,
            maximum_observed_exchange_matrix_error=result.receipt.maximum_observed_exchange_matrix_error,
            maximum_observed_exchange_energy_error=result.receipt.maximum_observed_exchange_energy_error)
        del result, plan, prepared
    row.update(preparation_s=statistics.median(construction), hf_s=statistics.median(scf),
        complete_s=statistics.median(complete), threads=1, python=platform.python_version(),
        platform=platform.system(), architecture=platform.machine(),
        native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
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
    print(json.dumps(dict(benchmark='manta-periodic-bupo-preparation', results=rows), indent=2))


if __name__ == '__main__':
    main()
