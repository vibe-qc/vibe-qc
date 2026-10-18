#!/usr/bin/env python3
"""Certified exchange-sphere selection and complete bounded BUPO HF cost."""
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
    import numpy as np
    from vibeqc import manta, _vibeqc_core as core
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from tests.test_manta_periodic_bupo_domains import fixture
    native = Path(core.__file__).resolve()
    if not native.is_relative_to(root):
        raise RuntimeError('Use this checkout native runtime')
    a, _ = fixture(spin='restricted' if name == 'rhf-gamma' else 'unrestricted',
                   z=2 if name == 'rhf-gamma' else 3 if name.endswith('capped') else 1,
                   shifted=name != 'rhf-gamma', q0='BvK-ewald', iterations=1 if name.endswith('capped') else 40)
    cutoff = 900. if name.endswith('capped') else 500.
    planning, generation, scf = [], [], []
    for _ in range(repeats):
        start = time.perf_counter()
        plan = manta.plan_periodic_bupo_hf_from_cutoff(**a, reciprocal_energy_cutoff=cutoff)
        planning.append(time.perf_counter()-start)
        start = time.perf_counter()
        domains = manta.build_periodic_bupo_hf_exchange_domains(**a, reciprocal_energy_cutoff=cutoff)
        generation.append(time.perf_counter()-start)
        start = time.perf_counter()
        result = core._manta.run_periodic_bupo_hf_from_cutoff(**a, reciprocal_energy_cutoff=cutoff)
        scf.append(time.perf_counter()-start)
        assert result.converged == (not name.endswith('capped'))
        assert result.receipt.exchange_accuracy_passed
    return dict(case=name, repeats=repeats, reciprocal_energy_cutoff=cutoff,
                n_basis=plan.hf.n_basis, nk=plan.momenta, spins=plan.hf.spins,
                candidates=plan.exchange_candidates, vectors=domains.receipt.exchange_vectors,
                conjugacy_vectors=domains.receipt.conjugacy_vectors,
                completion_cube_radius=plan.exchange_bound_cube_radius,
                plan_s=statistics.median(planning), generation_s=statistics.median(generation),
                complete_hf_s=statistics.median(scf), planned_peak_bytes=plan.modeled_peak_bytes,
                actual_support_modeled_peak_bytes=result.plan.modeled_peak_bytes,
                domain_live_bytes=plan.domain_live_bytes,
                maximum_observed_exchange_matrix_error=result.receipt.maximum_observed_exchange_matrix_error,
                maximum_observed_exchange_energy_error=result.receipt.maximum_observed_exchange_energy_error,
                iterations=result.solver_snapshot.evaluated_iteration, converged=result.converged,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system() == 'Darwin' else 1024),
                source_identity=result.receipt.source_identity, domain_identity=domains.receipt.identity,
                native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
                python=platform.python_version(), numpy=np.__version__, platform=platform.system(),
                architecture=platform.machine(), threads=1)


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
    for case in CASES:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', case,
                                 '--repeats', str(args.repeats)], check=True, text=True, capture_output=True)
        rows.append(json.loads(result.stdout))
    print(json.dumps(dict(benchmark='manta-periodic-bupo-domains', results=rows), indent=2))


if __name__ == '__main__':
    main()
