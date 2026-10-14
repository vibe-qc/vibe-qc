"""Independent periodic RI J/K targets: same finite sources and measured work."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


PROFILES = tuple(f'{target}-{nk}' for nk in (1, 2, 4) for target in ('J', 'K', 'JK'))


def worker(profile, repeats):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import numpy as np
    from vibeqc import manta, _vibeqc_core as core
    from tests.test_manta_periodic_ri_targets import case, oracle
    from tests.test_manta_periodic_ri_q import densities
    from tests.test_manta_periodic_gdf import independent, legacy

    native = Path(core.__file__).resolve()
    if not native.is_relative_to(root) or native.stat().st_mtime <= max(
            p.stat().st_mtime for p in (root/'cpp').rglob('*') if p.suffix in ('.cpp', '.hpp', '.h')):
        raise RuntimeError('Require a fresh native build from the current checkout')
    target, nk = profile.split('-'); nk = int(nk)
    a = case(target, nk)
    h, d = densities(nk, spins=1, seed=611)[0], densities(nk)
    start = time.perf_counter(); builder = manta.PeriodicRIFockBuilder(**a)
    setup = time.perf_counter()-start
    start = time.perf_counter(); result = builder.build(h, d)
    cold = time.perf_counter()-start
    samples = []
    for _ in range(repeats):
        start = time.perf_counter(); result = builder.build(h, d)
        samples.append(time.perf_counter()-start)
    errors = {}; reference_times = {}
    for name, source in (('gaussian', independent), ('libint', legacy)):
        start = time.perf_counter(); j, k, f, energy = oracle(a, h, d, source=source)
        reference_times[name] = time.perf_counter()-start
        values = dict(F=float(np.max(abs(f-result.values))), energy=abs(float(energy[-1])-result.receipt.electronic_energy),
            components=float(np.max(abs(energy[:3]-np.array([result.receipt.one_electron_energy,
                result.receipt.coulomb_energy, result.receipt.exchange_energy])))))
        if a['make_j']:
            values['J'] = float(np.max(abs(j-result.J)))
        if a['make_k']:
            values['K'] = float(np.max(abs(k-result.K)))
        if max(values.values()) > 1e-9:
            raise RuntimeError(f'Independent {name} finite-source parity failed: {values}')
        errors[name] = values
    p, info, receipt = builder.plan, builder.info, result.receipt
    return dict(profile=profile, repeats=repeats, has_j=p.has_j, has_k=p.has_k,
        n_orb=p.n_basis, n_aux=p.n_auxiliary, momenta=nk, channels=p.channels,
        gamma_vectors=len(a['gamma_vectors']), pair_vectors=len(a['pair_vectors']),
        metric_ranks=[m.rank for m in info.metrics], errors=errors,
        setup_s=setup, cold_build_s=cold,
        warm_build=dict(median_s=statistics.median(samples), min_s=min(samples), max_s=max(samples)),
        independent_reference_s=reference_times,
        modeled_peak_bytes=p.modeled_peak_bytes, retained_source_bytes=p.retained_source_bytes,
        metric_shell_calls=info.metric_shell_calls, build_shell_calls=receipt.shell_calls,
        metric_primitive_products=info.metric_primitive_products, build_primitive_products=receipt.primitive_products,
        coupling_passes=receipt.coupling_passes, shell_bound=p.shell_calls, primitive_bound=p.primitive_products,
        scalar_bound=p.scalar_terms, kernel_bound=p.kernel_terms,
        core_sha256=hashlib.sha256(native.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=PROFILES)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error('At least three repeats required')
    if args.worker:
        print(json.dumps(worker(args.worker, args.repeats))); return
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    rows = []
    for profile in PROFILES:
        completed = subprocess.run([sys.executable, __file__, '--worker', profile, '--repeats', str(args.repeats)],
            env=env, text=True, capture_output=True, check=True)
        rows.append(json.loads(completed.stdout))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
