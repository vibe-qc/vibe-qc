"""Public fitted KS: complete native/public jobs and independent functional SCF."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time

PROFILES = tuple(f'{source}-{spin}' for source in ('pure', 'J', 'K', 'JK') for spin in ('rks', 'uks', 'empty'))


def worker(profile, repeats):
    root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
    import numpy as np
    import vibeqc as vq
    from vibeqc import _vibeqc_core as core
    from vibeqc.guess_read import _spin_density_k_from_qvf_bloch_payload
    from vibeqc.output import validate_qvf
    from benchmarks.manta_periodic_ri_ks import native_binary, measure
    from tests.test_manta_periodic_ri_ks_job import job_case
    from tests.test_manta_periodic_ri_ks_domains import function, explicit, ARRAYS
    from tests.test_manta_periodic_ri_ks import reference, independent_scf, components
    from tests.test_manta_periodic_rhf import _nuclear_reference
    binary = native_binary(root, core)
    source, spin = profile.split('-'); spins = 1 if spin == 'rks' else 2
    with tempfile.TemporaryDirectory(prefix='manta-fitted-ks-job-') as directory:
        job, args, a, k, old = job_case(Path(directory), source, spins, empty=spin == 'empty')
        start = time.perf_counter(); result = vq.run_periodic_job(**job); cold = time.perf_counter()-start
        result, public_wall = measure(lambda: vq.run_periodic_job(**job), repeats)
        native, native_wall = measure(lambda: function(args)(**args), repeats)
        admission = function(args, 'plan')(**args)
        if not all(r.converged and r.receipt.confirmed for r in (native, result.native)):
            raise RuntimeError('Public fitted KS failed convergence or confirmation')
        for field in ARRAYS: np.testing.assert_array_equal(getattr(native, field), getattr(result.native, field))
        if native.receipt.energy != result.energy: raise RuntimeError('Native/public method energies differ')
        domains, grid = explicit(args, a, old)
        start = time.perf_counter(); s, h, build = reference(a, k, old)
        pop = (native.plan.n_alpha,) if spins == 1 else (native.plan.n_alpha, native.plan.n_beta)
        density = independent_scf(s, h, build, pop, spins); elapsed = time.perf_counter()-start
        f, energy, counts = build(native.density)
        expected_energy = sum(_nuclear_reference(old)[0])+sum(build(density)[1])
        errors = dict(S=float(np.max(abs(s-native.overlap))), H=float(np.max(abs(h-native.hcore))),
            D=float(np.max(abs(density-native.density))), F=float(np.max(abs(f-native.fock))),
            components=float(np.max(abs(np.asarray(components(native))-energy))), energy=abs(expected_energy-native.receipt.energy),
            grid_populations=float(np.max(abs(np.asarray(native.receipt.xc.electrons_grid)-counts))))
        if any(value > (1e-8 if key == 'D' else 1e-9) for key, value in errors.items()):
            raise RuntimeError(f'Independent public fitted KS acceptance failed: {errors}')
        path = job['output'].with_suffix('.qvf')
        if not validate_qvf(path)['valid']: raise RuntimeError('Invalid public QVF')
        expected = native.density if spins == 2 else np.repeat(.5*native.density, 2, axis=0)
        np.testing.assert_array_equal(_spin_density_k_from_qvf_bloch_payload(path), expected)
        p, r = native.plan, native.receipt
        outputs = [p for p in job['output'].parent.rglob('*') if p.is_file()]
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)
        return dict(profile=profile, functional=job['functional'], spins=spins, repeats=repeats, errors=errors,
            cold_public_s=cold, repeated_native=native_wall, repeated_public=public_wall, independent_reference_s=elapsed,
            native_public_bitwise=True, qvf_density_bitwise=True, energy=r.energy, iterations=result.iterations,
            fock_calls=r.fock_calls, confirmation_fock_error=r.confirmation_fock_error,
            confirmation_energy_error=r.confirmation_energy_error, grid_population_error=r.maximum_grid_population_error,
            grid_points=p.xc.grid_points, gamma_vectors=r.domain_hartree_vectors, exchange_vectors=r.domain_exchange_vectors,
            channels=p.fitting.channels, metric_ranks=[m.rank for m in r.fitting_info.metrics],
            native_plan_bytes=p.modeled_peak_bytes, admission_bytes=admission.modeled_peak_bytes,
            public_plan_bytes=result.native.plan.modeled_peak_bytes, peak_rss_bytes=rss,
            job_reserved_bytes=result.run_fields['job_reserved_bytes'],
            python_reserved_bytes=result.integral_provenance['python_reserved_bytes'],
            retained_fit_bytes=p.fitting.retained_source_bytes, shell_calls=r.shell_calls,
            shell_bound=admission.shell_calls, primitive_bound=admission.primitive_work,
            scalar_bound=admission.scalar_terms, kernel_bound=admission.kernel_terms,
            output_files=len(outputs), output_bytes=sum(p.stat().st_size for p in outputs),
            serialized_evidence_bytes=len(json.dumps(result.integral_provenance, allow_nan=False).encode()),
            core_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            scientific_adapter_sha256=hashlib.sha256((root/'python/vibeqc/manta_periodic.py').read_bytes()).hexdigest(),
            job_adapter_sha256=hashlib.sha256((root/'python/vibeqc/manta_periodic_job.py').read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=PROFILES); parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 3: parser.error('At least three repeats required')
    if args.worker:
        print(json.dumps(worker(args.worker, args.repeats))); return
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    rows = []
    for profile in PROFILES:
        result = subprocess.run([sys.executable, __file__, '--worker', profile, '--repeats', str(args.repeats)],
                                env=env, capture_output=True, text=True, check=True)
        rows.append(json.loads(result.stdout))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__': main()
