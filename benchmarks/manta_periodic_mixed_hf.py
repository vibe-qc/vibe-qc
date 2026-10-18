"""Mixed exact/RI periodic RHF/UHF: complete native/public jobs and independent SCF."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time


PROFILES = tuple(f'{method}-{target}-{correction}' for method in ('rhf', 'uhf')
                 for target in ('J', 'K') for correction in ('strict', 'BvK'))


def measure(call, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter(); result = call(); samples.append(time.perf_counter()-start)
    return result, dict(median_s=statistics.median(samples), min_s=min(samples), max_s=max(samples))


def worker(profile, repeats):
    root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
    import numpy as np
    import vibeqc as vq
    from vibeqc import manta, _vibeqc_core as core
    from vibeqc.guess_read import _spin_density_k_from_qvf_bloch_payload
    from vibeqc.output import validate_qvf
    from tests.test_manta_periodic_mixed_hf import job_case, solve_reference
    from tests.test_manta_periodic_ri_domains import explicit
    from tests.test_manta_periodic_rhf import _nuclear_reference
    binary = Path(core.__file__).resolve()
    if not binary.is_relative_to(root) or binary.stat().st_mtime <= max(
            p.stat().st_mtime for p in (root/'cpp').rglob('*') if p.suffix in ('.cpp', '.hpp', '.h')):
        raise RuntimeError('Require a fresh native build from the current checkout')
    method, target, correction = profile.split('-')
    with tempfile.TemporaryDirectory(prefix='manta-mixed-hf-') as directory:
        job, a, k = job_case(Path(directory), method, target, 'BvK-ewald' if correction == 'BvK' else 'strict-zero-mode')
        start = time.perf_counter(); result = vq.run_periodic_job(**job)
        cold = time.perf_counter()-start
        result, public_wall = measure(lambda: vq.run_periodic_job(**job), repeats)
        native, native_wall = measure(lambda: getattr(core._manta, 'run_periodic_ri_'+method+'_from_cutoffs')(**a), repeats)
        if not result.converged or not native.converged:
            raise RuntimeError('Mixed HF failed convergence/confirmation')
        for field in ('density', 'fock', 'hcore', 'overlap'):
            np.testing.assert_array_equal(getattr(native, field), getattr(result.native, field))
        domains = getattr(manta, 'build_periodic_ri_'+method+'_domains')(**a); c = explicit(a, domains)
        start = time.perf_counter(); d, s, h, build = solve_reference(c, k, method)
        reference_time = time.perf_counter()-start
        f, components = build(native.density)
        actual = [getattr(native.receipt, name) for name in ('one_electron_energy', 'coulomb_energy',
            'uncorrected_exchange_energy', 'exchange_correction_energy')]
        expected_energy = sum(_nuclear_reference(c)[0])+sum(build(d)[1])
        errors = dict(S=float(np.max(abs(s-native.overlap))), H=float(np.max(abs(h-native.hcore))),
            D=float(np.max(abs(d-native.density))), F=float(np.max(abs(f-native.fock))),
            components=float(np.max(abs(np.asarray(actual)-components))), energy=abs(expected_energy-native.receipt.energy))
        if any(value > (1e-8 if name == 'D' else 1e-9) for name, value in errors.items()):
            raise RuntimeError(f'Independent mixed HF acceptance failed: {errors}')
        qvf = job['output'].with_suffix('.qvf')
        if not validate_qvf(qvf)['valid']:
            raise RuntimeError('Invalid public QVF')
        saved = _spin_density_k_from_qvf_bloch_payload(qvf)
        np.testing.assert_array_equal(saved, np.stack([.5*native.density]*2) if method == 'rhf' else native.density)
        receipt, plan = native.receipt, native.plan
        output_files = sum(p.is_file() for p in job['output'].parent.rglob('*'))
        direct = receipt.exact_hartree if target == 'K' else receipt.exact_exchange
        return dict(profile=profile, repeats=repeats, errors=errors, cold_public_s=cold,
            repeated_public=public_wall, repeated_native=native_wall, independent_reference_s=reference_time,
            energy=receipt.energy, iterations=native.solver_snapshot.evaluated_iteration, fock_calls=receipt.fock_calls,
            confirmation_fock_error=receipt.confirmation_fock_error, confirmation_energy_error=receipt.confirmation_energy_error,
            nao=plan.n_basis, naux=a['auxiliary'].nbasis, nk=plan.momenta, metric_owners=plan.fock.channels,
            metric_ranks=[m.rank for m in receipt.fitting_info.metrics],
            generated_peak_bytes=domains.plan.modeled_peak_bytes, executed_peak_bytes=plan.modeled_peak_bytes,
            public_peak_bytes=result.native.plan.modeled_peak_bytes, two_electron_peak_bytes=plan.two_electron_peak_bytes,
            retained_fit_bytes=plan.fock.retained_source_bytes, shell_calls=receipt.shell_calls, primitive_pairs=receipt.primitive_pairs,
            shell_bound=plan.shell_calls, primitive_bound=plan.primitive_pairs, scalar_bound=plan.scalar_terms,
            kernel_bound=plan.kernel_terms, terminal_direct_shell_calls=direct.shell_calls,
            gamma_vectors=len(domains.reciprocal_vectors), pair_vectors=len(domains.pair_vectors),
            output_files=output_files, qvf_density_bitwise=True,
            native_public_bitwise=True, core_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            adapter_sha256=hashlib.sha256((root/'python/vibeqc/manta_periodic.py').read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=PROFILES); parser.add_argument('--repeats', type=int, default=5)
    a = parser.parse_args()
    if a.repeats < 3:
        parser.error('At least three repeats required')
    if a.worker:
        print(json.dumps(worker(a.worker, a.repeats))); return
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    rows = []
    for profile in PROFILES:
        run = subprocess.run([sys.executable, __file__, '--worker', profile, '--repeats', str(a.repeats)],
            env=env, text=True, capture_output=True, check=True)
        rows.append(json.loads(run.stdout))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
