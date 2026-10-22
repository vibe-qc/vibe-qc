"""Whole-mesh streamed RI Fock and physical UHF: finite-source acceptance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


PROFILES = ('fock-1', 'fock-2', 'fock-4', 'H-gamma-strict', 'H-shifted-strict',
            'Li-shifted-strict', 'H-gamma-BvK', 'H-shifted-BvK', 'Li-shifted-BvK')


def measure(call, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter(); result = call(); samples.append(time.perf_counter()-start)
    return result, dict(median_s=statistics.median(samples), min_s=min(samples), max_s=max(samples))


def worker(profile, repeats):
    root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
    import numpy as np
    from vibeqc import manta, _vibeqc_core as core

    native = Path(core.__file__).resolve()
    if not native.is_relative_to(root) or native.stat().st_mtime <= max(
            p.stat().st_mtime for p in (root/'cpp').rglob('*') if p.suffix in ('.cpp', '.hpp', '.h')):
        raise RuntimeError('Require a fresh native build from the current checkout')
    if profile.startswith('fock-'):
        from tests.test_manta_periodic_ri_fock import case, oracle
        from tests.test_manta_periodic_ri_q import densities
        a = case(int(profile[-1])); nk = len(a['momenta'])
        start = time.perf_counter(); builder = manta.PeriodicRIFockBuilder(**a)
        setup = time.perf_counter()-start
        h = densities(nk, spins=1, seed=831)[0]; d = densities(nk)
        start = time.perf_counter(); r = builder.build(h, d); cold = time.perf_counter()-start
        r, wall = measure(lambda: builder.build(h, d), repeats)
        start = time.perf_counter(); j, k, f, e = oracle(a, h, d)
        oracle_time = time.perf_counter()-start
        errors = dict(J=float(np.max(abs(j-r.J))), K=float(np.max(abs(k-r.K))),
                      F=float(np.max(abs(f-r.values))), energy=abs(float(e[-1])-r.receipt.electronic_energy))
        p = builder.plan
        row = dict(setup_s=setup, cold_build_s=cold, warm_build=wall, independent_oracle_s=oracle_time,
            n_orb=p.n_basis, n_aux=p.n_auxiliary, momenta=nk, channels=p.channels,
            metric_ranks=[m.rank for m in builder.info.metrics], errors=errors,
            modeled_peak_bytes=p.modeled_peak_bytes, retained_source_bytes=p.retained_source_bytes,
            metric_shell_calls=builder.info.metric_shell_calls, build_shell_calls=r.receipt.shell_calls,
            metric_primitive_products=builder.info.metric_primitive_products,
            build_primitive_products=r.receipt.primitive_products, coupling_passes=r.receipt.coupling_passes,
            shell_bound=p.shell_calls, primitive_bound=p.primitive_products,
            scalar_bound=p.scalar_terms, kernel_bound=p.kernel_terms)
    else:
        from tests.test_manta_periodic_ri_uhf import case, reference
        from tests.test_manta_periodic_uhf import _scf_reference
        from tests.test_manta_periodic_rhf import _nuclear_reference
        species, mesh, correction = profile.split('-')
        shape, shift = ((1, 1, 1), (0, 0, 0)) if mesh == 'gamma' else ((2, 1, 1), (1, 0, 0))
        a, k = case(shape, shift, species, 'BvK-ewald' if correction == 'BvK' else 'strict-zero-mode')
        start = time.perf_counter(); r = core._manta.run_periodic_ri_uhf(**a)
        cold = time.perf_counter()-start
        r, wall = measure(lambda: core._manta.run_periodic_ri_uhf(**a), repeats)
        if not r.converged or not r.receipt.confirmed:
            raise RuntimeError('Native fitted UHF did not pass its terminal gates')
        start = time.perf_counter(); s, h, build, _ = reference(a, k)
        nuclear = sum(_nuclear_reference(a)[0]); reference_setup = time.perf_counter()-start
        start = time.perf_counter(); d = _scf_reference(s, h, build, (r.plan.n_alpha, r.plan.n_beta))
        reference_scf = time.perf_counter()-start
        f, energies = build(r.density)
        actual = (r.receipt.one_electron_energy, r.receipt.coulomb_energy,
                  r.receipt.uncorrected_exchange_energy, r.receipt.exchange_correction_energy)
        errors = dict(S=float(np.max(abs(s-r.overlap))), H=float(np.max(abs(h-r.hcore))),
            D=float(np.max(abs(d-r.density))), F=float(np.max(abs(f-r.fock))),
            same_density_components=float(np.max(abs(np.asarray(actual)-energies))),
            independent_scf_energy=abs(nuclear+sum(build(d)[1])-r.receipt.energy))
        p = r.plan
        row = dict(cold_scf_s=cold, repeated_scf=wall, reference_setup_s=reference_setup,
            reference_scf_s=reference_scf, n_orb=p.n_basis, n_aux=a['auxiliary'].nbasis,
            momenta=p.momenta, channels=p.fock.channels,
            metric_ranks=[m.rank for m in r.receipt.fitting_info.metrics],
            errors=errors, energy=r.receipt.energy, iterations=r.solver_snapshot.evaluated_iteration,
            fock_calls=r.receipt.fock_calls, confirmation_fock_error=r.receipt.confirmation_fock_error,
            confirmation_energy_error=r.receipt.confirmation_energy_error,
            density_closure=r.solver_snapshot.maximum_density_closure_frobenius,
            modeled_peak_bytes=p.modeled_peak_bytes, retained_source_bytes=p.fock.retained_source_bytes,
            metric_shell_calls=r.receipt.fitting_info.metric_shell_calls,
            total_shell_calls=r.receipt.shell_calls, total_primitive_products=r.receipt.primitive_pairs,
            shell_bound=p.shell_calls, primitive_bound=p.primitive_pairs,
            scalar_bound=p.scalar_terms, kernel_bound=p.kernel_terms)
    if any(value > (1e-8 if name == 'D' else 1e-9) for name, value in errors.items()):
        raise RuntimeError(f'Independent finite-source parity failed: {errors}')
    return dict(profile=profile, repeats=repeats, **row,
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
        result = subprocess.run([sys.executable, __file__, '--worker', profile, '--repeats', str(args.repeats)],
            env=env, text=True, capture_output=True, check=True)
        rows.append(json.loads(result.stdout))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
