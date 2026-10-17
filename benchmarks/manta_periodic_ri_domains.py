"""Certified RI UHF cutoff sources: independent finite-source acceptance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROFILES = ('H-gamma-strict', 'H-shifted-strict', 'Li-shifted-strict',
            'H-gamma-BvK', 'H-shifted-BvK', 'Li-shifted-BvK')


def worker(profile, repeats):
    root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
    import numpy as np
    from vibeqc import manta, _vibeqc_core as core
    from benchmarks.manta_periodic_ri_uhf import measure
    from tests.test_manta_periodic_ri_domains import arguments, explicit
    from tests.test_manta_periodic_ri_uhf import reference
    from tests.test_manta_periodic_uhf import _scf_reference
    from tests.test_manta_periodic_rhf import _nuclear_reference

    native = Path(core.__file__).resolve()
    if not native.is_relative_to(root) or native.stat().st_mtime <= max(
            p.stat().st_mtime for p in (root/'cpp').rglob('*') if p.suffix in ('.cpp', '.hpp', '.h')):
        raise RuntimeError('Require a fresh native build from the current checkout')
    species, mesh, correction = profile.split('-')
    shape, shift = ((1, 1, 1), (0, 0, 0)) if mesh == 'gamma' else ((2, 1, 1), (1, 0, 0))
    a, k = arguments(shape, shift, species, 'BvK-ewald' if correction == 'BvK' else 'strict-zero-mode')
    p, planning = measure(lambda: manta.plan_periodic_ri_uhf_from_cutoffs(**a), repeats)
    domains, generation = measure(lambda: manta.build_periodic_ri_uhf_domains(**a), repeats)
    c = explicit(a, domains)
    start = time.perf_counter(); r = core._manta.run_periodic_ri_uhf_from_cutoffs(**a)
    cold = time.perf_counter()-start
    r, repeated = measure(lambda: core._manta.run_periodic_ri_uhf_from_cutoffs(**a), repeats)
    if not r.converged or not r.receipt.confirmed:
        raise RuntimeError('Native cutoff RI UHF did not pass its terminal gates')
    raw = core._manta.run_periodic_ri_uhf(**c)
    if r.receipt.source_identity != raw.receipt.source_identity:
        raise RuntimeError('Generated and explicit scientific sources differ')
    np.testing.assert_array_equal(r.fock, raw.fock)
    np.testing.assert_array_equal(r.density, raw.density)
    start = time.perf_counter(); s, h, build, _ = reference(c, k)
    nuclear = sum(_nuclear_reference(c)[0]); ref_setup = time.perf_counter()-start
    start = time.perf_counter(); d = _scf_reference(s, h, build, (r.plan.n_alpha, r.plan.n_beta))
    ref_scf = time.perf_counter()-start
    f, energies = build(r.density)
    actual = (r.receipt.one_electron_energy, r.receipt.coulomb_energy,
              r.receipt.uncorrected_exchange_energy, r.receipt.exchange_correction_energy)
    errors = dict(S=float(np.max(abs(s-r.overlap))), H=float(np.max(abs(h-r.hcore))),
        D=float(np.max(abs(d-r.density))), F=float(np.max(abs(f-r.fock))),
        same_density_components=float(np.max(abs(np.asarray(actual)-energies))),
        independent_scf_energy=abs(nuclear+sum(build(d)[1])-r.receipt.energy))
    if any(value > (1e-8 if name == 'D' else 1e-9) for name, value in errors.items()):
        raise RuntimeError(f'Independent finite-source parity failed: {errors}')
    return dict(profile=profile, repeats=repeats, core_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        planning=planning, generation=generation, cold_scf_s=cold, repeated_scf=repeated,
        reference_setup_s=ref_setup, reference_scf_s=ref_scf, errors=errors,
        n_orb=r.plan.n_basis, n_aux=a['auxiliary'].nbasis, momenta=len(k), channels=r.plan.fock.channels,
        reciprocal_energy_cutoff=a['reciprocal_energy_cutoff'], energy=r.receipt.energy,
        iterations=r.solver_snapshot.evaluated_iteration, fock_calls=r.receipt.fock_calls,
        translations=len(domains.translations), gamma_vectors=len(domains.reciprocal_vectors),
        pair_vectors=len(domains.pair_vectors), conjugacy_vectors=domains.receipt.conjugacy_vectors,
        metric_ranks=[m.rank for m in r.receipt.fitting_info.metrics],
        domain_live_bytes=p.domain_live_bytes, candidate_peak_bytes=p.modeled_peak_bytes,
        executed_plan_peak_bytes=r.plan.modeled_peak_bytes,
        retained_source_bytes=r.plan.fock.retained_source_bytes,
        candidate_shell_bound=p.shell_calls, executed_shell_bound=r.plan.shell_calls,
        total_shell_calls=r.receipt.shell_calls, total_primitive_products=r.receipt.primitive_pairs,
        candidate_visits=p.candidate_visits, scalar_bound=p.scalar_terms, kernel_bound=p.kernel_terms,
        confirmation_fock_error=r.receipt.confirmation_fock_error,
        confirmation_energy_error=r.receipt.confirmation_energy_error,
        density_closure=r.solver_snapshot.maximum_density_closure_frobenius)


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
