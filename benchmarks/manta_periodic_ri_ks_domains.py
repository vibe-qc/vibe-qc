"""Generated fitted KS: joint quadrature/domain/SCF and selected output costs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

PROFILES = tuple(f'{source}-{spin}' for source in ('pure', 'J', 'K', 'JK') for spin in ('rks', 'uks', 'empty'))


def worker(profile, repeats):
    root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
    import numpy as np
    from vibeqc import manta, _vibeqc_core as core
    from vibeqc.manta_periodic import _periodic_ri_ks_configuration, periodic_ks_provenance
    from benchmarks.manta_periodic_ri_ks import native_binary, measure
    from tests.test_manta_periodic_ri_ks_domains import arguments, function, explicit, ARRAYS
    from tests.test_manta_periodic_ri_ks import reference, independent_scf, components, run
    from tests.test_manta_periodic_rhf import _nuclear_reference
    binary = native_binary(root, core)
    source, spin = profile.split('-'); spins = 1 if spin == 'rks' else 2
    args, a, k, old = arguments(source, spins, grid=True, species='H' if spin == 'empty' else None)
    admission = function(args, 'plan')(**args)
    start = time.perf_counter(); native = function(args)(**args); cold = time.perf_counter()-start
    native, wall = measure(lambda: function(args)(**args), repeats)
    selected_args = dict(args); budget = selected_args.pop('budget')
    engine = manta.IntegralEngineOptions(backend='manta', budget=budget); selected_args['integral_engine'] = engine
    selected, selected_wall = measure(lambda: function(args, namespace=manta)(**selected_args), repeats)
    domains, grid = explicit(args, a, old); separate = run(a)
    for field in ARRAYS:
        np.testing.assert_array_equal(getattr(native, field), getattr(selected, field))
        np.testing.assert_array_equal(getattr(native, field), getattr(separate, field))
    if not all(r.converged and r.receipt.confirmed for r in (native, selected, separate)):
        raise RuntimeError('Generated fitted KS failed convergence or confirmation')
    if not native.receipt.energy == selected.receipt.energy == separate.receipt.energy:
        raise RuntimeError('Generated/explicit/selected method energies differ')
    _, _, python_bytes = _periodic_ri_ks_configuration(args['options'], engine, args['mesh'], source != 'pure')
    evidence = periodic_ks_provenance(selected, args['mesh'], args['options'], python_bytes)
    start = time.perf_counter(); s, h, build = reference(a, k, old)
    populations = (native.plan.n_alpha,) if spins == 1 else (native.plan.n_alpha, native.plan.n_beta)
    density = independent_scf(s, h, build, populations, spins); elapsed = time.perf_counter()-start
    f, energy, counts = build(native.density)
    expected_energy = sum(_nuclear_reference(old)[0])+sum(build(density)[1])
    errors = dict(S=float(np.max(abs(s-native.overlap))), H=float(np.max(abs(h-native.hcore))),
        D=float(np.max(abs(density-native.density))), F=float(np.max(abs(f-native.fock))),
        components=float(np.max(abs(np.asarray(components(native))-energy))), energy=abs(expected_energy-native.receipt.energy),
        grid_populations=float(np.max(abs(np.asarray(native.receipt.xc.electrons_grid)-counts))))
    if any(value > (1e-8 if key == 'D' else 1e-9) for key, value in errors.items()):
        raise RuntimeError(f'Independent generated fitted KS acceptance failed: {errors}')
    p, r = native.plan, native.receipt
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)
    return dict(profile=profile, functional=args['functional'].name, spins=spins, repeats=repeats, errors=errors,
        cold_native_s=cold, repeated_native=wall, repeated_selected=selected_wall, independent_reference_s=elapsed,
        explicit_generated_selected_bitwise=True, energy=r.energy, iterations=native.solver_snapshot.evaluated_iteration,
        fock_calls=r.fock_calls, confirmation_fock_error=r.confirmation_fock_error,
        confirmation_energy_error=r.confirmation_energy_error, grid_population_error=r.maximum_grid_population_error,
        grid_points=p.xc.grid_points, radial_points=args['grid_options'].radial_points,
        lebedev_order=args['grid_options'].lebedev_order, partition_radius=args['grid_options'].partition_radius_bohr,
        cutoff=args['reciprocal_energy_cutoff'], xc_radius=args['xc_image_cutoff_bohr'],
        translations=r.domain_translations, gamma_vectors=r.domain_hartree_vectors,
        exchange_vectors=r.domain_exchange_vectors, xc_images=r.domain_xc_images,
        channels=p.fitting.channels, metric_ranks=[m.rank for m in r.fitting_info.metrics],
        actual_plan_bytes=p.modeled_peak_bytes, admission_bytes=admission.modeled_peak_bytes,
        selected_plan_bytes=selected.plan.modeled_peak_bytes, domain_bytes=p.domain_live_bytes,
        grid_bytes=p.quadrature_live_bytes, retained_fit_bytes=p.fitting.retained_source_bytes,
        peak_rss_bytes=rss, shell_calls=r.shell_calls, shell_bound=admission.shell_calls,
        primitive_bound=admission.primitive_work, scalar_bound=admission.scalar_terms, kernel_bound=admission.kernel_terms,
        serialized_evidence_bytes=len(json.dumps(evidence, allow_nan=False).encode()),
        core_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256((root/'python/vibeqc/manta_periodic.py').read_bytes()).hexdigest())


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
