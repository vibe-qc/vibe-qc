"""Generated range KS: complete native/selected solves and independent SCF."""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def worker(profile, spins, name):
    root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
    import numpy as np
    from vibeqc import manta, _vibeqc_core as core
    from vibeqc.manta_periodic import _periodic_ri_range_ks_configuration, periodic_ks_provenance
    from benchmarks.manta_periodic_ri_ks import native_binary, measure
    from tests.test_manta_periodic_ri_range_ks_domains import (
        arguments, function, explicit, explicit_run, ARRAYS, check_reference, MIXED_RANGE,
    )
    binary = native_binary(root, core)
    args, a, k, old = arguments(profile, spins, grid=True, name=MIXED_RANGE if name == 'mixed' else name)
    admission = function(args, 'plan')(**args)
    start = time.perf_counter(); native = function(args)(**args); cold = time.perf_counter()-start
    native, wall = measure(lambda: function(args)(**args), 5)
    selected_args = dict(args); budget = selected_args.pop('budget')
    engine = manta.IntegralEngineOptions(backend='manta', budget=budget); selected_args['integral_engine'] = engine
    selected, selected_wall = measure(lambda: function(args, namespace=manta)(**selected_args), 5)
    domains, grid = explicit(args, a, old); separate = explicit_run(a)
    for field in ARRAYS:
        np.testing.assert_array_equal(getattr(native, field), getattr(selected, field))
        np.testing.assert_array_equal(getattr(native, field), getattr(separate, field))
    if not all(r.converged and r.receipt.confirmed for r in (native, selected, separate)):
        raise RuntimeError('Generated range KS failed convergence or confirmation')
    if not native.receipt.energy == selected.receipt.energy == separate.receipt.energy:
        raise RuntimeError('Generated/explicit/selected method energies differ')
    _, _, python_bytes = _periodic_ri_range_ks_configuration(args['options'], engine, args['mesh'], args['functional'])
    evidence = periodic_ks_provenance(selected, args['mesh'], args['options'], python_bytes)
    start = time.perf_counter(); errors = check_reference(native, a, k, old); elapsed = time.perf_counter()-start
    p, r = native.plan, native.receipt
    fits = ([r.fitting_info] if r.fitting_info else [])+[c.fitting_info for c in r.exchange_components if c.fitting_info]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)
    return dict(profile=profile, functional=args['functional'].name, spins=spins, repeats=5, errors=errors,
        cold_native_s=cold, repeated_native=wall, repeated_selected=selected_wall, independent_reference_s=elapsed,
        explicit_generated_selected_bitwise=True, energy=r.energy, iterations=native.solver_snapshot.evaluated_iteration,
        fock_calls=r.fock_calls, confirmation_fock_error=r.confirmation_fock_error,
        confirmation_energy_error=r.confirmation_energy_error, grid_population_error=r.maximum_grid_population_error,
        grid_points=p.xc.grid_points, radial_points=args['grid_options'].radial_points,
        lebedev_order=args['grid_options'].lebedev_order, partition_radius=args['grid_options'].partition_radius_bohr,
        cutoff=args['reciprocal_energy_cutoff'], xc_radius=args['xc_image_cutoff_bohr'],
        translations=r.domain_translations, gamma_vectors=r.domain_hartree_vectors,
        exchange_vectors=r.domain_exchange_vectors, xc_images=r.domain_xc_images,
        fit_sources=len(fits), metric_channels=sum(len(f.metrics) for f in fits),
        metric_ranks=[[m.rank for m in f.metrics] for f in fits],
        actual_plan_bytes=p.modeled_peak_bytes, admission_bytes=admission.modeled_peak_bytes,
        selected_plan_bytes=selected.plan.modeled_peak_bytes, domain_bytes=p.domain_live_bytes,
        grid_bytes=p.quadrature_live_bytes, provider_retained_bytes=p.solver.provider_retained_bytes,
        peak_rss_bytes=rss, shell_calls=r.shell_calls, shell_bound=admission.shell_calls,
        primitive_bound=admission.primitive_work, scalar_bound=admission.scalar_terms, kernel_bound=admission.kernel_terms,
        serialized_evidence_bytes=len(json.dumps(evidence, allow_nan=False).encode()),
        source_identity=r.source_identity, grid_identity=r.quadrature.source_identity, domain_identity=r.domain_identity,
        core_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256((root/'python/vibeqc/manta_periodic.py').read_bytes()).hexdigest())


if __name__ == '__main__':
    if len(sys.argv) > 1: print(json.dumps(worker(sys.argv[1], int(sys.argv[2]), sys.argv[3])))
    else:
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
        rows = []
        for profile in ('J', 'K', 'JK'):
            for spins in (1, 2):
                for name in ('hse06', 'cam-b3lyp', 'mixed'):
                    result = subprocess.run([sys.executable, __file__, profile, str(spins), name], env=env,
                                            capture_output=True, text=True, check=True)
                    rows.append(json.loads(result.stdout))
        print(json.dumps(rows, indent=2))
