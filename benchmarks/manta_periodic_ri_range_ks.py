"""Bounded range KS: isolated complete native/selected builds and reference SCF."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def worker(profile, spins, name):
    root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
    import numpy as np
    from vibeqc import manta, _vibeqc_core as core
    from benchmarks.manta_periodic_ri_ks import native_binary, measure
    from tests.test_manta_periodic_ri_range_ks import case, run, check_scf, MIXED_RANGE
    binary = native_binary(root, core)
    a, k, old = case(profile, spins, MIXED_RANGE if name == 'mixed' else name)
    native, native_wall = measure(lambda: run(a), 5)
    args = dict(a); budget = args.pop('budget'); args['integral_engine'] = manta.IntegralEngineOptions(backend='manta', budget=budget)
    selected, selected_wall = measure(lambda: manta.run_periodic_ri_range_separated_ks(**args), 5)
    for field in ('density', 'fock', 'hcore', 'overlap', 'coefficients', 'orbital_energies'):
        np.testing.assert_array_equal(getattr(native, field), getattr(selected, field))
    start = time.perf_counter(); _, errors = check_scf(a, k, old); reference_time = time.perf_counter()-start
    p, r = native.plan, native.receipt
    fits = ([r.fitting_info] if r.fitting_info else [])+[c.fitting_info for c in r.exchange_components if c.fitting_info]
    return dict(profile=profile, spins=spins, functional=a['functional'].name, repeats=5, errors=errors,
        native_wall=native_wall, selected_wall=selected_wall, reference_s=reference_time, native_selected_bitwise=True,
        energy=r.energy, components={key:getattr(r, key) for key in ('one_electron_energy', 'coulomb_energy', 'xc_energy',
            'uncorrected_exchange_energy', 'exchange_correction_energy', 'nuclear_energy')},
        momenta=p.momenta, grid_points=p.xc.grid_points, grid_population_error=r.maximum_grid_population_error,
        iterations=native.solver_snapshot.evaluated_iteration, fock_calls=r.fock_calls,
        confirmation_fock_error=r.confirmation_fock_error, confirmation_energy_error=r.confirmation_energy_error,
        exchange_q0=r.exchange_q0, fit_sources=len(fits), metric_channels=sum(len(f.metrics) for f in fits),
        metric_ranks=[[m.rank for m in f.metrics] for f in fits], metric_shell_calls=sum(f.metric_shell_calls for f in fits),
        shell_calls=r.shell_calls, shell_bound=p.shell_calls, primitive_pairs=r.primitive_pairs, primitive_bound=p.primitive_work,
        kernel_bound=p.kernel_terms, scalar_bound=p.scalar_terms, native_peak_bytes=p.modeled_peak_bytes,
        selected_peak_bytes=selected.plan.modeled_peak_bytes, provider_retained_bytes=p.solver.provider_retained_bytes,
        provider_workspace_bytes=p.provider_workspace_bytes, source_identity=r.source_identity,
        fitting_omega=a['options'].fitting.omega, pair_cutoff_bohr=a['options'].fitting.pair_cutoff_bohr,
        auxiliary_cutoff_bohr=a['options'].fitting.auxiliary_cutoff_bohr,
        core_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        python_sha256=hashlib.sha256((root/'python/vibeqc/manta_periodic.py').read_bytes()).hexdigest())


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
