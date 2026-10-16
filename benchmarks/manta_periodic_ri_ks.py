"""Explicit fitted KS: native and selected scientific calls, independent SCF."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time


PROFILES = tuple(f'{source}-{spin}' for source in ('pure', 'J', 'K', 'JK')
                 for spin in ('rks', 'uks', 'empty'))


def native_binary(root, core):
    binary = Path(core.__file__).resolve()
    if not binary.is_relative_to(root) or binary.stat().st_mtime <= max(
            p.stat().st_mtime for p in (root/'cpp').rglob('*') if p.suffix in ('.cpp', '.hpp', '.h')):
        raise RuntimeError('Require a fresh native build from the current checkout')
    return binary


def measure(call, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter(); result = call(); samples.append(time.perf_counter()-start)
    return result, dict(median_s=statistics.median(samples), min_s=min(samples), max_s=max(samples))


def worker(profile, repeats):
    root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
    import numpy as np
    from vibeqc import manta, _vibeqc_core as core
    from vibeqc.manta_periodic import periodic_ks_provenance, _periodic_ri_ks_configuration
    from tests.test_manta_periodic_ri_ks import case, run, reference, components, independent_scf, EMPTY_HYBRID
    from tests.test_manta_periodic_rhf import _nuclear_reference
    binary = native_binary(root, core)
    source, spin = profile.split('-'); spins = 1 if spin == 'rks' else 2
    name = ('blyp' if source == 'pure' else EMPTY_HYBRID) if spin == 'empty' else ('pbe' if source == 'pure' else 'pbe0')
    a, k, old = case(source, spins, name, species='H' if spin == 'empty' else None, order=10)
    start = time.perf_counter(); native = run(a); cold = time.perf_counter()-start
    native, native_wall = measure(lambda: run(a), repeats)
    engine = manta.IntegralEngineOptions(backend='manta', budget=a['budget'])
    args = dict(a); args.pop('budget'); args['integral_engine'] = engine
    selected_call = getattr(manta, 'run_periodic_ri_'+('ks' if source == 'pure' else 'global_hybrid_ks'))
    selected, selected_wall = measure(lambda: selected_call(**args), repeats)
    if not native.converged or not native.receipt.confirmed or not selected.converged:
        raise RuntimeError('Fitted KS failed its method convergence or confirmation gate')
    for field in ('density', 'fock', 'hcore', 'overlap', 'coefficients', 'orbital_energies'):
        np.testing.assert_array_equal(getattr(native, field), getattr(selected, field))
    _, _, python_bytes = _periodic_ri_ks_configuration(a['options'], engine, a['mesh'], source != 'pure')
    evidence = periodic_ks_provenance(selected, a['mesh'], a['options'], python_bytes)
    evidence_bytes = len(json.dumps(evidence, allow_nan=False).encode())
    start = time.perf_counter(); s, h, build = reference(a, k, old)
    populations = (native.plan.n_alpha,) if spins == 1 else (native.plan.n_alpha, native.plan.n_beta)
    d = independent_scf(s, h, build, populations, spins); reference_time = time.perf_counter()-start
    f, energy, grid_electrons = build(native.density)
    expected_energy = sum(_nuclear_reference(old)[0])+sum(build(d)[1])
    errors = dict(S=float(np.max(abs(s-native.overlap))), H=float(np.max(abs(h-native.hcore))),
        D=float(np.max(abs(d-native.density))), F=float(np.max(abs(f-native.fock))),
        components=float(np.max(abs(np.asarray(components(native))-energy))), energy=abs(expected_energy-native.receipt.energy),
        grid_populations=float(np.max(abs(np.asarray(native.receipt.xc.electrons_grid)-grid_electrons))))
    if any(value > (1e-8 if key == 'D' else 1e-9) for key, value in errors.items()):
        raise RuntimeError(f'Independent fitted KS acceptance failed: {errors}')
    p, r = native.plan, native.receipt
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)
    return dict(profile=profile, functional=name, spins=spins, repeats=repeats, errors=errors,
        cold_native_s=cold, repeated_native=native_wall, repeated_selected=selected_wall,
        independent_reference_s=reference_time, native_selected_bitwise=True, energy=r.energy,
        iterations=native.solver_snapshot.evaluated_iteration, fock_calls=r.fock_calls,
        confirmation_fock_error=r.confirmation_fock_error, confirmation_energy_error=r.confirmation_energy_error,
        grid_population_error=r.maximum_grid_population_error, grid_points=p.xc.grid_points,
        n_basis=p.n_basis, n_auxiliary=a['auxiliary'].nbasis, momenta=p.momenta, channels=p.fitting.channels,
        metric_ranks=[m.rank for m in r.fitting_info.metrics], exchange_q0=r.exchange_q0,
        modeled_peak_bytes=p.modeled_peak_bytes, selected_peak_bytes=selected.plan.modeled_peak_bytes,
        provider_workspace_bytes=p.provider_workspace_bytes, retained_fit_bytes=p.fitting.retained_source_bytes,
        peak_rss_bytes=rss, metric_shell_calls=r.fitting_info.metric_shell_calls,
        shell_calls=r.shell_calls, primitive_pairs=r.primitive_pairs, primitive_points=r.primitive_points,
        shell_bound=p.shell_calls, primitive_bound=p.primitive_work, scalar_bound=p.scalar_terms, kernel_bound=p.kernel_terms,
        serialized_evidence_bytes=evidence_bytes, core_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256((root/'python/vibeqc/manta_periodic.py').read_bytes()).hexdigest())


def conditioning():
    root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
    import numpy as np
    from vibeqc import _vibeqc_core as core
    from tests.test_manta_periodic_ri_ks import case, run, reference, minority_conditioning, components
    native_binary(root, core)
    rows = []
    for profile, name in [('pure', 'pbe'), ('JK', 'pbe0'), ('JK', 'b3lyp')]:
        for mesh, shift in [((1, 1, 1), (0, 0, 0)), ((2, 1, 1), (1, 0, 0))]:
            a, k, old = case(profile, 2, name, mesh, shift, species='H')
            r = run(a); _, _, build = reference(a, k, old); f, energies, _ = build(r.density)
            _, sensitivity = minority_conditioning(a, k, r.density)
            error = float(np.max(abs(r.fock[1]-f[1])))
            occupied = float(np.max(abs(r.fock[0]-f[0])))
            energy = float(np.max(abs(np.array(components(r))-energies)))
            if not r.converged or error > max(1e-9, 2*sensitivity) or max(occupied, energy) > 8e-11:
                raise RuntimeError('Empty-spin KS conditioning witness failed')
            rows.append(dict(source=profile, functional=name, momenta=len(k), minority_error=error,
                epsilon_sensitivity=float(sensitivity), occupied_error=occupied, component_error=energy))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=PROFILES); parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--conditioning', action='store_true', help='Measure empty-spin PBE/PBE0/B3LYP endpoints separately')
    a = parser.parse_args()
    if a.repeats < 3: parser.error('At least three repeats required')
    if a.conditioning:
        if a.worker: parser.error('--conditioning does not select a timing worker')
        print(json.dumps(conditioning(), indent=2)); return
    if a.worker:
        print(json.dumps(worker(a.worker, a.repeats))); return
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    rows = []
    for profile in PROFILES:
        completed = subprocess.run([sys.executable, __file__, '--worker', profile, '--repeats', str(a.repeats)],
            env=env, text=True, capture_output=True, check=True)
        rows.append(json.loads(completed.stdout))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
