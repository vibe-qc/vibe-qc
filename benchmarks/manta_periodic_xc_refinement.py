"""Joint frozen-density grid assessment, independent XC contractions and timing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def fresh_core():
    root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
    from vibeqc import _vibeqc_core as core
    native = Path(core.__file__).resolve()
    if not native.is_relative_to(root) or native.stat().st_mtime <= max(
        p.stat().st_mtime for p in (root/'cpp').rglob('*') if p.suffix in ('.cpp', '.hpp', '.h')):
        raise RuntimeError('Require fresh own-checkout native build')
    return hashlib.sha256(native.read_bytes()).hexdigest()


def worker(profile, repeats):
    core_hash = fresh_core()
    import numpy as np
    from vibeqc import manta
    from tests.test_manta_periodic_quadrature import xc_case, tensor_reference
    from tests.test_manta_periodic_semilocal_xc import reference
    from tests.test_manta_periodic_range_separated_ks import EMPTY_RANGE
    name, spins = {'lda-rks': ('lda', 1), 'pbe-uks': ('pbe', 2), 'empty-blyp': ('blyp', 2),
        'pbe0-rks': ('pbe0', 1), 'cam-uks': ('cam-b3lyp', 2), 'empty-lc-blyp': (EMPTY_RANGE, 2)}[profile]
    hybrid = profile in ('pbe0-rks', 'cam-uks', 'empty-lc-blyp')
    assess = manta.assess_periodic_hybrid_xc_refinement if hybrid else manta.assess_periodic_xc_refinement
    system, c = xc_case(name, spins)
    if profile.startswith('empty-'): c['densities'][1] = 0
    grids = []
    for radial, angular, radius in ((16, 29, 8.), (24, 35, 8.), (40, 59, 12.)):
        o = manta.PeriodicQuadratureOptions()
        o.radial_points = radial; o.lebedev_order = angular; o.partition_radius_bohr = radius
        grids.append(o)
    args = {key: c[key] for key in ('basis', 'functional', 'images', 'momenta', 'weights', 'densities', 'grid_batch', 'budget')}
    args.update(system=system, grid_options=grids, reference_index=2)
    start = time.perf_counter(); assess(**args); cold = time.perf_counter()-start
    wall, cpu = [], []
    for _ in range(repeats):
        result = None; start = time.perf_counter(); t = time.process_time()
        result = assess(**args)
        wall.append(time.perf_counter()-start); cpu.append(time.process_time()-t)
    independent = []
    for options in grids:
        grid = manta.build_periodic_quadrature(system, options, c['budget'])
        points, weights = grid.points, grid.weights
        energy, potential, populations = 0., np.zeros_like(c['densities']), np.zeros(spins)
        for start in range(0, len(points), 2048):
            part = dict(c, points=points[start:start+2048], quadrature_weights=weights[start:start+2048])
            e, v, n = reference(part, c['functional']); energy += e; potential += v; populations += n
        independent.append((energy, potential, populations))
    errors = dict(energy=0., potential=0., populations=0., comparisons=0.)
    target = independent[2]
    errors['potential'] = float(np.max(abs(result.reference_potential-target[1])))
    rows = []
    for step, qplan, witness in zip(result.steps, result.plan.quadrature, independent):
        e, v, n = witness
        errors['energy'] = max(errors['energy'], abs(step.xc.energy-e))
        errors['populations'] = max(errors['populations'], float(np.max(abs(np.array(step.xc.electrons_grid)-n))))
        delta = v-target[1]
        errors['comparisons'] = max(errors['comparisons'], abs(step.energy_difference-e+target[0]),
            float(np.max(abs(np.array(step.potential_max_abs).reshape(spins, 2)-np.max(abs(delta), axis=(-1, -2))))),
            float(np.max(abs(np.array(step.potential_frobenius).reshape(spins, 2)-np.linalg.norm(delta, axis=(-1, -2))))))
        rows.append(dict(radial=step.options.radial_points, angular=step.options.lebedev_order,
            radius=step.options.partition_radius_bohr, points=qplan.grid_points, energy_difference=step.energy_difference,
            maximum_potential_difference=max(step.potential_max_abs), maximum_frobenius_difference=max(step.potential_frobenius),
            maximum_population_difference=max(abs(x) for x in step.population_differences)))
    if max(errors.values()) > 3e-11: raise RuntimeError('Independent finite-grid XC assessment failed')
    reference64 = tensor_reference(c, 64); reference80 = tensor_reference(c, 80)
    reference_change = dict(energy=abs(reference64[0]-reference80[0]),
        potential=float(np.max(abs(reference64[1]-reference80[1]))))
    grid_difference = dict(energy=abs(target[0]-reference80[0]), potential=float(np.max(abs(target[1]-reference80[1]))))
    return dict(profile=profile, cold_s=cold, median_s=statistics.median(wall), min_s=min(wall), max_s=max(wall),
        comparison=result.comparison, semilocal_component_only=result.plan.semilocal_component_only,
        required_exact_exchange_fraction=result.steps[0].xc.required_exact_exchange_fraction,
        required_exchange_components=result.steps[0].xc.required_exchange_components,
        cpu_median_s=statistics.median(cpu), repeats=repeats, grid_comparisons=rows, independent_errors=errors,
        tensor_64_to_80=reference_change, selected_grid_vs_tensor80=grid_difference,
        modeled_peak_bytes=result.plan.modeled_peak_bytes, driver_live_bytes=result.plan.driver_live_bytes,
        shell_calls=result.shell_calls, primitive_points=result.primitive_points, scalar_bound=result.plan.scalar_terms,
        kernel_bound=result.plan.kernel_terms, core_sha256=core_hash)


def physical_xc_case():
    """One confirmed physical density reused unchanged by grid diagnostics."""
    import numpy as np
    from vibeqc import manta
    from tests.test_manta_periodic_ks_grid import case, explicit
    args, c, k, old = case('HeH', 2)
    explicit(args, c, old)
    state = manta.run_periodic_ks_with_quadrature(**args)
    if not state.converged or not state.receipt.confirmed:
        raise RuntimeError('Require confirmed finite-source starting density')
    xc = dict(basis=c['basis'], functional=c['functional'], lattice=c['system'].lattice,
        images=c['xc_images'], momenta=k, weights=np.full(len(k), 1/len(k)),
        densities=state.density, grid_batch=128, budget=c['budget'])
    return c['system'], xc


def physical_refinement():
    """A harder fixed physical UKS state; one diagnostic run, not a speed claim."""
    core_hash = fresh_core()
    import numpy as np
    from vibeqc import manta
    from tests.test_manta_periodic_quadrature import tensor_reference
    system, xc = physical_xc_case()
    grids = []
    for radial, order, radius in ((40, 59, 8.), (80, 95, 8.), (80, 95, 16.), (120, 95, 20.)):
        o = manta.PeriodicQuadratureOptions()
        o.radial_points = radial; o.lebedev_order = order; o.partition_radius_bohr = radius
        grids.append(o)
    controls = {key: xc[key] for key in ('basis', 'functional', 'images', 'momenta', 'weights', 'densities', 'grid_batch', 'budget')}
    controls.update(system=system, grid_options=grids, reference_index=3)
    controls['budget'].scalar_terms = 10**18; controls['budget'].kernel_terms = 2**64-1
    start = time.perf_counter(); result = manta.assess_periodic_xc_refinement(**controls)
    elapsed = time.perf_counter()-start
    rows = [dict(radial=s.options.radial_points, angular=s.options.lebedev_order,
        radius=s.options.partition_radius_bohr, points=p.grid_points, energy=s.xc.energy,
        energy_difference=s.energy_difference, potential_max_abs=s.potential_max_abs,
        potential_frobenius=s.potential_frobenius, populations=s.xc.electrons_grid)
        for s, p in zip(result.steps, result.plan.quadrature)]
    tensors, changes = [], []
    previous = None
    for order in (64, 96, 128):
        start = time.perf_counter(); current = tensor_reference(xc, order)
        tensors.append(dict(order=order, seconds=time.perf_counter()-start, energy=current[0],
            populations=current[2].tolist(), grid_energy_difference=abs(rows[3]['energy']-current[0]),
            grid_potential_difference=float(np.max(abs(result.reference_potential-current[1])))))
        if previous is not None:
            changes.append(dict(from_order=previous[0], to_order=order,
                energy=abs(current[0]-previous[1][0]),
                potential_max_abs=np.max(abs(current[1]-previous[1][1]), axis=(-1, -2)).tolist(),
                populations=float(np.max(abs(current[2]-previous[1][2])))))
        previous = order, current
    return dict(profile='heh-refinement', diagnostic_seconds=elapsed, grid_comparisons=rows,
        tensors=tensors, tensor_changes=changes, modeled_peak_bytes=result.plan.modeled_peak_bytes,
        reference_potential_real=result.reference_potential.real.tolist(),
        reference_potential_imag=result.reference_potential.imag.tolist(),
        shell_calls=result.shell_calls, primitive_points=result.primitive_points,
        scalar_bound=result.plan.scalar_terms, kernel_bound=result.plan.kernel_terms, core_sha256=core_hash)


def adaptive_refinement():
    """Two different adaptive rules; their estimates are not error certificates."""
    core_hash = fresh_core()
    import numpy as np
    import scipy
    from benchmarks.manta_periodic_xc_reference import adaptive_reference
    _, xc = physical_xc_case()
    rows, changes = [], []
    previous = None
    for rule, atol, limit in (('genz-malik', 1e-7, 6000), ('gk15', 2e-8, 300), ('gk21', 2e-9, 300)):
        r = adaptive_reference(xc, rule=rule, atol=atol, max_subdivisions=limit)
        if previous is not None:
            changes.append(dict(from_rule=previous['rule'], to_rule=rule,
                energy=abs(r['energy']-previous['energy']),
                potential_max_abs=np.max(abs(r['potential']-previous['potential']), axis=(-1, -2)).tolist(),
                populations=float(np.max(abs(r['populations']-previous['populations'])))))
        rows.append({key: value for key, value in r.items() if key not in ('potential', 'populations')})
        rows[-1].update(populations=r['populations'].tolist(), potential_real=r['potential'].real.tolist(),
                       potential_imag=r['potential'].imag.tolist())
        previous = r
    return dict(profile='heh-adaptive', references=rows, changes=changes,
                scipy_version=scipy.__version__, core_sha256=core_hash)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', choices=('lda-rks', 'pbe-uks', 'empty-blyp', 'pbe0-rks', 'cam-uks', 'empty-lc-blyp', 'heh-refinement', 'heh-adaptive'))
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--physical-refinement', action='store_true', help='Run the slower physical HeH/PBE diagnostic instead of timing profiles')
    mode.add_argument('--adaptive-reference', action='store_true', help='Compare independent adaptive integration rules on the frozen HeH/PBE density')
    mode.add_argument('--hybrid', action='store_true', help='Assess explicit semilocal components of PBE0, CAM-B3LYP and empty-spin LC-BLYP')
    p.add_argument('--repeats', type=int, default=5); a = p.parse_args()
    if a.repeats < 3: p.error('At least three repeats required')
    if a.worker:
        if a.worker == 'heh-refinement': result = physical_refinement()
        elif a.worker == 'heh-adaptive': result = adaptive_refinement()
        else: result = worker(a.worker, a.repeats)
        print(json.dumps(result)); return
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    rows = []
    profiles = ('lda-rks', 'pbe-uks', 'empty-blyp')
    if a.hybrid: profiles = ('pbe0-rks', 'cam-uks', 'empty-lc-blyp')
    if a.physical_refinement: profiles = ('heh-refinement',)
    if a.adaptive_reference: profiles = ('heh-adaptive',)
    for profile in profiles:
        out = subprocess.run([sys.executable, __file__, '--worker', profile, '--repeats', str(a.repeats)],
            env=env, text=True, capture_output=True, check=True, timeout=900)
        rows.append(json.loads(out.stdout))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__': main()
