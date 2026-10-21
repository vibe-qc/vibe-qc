"""Complete public KS jobs, standard artifacts and independent finite-source SCF."""
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


def worker(profile, repeats, hybrid_q0=None, range_separated=False):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import numpy as np
    import vibeqc as vq
    from vibeqc import _vibeqc_core as core
    from vibeqc.output import validate_qvf
    from vibeqc.guess_read import _spin_density_k_from_qvf_bloch_payload
    from tests.test_manta_periodic_ks_job import job_case
    from tests.test_manta_periodic_ks_grid import explicit
    from tests.test_manta_periodic_ks import oracle, independent_scf
    from tests.test_manta_periodic_rhf import _nuclear_reference
    if range_separated:
        from tests.test_manta_periodic_range_separated_job import job_case
        from tests.test_manta_periodic_range_separated_domains import explicit
        from tests.test_manta_periodic_range_separated_ks import oracle
    elif hybrid_q0:
        from tests.test_manta_periodic_global_hybrid_job import job_case
        from tests.test_manta_periodic_global_hybrid_domains import explicit
        from tests.test_manta_periodic_global_hybrid_ks import oracle
    native = Path(core.__file__).resolve()
    if not native.is_relative_to(root) or native.stat().st_mtime <= max(
        p.stat().st_mtime for p in (root/'cpp').rglob('*') if p.suffix in ('.cpp', '.hpp', '.h')):
        raise RuntimeError('Require fresh own-checkout native build')
    species, spins = {'rks': ('H2', 1), 'uks': ('HeH', 2), 'empty-spin': ('H', 2)}[profile]
    with tempfile.TemporaryDirectory(prefix='manta-ks-job-benchmark-') as directory:
        controls = dict(correction=hybrid_q0) if hybrid_q0 else {}
        job, args, c, k, old = job_case(Path(directory), species, spins, **controls)
        start = time.perf_counter(); vq.run_periodic_job(**job); cold = time.perf_counter()-start
        wall, cpu = [], []
        for _ in range(repeats):
            result = None
            start = time.perf_counter(); t = time.process_time(); result = vq.run_periodic_job(**job)
            wall.append(time.perf_counter()-start); cpu.append(time.process_time()-t)
        r = result.native
        explicit(args, c, old); s, h, build = oracle(c, k, old)[:3]
        pop = (r.plan.n_alpha,) if spins == 1 else (r.plan.n_alpha, r.plan.n_beta)
        d = independent_scf(s, h, build, pop, spins); f, e, n = build(r.density)
        nuclear, _ = _nuclear_reference(old)
        path = job['output'].with_suffix('.qvf')
        restored = _spin_density_k_from_qvf_bloch_payload(path)
        expected = r.density if spins == 2 else np.repeat(.5*r.density, 2, axis=0)
        components = [r.receipt.one_electron_energy, r.receipt.coulomb_energy, r.receipt.xc_energy]
        if hybrid_q0:
            components += [r.receipt.uncorrected_exchange_energy, r.receipt.exchange_correction_energy]
        errors = dict(density=float(np.max(abs(r.density-d))), potential=float(np.max(abs(r.fock-f))),
            energy=abs(result.energy-sum(nuclear)-sum(e)),
            components=float(np.max(abs(np.array(components)-e))),
            populations=float(np.max(abs(np.array(r.receipt.xc.electrons_grid)-n))),
            qvf_density=float(np.max(abs(restored-expected))))
        if (not result.converged or not validate_qvf(path)['valid'] or errors['density'] > 3e-9
            or max(errors[key] for key in ('potential', 'energy', 'components', 'populations')) > 2e-11 or errors['qvf_density']):
            raise RuntimeError('Independent job/SCF/output acceptance failed')
        return dict(profile=profile, functional=job['functional'], exchange_q0=result.exchange_q0,
            exchange_q0_requested=r.receipt.exchange_q0_requested, range_separated=r.receipt.range_separated,
            long_range_exchange_fraction=r.receipt.long_range_exchange_fraction,
            short_range_exchange_fraction=r.receipt.short_range_exchange_fraction,
            exchange_components=result.integral_provenance.get('exchange_components', []),
            exact_exchange_fraction=r.receipt.exact_exchange_fraction,
            exchange_vectors=r.receipt.domain_exchange_vectors,
            repeats=repeats, cold_s=cold, median_s=statistics.median(wall),
            min_s=min(wall), max_s=max(wall), cpu_median_s=statistics.median(cpu), errors=errors,
            iterations=result.iterations, fock_calls=r.receipt.fock_calls, shell_calls=r.receipt.shell_calls,
            primitive_pairs=r.receipt.primitive_pairs, primitive_points=r.receipt.primitive_points,
            actual_source_modeled_peak=r.plan.modeled_peak_bytes, job_reserved_bytes=result.run_fields['job_reserved_bytes'],
            grid_points=r.plan.quadrature.grid_points, n_basis=r.plan.n_basis, momenta=r.plan.momenta,
            artifact_count=len(list(job['output'].parent.iterdir())),
            grid_population_error=r.receipt.maximum_grid_population_error,
            core_sha256=hashlib.sha256(native.read_bytes()).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', choices=('rks', 'uks', 'empty-spin'))
    p.add_argument('--hybrid-q0', choices=('strict-zero-mode', 'BvK-ewald'),
                   help='Run complete hybrid jobs with this exchange convention')
    p.add_argument('--range-separated', action='store_true', help='Use HSE06, CAM-B3LYP and LC-BLYP profiles')
    p.add_argument('--repeats', type=int, default=5); a = p.parse_args()
    if a.repeats < 3: p.error('At least three repeats required')
    if a.range_separated and not a.hybrid_q0: p.error('--range-separated requires --hybrid-q0')
    if a.worker:
        print(json.dumps(worker(a.worker, a.repeats, a.hybrid_q0, a.range_separated))); return
    rows = []
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    for profile in ('rks', 'uks', 'empty-spin'):
        extra = ['--hybrid-q0', a.hybrid_q0] if a.hybrid_q0 else []
        if a.range_separated: extra += ['--range-separated']
        r = subprocess.run([sys.executable, __file__, '--worker', profile, '--repeats', str(a.repeats), *extra],
            check=True, capture_output=True, text=True, env=env)
        rows.append(json.loads(r.stdout))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__': main()
