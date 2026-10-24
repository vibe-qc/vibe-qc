"""Finite-source physical UHF timings with independent integral/SCF witnesses."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time


def worker(species, mesh_name, correction, repeats, source):
    began = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import numpy as np
    from vibeqc import _vibeqc_core as core
    from tests.test_manta_periodic_uhf import _case, _reference, _scf_reference, _nuclear_reference
    native = Path(core.__file__).resolve()
    if not native.is_relative_to(root) or native.stat().st_mtime <= max(
            p.stat().st_mtime for p in (root/'cpp').rglob('*') if p.suffix in ('.cpp', '.hpp', '.h')):
        raise RuntimeError('Use a freshly rebuilt editable core from this checkout')
    mesh = (1, 1, 1) if mesh_name == 'gamma' else (2, 1, 1)
    shift = (1, 0, 0) if mesh_name == 'shifted' else (0, 0, 0)
    if source in ('cutoffs', 'job'):
        from tests.test_manta_periodic_uhf_domains import _arguments
        arguments, k = _arguments(mesh, shift, species, correction)
        run = lambda: core._manta.run_periodic_uhf_from_cutoffs(**arguments)
        if source == 'job':
            import tempfile
            import vibeqc as vq
            from tests.test_manta_periodic_uhf_job import _job
            temporary = tempfile.TemporaryDirectory(prefix='manta-uhf-job-')
            job, _, _ = _job(Path(temporary.name), mesh, shift, species)
            job['exchange_exxdiv'] = 'ewald' if correction == 'BvK-ewald' else 'none'
            run = lambda: vq.run_periodic_job(**job).native
    else:
        c, k = _case(mesh, shift, species, correction)
        run = lambda: core._manta.run_periodic_uhf(**c)
    start = time.perf_counter(); run(); cold = time.perf_counter()-start
    walls, cpus = [], []
    for _ in range(repeats):
        start, cpu = time.perf_counter(), time.process_time()
        r = run()
        walls.append(time.perf_counter()-start); cpus.append(time.process_time()-cpu)
        if not r.converged:
            raise RuntimeError('Native physical UHF did not converge')
    if source in ('cutoffs', 'job'):
        from tests.test_manta_periodic_domains import _explicit
        domains = core._manta.build_periodic_uhf_domains(**arguments)
        c = _explicit(arguments, domains)
    s, h, build, _ = _reference(c, k)
    reference = _scf_reference(s, h, build, (r.receipt.n_alpha, r.receipt.n_beta))
    nucleus, _ = _nuclear_reference(c)
    f, components = build(r.density)
    error = dict(density_max=float(np.max(abs(reference-r.density))),
                 fock_max=float(np.max(abs(f-r.fock))),
                 energy_abs=abs(sum(nucleus)+sum(build(reference)[1])-r.receipt.energy),
                 component_max=float(np.max(abs(np.array(components)-np.array([
                     r.receipt.one_electron_energy, r.receipt.coulomb_energy,
                     r.receipt.uncorrected_exchange_energy, r.receipt.exchange_correction_energy])))))
    if error['density_max'] > 4e-9 or error['fock_max'] > 4e-12 or error['energy_abs'] > 4e-11 or error['component_max'] > 4e-12:
        raise RuntimeError('Independent physical UHF acceptance failed')
    if source == 'job':
        from vibeqc.guess_read import _spin_density_k_from_qvf_bloch_payload
        restored = _spin_density_k_from_qvf_bloch_payload(job['output'].with_suffix('.qvf'))
        if not np.array_equal(restored, r.density):
            raise RuntimeError('Timed UHF job QVF did not preserve spin state')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != 'darwin': rss *= 1024
    return dict(source=source, species=species, mesh=mesh_name, exchange_q0=correction, repeats=repeats,
                basis=c['basis'].nbasis, alpha=r.receipt.n_alpha, beta=r.receipt.n_beta,
                cold_s=cold, median_s=statistics.median(walls), min_s=min(walls), max_s=max(walls),
                cpu_median_s=statistics.median(cpus), worker_s=time.perf_counter()-began,
                peak_rss_bytes=rss, modeled_peak_bytes=r.plan.modeled_peak_bytes,
                iterations=r.solver_snapshot.evaluated_iteration, fock_calls=r.receipt.fock_calls,
                shell_calls=r.receipt.shell_calls, primitive_pairs=r.receipt.primitive_pairs,
                history_reserved_bytes=r.plan.history_live_bytes, errors=error,
                domain_live_bytes=r.plan.domain_live_bytes,
                translations=r.receipt.domain_translations, hartree_vectors=r.receipt.domain_hartree_vectors,
                exchange_vectors=r.receipt.domain_exchange_vectors,
                core_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
                python=platform.python_version(), numpy=np.__version__, platform=sys.platform,
                architecture=platform.machine())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', choices=('explicit', 'cutoffs', 'job'), default='explicit')
    p.add_argument('--worker', choices=('H', 'Li'))
    p.add_argument('--mesh', choices=('gamma', 'multi-k', 'shifted'), default='gamma')
    p.add_argument('--exchange-q0', choices=('strict-zero-mode', 'BvK-ewald'), default='BvK-ewald')
    p.add_argument('--repeats', type=int, default=5)
    args = p.parse_args()
    if args.repeats < 3: p.error('At least three repeats required')
    if args.worker:
        print(json.dumps(worker(args.worker, args.mesh, args.exchange_q0, args.repeats, args.source)))
        return
    rows = []
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    for species in ('H', 'Li'):
        for mesh in ('gamma', 'multi-k', 'shifted'):
            for correction in ('strict-zero-mode', 'BvK-ewald'):
                r = subprocess.run([sys.executable, __file__, '--worker', species, '--mesh', mesh,
                    '--exchange-q0', correction, '--repeats', str(args.repeats), '--source', args.source], env=env,
                    check=True, capture_output=True, text=True)
                rows.append(json.loads(r.stdout))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
