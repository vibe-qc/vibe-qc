"""Public RI UHF jobs with complete standard output and independent SCF."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


PROFILES = ('H-gamma-strict', 'H-shifted-strict', 'Li-shifted-strict',
            'H-gamma-BvK', 'H-shifted-BvK', 'Li-shifted-BvK')


def worker(profile, repeats):
    root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
    import numpy as np
    import vibeqc as vq
    from vibeqc import manta, _vibeqc_core as core
    from vibeqc.output import validate_qvf
    from vibeqc.guess_read import _spin_density_k_from_qvf_bloch_payload
    from benchmarks.manta_periodic_ri_uhf import measure
    from tests.test_manta_periodic_ri_job import job_case
    from tests.test_manta_periodic_ri_domains import explicit
    from tests.test_manta_periodic_ri_uhf import reference
    from tests.test_manta_periodic_uhf import _scf_reference
    from tests.test_manta_periodic_rhf import _nuclear_reference

    native = Path(core.__file__).resolve()
    if not native.is_relative_to(root) or native.stat().st_mtime <= max(
            p.stat().st_mtime for p in (root/'cpp').rglob('*') if p.suffix in ('.cpp', '.hpp', '.h')):
        raise RuntimeError('Require a fresh native build from the current checkout')
    species, mesh, correction = profile.split('-')
    shape, shift = ((1, 1, 1), (0, 0, 0)) if mesh == 'gamma' else ((2, 1, 1), (1, 0, 0))
    with tempfile.TemporaryDirectory(prefix='manta-ri-job-benchmark-') as temporary:
        directory = Path(temporary)
        job, a, k = job_case(directory, shape, shift, species,
                             'BvK-ewald' if correction == 'BvK' else 'strict-zero-mode')
        count = 0
        def run():
            nonlocal count
            count += 1; job['output'] = directory/str(count)/'job'
            return vq.run_periodic_job(**job)
        start = time.perf_counter(); r = run(); cold = time.perf_counter()-start
        r, wall = measure(run, repeats)
        if not r.converged or not r.native.receipt.confirmed:
            raise RuntimeError('Public RI UHF did not pass its terminal gates')
        path = job['output'].with_suffix('.qvf')
        if not validate_qvf(path)['valid']:
            raise RuntimeError('Public fitted QVF failed validation')
        np.testing.assert_array_equal(_spin_density_k_from_qvf_bloch_payload(path), r.native.density)
        c = explicit(a, manta.build_periodic_ri_uhf_domains(**a))
        start = time.perf_counter(); s, h, build, _ = reference(c, k)
        nuclear = sum(_nuclear_reference(c)[0]); ref_setup = time.perf_counter()-start
        start = time.perf_counter(); d = _scf_reference(s, h, build, (r.native.plan.n_alpha, r.native.plan.n_beta))
        ref_scf = time.perf_counter()-start
        e = r.integral_provenance
        errors = dict(D=float(np.max(abs(d-r.native.density))), F=float(np.max(abs(build(r.native.density)[0]-r.native.fock))),
            energy=abs(nuclear+sum(build(d)[1])-r.energy))
        if any(value > (1e-8 if name == 'D' else 1e-9) for name, value in errors.items()):
            raise RuntimeError(f'Independent public RI UHF parity failed: {errors}')
        return dict(profile=profile, repeats=repeats, cold_job_s=cold, repeated_job=wall,
            reference_setup_s=ref_setup, reference_scf_s=ref_scf, errors=errors,
            energy=r.energy, iterations=r.iterations, momenta=len(k),
            metric_ranks=[m['rank'] for m in e['fitting']['metrics']],
            native_modeled_peak_bytes=r.native.plan.modeled_peak_bytes,
            python_reserved_bytes=e['python_reserved_bytes'], job_reserved_bytes=e['job_reserved_bytes'],
            output_file_count=len(list(job['output'].parent.iterdir())),
            output_bytes=sum(p.stat().st_size for p in job['output'].parent.iterdir()),
            qvf_density_error=0.,
            core_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
            adapter_sha256=hashlib.sha256((root/'python/vibeqc/manta_periodic_job.py').read_bytes()).hexdigest())


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
