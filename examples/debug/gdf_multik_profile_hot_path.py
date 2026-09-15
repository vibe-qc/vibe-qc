"""Measure the production SR/LR GDF source and SCF on a full allocated node.

Run through vq with the node's complete CPU allocation. Each thread count
uses a fresh process, so peak RSS and CPU/wall ratios are comparable. The
optional wide-scaling gate requires 32 and 64 threads to improve the source
build by at least 20 percent over 16 threads. This is a performance witness;
thread-invariant energies do not replace cross-code scientific validation.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system', choices=('lih', 'mgo', 'si', 'cao', 'agcl'), default='lih')
    parser.add_argument('--basis', default='pob-tzvp-rev2')
    parser.add_argument('--aux-basis', default='def2-svp-jk')
    parser.add_argument('--method', choices=('RHF', 'RKS'), default='RKS')
    parser.add_argument('--functional', default='pbe')
    parser.add_argument('--kmesh', type=int, nargs=3, default=(2, 1, 1))
    parser.add_argument('--threads', type=int, nargs='+', default=(1, 8, 16, 32, 64))
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--require-wide-scaling', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', type=int, help=argparse.SUPPRESS)
    return parser


def _system(name):
    import numpy as np
    import vibeqc as vq

    # Explicit diagnostic cells, not inferred historical canary geometries.
    a_ang, z, fraction = {
        'lih': (4.084, (3, 1), (.5, 0., 0.)),
        'mgo': (4.211, (12, 8), (.5, 0., 0.)),
        'si': (5.431, (14, 14), (.25, .25, .25)),
        'cao': (4.81, (20, 8), (.5, 0., 0.)),
        'agcl': (5.55, (47, 17), (.5, 0., 0.)),
    }[name]
    a = a_ang / 0.529177210903
    lattice = a * np.array([[0., .5, .5], [.5, 0., .5], [.5, .5, 0.]])
    return vq.PeriodicSystem(3, lattice, [
        vq.Atom(z[0], [0., 0., 0.]), vq.Atom(z[1], list(a * np.asarray(fraction))),
    ])


@contextlib.contextmanager
def _timed_calls(module, names, records):
    originals = {}
    for name in names:
        original = getattr(module, name)
        originals[name] = original
        def timed(*args, _name=name, _original=original, **kwargs):
            wall, cpu = time.perf_counter(), time.process_time()
            try:
                return _original(*args, **kwargs)
            finally:
                row = records.setdefault(_name, dict(calls=0, wall_seconds=0., cpu_seconds=0.))
                row['calls'] += 1
                row['wall_seconds'] += time.perf_counter() - wall
                row['cpu_seconds'] += time.process_time() - cpu
        setattr(module, name, timed)
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(module, name, original)


def _worker(args):
    import numpy as np
    import vibeqc as vq
    from vibeqc import _vibeqc_core as core
    from vibeqc import periodic_k_gdf as driver
    from vibeqc.periodic_runner import run_periodic_job

    core.set_num_threads(args.worker)
    if core.get_num_threads() != args.worker:
        raise RuntimeError('Runtime did not honor the requested thread count')
    system = _system(args.system)
    basis = vq.BasisSet(system.unit_cell_molecule(), args.basis)
    records = {}
    wall, cpu = time.perf_counter(), time.process_time()
    # Wrappers observe executed production calls; they do not reconstruct a
    # surrogate fit. Native SR/LR source time is nested within cache time.
    with _timed_calls(core, ('compute_gdf_range_separated_integrals',
                            'build_xc_periodic', 'build_xc_periodic_uks'), records):
        with _timed_calls(driver, ('_build_scf_range_separated_lpq_cache',
                                  '_k_from_signed_factors', 'build_xc_periodic'), records):
            result = run_periodic_job(
                system, basis, method=args.method,
                functional=args.functional if args.method == 'RKS' else None,
                kpoints=args.kmesh, jk_method='gdf', gdf_method='rsgdf',
                aux_basis=args.aux_basis, max_iter=args.max_iter,
                conv_tol_energy=1e-10,
                output=args.output / 'scf', output_qvf=False, progress=False,
            )
    total_wall, total_cpu = time.perf_counter() - wall, time.process_time() - cpu
    for row in records.values():
        row['effective_threads'] = row['cpu_seconds'] / max(row['wall_seconds'], 1e-12)
    report = dict(
        system=args.system, basis=args.basis, aux_basis=args.aux_basis,
        lattice=np.asarray(system.lattice).tolist(),
        atoms=[dict(Z=int(a.Z), xyz=list(a.xyz)) for a in system.unit_cell],
        method=args.method, functional=args.functional if args.method == 'RKS' else None,
        kmesh=args.kmesh, threads=args.worker, nbasis=int(basis.nbasis),
        converged=bool(result.converged), energy=float(result.energy),
        iterations=int(result.n_iter), wall_seconds=total_wall, cpu_seconds=total_cpu,
        effective_threads=total_cpu / total_wall,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == 'darwin' else 1024),
        core_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        version=vq.__version__, phases=records,
    )
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    if not result.converged:
        raise RuntimeError('Unconverged SCF: report retained, scaling not certified')
    if 'compute_gdf_range_separated_integrals' not in records:
        raise RuntimeError('Production calculation did not execute the SR/LR source')
    return 0


def main():
    args = _parser().parse_args()
    if any(n <= 0 for n in (*args.threads, *args.kmesh)):
        raise ValueError('Thread counts and mesh dimensions must be positive')
    if args.worker is not None:
        return _worker(args)
    if len(set(args.threads)) != len(args.threads):
        raise ValueError('Thread counts must be unique')
    allocated = int(os.environ.get('VQ_CPUS', os.cpu_count() or 1))
    if max(args.threads) > allocated:
        raise ValueError(f'Requested {max(args.threads)} threads but allocation has {allocated}')
    if args.require_wide_scaling and not {16, 32, 64}.issubset(args.threads):
        raise ValueError('Wide-scaling acceptance requires 16, 32 and 64 threads')
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for threads in args.threads:
        output = args.output / str(threads)
        output.mkdir()
        command = [sys.executable, str(Path(__file__).resolve()),
                   '--worker', str(threads), '--system', args.system, '--basis', args.basis,
                   '--aux-basis', args.aux_basis, '--method', args.method,
                   '--functional', args.functional, '--max-iter', str(args.max_iter),
                   '--kmesh', *map(str, args.kmesh), '--output', str(output.resolve())]
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', BLIS_NUM_THREADS='1')
        with (output / 'worker.log').open('w') as log:
            child = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        if child.returncode:
            raise RuntimeError(f'{threads}-thread worker failed; see {output / "worker.log"}')
        row = json.loads((output / 'report.json').read_text())
        rows.append(row)
        if row['core_sha256'] != rows[0]['core_sha256']:
            raise RuntimeError('Native core changed during the thread sweep')
        (args.output / 'reports.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(json.dumps(row), flush=True)
    energies = [row['energy'] for row in rows]
    if max(energies) - min(energies) > 1e-8:
        raise RuntimeError('Thread sweep changed the accepted energy by more than 1e-8 Ha')
    if args.require_wide_scaling:
        source = {row['threads']: row['phases']['compute_gdf_range_separated_integrals']['wall_seconds']
                  for row in rows}
        for threads in (32, 64):
            if source[16] / source[threads] < 1.2:
                raise RuntimeError(f'{threads}-thread source failed the 1.2x speedup over 16 threads')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
