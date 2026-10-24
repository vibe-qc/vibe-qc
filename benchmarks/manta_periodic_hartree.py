#!/usr/bin/env python3
"""Finite periodic Hartree benchmark; portable JSON goes to stdout.

Each row runs in a fresh single-thread process. Build timing includes the
reciprocal grid, admission, Fourier generation, contractions and J(k) export.
No SCF, exchange, BUPO or symmetry acceleration is measured.
"""
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


def worker(case, variant, batch, repeats):
    began = time.perf_counter()
    import numpy as np
    import vibeqc as vq
    from vibeqc import _vibeqc_core as core
    from vibeqc.manta_periodic import _selected_periodic_hartree
    from vibeqc.periodic_fock_multi_k import build_periodic_j_ewald3d_k_from_k_density

    root = Path(__file__).resolve().parents[1]
    native = Path(core.__file__).resolve()
    if not native.is_relative_to(root):
        raise RuntimeError("Use the rebuilt editable extension from this checkout")
    newest = max(p.stat().st_mtime for p in (root/'cpp').rglob('*') if p.suffix in ('.cpp', '.hpp', '.h'))
    if native.stat().st_mtime <= newest:
        raise RuntimeError("Rebuild the native extension before benchmarking")
    start = time.perf_counter()
    positions = [[.1, -.2, .3], [.4, .2, -.1], [-.2, .5, .7]]
    lattice = np.array([[2.5, .2, .1], [0., 2.7, .3], [.2, .1, 2.9]])
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(2, p) for p in positions])
    shells = []
    for i, origin in enumerate(positions):
        s = core.ShellInfo(); s.atom_index = i; s.l = i; s.pure = True; s.origin = origin
        s.exponents = [.7+.1*i, 1.6+.2*i]; s.coefficients = [.8, -.18]; shells.append(s)
    basis = vq.BasisSet(system.unit_cell_molecule(), shells, 'manta-periodic-benchmark', False)
    indices = [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    cells = [vq.LatticeCell(system, index) for index in indices]
    k = np.array([[0., 0., 0.], [.2, -.1, .07], [-.17, .12, .14]])
    weights = np.array([.2, .3, .5])
    if case == 'gamma':
        k = k[:1].copy(); weights = np.ones(1)
    rng = np.random.default_rng(713)
    x = rng.normal(size=(len(k), basis.nbasis, basis.nbasis))+1j*rng.normal(size=(len(k), basis.nbasis, basis.nbasis))
    density = np.ascontiguousarray(x+x.conj().transpose(0, 2, 1))
    engine = vq.IntegralEngineOptions(backend='manta')
    fixture_wall = time.perf_counter()-start

    def build(provider):
        if provider == 'reference':
            return np.asarray(build_periodic_j_ewald3d_k_from_k_density(
                basis, system, density, k, weights, cells, .5, ke_cutoff=6.,
                reciprocal_chunk_size=batch, cell_chunk_size=1)), None
        result = _selected_periodic_hartree(basis, system, density, k, weights, cells, .5,
            lattice_opts=None, ke_cutoff=6., reciprocal_batch=batch, cell_chunk_size=1, integral_engine=engine)
        return result.values, result.receipt

    build(variant)
    wall, cpu = [], []
    for _ in range(repeats):
        start, process = time.perf_counter(), time.process_time()
        values, receipt = build(variant)
        wall.append(time.perf_counter()-start); cpu.append(time.process_time()-process)
    start = time.perf_counter()
    reference, _ = build('reference')
    error = float(np.max(np.abs(values-reference)))
    energy_error = float(abs(.5*np.einsum('k,kmn,knm->', weights, density, values-reference)))
    if error > 3e-11 or energy_error > 3e-11:
        raise RuntimeError("Periodic Hartree benchmark numerical gate failed")
    validation_wall = time.perf_counter()-start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system() == 'Darwin' else 1024)
    fields = ('panel_calls', 'shell_calls', 'primitive_pairs', 'vector_primitive_products',
              'phase_products', 'phase_accumulations', 'contraction_terms', 'modeled_peak_bytes')
    return dict(case=case, variant=variant, reciprocal_batch=batch, repeats=repeats,
                n_basis=basis.nbasis, n_cells=len(cells), n_k=len(k), ke_cutoff=6.,
                fixture_wall_s=fixture_wall, build_median_wall_s=statistics.median(wall),
                build_min_wall_s=min(wall), build_max_wall_s=max(wall),
                build_median_cpu_s=statistics.median(cpu), reference_validation_wall_s=validation_wall,
                total_worker_wall_s=time.perf_counter()-began, peak_process_rss_bytes=rss,
                max_abs_j_error=error, abs_hartree_energy_error=energy_error,
                work=None if receipt is None else {f: getattr(receipt, f) for f in fields},
                core_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
                platform=platform.system(), architecture=platform.machine(),
                python=platform.python_version(), numpy=np.__version__, threads=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['reference', 'manta'])
    parser.add_argument('--case', choices=['gamma', 'multi-k'], default='multi-k')
    parser.add_argument('--batch', type=int, choices=[4, 32], default=32)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if not 3 <= args.repeats <= 100:
        parser.error('repeats must be between 3 and 100')
    for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS']:
        os.environ[name] = '1'
    if args.worker:
        print(json.dumps(worker(args.case, args.worker, args.batch, args.repeats)))
        return
    rows = []
    for case in ['gamma', 'multi-k']:
        for batch in [4, 32]:
            for variant in ['reference', 'manta']:
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', variant,
                    '--case', case, '--batch', str(batch), '--repeats', str(args.repeats)],
                    check=True, capture_output=True, text=True, timeout=300)
                rows.append(json.loads(result.stdout))
    print(json.dumps(dict(schema='manta-periodic-hartree-benchmark-v1', results=rows), indent=2))


if __name__ == '__main__':
    main()
