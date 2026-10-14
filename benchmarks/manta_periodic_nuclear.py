#!/usr/bin/env python3
"""Finite 3D Ewald nuclear benchmark; portable JSON goes to stdout.

Each row runs in a fresh single-thread process. Build timing includes the
admission, screened/reciprocal/constant terms, Bloch folding and cell/k export.
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
    from vibeqc import manta
    from vibeqc.aux_basis import rsgdf_dense_g_mesh
    from vibeqc.periodic_v_ne import compute_v_ne_ewald_3d_ft_lattice

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
    options = vq.LatticeSumOptions(); options.cutoff_bohr = 2.4
    options.nuclear_cutoff_bohr = 4.5; options.pair_complete_1e = True
    cells = vq.pair_complete_lattice_cells(basis, system, options.cutoff_bohr)
    translations = np.array([c.r_cart for c in cells])
    k = np.array([[0., 0., 0.], [.2, -.1, .07], [-.17, .12, .14]])
    if case == 'gamma': k = k[:1].copy()
    g = rsgdf_dense_g_mesh(system, 6.)
    g = np.ascontiguousarray(g[np.any(g != 0., axis=1)])
    alpha = .7; ewald = vq.EwaldOptions(); ewald.alpha = alpha
    budget = manta.ResourceBudget(); budget.kernel_terms = 100_000_000_000_000
    fixture_wall = time.perf_counter()-start

    def build(provider):
        if provider == 'reference':
            reference = compute_v_ne_ewald_3d_ft_lattice(basis, system, options,
                ewald_options=ewald, ke_cutoff=6., screen_rel=0., domain_rel=0.)
            blocks = np.asarray(reference.blocks)
            return np.einsum('kr,rij->kij', np.exp(1j*k@translations.T), blocks), blocks, None
        result = manta.compute_periodic_nuclear(basis, system, g, translations, k, alpha,
            options.nuclear_cutoff_bohr, batch, options.cutoff_bohr, budget)
        return result.values, result.cell_values, result.receipt

    build(variant)
    wall, cpu = [], []
    for _ in range(repeats):
        start, process = time.perf_counter(), time.process_time()
        values, blocks, receipt = build(variant)
        wall.append(time.perf_counter()-start); cpu.append(time.process_time()-process)
    start = time.perf_counter()
    reference, reference_blocks, _ = build('reference')
    error = float(np.max(np.abs(values-reference)))
    cell_error = float(np.max(np.abs(blocks-reference_blocks)))
    if error > 3e-11 or cell_error > 3e-11:
        raise RuntimeError('Ewald nuclear benchmark numerical gate failed')
    validation_wall = time.perf_counter()-start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system() == 'Darwin' else 1024)
    fields = ('nuclear_shell_calls', 'overlap_shell_calls', 'fourier_shell_calls', 'primitive_pairs',
              'primitive_charge_products', 'nuclear_images_retained', 'vector_primitive_products', 'contraction_terms', 'modeled_peak_bytes')
    return dict(case=case, variant=variant, reciprocal_batch=batch, repeats=repeats,
                alpha=alpha, nuclear_cutoff_bohr=options.nuclear_cutoff_bohr, pair_cutoff_bohr=options.cutoff_bohr,
                kernel_budget=budget.kernel_terms,
                n_basis=basis.nbasis, n_cells=len(cells), n_k=len(k), n_g=len(g), ke_cutoff=6.,
                fixture_wall_s=fixture_wall, build_median_wall_s=statistics.median(wall),
                build_min_wall_s=min(wall), build_max_wall_s=max(wall),
                build_median_cpu_s=statistics.median(cpu), reference_validation_wall_s=validation_wall,
                total_worker_wall_s=time.perf_counter()-began, peak_process_rss_bytes=rss,
                max_abs_k_matrix_error=error, max_abs_cell_matrix_error=cell_error,
                work=None if receipt is None else {f: getattr(receipt, f) for f in fields},
                core_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
                platform=platform.system(), architecture=platform.machine(),
                python=platform.python_version(), numpy=np.__version__, threads=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['reference', 'manta'])
    parser.add_argument('--case', choices=['gamma', 'multi-k'], default='multi-k')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1 or args.batch < 1: parser.error('repeats and batch must be positive')
    if args.worker:
        print(json.dumps(worker(args.case, args.worker, args.batch, args.repeats)))
        return
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    rows = []
    for case in ('gamma', 'multi-k'):
        for provider, batch in [('reference', 32), ('manta', 4), ('manta', 32)]:
            output = subprocess.check_output([sys.executable, __file__, '--worker', provider,
                '--case', case, '--batch', str(batch), '--repeats', str(args.repeats)], env=env, text=True)
            rows.append(json.loads(output))
    print(json.dumps(dict(schema_version=1, rows=rows), indent=2))


if __name__ == '__main__':
    main()
