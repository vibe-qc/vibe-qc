#!/usr/bin/env python3
"""Finite whole-mesh reciprocal exchange benchmark; portable JSON goes to stdout.

Each row runs in a fresh single-thread process. Build timing includes the
admission, density snapshots, Fourier generation, exchange contraction and export.
No SCF, Madelung correction, BUPO or symmetry acceleration is measured.
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


def worker(case, variant, batch, states, repeats, range_kernel='coulomb', omega=0.):
    began = time.perf_counter()
    import numpy as np
    import vibeqc as vq
    from vibeqc import _vibeqc_core as core
    from vibeqc import manta
    from vibeqc.bipole_fock_ewald import _contract_K_long_range_channel_linear

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
    if case == 'gamma': k = k[:1].copy()
    weights = np.ones(1) if case == 'gamma' else np.array([.2, .3, .5])
    translations = np.array([c.r_cart for c in cells])
    labels = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    reciprocal = 2*np.pi*np.linalg.inv(lattice).T
    chunks = []
    for target in k:
        for source in k:
            wave = labels@reciprocal.T+target-source
            chunks.append(wave if range_kernel=='erfc' and omega>0 else wave[np.any(wave != 0., axis=1)])
    q = np.ascontiguousarray(np.concatenate(chunks))
    offsets = np.r_[0, np.cumsum([len(wave) for wave in chunks])].astype(np.int64)
    rng = np.random.default_rng(913)
    x = rng.normal(size=(states, len(k), basis.nbasis, basis.nbasis))+1j*rng.normal(size=(states, len(k), basis.nbasis, basis.nbasis))
    density = np.ascontiguousarray(x+x.conj().transpose(0, 1, 3, 2))
    scales = np.concatenate([np.full(2*s.l+1, np.sqrt(4*np.pi/(2*s.l+1)) if s.l else 1.) for s in basis.shells()])
    fixture_wall = time.perf_counter()-start

    def build(provider):
        if provider == 'reference':
            values = np.zeros_like(density)
            for target in range(len(k)):
                for source in range(len(k)):
                    pair = target*len(k)+source; wave = q[offsets[pair]:offsets[pair+1]]
                    raw = core.ao_pair_fourier_transform_bloch_multi_cxx(basis, wave, translations, -k[source:source+1], 0., None, -1.)[0]
                    b = raw*scales[:, None, None]*scales[None, :, None]
                    norm = np.sum(wave*wave, axis=1)
                    coefficient = np.zeros(len(norm))
                    nonzero = norm>0
                    coefficient[nonzero] = 4*np.pi/norm[nonzero]
                    if range_kernel=='erf':
                        coefficient[nonzero] *= np.exp(-norm[nonzero]/(4*omega*omega)) if omega else 0.
                    elif range_kernel=='erfc' and omega:
                        coefficient[nonzero] *= -np.expm1(-norm[nonzero]/(4*omega*omega))
                        coefficient[~nonzero] = np.pi/(omega*omega)
                    kernel = weights[source]/abs(np.linalg.det(lattice))*coefficient
                    for state in range(states):
                        values[state, target] += _contract_K_long_range_channel_linear(kernel, b, density[state, source])
            energy = -.5*np.einsum('k,skij,skji->s', weights, density, values).real
            return values, energy, None
        result = manta.compute_periodic_exchange(basis, lattice, q, offsets, translations, k, weights, density,
                                                batch, kernel=range_kernel, omega=omega)
        return result.values, np.array(result.receipt.energies), result.receipt

    build(variant)
    wall, cpu = [], []
    for _ in range(repeats):
        start, process = time.perf_counter(), time.process_time()
        values, energies, receipt = build(variant)
        wall.append(time.perf_counter()-start); cpu.append(time.process_time()-process)
    start = time.perf_counter()
    reference, reference_energies, _ = build('reference')
    error = float(np.max(np.abs(values-reference)))
    energy_error = float(np.max(np.abs(energies-reference_energies)))
    if error > 3e-10 or energy_error > 3e-9:
        raise RuntimeError('Periodic exchange benchmark numerical gate failed')
    validation_wall = time.perf_counter()-start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system() == 'Darwin' else 1024)
    fields = ('transfer_vectors', 'zero_vectors', 'panel_calls', 'shell_calls', 'primitive_pairs', 'vector_primitive_products',
              'phase_products', 'phase_accumulations', 'matrix_products', 'contraction_terms', 'modeled_peak_bytes')
    return dict(case=case, variant=variant, reciprocal_batch=batch, density_channels=states, repeats=repeats,
                kernel=range_kernel, omega=omega,
                n_basis=basis.nbasis, n_cells=len(cells), n_k=len(k), transfer_vectors=len(q),
                fixture_wall_s=fixture_wall, build_median_wall_s=statistics.median(wall),
                build_min_wall_s=min(wall), build_max_wall_s=max(wall),
                build_median_cpu_s=statistics.median(cpu), reference_validation_wall_s=validation_wall,
                total_worker_wall_s=time.perf_counter()-began, peak_process_rss_bytes=rss,
                max_abs_k_matrix_error=error, max_abs_exchange_energy_error=energy_error,
                work=None if receipt is None else {f: getattr(receipt, f) for f in fields},
                core_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
                platform=platform.system(), architecture=platform.machine(),
                python=platform.python_version(), numpy=np.__version__, threads=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['reference', 'manta'])
    parser.add_argument('--case', choices=['gamma', 'multi-k'], default='multi-k')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--states', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--kernel', choices=['coulomb', 'erf', 'erfc'], default='coulomb')
    parser.add_argument('--omega', type=float, default=0.)
    args = parser.parse_args()
    if min(args.repeats, args.batch, args.states)<1: parser.error('repeats, batch and states must be positive')
    if not (0<=args.omega<float('inf')) or (args.kernel=='coulomb' and args.omega!=0):
        parser.error('omega must be finite and nonnegative, and zero for Coulomb')
    if args.worker:
        print(json.dumps(worker(args.case, args.worker, args.batch, args.states, args.repeats, args.kernel, args.omega)))
        return
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    rows = []
    for case in ('gamma', 'multi-k'):
        for states in (1, 2):
            for provider, batch in [('reference', 32), ('manta', 4), ('manta', 32)]:
                output = subprocess.check_output([sys.executable, __file__, '--worker', provider,
                    '--case', case, '--batch', str(batch), '--states', str(states), '--repeats', str(args.repeats),
                    '--kernel', args.kernel, '--omega', str(args.omega)], env=env, text=True)
                rows.append(json.loads(output))
    print(json.dumps(dict(schema_version=1, rows=rows), indent=2))


if __name__ == '__main__':
    main()
