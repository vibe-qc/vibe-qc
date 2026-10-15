#!/usr/bin/env python3
"""Benchmark matched ordered/permutation/spatial builds over MANTA P0 tiles.

Run one mode per fresh process. Qualification evaluates the full source and
is reported separately; repeated-build timings never include that setup.
The independent dense oracle is bounded and exists only in this benchmark.
"""
import argparse
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
import resource
import runpy
import statistics
import sys
import tracemalloc
from time import perf_counter

import numpy as np

from vibeqc.manta import ReferenceJKBuilder
from vibeqc.symmetry_quartets import (
    QuartetJKBatchResult, QuartetWork, manta_quartet_requirements, qualify_manta_quartets,
)
from vibeqc.symmetry_quartets_batch import quartet_batch_requirements
from vibeqc.symmetry_quartets_native import native_quartet_requirements
from vibeqc.symmetry_quartets_screening import QuartetScreening


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--molecule', choices=('water', 'ammonia', 'cartesian'), default='ammonia')
    parser.add_argument('--basis', default='sto-3g')
    parser.add_argument('--mode', choices=('none', 'permutation', 'spatial'), default='spatial')
    parser.add_argument('--schedule', choices=('stored', 'streaming'), default='stored')
    parser.add_argument('--executor', choices=('numpy', 'native'), default='numpy')
    parser.add_argument('--densities', type=int, default=1)
    parser.add_argument('--batch-policy', choices=('shared', 'separate'), default='separate')
    parser.add_argument('--target-policy', choices=('all', 'j', 'k', 'spin'), default='all')
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--screen-norm-budget', type=float, default=None,
        help='explicit fixed-orbit AO tensor omission budget; stored spatial shared batches only')
    parser.add_argument('--trace-admission-memory', action='store_true',
        help='separate memory probe; tracing perturbs admission timing')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    if not 1 <= args.densities <= 64:
        parser.error('benchmark density count must be between 1 and 64')
    if args.target_policy != 'all' and args.batch_policy != 'shared':
        parser.error('selective target measurements require a shared batch')
    if args.target_policy == 'spin' and args.densities != 2:
        parser.error('spin targets require two input spin densities')
    if args.screen_norm_budget is not None and (args.schedule != 'stored'
            or args.mode != 'spatial' or args.batch_policy != 'shared'):
        parser.error('screening requires a stored spatial shared batch')
    fixture = runpy.run_path(str(Path(__file__).resolve().parents[1]/'tests/test_symmetry_quartets.py'))
    started = perf_counter()
    mol, basis, rotations, translations = fixture['molecular_case'](args.molecule, args.basis)
    n, ns = basis.nbasis, basis.nshells
    if n**4 > 20_000_000:
        parser.error('independent dense reference exceeds benchmark cap')
    provider = ReferenceJKBuilder(basis)
    factory_budget = manta_quartet_requirements(mol, basis, provider, len(rotations),
        budget=fixture['BUDGET'], mode=args.mode, schedule=args.schedule)
    preparation = perf_counter()-started
    if args.trace_admission_memory:
        tracemalloc.start()
    started = perf_counter()
    plan = qualify_manta_quartets(mol, basis, rotations, translations, provider=provider,
        source_tolerance=1e-10, budget=fixture['BUDGET'], mode=args.mode, schedule=args.schedule)
    setup = perf_counter()-started
    started = perf_counter()
    screening = (None if args.screen_norm_budget is None else QuartetScreening(plan,
        maximum_omitted_norm=args.screen_norm_budget,budget=fixture['BUDGET']))
    screening_time = perf_counter()-started
    python_live = python_peak = None
    if args.trace_admission_memory:
        python_live, python_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    count = 3 if args.target_policy == 'spin' else args.densities
    targets = (('j','k','k') if args.target_policy == 'spin' else
               (args.target_policy,)*count if args.target_policy in ('j','k') else None)
    j_channels = tuple(i for i in range(count) if targets is None or 'j' in targets[i])
    k_channels = tuple(i for i in range(count) if targets is None or 'k' in targets[i])
    if args.batch_policy == 'shared':
        execution_budget = quartet_batch_requirements(plan, count,
            budget=fixture['BUDGET'], executor=args.executor, targets=targets, screening=screening)
    else:
        execution_budget = (native_quartet_requirements(plan, budget=fixture['BUDGET'])
                            if args.executor == 'native' else fixture['BUDGET'])
    rng = np.random.default_rng(679)
    # Independent channels, with channel zero identical to the scalar benchmark.
    d = []
    for _ in range(args.densities):
        raw = rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
        d.append(raw+raw.conj().T)
    d = np.stack(d)
    spin_densities = d if args.densities == 2 else None
    if args.target_policy == 'spin':
        d = np.stack([d.sum(axis=0),d[0],d[1]])
    full = np.empty((n,)*4)
    reference_requests = reference_primitives = 0
    session = provider.session()
    for key in product(range(ns), repeat=4):
        receipt = session.evaluate(key)
        full[tuple(slice(plan.offsets[s], plan.offsets[s+1]) for s in key)] = receipt['values']
        reference_requests += receipt['shell_calls']
        reference_primitives += receipt['primitive_quartets_requested']
    del session
    def direct_reference():
        references = [fixture['dense_jk'](full, channel) for channel in d]
        return tuple(np.stack([references[channel][which] for channel in channels])
            if channels else np.empty((0,n,n),complex)
            for which,channels in enumerate((j_channels,k_channels)))
    uncut_reference = direct_reference()
    independent_omitted_norm = 0.
    if screening is not None:
        from math import hypot
        for keep,orbit in zip(screening.keep,plan.orbits):
            if not keep:
                for member in orbit.members:
                    selection = tuple(slice(plan.offsets[s],plan.offsets[s+1]) for s in member.shells)
                    independent_omitted_norm = hypot(independent_omitted_norm,float(np.linalg.norm(full[selection])))
                    full[selection] = 0.
    reference = direct_reference() if screening is not None else uncut_reference
    def build():
        if args.batch_policy == 'shared':
            return plan.jk_batch(d, budget=execution_budget, executor=args.executor,
                                 targets=targets, screening=screening)
        j, k, bounds = np.empty_like(d), np.empty_like(d), np.empty(args.densities)
        work, scalar, blocks, transport = [0]*5, 0, 0, 0
        for i, channel in enumerate(d):
            result = plan.jk(channel, budget=execution_budget, executor=args.executor)
            j[i], k[i], bounds[i] = result.j, result.k, result.error_bound
            work = [a+b for a,b in zip(work, asdict(result.work).values())]
            scalar += result.scalar_products
            blocks += result.block_contractions
            transport += result.transport_scalar_products
        return QuartetJKBatchResult(j, k, QuartetWork(*work), scalar, blocks,
            transport, bounds, plan.source_identity, args.executor, j_channels, k_channels)
    result = build()
    errors = [float(np.max(np.abs(a-b))) if a.size else None
              for a, b in zip((result.j, result.k), reference)]
    uncut_errors = [float(np.max(np.abs(a-b))) if a.size else None
                   for a,b in zip((result.j,result.k),uncut_reference)]
    uncut_norm_errors = [[float(np.linalg.norm(a-b)) for a,b in zip(actual,expected)]
                        for actual,expected in zip((result.j,result.k),uncut_reference)]
    def energy(j,k):
        if args.target_policy in ('j','k'):
            return None
        if spin_densities is not None:
            total_j = j[0] if args.target_policy == 'spin' else j.sum(axis=0)
            return .5*np.einsum('pq,qp',spin_densities.sum(axis=0),total_j)-.5*np.einsum('bpq,bqp',spin_densities,k)
        return .5*np.einsum('bpq,bqp->b',d,j-.5*k)
    energy_error = None if args.target_policy in ('j','k') else float(np.max(np.abs(energy(result.j,result.k)-energy(*reference))))
    uncut_energy_error = None if args.target_policy in ('j','k') else float(np.max(np.abs(energy(result.j,result.k)-energy(*uncut_reference))))
    times = []
    for _ in range(args.repeats):
        started = perf_counter()
        result = build()
        times.append(perf_counter()-started)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != 'darwin':
        rss *= 1024
    print(json.dumps(dict(molecule=args.molecule, basis=args.basis, mode=args.mode,
        schedule=args.schedule,
        executor=args.executor,
        density_count=count, batch_policy=args.batch_policy, target_policy=args.target_policy,
        j_channels=result.j_channels, k_channels=result.k_channels,
        aggregate_execution_logical_bytes=(execution_budget.maximum_bytes
            if args.executor == 'native' or args.batch_policy == 'shared' else plan.required_bytes),
        benchmark_density_and_result_bytes=d.nbytes+result.j.nbytes+result.k.nbytes,
        benchmark_extra_spin_input_bytes=(spin_densities.nbytes if args.target_policy=='spin' else 0),
        target_array_bytes=result.j.nbytes+result.k.nbytes,
        admission_memory_traced=args.trace_admission_memory,
        python_admission_live_bytes=python_live, python_admission_peak_bytes=python_peak,
        n_basis=n, n_shells=ns, group_order=plan.group.order, orbits=len(plan.orbits),
        qualification=asdict(plan.qualification_work), build=asdict(result.work),
        scalar_products=result.scalar_products, block_contractions=result.block_contractions,
        transport_scalar_products=result.transport_scalar_products,
        compact_action_array_bytes=plan.retained_array_bytes,
        stabilizer_mask_bytes=(sum(len(o.stabilizer.masks) for o in plan.orbits)
                               if args.schedule == 'stored' else 0),
        pair_schedule_array_bytes=(plan.orbits.retained_array_bytes if args.schedule == 'streaming' else 0),
        aggregate_factory_logical_bytes=factory_budget.maximum_bytes,
        aggregate_logical_bytes=plan.required_bytes, provider_modeled_bytes=provider.plan.modeled_peak_bytes,
        independent_dense_reference_bytes=full.nbytes, process_peak_rss_bytes=rss,
        independent_jk_reference_bytes=sum(a.nbytes for a in reference),
        extra_unscreened_reference_bytes=(sum(a.nbytes for a in uncut_reference) if screening is not None else 0),
        independent_reference_shell_calls=reference_requests,
        independent_reference_primitive_quartets_requested=reference_primitives,
        preparation_seconds=preparation, admission_seconds=setup,
        screening_seconds=screening_time,
        screening_receipt=None if screening is None else screening.receipt,
        independent_omitted_norm=independent_omitted_norm,
        median_build_seconds=statistics.median(times), build_seconds=times, repeats=args.repeats,
        source_residual_norm=plan.source_residual_norm,
        stabilizer_residual=plan.maximum_stabilizer_residual,
        jk_error_bound=float(np.max(result.error_bounds)),
        jk_error_bounds=result.error_bounds.tolist(), jk_max_abs_error=errors,
        interaction_energy_abs_error=energy_error,
        unscreened_jk_max_abs_error=uncut_errors, unscreened_jk_frobenius_errors=uncut_norm_errors,
        unscreened_interaction_energy_abs_error=uncut_energy_error,
        source_identity=result.source_identity), indent=2))


if __name__ == '__main__':
    main()
