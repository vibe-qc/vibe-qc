#!/usr/bin/env python3
"""Small, reproducible same-source block J/K benchmark; not a production gate.

Uses the independent molecular fixtures in tests/test_symmetry_abelian.py.
Run each case in a fresh process for a meaningful process peak RSS. Prints
numerical evidence only; redirect to external evidence storage if desired.
"""
import argparse
import json
from pathlib import Path
import resource
import runpy
import statistics
import sys
from time import perf_counter

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--molecule', choices=('water', 'ammonia', 'cartesian'), default='ammonia')
    parser.add_argument('--basis', default='cc-pvdz')
    parser.add_argument('--repeats', type=int, default=21)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    fixtures = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tests/test_symmetry_abelian.py'))
    start = perf_counter()
    _, _, ir, labels, c, eri, ao, h, _ = fixtures['molecular_source'](args.molecule, args.basis)
    preparation = perf_counter()-start
    start = perf_counter()
    plan = fixtures['plan'](eri, labels, ir)
    setup = perf_counter()-start
    rng = np.random.default_rng(6128)
    d = rng.normal(size=(len(c),)*2)+1j*rng.normal(size=(len(c),)*2)
    d = np.ascontiguousarray(d+d.conj().T)
    reference = fixtures['dense_jk'](eri, d)
    reduced = plan.jk(d, budget=fixtures['BUDGET'])
    independent = fixtures['dense_jk'](ao, c@d@c.conj().T)
    errors = [float(np.max(np.abs(a-c.conj().T@b@c))) for a, b in zip((reduced.j, reduced.k), independent)]
    energy = lambda j, k: .5*np.einsum('pq,qp', d, 2*c.conj().T@h@c+j-.5*k)
    timings = {'dense': [], 'blocks': []}
    for _ in range(args.repeats):
        start = perf_counter()
        fixtures['dense_jk'](eri, d)
        timings['dense'].append(perf_counter()-start)
        start = perf_counter()
        plan.jk(d, budget=fixtures['BUDGET'])
        timings['blocks'].append(perf_counter()-start)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != 'darwin':
        rss *= 1024
    print(json.dumps(dict(molecule=args.molecule, basis=args.basis, orbitals=len(c),
        populations=np.bincount(labels, minlength=ir.group.order).tolist(),
        full_eri_builder_calls=1, integral_values_materialized=eri.size,
        native_quartet_evaluations='not instrumented; no integral work reduction',
        stored_eri_values=plan.stored_elements, dense_scalar_products=2*eri.size,
        block_scalar_products=reduced.scalar_products, block_contractions=reduced.block_contractions,
        dense_eri_bytes=eri.nbytes, retained_array_bytes=plan.retained_bytes,
        setup_logical_budget_bytes=plan.required_bytes, process_peak_rss_bytes=rss,
        preparation_seconds=preparation, plan_seconds=setup,
        median_seconds={k:statistics.median(v) for k,v in timings.items()},
        omitted_frobenius=plan.omitted_norm, jk_error_bound=reduced.error_bound,
        jk_max_abs_error_ao=errors,
        energy_abs_error=float(abs(energy(reduced.j,reduced.k)-energy(*reference))),
        repeats=args.repeats), indent=2))


if __name__ == '__main__':
    main()
