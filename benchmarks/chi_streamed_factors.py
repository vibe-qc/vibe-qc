"""Bounded synthetic retained-factor ladder, not scientific acceptance.

Run with the source checkout's Python; output uses vibeqc.output. The model
and source construction precede measurements. Tracemalloc includes Python
overhead and is reported separately from the provider's numerical-payload
ledger. Neither measurement is process RSS or a total-memory guarantee.
"""
import json
from time import perf_counter
import tracemalloc

import numpy as np

from vibeqc.dlpno.translation_bloch_factors import TranslationBlochFactors
from vibeqc.periodic.chi.canonical_mp2 import CanonicalFactorAdapter
from vibeqc.periodic.chi.local_mp2 import _home_factors
from vibeqc.periodic_fit_source import RSGDFFitSource, factor_digest
from vibeqc.periodic_fitted_support import ChiFittedOperator


def model(n, b=16, a=32):
    rng = np.random.default_rng(508+n)
    mesh = (n,1,1)
    cells = np.array([(r,0,0) for r in range(n)])
    k = cells / np.asarray(mesh)
    phase = np.exp(2j*np.pi*(k @ cells.T))
    home = rng.normal(size=(a,n*b,n*b))*.01
    home = (home+home.transpose(0,2,1))*.5
    home[-1] = 0.  # Retain the canonical null mode, not a compressed eigenframe.
    grid = home.reshape(a,n,b,n,b)
    factors = {(i,j):np.einsum('r,s,Prmsv->Pmv',phase[i].conj(),phase[j],grid)
               for i in range(n) for j in range(n)}
    groups = tuple((tuple(k[q]),tuple((i,j) for i,j in factors if (j-i)%n==q))
                   for q in range(n))
    records = [dict(q=q,lattice=(np.eye(3)*2*np.pi).tolist(),
                    metric_eigenvalues=[1.]*(a-1)+[0.],metric_keep=[True]*(a-1)+[False],rank=a-1)
               for q,_ in groups]
    descriptor = RSGDFFitSource('native-chi-classic-finite-contract',1,'classic','synthetic',
        'synthetic','synthetic','synthetic-aux','synthetic-aux',(),1e-9,'canonical',groups,
        tuple(map(tuple,k)),10.,None,tuple((p,v.shape) for p,v in sorted(factors.items())),
        factor_digest(factors),executed_domains_json=json.dumps(dict(q_records=records)))
    operator = ChiFittedOperator(factors,descriptor,np.ones(n)/n)
    return CanonicalFactorAdapter(operator), phase, cells, mesh


def measure(function):
    tracemalloc.start()
    start = perf_counter()
    result = function()
    elapsed = perf_counter()-start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak


def main():
    import argparse
    import subprocess
    from vibeqc.banner import loaded_process_provenance
    from vibeqc.output import OutputChannel, write
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells',type=int,nargs='+',default=[1,2,4,8])
    parser.add_argument('--aos',type=int,default=16)
    parser.add_argument('--aux',type=int,default=32)
    parser.add_argument('--budgets',nargs='+',default=['2097152'],
                        help='byte caps or minimum (audit minimum, may refuse pair output)')
    parser.add_argument('--cache-bytes',type=int,default=1024**2)
    parser.add_argument('--baseline',help='Git revision of the former complete-table provider')
    args = parser.parse_args()
    baseline = None
    if args.baseline:
        # Explicit small-system oracle; no checkout, installed module or source
        # state is changed. Operational measurements stay outside the checkout.
        source = subprocess.check_output(['git','show',
            args.baseline+':python/vibeqc/dlpno/translation_bloch_factors.py'],text=True)
        namespace = dict(__name__='vibeqc.dlpno._factor_benchmark_oracle',__package__='vibeqc.dlpno')
        exec(compile(source,'<complete-table-factor-oracle>','exec'),namespace)
        baseline = namespace['TranslationBlochFactors']
    rows = []
    for n in args.cells:
        adapter, phase, cells, mesh = model(n,b=args.aos,a=args.aux)
        (dense,_), dense_prepare, dense_peak = measure(lambda:_home_factors(adapter,phase,cells,mesh))
        rng = np.random.default_rng(19)
        left, right = rng.normal(size=(args.aos*n,2)),rng.normal(size=(args.aos*n,8))
        expected, dense_pair, dense_pair_peak = measure(lambda:dense.pair_exchange(left,right))
        old = old_record = None
        if baseline is not None:
            old, old_time, old_peak = measure(lambda:baseline(adapter,mesh,
                max_bytes=2*1024**2,auxiliary_tile=4,cache_bytes=args.cache_bytes))
            old_value, old_pair_time, old_pair_peak = measure(lambda:old.pair_exchange(left,right))
            np.testing.assert_allclose(old_value,expected,rtol=2e-12,atol=1e-10)
            old_record = dict(preparation_s=old_time,traced_peak=old_peak,
                              pair_s=old_pair_time,pair_traced_peak=old_pair_peak,provider=old.provenance)
        minimum = TranslationBlochFactors._iterator_bytes()+320*n
        try:
            TranslationBlochFactors(adapter,mesh,max_bytes=minimum-1)
        except MemoryError:
            below_minimum_refused = True
        else:
            raise AssertionError('one-byte-below audit budget must refuse')
        for requested in args.budgets:
            budget = minimum if requested=='minimum' else int(requested)
            streamed, stream_prepare, stream_peak = measure(lambda:TranslationBlochFactors(
                adapter,mesh,max_bytes=budget,auxiliary_tile=4,cache_bytes=args.cache_bytes))
            preparation_record = streamed.provenance
            pair = {}
            try:
                actual, elapsed, peak = measure(lambda:streamed.pair_exchange(left,right))
            except MemoryError as error:
                # Minimum audit space need not fit the unavoidable pair output.
                tracemalloc.stop()
                pair['refusal'] = str(error)
            else:
                np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=1e-10)
                pair = dict(seconds=elapsed,traced_peak=peak,max_error=float(np.max(abs(actual-expected))))
            rows.append(dict(cells=n,primitive_aos=args.aos,primitive_auxiliary=args.aux,
                budget=budget,minimum_audit_bytes=minimum,below_minimum_refused=below_minimum_refused,
                preparation_accounted_bytes=preparation_record['peak_accounted_bytes'],
                preparation_cache_bytes=preparation_record['cache_bytes'],
                dense_preparation_s=dense_prepare,dense_pair_s=dense_pair,
                dense_preparation_traced_peak=dense_peak,dense_pair_traced_peak=dense_pair_peak,
                streamed_preparation_s=stream_prepare,streamed_preparation_traced_peak=stream_peak,
                pair=pair,provider=streamed.provenance,complete_table=old_record,
                audit_differences=None if old is None else {
                    key:streamed.audit[key]-old.audit[key] for key in old.audit}))
    with OutputChannel.to_stdout():
        write(json.dumps(dict(runtime=loaded_process_provenance(),baseline=args.baseline,rows=rows),indent=2)+'\n')


if __name__ == '__main__':
    main()
