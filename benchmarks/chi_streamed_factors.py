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
    from vibeqc.banner import loaded_process_provenance
    from vibeqc.output import OutputChannel, write
    rows = []
    for n in (1,2,4,8):
        adapter, phase, cells, mesh = model(n)
        (dense,_), dense_prepare, dense_peak = measure(lambda:_home_factors(adapter,phase,cells,mesh))
        streamed, stream_prepare, stream_peak = measure(lambda:TranslationBlochFactors(
            adapter,mesh,max_bytes=2*1024**2,auxiliary_tile=4,cache_bytes=32*n*n))
        rng = np.random.default_rng(19)
        left, right = rng.normal(size=(16*n,2)),rng.normal(size=(16*n,8))
        expected, dense_pair, dense_pair_peak = measure(lambda:dense.pair_exchange(left,right))
        actual, stream_pair, stream_pair_peak = measure(lambda:streamed.pair_exchange(left,right))
        error = float(np.max(abs(actual-expected)))
        np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=1e-10)
        rows.append(dict(cells=n,primitive_aos=16,primitive_auxiliary=32,
            dense_preparation_s=dense_prepare,streamed_preparation_s=stream_prepare,
            dense_pair_s=dense_pair,streamed_pair_s=stream_pair,
            dense_preparation_traced_peak=dense_peak,streamed_preparation_traced_peak=stream_peak,
            dense_pair_traced_peak=dense_pair_peak,streamed_pair_traced_peak=stream_pair_peak,
            max_pair_error=error,provider=streamed.provenance))
    with OutputChannel.to_stdout():
        write(json.dumps(dict(runtime=loaded_process_provenance(),rows=rows),indent=2)+'\n')


if __name__ == '__main__':
    main()
