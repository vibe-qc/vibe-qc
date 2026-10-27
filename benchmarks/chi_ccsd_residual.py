"""Single-snapshot internal CCSD residual cost; not iterative or scientific acceptance.

Use OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1. Tracemalloc excludes native
Eigen/library allocations. The unreduced engine is an explicit small oracle.
"""
import argparse
import json
from time import perf_counter
import tracemalloc

import numpy as np

from tests.test_translation_ccsd_residual import model, amplitudes, unfolded_oracle
from vibeqc.dlpno.translation_ccsd import evaluate_translation_ccsd
from vibeqc.dlpno.translation_transport import prepare_translation_transport


def measured(call):
    tracemalloc.start()
    start = perf_counter()
    try:
        result = call()
        return result,perf_counter()-start,tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def main():
    from vibeqc.banner import loaded_process_provenance
    from vibeqc.output import OutputChannel,write
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells',type=int,nargs='+',default=[1,2,3,4,8])
    args=parser.parse_args()
    rows=[]
    for n in args.cells:
        source,build=model((n,1,1),True)
        singles,doubles=amplitudes(build)
        transport,prep_s,prep_peak=measured(lambda:prepare_translation_transport(build,backend='bounded'))
        actual,seconds,peak=measured(lambda:evaluate_translation_ccsd(
            source,build,singles,doubles,transport=transport))
        oracle,oracle_s,oracle_peak=measured(lambda:unfolded_oracle(source,build,singles,doubles,terms=False))
        result,data=oracle
        defect=max(float(np.max(abs(a-result['R2'][data['pairs'].index(pair)])))
                   for a,pair in zip(actual['R2'],build.spaces.representative_pairs))
        defect=max(defect,max(float(np.max(abs(a-result['R1'][i]))) for i,a in enumerate(actual['R1'])))
        assert defect<2e-11
        assert abs(actual['energy']-result['energy']/n)<2e-12
        rows.append(dict(cells=n,transport_prepare_s=prep_s,transport_prepare_traced_bytes=prep_peak,
            evaluate_s=seconds,evaluate_traced_peak_bytes=peak,oracle_s=oracle_s,
            oracle_traced_peak_bytes=oracle_peak,max_residual_difference=defect,
            raw_energy_difference=abs(actual['energy']-result['energy']/n),
            output_bytes=sum(x.nbytes for x in (*actual['R1'],*actual['R2'])),
            occupied_fock_bytes=build.spaces.occupied_fock.nbytes,
            projected_recipe_bytes=build.spaces.pair_coefficients.nbytes,
            transport_required_bytes=transport.required_bytes,cc=actual['provenance']))
    with OutputChannel.to_stdout():
        write(json.dumps(dict(runtime=loaded_process_provenance(),rows=rows),indent=2)+'\n')


if __name__=='__main__':
    main()
