"""Transport payload and recomputation ladder on the existing finite model.

Source and pair construction are reported separately from transport preparation
and compiled iteration. Tracemalloc does not see Eigen allocations; reservations
are controlled payload bounds, not process-memory guarantees. No chemistry
acceptance is claimed. Run with single-thread BLAS for reproducibility.
"""
import argparse
import json
from time import perf_counter
import tracemalloc

import numpy as np

from tests.test_dlpno_source import _automatic_translation_source, _translation_block_source
from vibeqc.dlpno.translation_driver import run_translation_dlpno_mp2
from vibeqc.dlpno.translation_factors import TranslationHomeFactors
from vibeqc.dlpno.translation_builder import _record
from vibeqc.dlpno.translation_transport import prepare_translation_transport
from vibeqc.dlpno.translation_solver import solve_translation_mp2


def measured(call):
    tracemalloc.start()
    start = perf_counter()
    try:
        value = call()
        return value,perf_counter()-start,tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def main():
    from vibeqc.banner import loaded_process_provenance
    from vibeqc.output import OutputChannel, write
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells',type=int,nargs='+',default=[1,2,4,8,16])
    args = parser.parse_args()
    rows = []
    for cells in args.cells:
        mesh = (cells,1,1)
        source,_,core,home = _automatic_translation_source(mesh,True)
        source.factors = TranslationHomeFactors(home,mesh)
        source = _translation_block_source(source,mesh)
        initial,build_s,build_peak = measured(lambda: run_translation_dlpno_mp2(
            source,mesh,n_frozen=core.shape[1],compact_storage=True,max_iter=1,transport_backend="bounded"))
        build = initial.pair_build
        energies = []
        for backend in ('prepared','bounded'):
            transport,prep_s,prep_peak = measured(lambda: prepare_translation_transport(build,backend=backend))
            solved,solve_s,solve_peak = measured(lambda: solve_translation_mp2(build,
                transport=transport,transport_backend=backend,
                max_transport_bytes=transport.required_bytes,
                conv_tol_energy=1e-13,conv_tol_residual=1e-12))
            assert solved.converged
            energies.append(solved.e_pair_per_cell)
            if backend == 'prepared':
                oracle = solved
            else:
                for actual,expected in zip((*solved.amplitudes,*solved.residuals),
                                           (*oracle.amplitudes,*oracle.residuals)):
                    np.testing.assert_array_equal(actual,expected)
                assert solved.e_pair_per_cell == oracle.e_pair_per_cell
                assert solved.e_pno_correction_per_cell == oracle.e_pno_correction_per_cell
                work = solved.provenance['transport_preparation']
                assert work['target_metric_constructions'] <= solved.n_iter*len(build.pno_ranks)
            # Exact cap exercises both Python and native planning. One byte less
            # must refuse, rather than silently switching implementation.
            exact = prepare_translation_transport(build,backend=backend,max_bytes=transport.required_bytes)
            try:
                prepare_translation_transport(build,backend=backend,max_bytes=transport.required_bytes-1)
            except (ValueError,MemoryError):
                pass
            else:
                raise AssertionError('below-budget transport accepted')
            _,reconstruction_s,reconstruction_peak = measured(lambda: build.spaces.pair_coefficients[0])
            rows.append(dict(cells=cells,backend=backend,
                source_pair_build_and_initial_snapshot_s=build_s,build_traced_peak_bytes=build_peak,
                transport_preparation_s=prep_s,iteration_s=solve_s,
                total_preparation_iteration_s=prep_s+solve_s,
                preparation_traced_peak_bytes=prep_peak,iteration_traced_peak_bytes=solve_peak,
                reconstruction_s=reconstruction_s,reconstruction_traced_peak_bytes=reconstruction_peak,
                single_pair_bytes=8*build.spaces._occupied.shape[0]*max(build.pno_ranks),
                occupied_fock_bytes=build.spaces.occupied_fock.nbytes,
                projected_recipe_bytes=build.spaces.pair_coefficients.nbytes,
                solver_returned_bytes=sum(x.nbytes for x in (*solved.amplitudes,*solved.residuals)),
                transport=_record(solved.provenance['transport_preparation']),
                minimum_feasible_bytes=exact.required_bytes,
                raw=solved.e_pair_per_cell,residual=solved.max_residual,
                pno_correction=solved.e_pno_correction_per_cell,iterations=solved.n_iter))
        assert abs(energies[0]-energies[1]) < 2e-11
        rows[-1]['below_prepared_cap'] = rows[-1]['minimum_feasible_bytes'] < rows[-2]['minimum_feasible_bytes']
        if rows[-1]['below_prepared_cap']:
            try:
                prepare_translation_transport(build,max_bytes=rows[-1]['minimum_feasible_bytes'])
            except ValueError:
                rows[-1]['prepared_refused_at_bounded_cap'] = True
            else:
                raise AssertionError('prepared backend unexpectedly fit the bounded cap')
            end_to_end,seconds,peak = measured(lambda: run_translation_dlpno_mp2(
                source,mesh,n_frozen=core.shape[1],compact_storage=True,
                transport_backend='bounded',max_transport_bytes=rows[-1]['minimum_feasible_bytes'],
                conv_tol_energy=1e-13,conv_tol_residual=1e-12))
            assert end_to_end.solution.converged
            assert abs(end_to_end.solution.e_pair_per_cell-energies[0]) < 2e-11
            rows[-1]['bounded_end_to_end_at_cap_s'] = seconds
            rows[-1]['bounded_end_to_end_at_cap_traced_peak_bytes'] = peak
    with OutputChannel.to_stdout():
        write(json.dumps(dict(runtime=loaded_process_provenance(),rows=rows),indent=2)+'\n')


if __name__ == '__main__':
    main()
