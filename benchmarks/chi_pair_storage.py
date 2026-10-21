"""Bounded retained-coefficient ladder using the existing finite-source fixture.

Requires the development test dependencies. This reuses the nonorthogonal,
occupied-coupled frozen-core model used by the canonical-limit source tests.
Model creation precedes measurement; no physical calculation is claimed.
"""
import json
from time import perf_counter
import tracemalloc

from tests.test_dlpno_source import _automatic_translation_source, _translation_block_source
from vibeqc.dlpno.translation_driver import run_translation_dlpno_mp2
from vibeqc.dlpno.translation_factors import TranslationHomeFactors
from vibeqc.dlpno.translation_builder import _record


def main():
    from vibeqc.banner import loaded_process_provenance
    from vibeqc.output import OutputChannel, write
    rows = []
    for cells in (1,2,4,8):
        mesh = (cells,1,1)
        source, _, core, home = _automatic_translation_source(mesh, True)
        source.factors = TranslationHomeFactors(home,mesh)
        source = _translation_block_source(source,mesh)
        energies = []
        for compact in (False,True):
            tracemalloc.start()
            start = perf_counter()
            result = run_translation_dlpno_mp2(source,mesh,n_frozen=core.shape[1],
                compact_storage=compact,conv_tol_energy=1e-13,conv_tol_residual=1e-12)
            elapsed = perf_counter()-start
            _,peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            assert result.solution.converged
            energies.append(result.solution.e_pair_per_cell)
            spaces,selection = result.pair_build.spaces,result.occupied_selection
            arrays = {id(value):value for value in (spaces._occupied,spaces._frozen_occupied,
                                                    selection.occupied,selection.frozen_occupied)}
            rows.append(dict(cells=cells,compact_storage=compact,elapsed_s=elapsed,
                traced_peak_bytes=peak,unique_selected_bytes=sum(v.nbytes for v in arrays.values()),
                storage=_record(result.pair_build.provenance['coefficient_storage']),
                transport=_record(result.solution.provenance['transport_preparation']),
                raw=energies[-1],residual=result.solution.max_residual,
                pno_correction=result.solution.e_pno_correction_per_cell))
        assert abs(energies[0]-energies[1]) < 2e-11
    with OutputChannel.to_stdout():
        write(json.dumps(dict(runtime=loaded_process_provenance(),rows=rows),indent=2)+'\n')


if __name__ == '__main__':
    main()
