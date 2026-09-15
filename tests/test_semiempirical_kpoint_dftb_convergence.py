"""K-mesh convergence fixtures for native full-k DFTB routes."""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc._vibeqc_core import Atom, PeriodicSystem, monkhorst_pack
from vibeqc._vibeqc_core import semiempirical as _se


_CHAIN_LENGTH = 4.1
_CUTOFF = 60.0
_MESH_SIZES = (4, 8, 12)


def _hli_chain() -> PeriodicSystem:
    return PeriodicSystem(
        1,
        np.diag([_CHAIN_LENGTH, 30.0, 30.0]),
        [
            Atom(1, [0.17, 0.31, 0.0]),
            Atom(3, [1.39, -0.22, 0.0]),
        ],
        0,
        1,
    )


@pytest.fixture(scope="module")
def parameters():
    return _se.SemiempiricalParameters.dftb0_default()


def _mesh_value(parameters, method: str, mesh_size: int, temperature: float) -> float:
    system = _hli_chain()
    kmesh = monkhorst_pack(system, (mesh_size, 1, 1))
    occupation_options = _se.KPointOccupationOptions()
    occupation_options.smearing_temperature = temperature

    if method == "dftb0":
        result = _se.run_dftb0_kpoints(
            system,
            parameters,
            kmesh,
            _CUTOFF,
            occupation_options,
        )
    elif method == "scc_dftb":
        scc_options = _se.SCCOptions()
        scc_options.max_iter = 500
        scc_options.conv_tol_charge = 1.0e-10
        result = _se.run_scc_dftb_kpoints(
            system,
            parameters,
            kmesh,
            scc_options,
            _CUTOFF,
            occupation_options,
        )
        assert result.converged
    else:
        raise AssertionError(f"unexpected convergence fixture method: {method}")

    return result.free_energy if temperature > 0.0 else result.energy


@pytest.mark.parametrize(
    ("method", "temperature", "dense_limit"),
    # Independent 24-point values for the fixed chain above, all taken at a
    # fixed 60-bohr lattice cutoff.  The 48-point values agree with them to
    # better than 2e-15 Ha, so they are converged in the k-mesh, which is the
    # only axis this test measures.
    #
    # They are deliberately NOT infinite-cutoff limits.  The two DFTB0 rows do
    # happen to be cutoff-saturated (unchanged to 1e-16 from 60 out to 2560
    # bohr), but SCC-DFTB still carries 35.4 microhartree (T=0) and 9.4
    # microhartree (T=0.05) of cutoff error against its leading-1/R^2 limit.
    # The cutoff-free SCC-DFTB T=0 value is -0.107889099350429 Ha and lives in
    # tests/test_ccm_semiempirical.py, which is where a supercell-convergence
    # test needs it.  Conflating the two is what issue #312 was filed about.
    [
        ("dftb0", 0.0, -0.129238654463388),
        # Re-pinned 2026-09-07 (D1/#425): the periodic SCC-DFTB gamma is the
        # Ewald-split Elstner form, so the dense limit moved. Measured on a
        # 48-point mesh, where the 12-point error is already 1.5e-09.
        ("scc_dftb", 0.0, -0.107171439016323),
        ("dftb0", 0.05, -0.201731944060102),
        ("scc_dftb", 0.05, -0.197570260683130),
    ],
)
def test_even_kmesh_sequence_converges_to_dense_limit(
    parameters,
    method,
    temperature,
    dense_limit,
):
    values = np.array(
        [
            _mesh_value(parameters, method, mesh_size, temperature)
            for mesh_size in _MESH_SIZES
        ]
    )
    errors = np.abs(values - dense_limit)

    assert errors[1] < 0.01 * errors[0]
    assert errors[2] < 0.02 * errors[1]
    # Absolute backstop behind the relative gates above.  Moving the fixture
    # from a 12-bohr to a 60-bohr cutoff (issue #312) raised the two T=0
    # residuals to 1.76e-9 and 1.67e-9 Ha, which left the historical 2.0e-9
    # bound only ~1.1x headroom; 2.5e-9 restores ~1.4x.  The T=0.05 rows sit
    # near 1e-14 and are unaffected.
    assert errors[2] < 2.5e-9
