"""K-mesh convergence fixtures for native full-k DFTB routes."""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc._vibeqc_core import Atom, PeriodicSystem, monkhorst_pack
from vibeqc._vibeqc_core import semiempirical as _se


_CHAIN_LENGTH = 4.1
_CUTOFF = 60.0
# #478 (the Wolfsberg-Helmholz hopping without the duplicated 1/2) widened
# this chain's bands, so the exponential k-convergence sets in on denser
# meshes: on (4, 8, 12) the 8/4 error ratio was 0.036 at T=0 and 0.41 at
# T=0.05. The gates below are unchanged; the sequence moved (#561).
_MESH_SIZES = (8, 16, 24)


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
    # Independent 96-point values for the fixed chain above, all taken at a
    # fixed 60-bohr lattice cutoff, re-measured after #478 (#561).  The
    # 48-point values agree with them to better than 2e-15 Ha, so they are
    # converged in the k-mesh, which is the only axis this test measures.
    #
    # They are deliberately NOT infinite-cutoff limits: SCC-DFTB still carries
    # lattice-cutoff error against its leading-1/R^2 limit, and a cutoff-free
    # value belongs with the supercell-convergence tests in
    # tests/test_ccm_semiempirical.py.  Conflating the two is what issue #312
    # was filed about.
    [
        ("dftb0", 0.0, -0.0971296477811251),
        ("scc_dftb", 0.0, -0.08824376448128257),
        ("dftb0", 0.05, -0.12201502592535109),
        ("scc_dftb", 0.05, -0.11557743948151629),
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
    # Absolute backstop behind the relative gates above (2.5e-9 since issue
    # #312).  Measured on the 24-point mesh after #478: 1.5e-13 and 1.2e-12 Ha
    # at T=0, 6.0e-10 and 1.1e-9 Ha at T=0.05.
    assert errors[2] < 2.5e-9
