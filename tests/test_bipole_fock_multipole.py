"""Driver guards for the retired BIPOLE multipole far-field branches."""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import InitialGuess, monkhorst_pack


def _h2_3d(box: float = 14.0):
    """H₂ in a large cubic box so there are many image cells."""
    lattice = np.eye(3) * box
    system = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _rhf_opts(cutoff: float = 12.0) -> vq.PeriodicRHFOptions:
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = cutoff
    opts.lattice_opts.nuclear_cutoff_bohr = cutoff
    opts.max_iter = 1
    opts.initial_guess = InitialGuess.SAD
    opts.use_diis = False
    return opts


@pytest.mark.parametrize(
    ("module_name", "driver_name"),
    [
        ("vibeqc.pbc_bipole", "run_pbc_bipole_rhf"),
        ("vibeqc.pbc_bipole_rks", "run_pbc_bipole_rks"),
        ("vibeqc.pbc_bipole_uhf", "run_pbc_bipole_uhf"),
        ("vibeqc.pbc_bipole_uks", "run_pbc_bipole_uks"),
    ],
)
def test_multipole_far_field_fails_closed_before_scf_setup(
    module_name: str,
    driver_name: str,
):
    """An explicit approximation request must never silently run exact."""
    driver = getattr(importlib.import_module(module_name), driver_name)
    with pytest.raises(
        NotImplementedError,
        match="three-translation periodic Fock domain",
    ):
        driver(None, None, None, use_multipole_far_field=True)


def test_multipole_far_field_off_by_default_rhf():
    """Smoke: explicitly disabling the retired branch produces a finite energy.

    The supported default and explicit ``False`` both select the exact
    periodic route."""
    import warnings

    from vibeqc.pbc_bipole import run_pbc_bipole_rhf

    system, basis = _h2_3d()
    kmesh = monkhorst_pack(system, [1, 1, 1])
    opts = _rhf_opts()

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # any retirement warn -> fail
        result = run_pbc_bipole_rhf(
            system,
            basis,
            kmesh,
            opts,
            use_ewald_j_split=True,
            use_exchange_ewald_split=False,
            ewald_precision=1e-8,
            use_multipole_far_field=False,  # explicit disable
            progress=False,
        )
    assert result.n_iter == 1
    assert math.isfinite(result.energy)
