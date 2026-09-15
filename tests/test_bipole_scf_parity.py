"""Slow end-to-end coverage for the supported exact BIPOLE route."""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.pbc_bipole import run_pbc_bipole_rhf


def _h2_box():
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 12.0,
        [vq.Atom(1, [-0.7, 0.0, 0.0]), vq.Atom(1, [0.7, 0.0, 0.0])],
    )
    return system, vq.BasisSet(system.unit_cell_molecule(), "sto-3g")


@pytest.mark.slow
def test_default_exact_scf_matches_explicit_opt_out():
    system, basis = _h2_box()
    kmesh = vq.monkhorst_pack(system, [1, 1, 1])
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 12.0
    opts.max_iter = 30
    opts.use_diis = True
    opts.conv_tol_energy = 1.0e-8
    opts.initial_guess = vq.InitialGuess.SAD

    default = run_pbc_bipole_rhf(
        system,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        progress=False,
    )
    explicit = run_pbc_bipole_rhf(
        system,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        use_multipole_far_field=False,
        progress=False,
    )
    assert default.converged and explicit.converged
    assert default.energy == explicit.energy


@pytest.mark.slow
@pytest.mark.parametrize("order", [0, 2, 4, 6])
def test_explicit_far_field_orders_all_fail_closed(order: int):
    with pytest.raises(NotImplementedError, match="three-translation"):
        run_pbc_bipole_rhf(
            None,
            None,
            None,
            use_multipole_far_field=True,
            multipole_l_max=order,
        )
