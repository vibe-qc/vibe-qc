"""Route-level guards for the unavailable BIPOLE quartet far-field."""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


def _h2_box(box_bohr: float = 12.0):
    lattice = np.eye(3) * box_bohr
    system = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [1.4, 0.0, 0.0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def test_explicit_far_field_request_fails_before_integral_setup():
    """No separation or cutoff may silently turn an explicit request exact."""
    from vibeqc.pbc_bipole import run_pbc_bipole_rhf

    with pytest.raises(NotImplementedError, match="three-translation"):
        run_pbc_bipole_rhf(None, None, None, use_multipole_far_field=True)


def test_default_and_explicit_exact_route_match():
    """The supported default is bit-identical to explicit opt-out."""
    from vibeqc.pbc_bipole import run_pbc_bipole_rhf

    system, basis = _h2_box()
    kmesh = vq.monkhorst_pack(system, [1, 1, 1])
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 12.0
    opts.max_iter = 1
    opts.initial_guess = vq.InitialGuess.SAD
    opts.use_diis = False

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
    assert default.energy == explicit.energy
