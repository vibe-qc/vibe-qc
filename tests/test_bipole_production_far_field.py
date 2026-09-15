"""Legacy-named production checks for the supported BIPOLE route.

The quartet multipole far field is neither automatic nor available through
the public drivers. The MgO checks use a fold-converged operator cutoff with
an explicitly bounded exact bielectronic zone; farther cells retain the
reciprocal Ewald-J channel. Diamond retains the full exact-zone route.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc._vibeqc_core import (
    PeriodicRHFOptions,
    monkhorst_pack,
)

ANG2BOHR = 1.0 / 0.529177210903

# The precision-derived SR pad at the historical 14-bohr operator cutoff did
# not complete its first exact build inside the 5,400-second per-test budget.
# At 12 bohr this fixture's measured overlap-fold drift is 5e-3, below the
# 1e-2 reliability gate. The explicit 14-bohr SR pad and 8-bohr exact zone
# follow the established bounded-MgO smoke pattern while the 12-iteration cap
# still reaches the convergence and energy contracts asserted below.
MGO_OPERATOR_CUTOFF_BOHR = 12.0
MGO_SR_IMAGE_EXTENT_BOHR = 14.0
MGO_EXACT_ZONE_BOHR = 8.0
MGO_MAX_ITER = 12


def _make_mgo(a_ang=4.21):
    a = a_ang * ANG2BOHR
    lattice = (a / 2.0) * np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    atoms = [vq.Atom(12, [0, 0, 0]), vq.Atom(8, [a / 2] * 3)]
    system = vq.PeriodicSystem(3, lattice, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _make_diamond(a_ang=3.567):
    a = a_ang * ANG2BOHR
    lattice = (a / 2.0) * np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    atoms = [vq.Atom(6, [0, 0, 0]), vq.Atom(6, [a / 4] * 3)]
    system = vq.PeriodicSystem(3, lattice, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _mgo_rhf_options():
    """Return the bounded, fold-reliable MgO production fixture."""
    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = MGO_OPERATOR_CUTOFF_BOHR
    opts.lattice_opts.nuclear_cutoff_bohr = MGO_OPERATOR_CUTOFF_BOHR
    opts.lattice_opts.sr_range_screening = True
    opts.max_iter = MGO_MAX_ITER
    opts.use_diis = True
    opts.conv_tol_energy = 1e-6
    opts.initial_guess = vq.InitialGuess.SAD
    return opts


def _bounded_mgo_kwargs():
    """Pin the expensive SR traversal without changing the public defaults."""
    return {
        "use_ewald_j_split": True,
        "ewald_precision": 1e-6,
        "sr_image_extent_bohr": MGO_SR_IMAGE_EXTENT_BOHR,
        "exact_zone_bohr": MGO_EXACT_ZONE_BOHR,
        "progress": False,
    }


def _assert_bounded_mgo_result(result, *, context):
    assert result.converged, (
        f"{context} SCF failed after {result.n_iter} iterations"
    )
    assert result.exchange_ewald_split is True
    assert result.sr_image_extent_bohr == pytest.approx(
        MGO_SR_IMAGE_EXTENT_BOHR
    )
    assert result.exact_zone_bohr == pytest.approx(MGO_EXACT_ZONE_BOHR)
    assert -280.0 < result.energy < -260.0, (
        f"Energy {result.energy:.4f} out of expected range"
    )
    assert np.isfinite(result.energy)


class TestMgOProductionFarField:
    """MgO SCF on the supported bounded exact-zone route."""

    @pytest.mark.slow
    def test_mgo_rhf_sto3g_gamma_far_field_converges(self):
        """MgO RHF/STO-3G Gamma: the bounded route converges."""
        system, basis = _make_mgo(4.21)
        kmesh = monkhorst_pack(system, [1, 1, 1])
        opts = _mgo_rhf_options()

        from vibeqc.pbc_bipole import run_pbc_bipole_rhf
        result = run_pbc_bipole_rhf(
            system,
            basis,
            kmesh,
            opts,
            **_bounded_mgo_kwargs(),
        )
        _assert_bounded_mgo_result(result, context="Gamma")

    @pytest.mark.slow
    @pytest.mark.parametrize("kmesh_size", [(2, 2, 2)])
    def test_mgo_rhf_sto3g_multik_far_field_converges(self, kmesh_size):
        """MgO RHF/STO-3G multi-k (2,2,2): the bounded route converges."""
        system, basis = _make_mgo(4.21)
        kmesh = monkhorst_pack(system, kmesh_size)
        opts = _mgo_rhf_options()

        from vibeqc.pbc_bipole import run_pbc_bipole_rhf
        result = run_pbc_bipole_rhf(
            system,
            basis,
            kmesh,
            opts,
            **_bounded_mgo_kwargs(),
        )
        _assert_bounded_mgo_result(result, context=f"Multi-k {kmesh_size}")

    @pytest.mark.slow
    def test_mgo_rhf_sto3g_explicit_disable_far_field(self):
        """MgO RHF: explicit disable works."""
        system, basis = _make_mgo(4.21)
        kmesh = monkhorst_pack(system, [1, 1, 1])
        opts = _mgo_rhf_options()

        from vibeqc.pbc_bipole import run_pbc_bipole_rhf
        result = run_pbc_bipole_rhf(
            system,
            basis,
            kmesh,
            opts,
            use_multipole_far_field=False,
            **_bounded_mgo_kwargs(),
        )
        _assert_bounded_mgo_result(result, context="Explicit-disable")


class TestDiamondProductionFarField:
    """Diamond SCF on the supported exact route."""

    @pytest.mark.slow
    def test_diamond_rhf_sto3g_gamma_far_field_converges(self):
        """Diamond RHF/STO-3G Gamma: the exact route converges."""
        system, basis = _make_diamond(3.567)
        kmesh = monkhorst_pack(system, [1, 1, 1])

        opts = PeriodicRHFOptions()
        opts.lattice_opts.cutoff_bohr = 9.0
        opts.max_iter = 30
        opts.use_diis = True
        opts.conv_tol_energy = 1e-6
        opts.initial_guess = vq.InitialGuess.SAD

        from vibeqc.pbc_bipole import run_pbc_bipole_rhf
        result = run_pbc_bipole_rhf(
            system, basis, kmesh, opts,
            use_ewald_j_split=True,
            ewald_precision=1e-6,
            progress=False,
        )
        assert result.converged, f"SCF failed after {result.n_iter} iters"
        # Diamond STO-3G RHF energy ~ -74.9 Ha
        assert -85.0 < result.energy < -65.0, (
            f"Energy {result.energy:.4f} out of expected range"
        )
        assert np.isfinite(result.energy)
