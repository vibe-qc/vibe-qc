"""Tests for the CRYSTAL-gauge BIPOLE UHF driver."""

from __future__ import annotations

import math

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import InitialGuess, monkhorst_pack
from vibeqc._vibeqc_core import PeriodicRHFOptions
from vibeqc.pbc_bipole import run_pbc_bipole_rhf
from vibeqc.pbc_bipole_uhf import run_pbc_bipole_uhf


def _opts(max_iter: int = 1) -> PeriodicRHFOptions:
    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 5.0
    opts.lattice_opts.nuclear_cutoff_bohr = 10.0
    opts.max_iter = max_iter
    opts.initial_guess = InitialGuess.HCORE
    opts.use_diis = False
    return opts


def test_pbc_bipole_uhf_is_exported():
    assert vq.run_pbc_bipole_uhf is run_pbc_bipole_uhf


def test_closed_shell_bipole_uhf_matches_rhf_one_iteration():
    """Multiplicity-1 UHF should reduce to RHF for the same local SAD."""
    lattice = np.eye(3) * 7.0
    system = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(2, [0.0, 0.0, 0.0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, [1, 1, 1])
    opts = _opts()
    opts.initial_guess = InitialGuess.SAD

    # Both drivers default to the corrected gauge (Ewald exchange
    # split) at Γ since option (b) Phase 4b — closed-shell UHF must
    # reduce to RHF in the production gauge.
    rhf = run_pbc_bipole_rhf(
        system,
        basis,
        kmesh,
        opts,
        ewald_precision=1e-6,
        progress=False,
    )
    uhf = run_pbc_bipole_uhf(
        system,
        basis,
        kmesh,
        opts,
        ewald_precision=1e-6,
        progress=False,
    )

    assert uhf.n_iter == rhf.n_iter == 1
    assert math.isclose(uhf.energy, rhf.energy, abs_tol=1e-6)
    assert math.isclose(uhf.e_electronic, rhf.e_electronic, abs_tol=1e-6)
    assert math.isclose(uhf.e_nuclear, rhf.e_nuclear, abs_tol=1e-6)
    assert math.isclose(uhf.s_squared, 0.0, abs_tol=1e-12)
    np.testing.assert_allclose(uhf.fock_alpha[0], rhf.fock[0], atol=1e-12)
    np.testing.assert_allclose(uhf.fock_beta[0], rhf.fock[0], atol=1e-12)

    c_rhf = rhf.energy_components[-1]
    c_uhf = uhf.energy_components[-1]
    assert math.isclose(c_uhf.e_kinetic, c_rhf.e_kinetic, abs_tol=1e-12)
    assert math.isclose(
        c_uhf.e_nuclear_attraction,
        c_rhf.e_nuclear_attraction,
        abs_tol=1e-12,
    )
    assert math.isclose(c_uhf.e_two_electron, c_rhf.e_two_electron, abs_tol=1e-12)
    assert math.isclose(c_uhf.e_j_short_range, c_rhf.e_j_short_range, abs_tol=1e-12)
    assert math.isclose(c_uhf.e_j_long_range, c_rhf.e_j_long_range, abs_tol=1e-12)
    assert math.isclose(c_uhf.e_exchange, c_rhf.e_exchange, abs_tol=1e-12)


def test_open_shell_bipole_uhf_doublet_smoke():
    lattice = np.eye(3) * 8.0
    system = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(1, [0.0, 0.0, 0.0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, [1, 1, 1])

    result = run_pbc_bipole_uhf(
        system,
        basis,
        kmesh,
        _opts(),
        ewald_precision=1e-6,
        progress=False,
    )

    assert result.n_iter == 1
    assert math.isfinite(result.energy)
    assert math.isclose(result.s_squared, 0.75, abs_tol=1e-12)
    assert math.isclose(result.s_squared_ideal, 0.75, abs_tol=1e-12)
    assert system.multiplicity == 1  # promotion does not mutate caller state
    comp = result.energy_components[-1]
    assert math.isclose(
        comp.e_electronic
        + comp.e_nuclear_repulsion
        + (comp.e_ext_el_spheropole or 0.0),
        comp.e_total,
        abs_tol=1e-10,
    )
    assert math.isclose(
        comp.e_j_short_range + comp.e_j_long_range + comp.e_exchange,
        comp.e_two_electron,
        abs_tol=1e-10,
    )


def test_open_shell_bipole_uhf_uses_shared_unrestricted_builder(monkeypatch):
    import vibeqc.pbc_bipole_uhf as uhf_mod

    calls = []
    original = uhf_mod.build_bipole_unrestricted_fock

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(uhf_mod, "build_bipole_unrestricted_fock", _spy)

    lattice = np.eye(3) * 8.0
    system = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(1, [0.0, 0.0, 0.0])],
        multiplicity=2,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, [1, 1, 1])

    result = run_pbc_bipole_uhf(
        system,
        basis,
        kmesh,
        _opts(),
        ewald_precision=1e-6,
        progress=False,
    )

    assert result.n_iter == 1
    assert calls


def test_open_shell_bipole_uhf_smearing_smoke():
    lattice = np.eye(3) * 8.0
    system = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(1, [0.0, 0.0, 0.0])],
        multiplicity=2,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, [1, 1, 1])
    opts = _opts()
    opts.smearing_temperature = 0.005

    result = run_pbc_bipole_uhf(
        system,
        basis,
        kmesh,
        opts,
        ewald_precision=1e-6,
        progress=False,
    )

    assert result.n_iter == 1
    assert result.smearing_temperature == pytest.approx(0.005)
    assert math.isfinite(result.energy)
    assert math.isfinite(result.free_energy)
    assert result.scf_trace[-1].energy == pytest.approx(result.free_energy)
    assert len(result.occupations_alpha) == 1
    assert len(result.occupations_beta) == 1
    assert result.occupations_alpha[0].sum() == pytest.approx(1.0, abs=1e-12)
    assert result.occupations_beta[0].sum() == pytest.approx(0.0, abs=1e-12)

    D_alpha_expected = vq._vibeqc_core.real_space_density_from_kpoints_fractional(
        result.mo_coeffs_alpha,
        result.occupations_alpha,
        kmesh,
        result.density_alpha.cells,
    )
    D_beta_expected = vq._vibeqc_core.real_space_density_from_kpoints_fractional(
        result.mo_coeffs_beta,
        result.occupations_beta,
        kmesh,
        result.density_beta.cells,
    )
    for got, expected in zip(result.density_alpha.blocks, D_alpha_expected.blocks):
        np.testing.assert_allclose(got, expected, atol=1e-9)
    for got, expected in zip(result.density_beta.blocks, D_beta_expected.blocks):
        np.testing.assert_allclose(got, expected, atol=1e-9)


def test_closed_shell_bipole_uhf_multik_matches_rhf():
    """Exercise padded multi-k J_SR for UHF's total Hartree density."""
    lattice = np.eye(3) * 7.0
    system = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(2, [0.0, 0.0, 0.0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, [2, 2, 2], use_symmetry=False)
    opts = _opts()

    rhf = run_pbc_bipole_rhf(
        system,
        basis,
        kmesh,
        opts,
        use_exchange_ewald_split=False,  # closed-shell UHF==RHF is gauge-independent; legacy avoids the [2,2,2] torus-cutoff requirement
        ewald_precision=1e-6,
        progress=False,
    )
    uhf = run_pbc_bipole_uhf(
        system,
        basis,
        kmesh,
        opts,
        use_exchange_ewald_split=False,  # closed-shell UHF==RHF is gauge-independent; legacy avoids the [2,2,2] torus-cutoff requirement
        ewald_precision=1e-6,
        progress=False,
    )

    assert len(uhf.mo_energies_alpha) == len(rhf.mo_energies) == 8
    assert uhf.sr_image_extent_bohr == pytest.approx(rhf.sr_image_extent_bohr)
    assert uhf.sr_image_extent_bohr is not None
    assert math.isclose(uhf.energy, rhf.energy, abs_tol=1e-6)
    uhf_comp = uhf.energy_components[-1]
    rhf_comp = rhf.energy_components[-1]
    for field in (
        "e_j_short_range",
        "e_j_long_range",
        "e_exchange",
        "e_two_electron",
    ):
        assert getattr(uhf_comp, field) == pytest.approx(
            getattr(rhf_comp, field),
            abs=1e-12,
        )
    for fock_a, fock_b, fock_r in zip(
        uhf.fock_alpha,
        uhf.fock_beta,
        rhf.fock,
    ):
        np.testing.assert_allclose(fock_a, fock_r, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(fock_b, fock_r, rtol=0.0, atol=1e-12)
