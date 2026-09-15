"""Fermi-Dirac occupation gates for native full-k DFTB routes."""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc._vibeqc_core import (
    Atom,
    PeriodicSystem,
    bloch_kmesh_from_lists,
    monkhorst_pack,
)
from vibeqc._vibeqc_core import semiempirical as _se
from vibeqc.semiempirical.periodic import finite_difference_gradient
from vibeqc.smearing import fermi_dirac_occupations_per_k


_CHAIN_LENGTH = 4.1
_CUTOFF = 12.0
_TEMPERATURE = 0.05


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


def _filled_argon(charge: int = 0) -> PeriodicSystem:
    return PeriodicSystem(
        3,
        np.diag([10.0, 10.0, 10.0]),
        [Atom(18, [0.0, 0.0, 0.0])],
        charge,
        1,
    )


def _occupation_options(temperature: float) -> _se.KPointOccupationOptions:
    options = _se.KPointOccupationOptions()
    options.smearing_temperature = temperature
    return options


def _weighted_electron_count(kmesh, occupations) -> float:
    return sum(
        weight * float(np.asarray(occupation).sum())
        for weight, occupation in zip(kmesh.weights, occupations)
    )


@pytest.fixture(scope="module")
def parameters():
    return _se.SemiempiricalParameters.dftb0_default()


def test_dftb0_zero_temperature_keeps_integer_aufbau(parameters):
    system = _hli_chain()
    kmesh = monkhorst_pack(system, (3, 1, 1))
    result = _se.run_dftb0_kpoints(system, parameters, kmesh, _CUTOFF)

    assert result.free_energy == result.energy
    assert result.entropy == 0.0
    assert result.smearing_temperature == 0.0
    for occupation in result.occupations_per_k:
        np.testing.assert_array_equal(occupation, [2.0, 0.0])


def test_dftb0_dense_filled_mesh_respects_accumulated_band_capacity(parameters):
    system = _filled_argon()
    kmesh = monkhorst_pack(system, (24, 24, 24))

    result = _se.run_dftb0_kpoints(system, parameters, kmesh, _CUTOFF)

    assert result.n_kpoints == 24**3
    assert _weighted_electron_count(kmesh, result.occupations_per_k) == (
        pytest.approx(8.0, abs=1.0e-12)
    )
    for occupation in result.occupations_per_k:
        np.testing.assert_array_equal(occupation, [2.0, 2.0, 2.0, 2.0])


def test_dftb0_dense_filled_mesh_reaches_finite_temperature_endpoint(parameters):
    system = _filled_argon()
    kmesh = monkhorst_pack(system, (22, 22, 22))
    options = _occupation_options(0.01)

    result = _se.run_dftb0_kpoints(
        system,
        parameters,
        kmesh,
        _CUTOFF,
        options,
    )

    assert result.n_kpoints == 22**3
    assert _weighted_electron_count(kmesh, result.occupations_per_k) == (
        pytest.approx(8.0, abs=1.0e-12)
    )
    assert result.entropy == 0.0
    assert result.free_energy == result.energy
    for occupation in result.occupations_per_k:
        np.testing.assert_array_equal(occupation, [2.0, 2.0, 2.0, 2.0])


def test_dftb0_band_capacity_still_rejects_real_overfilling(parameters):
    system = _filled_argon(charge=-2)
    kmesh = monkhorst_pack(system, (2, 2, 2))

    with pytest.raises(ValueError, match="electron count is outside band capacity"):
        _se.run_dftb0_kpoints(system, parameters, kmesh, _CUTOFF)


def test_zero_weight_kpoints_do_not_set_endpoint_fermi_levels(parameters):
    empty_system = _hli_chain()
    empty_system.charge = 2
    empty_mesh_points = monkhorst_pack(empty_system, (3, 1, 1)).kpoints
    endpoint_weights = [0.0, 1.0, 0.0]
    empty_mesh = bloch_kmesh_from_lists(empty_mesh_points, endpoint_weights)
    empty = _se.run_dftb0_kpoints(
        empty_system,
        parameters,
        empty_mesh,
        _CUTOFF,
    )
    assert empty.fermi_level == min(empty.eps_per_k[1])

    full_system = _filled_argon()
    full_mesh_points = monkhorst_pack(full_system, (3, 1, 1)).kpoints
    full_mesh = bloch_kmesh_from_lists(full_mesh_points, endpoint_weights)
    full = _se.run_dftb0_kpoints(
        full_system,
        parameters,
        full_mesh,
        _CUTOFF,
    )
    assert full.fermi_level == max(full.eps_per_k[1])


def test_dftb0_native_fermi_occupations_match_python(parameters):
    system = _hli_chain()
    kmesh = monkhorst_pack(system, (3, 1, 1))
    options = _occupation_options(_TEMPERATURE)
    result = _se.run_dftb0_kpoints(
        system,
        parameters,
        kmesh,
        _CUTOFF,
        options,
    )
    expected_occupations, expected_mu, expected_entropy = (
        fermi_dirac_occupations_per_k(
            result.eps_per_k,
            kmesh.weights,
            2.0,
            _TEMPERATURE,
        )
    )

    for actual, expected in zip(
        result.occupations_per_k,
        expected_occupations,
    ):
        np.testing.assert_allclose(actual, expected, atol=1.0e-14)
    assert result.fermi_level == pytest.approx(expected_mu, abs=1.0e-14)
    assert result.entropy == pytest.approx(expected_entropy, abs=1.0e-14)
    assert _weighted_electron_count(kmesh, result.occupations_per_k) == (
        pytest.approx(2.0, abs=1.0e-12)
    )
    assert any(
        np.any((np.asarray(occupation) > 0.1) & (np.asarray(occupation) < 1.9))
        for occupation in result.occupations_per_k
    )

    expected_electronic = sum(
        weight * float(np.dot(occupation, energies))
        for weight, occupation, energies in zip(
            kmesh.weights,
            expected_occupations,
            result.eps_per_k,
        )
    )
    assert result.e_electronic == pytest.approx(expected_electronic, abs=1.0e-14)
    assert result.energy == pytest.approx(
        expected_electronic + result.e_repulsive,
        abs=1.0e-14,
    )
    assert result.free_energy == pytest.approx(
        result.energy - _TEMPERATURE * result.entropy,
        abs=1.0e-14,
    )


def test_scc_dftb_fractional_density_matches_python_occupations(parameters):
    system = _hli_chain()
    kmesh = monkhorst_pack(system, (3, 1, 1))
    scc_options = _se.SCCOptions()
    scc_options.max_iter = 500
    scc_options.conv_tol_charge = 1.0e-9
    options = _occupation_options(_TEMPERATURE)
    result = _se.run_scc_dftb_kpoints(
        system,
        parameters,
        kmesh,
        scc_options,
        _CUTOFF,
        options,
    )
    expected_occupations, expected_mu, expected_entropy = (
        fermi_dirac_occupations_per_k(
            result.eps_per_k,
            kmesh.weights,
            2.0,
            _TEMPERATURE,
        )
    )

    assert result.converged
    for actual, expected in zip(
        result.occupations_per_k,
        expected_occupations,
    ):
        np.testing.assert_allclose(actual, expected, atol=1.0e-14)
    assert result.fermi_level == pytest.approx(expected_mu, abs=1.0e-14)
    assert result.entropy == pytest.approx(expected_entropy, abs=1.0e-14)
    assert _weighted_electron_count(kmesh, result.occupations_per_k) == (
        pytest.approx(2.0, abs=1.0e-12)
    )
    assert float(np.asarray(result.charges).sum()) == pytest.approx(0.0, abs=1.0e-12)
    assert result.energy == pytest.approx(
        result.e_electronic + result.e_scc + result.e_repulsive,
        abs=1.0e-14,
    )
    assert result.free_energy == pytest.approx(
        result.energy - _TEMPERATURE * result.entropy,
        abs=1.0e-14,
    )


def test_dftb0_smeared_native_fd_differentiates_free_energy(parameters):
    system = _hli_chain()
    kmesh = monkhorst_pack(system, (3, 1, 1))
    options = _occupation_options(_TEMPERATURE)

    def free_energy(candidate):
        return _se.run_dftb0_kpoints(
            candidate,
            parameters,
            kmesh,
            _CUTOFF,
            options,
        ).free_energy

    expected = finite_difference_gradient(system, free_energy)
    fd_options = _se.KPointDFTBFDBatchOptions()
    fd_options.compute_stress = False
    result = _se.compute_dftb0_kpoints_fd_batch(
        system,
        parameters,
        kmesh,
        _CUTOFF,
        fd_options,
        options,
    )

    np.testing.assert_allclose(result.gradient, expected, atol=1.0e-12)
    assert result.differentiated_free_energy
    assert result.energy_evaluations == 12


def test_scc_dftb_smeared_native_fd_differentiates_free_energy(parameters):
    system = _hli_chain()
    kmesh = monkhorst_pack(system, (3, 1, 1))
    options = _occupation_options(_TEMPERATURE)
    scc_options = _se.SCCOptions()
    scc_options.max_iter = 500
    scc_options.conv_tol_charge = 1.0e-9

    def free_energy(candidate):
        calculation = _se.run_scc_dftb_kpoints(
            candidate,
            parameters,
            kmesh,
            scc_options,
            _CUTOFF,
            options,
        )
        assert calculation.converged
        return calculation.free_energy

    expected = finite_difference_gradient(system, free_energy)
    fd_options = _se.KPointDFTBFDBatchOptions()
    fd_options.compute_stress = False
    result = _se.compute_scc_dftb_kpoints_fd_batch(
        system,
        parameters,
        kmesh,
        scc_options,
        _CUTOFF,
        fd_options,
        options,
    )

    np.testing.assert_allclose(result.gradient, expected, atol=1.0e-12)
    assert result.differentiated_free_energy
    assert result.energy_evaluations == 12


@pytest.mark.parametrize("temperature", [-1.0, np.nan, np.inf])
@pytest.mark.parametrize("method", ["dftb0", "scc_dftb"])
def test_invalid_smearing_temperature_fails_before_kmesh_dispatch(
    parameters,
    temperature,
    method,
):
    options = _occupation_options(temperature)
    empty_kmesh = bloch_kmesh_from_lists([], [])

    with pytest.raises(
        ValueError,
        match="smearing_temperature must be finite and >= 0",
    ):
        if method == "dftb0":
            _se.run_dftb0_kpoints(
                _hli_chain(),
                parameters,
                empty_kmesh,
                occupation_options=options,
            )
        else:
            _se.run_scc_dftb_kpoints(
                _hli_chain(),
                parameters,
                empty_kmesh,
                occupation_options=options,
            )
