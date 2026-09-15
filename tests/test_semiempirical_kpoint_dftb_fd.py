"""Native finite-difference gates for full-k DFTB routes."""

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
from vibeqc.semiempirical.periodic import (
    evaluate_periodic_energy_gradient,
    evaluate_periodic_kpoint_energy_gradient_stress,
    finite_difference_gradient,
)


_CHAIN_LENGTH = 4.1
_CUTOFF = 12.0
_STEP = 0.001
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


def _clone_system(
    system: PeriodicSystem,
    transform: np.ndarray,
) -> PeriodicSystem:
    return PeriodicSystem(
        system.dim,
        transform @ np.asarray(system.lattice),
        [
            Atom(atom.Z, transform @ np.asarray(atom.xyz, dtype=float))
            for atom in system.unit_cell
        ],
        system.charge,
        system.multiplicity,
    )


def _dftb0_stress_xx(
    system: PeriodicSystem,
    parameters,
    kmesh,
    *,
    transform_kpoints: bool,
) -> float:
    energies = []
    for sign in (1.0, -1.0):
        transform = np.eye(3)
        transform[0, 0] += sign * _STEP
        strained = _clone_system(system, transform)
        kpoints = list(kmesh.kpoints)
        if transform_kpoints:
            reciprocal_transform = np.linalg.inv(transform).T
            kpoints = [reciprocal_transform @ np.asarray(k) for k in kpoints]
        strained_kmesh = bloch_kmesh_from_lists(kpoints, kmesh.weights)
        energies.append(
            _se.run_dftb0_kpoints(
                strained,
                parameters,
                strained_kmesh,
                _CUTOFF,
            ).energy
        )

    volume = abs(np.linalg.det(np.asarray(system.lattice)))
    return (energies[0] - energies[1]) / (2.0 * _STEP * volume)


@pytest.fixture(scope="module")
def parameters():
    return _se.SemiempiricalParameters.dftb0_default()


def test_dftb0_native_kpoint_batch_matches_independent_fd(parameters):
    system = _hli_chain()
    kmesh = monkhorst_pack(system, (3, 1, 1))

    def energy(candidate):
        return _se.run_dftb0_kpoints(
            candidate,
            parameters,
            kmesh,
            _CUTOFF,
        ).energy

    expected_gradient = finite_difference_gradient(system, energy, h=_STEP)
    expected_stress_xx = _dftb0_stress_xx(
        system,
        parameters,
        kmesh,
        transform_kpoints=True,
    )
    options = _se.KPointDFTBFDBatchOptions()
    options.coordinate_step = _STEP
    options.strain_step = _STEP
    result = _se.compute_dftb0_kpoints_fd_batch(
        system,
        parameters,
        kmesh,
        _CUTOFF,
        options,
    )

    np.testing.assert_allclose(result.gradient, expected_gradient, atol=1.0e-12)
    assert result.stress[0, 0] == pytest.approx(expected_stress_xx, abs=1.0e-12)
    np.testing.assert_array_equal(
        np.asarray(result.stress)[1:, :],
        np.zeros((2, 3)),
    )
    assert result.energy_evaluations == 14
    assert result.system_workspace_copies == 1
    assert result.kmesh_workspace_copies == 1
    assert result.lattice_cell_setups == 14
    assert result.workspace_bytes > 0
    assert not result.memory_counters_complete
    assert not result.differentiated_free_energy


def test_dftb0_kpoint_stress_preserves_fractional_mesh(parameters):
    system = _hli_chain()
    kmesh = monkhorst_pack(system, (3, 1, 1))
    options = _se.KPointDFTBFDBatchOptions()
    options.compute_gradient = False
    result = _se.compute_dftb0_kpoints_fd_batch(
        system,
        parameters,
        kmesh,
        _CUTOFF,
        options,
    )

    expected = _dftb0_stress_xx(
        system,
        parameters,
        kmesh,
        transform_kpoints=True,
    )
    static_cartesian_k = _dftb0_stress_xx(
        system,
        parameters,
        kmesh,
        transform_kpoints=False,
    )

    assert result.stress[0, 0] == pytest.approx(expected, abs=1.0e-12)
    assert abs(result.stress[0, 0] - static_cartesian_k) > 1.0e-7


def test_scc_dftb_native_kpoint_gradient_matches_independent_fd(parameters):
    system = _hli_chain()
    kmesh = monkhorst_pack(system, (3, 1, 1))
    scc_options = _se.SCCOptions()
    scc_options.max_iter = 500
    scc_options.conv_tol_charge = 1.0e-9

    def energy(candidate):
        calculation = _se.run_scc_dftb_kpoints(
            candidate,
            parameters,
            kmesh,
            scc_options,
            _CUTOFF,
        )
        assert calculation.converged
        return calculation.energy

    expected_gradient = finite_difference_gradient(system, energy, h=_STEP)
    fd_options = _se.KPointDFTBFDBatchOptions()
    fd_options.compute_stress = False
    result = _se.compute_scc_dftb_kpoints_fd_batch(
        system,
        parameters,
        kmesh,
        scc_options,
        _CUTOFF,
        fd_options,
    )

    np.testing.assert_allclose(result.gradient, expected_gradient, atol=1.0e-12)
    assert result.energy_evaluations == 12
    assert result.system_workspace_copies == 1
    assert result.kmesh_workspace_copies == 1
    assert result.lattice_cell_setups == 12


def test_public_dftb0_kpoint_gradient_uses_native_batch(parameters):
    system = _hli_chain()
    kmesh = monkhorst_pack(system, (3, 1, 1))
    fd_options = _se.KPointDFTBFDBatchOptions()
    fd_options.coordinate_step = _STEP
    fd_options.compute_stress = False

    energy, gradient = evaluate_periodic_energy_gradient(
        "dftb0",
        system,
        kpoints=(3, 1, 1),
        cutoff_bohr=_CUTOFF,
        fd_step_bohr=_STEP,
    )
    expected_energy = _se.run_dftb0_kpoints(
        system,
        parameters,
        kmesh,
        _CUTOFF,
    ).energy
    expected_gradient = _se.compute_dftb0_kpoints_fd_batch(
        system,
        parameters,
        kmesh,
        _CUTOFF,
        fd_options,
    ).gradient

    assert energy == pytest.approx(expected_energy, abs=1.0e-12)
    np.testing.assert_allclose(gradient, expected_gradient, atol=1.0e-12)


@pytest.mark.parametrize("with_stress", [False, True])
@pytest.mark.parametrize(
    "kpoints",
    [
        (1.2, 1, 1),
        (0, 1, 1),
        (np.inf, 1, 1),
        (1, 1),
        np.ones((3, 1)),
        ("2", 1, 1),
    ],
)
def test_public_kpoint_derivatives_reject_invalid_mesh_before_parameters(
    monkeypatch: pytest.MonkeyPatch,
    kpoints,
    with_stress: bool,
) -> None:
    import vibeqc.semiempirical.parameters as parameter_module

    def fail_parameters():
        raise AssertionError("parameters must not load for an invalid mesh")

    monkeypatch.setattr(parameter_module, "default_parameters", fail_parameters)
    evaluator = (
        evaluate_periodic_kpoint_energy_gradient_stress
        if with_stress
        else evaluate_periodic_energy_gradient
    )
    with pytest.raises(ValueError, match="three finite positive integers"):
        evaluator(
            "dftb0",
            _hli_chain(),
            kpoints=kpoints,
            cutoff_bohr=_CUTOFF,
            fd_step_bohr=_STEP,
        )


def test_public_scc_dftb_kpoint_gradient_uses_native_batch(parameters):
    system = _hli_chain()
    kmesh = monkhorst_pack(system, (3, 1, 1))
    scc_options = _se.SCCOptions()
    fd_options = _se.KPointDFTBFDBatchOptions()
    fd_options.coordinate_step = _STEP
    fd_options.compute_stress = False

    energy, gradient = evaluate_periodic_energy_gradient(
        "scc-dftb",
        system,
        kpoints=kmesh,
        cutoff_bohr=_CUTOFF,
        fd_step_bohr=_STEP,
    )
    expected = _se.run_scc_dftb_kpoints(
        system,
        parameters,
        kmesh,
        scc_options,
        _CUTOFF,
    )
    expected_gradient = _se.compute_scc_dftb_kpoints_fd_batch(
        system,
        parameters,
        kmesh,
        scc_options,
        _CUTOFF,
        fd_options,
    ).gradient

    assert expected.converged
    assert energy == pytest.approx(expected.energy, abs=1.0e-12)
    np.testing.assert_allclose(gradient, expected_gradient, atol=1.0e-12)


def test_public_dftb0_kpoint_derivative_facade_returns_stress(parameters):
    system = _hli_chain()
    kmesh = monkhorst_pack(system, (3, 1, 1))
    fd_options = _se.KPointDFTBFDBatchOptions()
    fd_options.coordinate_step = _STEP
    fd_options.strain_step = _STEP

    result = evaluate_periodic_kpoint_energy_gradient_stress(
        "dftb0",
        system,
        kpoints=(3, 1, 1),
        cutoff_bohr=_CUTOFF,
        fd_step_bohr=_STEP,
        strain_step=_STEP,
    )
    expected_energy = _se.run_dftb0_kpoints(
        system,
        parameters,
        kmesh,
        _CUTOFF,
    ).energy
    expected = _se.compute_dftb0_kpoints_fd_batch(
        system,
        parameters,
        kmesh,
        _CUTOFF,
        fd_options,
    )

    assert result.energy == pytest.approx(expected_energy, abs=1.0e-12)
    np.testing.assert_allclose(result.gradient, expected.gradient, atol=1.0e-12)
    np.testing.assert_allclose(result.stress, expected.stress, atol=1.0e-12)
    assert result.route_plan.status_route == "periodic-dftb0-kpoint-gradient-fd"
    assert result.energy_evaluations == expected.energy_evaluations
    assert result.workspace_bytes == expected.workspace_bytes
    assert not result.differentiated_free_energy


def test_public_dftb0_kpoint_facade_can_differentiate_free_energy(parameters):
    system = _hli_chain()
    kmesh = monkhorst_pack(system, (3, 1, 1))
    occupation_options = _se.KPointOccupationOptions()
    occupation_options.smearing_temperature = _TEMPERATURE
    fd_options = _se.KPointDFTBFDBatchOptions()
    fd_options.coordinate_step = _STEP
    fd_options.strain_step = _STEP

    result = evaluate_periodic_kpoint_energy_gradient_stress(
        "dftb0",
        system,
        kpoints=(3, 1, 1),
        cutoff_bohr=_CUTOFF,
        fd_step_bohr=_STEP,
        strain_step=_STEP,
        smearing_temperature=_TEMPERATURE,
    )
    expected_energy = _se.run_dftb0_kpoints(
        system,
        parameters,
        kmesh,
        _CUTOFF,
        occupation_options,
    )
    expected = _se.compute_dftb0_kpoints_fd_batch(
        system,
        parameters,
        kmesh,
        _CUTOFF,
        fd_options,
        occupation_options,
    )

    assert result.energy == pytest.approx(expected_energy.energy, abs=1.0e-12)
    assert result.free_energy == pytest.approx(
        expected_energy.free_energy,
        abs=1.0e-12,
    )
    assert result.differentiated_potential == "free_energy"
    assert result.differentiated_free_energy
    np.testing.assert_allclose(result.gradient, expected.gradient, atol=1.0e-12)
    np.testing.assert_allclose(result.stress, expected.stress, atol=1.0e-12)


def test_public_dftb0_kpoint_gradient_returns_differentiated_potential(parameters):
    system = _hli_chain()
    kmesh = monkhorst_pack(system, (3, 1, 1))
    occupation_options = _se.KPointOccupationOptions()
    occupation_options.smearing_temperature = _TEMPERATURE
    fd_options = _se.KPointDFTBFDBatchOptions()
    fd_options.coordinate_step = _STEP
    fd_options.compute_stress = False

    potential, gradient = evaluate_periodic_energy_gradient(
        "dftb0",
        system,
        kpoints=(3, 1, 1),
        cutoff_bohr=_CUTOFF,
        fd_step_bohr=_STEP,
        smearing_temperature=_TEMPERATURE,
    )
    expected_energy = _se.run_dftb0_kpoints(
        system,
        parameters,
        kmesh,
        _CUTOFF,
        occupation_options,
    )
    expected = _se.compute_dftb0_kpoints_fd_batch(
        system,
        parameters,
        kmesh,
        _CUTOFF,
        fd_options,
        occupation_options,
    )

    assert potential == pytest.approx(expected_energy.free_energy, abs=1.0e-12)
    np.testing.assert_allclose(gradient, expected.gradient, atol=1.0e-12)


def test_public_kpoint_derivatives_reject_invalid_smearing_before_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vibeqc.semiempirical.parameters as parameter_module

    def fail_parameters():
        raise AssertionError("parameters must not load for invalid smearing")

    monkeypatch.setattr(parameter_module, "default_parameters", fail_parameters)
    with pytest.raises(ValueError, match="smearing_temperature must be finite"):
        evaluate_periodic_kpoint_energy_gradient_stress(
            "dftb0",
            _hli_chain(),
            kpoints=(3, 1, 1),
            cutoff_bohr=_CUTOFF,
            fd_step_bohr=_STEP,
            strain_step=_STEP,
            smearing_temperature=np.nan,
        )


def test_public_scc_dftb_kpoint_derivative_facade_returns_stress(parameters):
    system = _hli_chain()
    kmesh = monkhorst_pack(system, (3, 1, 1))
    scc_options = _se.SCCOptions()
    fd_options = _se.KPointDFTBFDBatchOptions()
    fd_options.coordinate_step = _STEP
    fd_options.strain_step = _STEP

    result = evaluate_periodic_kpoint_energy_gradient_stress(
        "scc-dftb",
        system,
        kpoints=kmesh,
        cutoff_bohr=_CUTOFF,
        fd_step_bohr=_STEP,
        strain_step=_STEP,
    )
    expected_energy = _se.run_scc_dftb_kpoints(
        system,
        parameters,
        kmesh,
        scc_options,
        _CUTOFF,
    ).energy
    expected = _se.compute_scc_dftb_kpoints_fd_batch(
        system,
        parameters,
        kmesh,
        scc_options,
        _CUTOFF,
        fd_options,
    )

    assert result.energy == pytest.approx(expected_energy, abs=1.0e-12)
    np.testing.assert_allclose(result.gradient, expected.gradient, atol=1.0e-12)
    np.testing.assert_allclose(result.stress, expected.stress, atol=1.0e-12)
    assert result.route_plan.status_route == "periodic-scc-dftb-kpoint-gradient-fd"
    assert result.energy_evaluations == expected.energy_evaluations


def test_scc_dftb_kpoint_batch_fails_on_displaced_nonconvergence(parameters):
    system = _hli_chain()
    kmesh = monkhorst_pack(system, (3, 1, 1))
    scc_options = _se.SCCOptions()
    scc_options.max_iter = 1
    fd_options = _se.KPointDFTBFDBatchOptions()
    fd_options.compute_stress = False

    with pytest.raises(RuntimeError, match="displaced SCC did not converge"):
        _se.compute_scc_dftb_kpoints_fd_batch(
            system,
            parameters,
            kmesh,
            scc_options,
            _CUTOFF,
            fd_options,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda options: (
                setattr(options, "compute_gradient", False),
                setattr(options, "compute_stress", False),
            ),
            "must request gradient or stress",
        ),
        (
            lambda options: (
                setattr(options, "coordinate_step", 0.0),
                setattr(options, "compute_stress", False),
            ),
            "coordinate_step must be finite and > 0",
        ),
    ],
)
def test_kpoint_fd_options_fail_before_kmesh_dispatch(parameters, mutate, message):
    options = _se.KPointDFTBFDBatchOptions()
    mutate(options)
    empty_kmesh = bloch_kmesh_from_lists([], [])

    with pytest.raises(ValueError, match=message):
        _se.compute_dftb0_kpoints_fd_batch(
            _hli_chain(),
            parameters,
            empty_kmesh,
            _CUTOFF,
            options,
        )
