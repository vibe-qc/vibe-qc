"""Regression tests for immutable DFTB parameter-content identity."""

from __future__ import annotations

import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc import Atom, Molecule
from vibeqc._vibeqc_core import PeriodicSystem
from vibeqc._vibeqc_core import semiempirical as _se
from vibeqc.semiempirical.dftb0 import (
    DFTB0Model,
    SCCDFTBModel,
    UDFTB0Model,
    USCCDFTBModel,
)
from vibeqc.semiempirical.periodic import (
    evaluate_periodic_energy_gradient,
    evaluate_periodic_kpoint_energy_gradient_stress,
)


DFTB_SCREENING_SHA256 = (
    "9b043a8e7a85244f5ffafc3b9f4965015db3b5e26346e0d4cb22702a8887fb40"
)
DFTB_SCREENING_IDENTITY = "vibeqc-inhouse-dftb-screening-v1"


def _assert_custom(params: _se.SemiempiricalParameters) -> str:
    digest = params.content_sha256()
    assert digest != DFTB_SCREENING_SHA256
    assert params.parameter_identity() == f"custom:{digest}"
    return digest


def _h2() -> Molecule:
    return Molecule(
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 2.0])],
        charge=0,
        multiplicity=1,
    )


def _hydrogen_atom() -> Molecule:
    return Molecule(
        [Atom(1, [0.0, 0.0, 0.0])],
        charge=0,
        multiplicity=2,
    )


def _periodic_h2() -> PeriodicSystem:
    return PeriodicSystem(
        1,
        np.diag([4.0, 20.0, 20.0]),
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [1.4, 0.0, 0.0])],
        0,
        1,
    )


def test_dftb_builtins_have_exact_content_identity() -> None:
    default = _se.SemiempiricalParameters.dftb0_default()
    production = _se.SemiempiricalParameters.dftb0_production()

    for params in (default, production):
        assert params.content_sha256() == DFTB_SCREENING_SHA256
        assert params.parameter_identity() == DFTB_SCREENING_IDENTITY


def test_dftb_kappa_mutation_is_custom_and_restore_is_published() -> None:
    params = _se.SemiempiricalParameters.dftb0_default()
    original_kappa = params.kappa

    params.kappa = original_kappa + 0.125
    _assert_custom(params)

    params.kappa = original_kappa
    assert params.content_sha256() == DFTB_SCREENING_SHA256
    assert params.parameter_identity() == DFTB_SCREENING_IDENTITY


@pytest.mark.parametrize(
    ("on_site", "zeta", "hubbard_u", "valence_electrons"),
    [
        ([-0.2067], [1.24], 0.4195, 1),
        ([-0.2066], [1.25], 0.4195, 1),
        ([-0.2066], [1.24], 0.4205, 1),
        ([-0.2066], [1.24], 0.4195, 2),
        ([-0.2066, 0.0], [1.24], 0.4195, 1),
        ([-0.2066], [1.24, 0.0], 0.4195, 1),
    ],
    ids=[
        "on-site-value",
        "zeta-value",
        "hubbard-value",
        "valence-value",
        "on-site-length",
        "zeta-length",
    ],
)
def test_dftb_element_record_mutations_are_custom(
    on_site: list[float],
    zeta: list[float],
    hubbard_u: float,
    valence_electrons: int,
) -> None:
    params = _se.SemiempiricalParameters.dftb0_default()
    params.add_element(1, on_site, zeta, hubbard_u, valence_electrons)
    _assert_custom(params)


def test_dftb_analytic_and_raw_spline_pair_mutations_change_hash() -> None:
    analytic_a = _se.SemiempiricalParameters.dftb0_default()
    analytic_a.set_repulsive_pair_analytic(1, 1, 5.125, 0.0)
    analytic_b = _se.SemiempiricalParameters.dftb0_default()
    analytic_b.set_repulsive_pair_analytic(1, 1, 5.0, 0.125)

    analytic_hashes = {_assert_custom(analytic_a), _assert_custom(analytic_b)}
    assert len(analytic_hashes) == 2

    spline = _se.SemiempiricalParameters.dftb0_default()
    spline.set_repulsive_pair_spline(1, 1, [1.0, 2.0, 3.0], [2.0, 0.5, 0.0])
    spline_r = _se.SemiempiricalParameters.dftb0_default()
    spline_r.set_repulsive_pair_spline(
        1, 1, [1.0, 2.125, 3.0], [2.0, 0.5, 0.0]
    )
    spline_v = _se.SemiempiricalParameters.dftb0_default()
    spline_v.set_repulsive_pair_spline(
        1, 1, [1.0, 2.0, 3.0], [2.0, 0.625, 0.0]
    )

    spline_hashes = {
        _assert_custom(spline),
        _assert_custom(spline_r),
        _assert_custom(spline_v),
    }
    assert len(spline_hashes) == 3


def _ordered_custom_parameters(*, reverse: bool) -> _se.SemiempiricalParameters:
    params = _se.SemiempiricalParameters()
    params.kappa = 2.125
    hydrogen = (1, [-0.2], [1.2], 0.4, 1)
    carbon = (6, [-0.5, -0.2], [1.6, 1.7], 0.35, 4)

    if reverse:
        params.add_element(*carbon)
        params.add_element(*hydrogen)
        params.set_repulsive_pair_spline(
            6, 1, [1.0, 2.0, 3.0], [2.0, 0.5, 0.0]
        )
        params.set_repulsive_pair_analytic(1, 1, 7.0, 0.5)
    else:
        params.add_element(*hydrogen)
        params.add_element(*carbon)
        params.set_repulsive_pair_analytic(1, 1, 7.0, 0.5)
        params.set_repulsive_pair_spline(
            1, 6, [1.0, 2.0, 3.0], [2.0, 0.5, 0.0]
        )
    return params


def test_dftb_element_pair_and_coordinate_order_are_canonical() -> None:
    forward = _ordered_custom_parameters(reverse=False)
    reverse = _ordered_custom_parameters(reverse=True)

    assert forward.content_sha256() == reverse.content_sha256()
    assert forward.parameter_identity() == reverse.parameter_identity()
    assert forward.parameter_identity() == f"custom:{forward.content_sha256()}"


def test_dftb_result_binds_snapshot_and_gradient_rejects_mismatch() -> None:
    params = _se.SemiempiricalParameters.dftb0_default()
    molecule = _h2()
    result = _se.run_dftb0(molecule, params)

    assert result.parameter_sha256 == DFTB_SCREENING_SHA256
    assert result.parameter_identity == DFTB_SCREENING_IDENTITY
    assert not hasattr(result, "parameter_fingerprint")
    gradient = np.asarray(_se.compute_dftb0_gradient(molecule, result, params))
    assert gradient.shape == (2, 3)
    assert np.all(np.isfinite(gradient))

    params.kappa += 0.125
    _assert_custom(params)
    with pytest.raises(ValueError, match="same immutable parameter snapshot"):
        _se.compute_dftb0_gradient(molecule, result, params)


@pytest.mark.parametrize(
    ("model_type", "molecule_factory"),
    [
        (DFTB0Model, _h2),
        (SCCDFTBModel, _h2),
        (UDFTB0Model, _hydrogen_atom),
        (USCCDFTBModel, _hydrogen_atom),
    ],
    ids=["dftb0", "scc-dftb", "udftb0", "uscc-dftb"],
)
def test_direct_dftb_models_project_last_native_parameter_identity(
    model_type,
    molecule_factory,
) -> None:
    params = _se.SemiempiricalParameters.dftb0_default()
    model = model_type(molecule_factory(), params=params)

    assert model.parameter_identity is None
    assert model.parameter_sha256 is None
    assert np.isfinite(model.energy())
    assert model.parameter_identity == DFTB_SCREENING_IDENTITY
    assert model.parameter_sha256 == DFTB_SCREENING_SHA256
    assert model.parameter_identity == str(model._last_result.parameter_identity)
    assert model.parameter_sha256 == str(model._last_result.parameter_sha256)

    params.kappa += 0.125
    custom_sha256 = _assert_custom(params)
    assert model.parameter_identity == DFTB_SCREENING_IDENTITY
    assert model.parameter_sha256 == DFTB_SCREENING_SHA256

    assert np.isfinite(model.energy())
    assert model.parameter_identity == f"custom:{custom_sha256}"
    assert model.parameter_sha256 == custom_sha256


def test_periodic_dftb_result_rejects_parameter_snapshot_mismatch() -> None:
    params = _se.SemiempiricalParameters.dftb0_default()
    options = _se.PeriodicDFTB0Options()
    options.gamma_only_0 = True
    system = _periodic_h2()
    result = _se.run_dftb0_gamma(system, params, options)

    params.kappa += 0.125
    with pytest.raises(ValueError, match="same immutable parameter snapshot"):
        _se.compute_periodic_dftb0_gradient(system, result, params)


def test_periodic_dftb_legacy_cutoff_error_precedes_snapshot_error() -> None:
    params = _se.SemiempiricalParameters.dftb0_default()
    system = _periodic_h2()

    derivative_calls = (
        (
            _se.compute_periodic_dftb0_gradient,
            _se.PeriodicDFTB0Result(),
        ),
        (
            _se.compute_periodic_dftb0_stress,
            _se.PeriodicDFTB0Result(),
        ),
        (
            _se.compute_periodic_udftb0_gradient,
            _se.PeriodicUDFTB0Result(),
        ),
    )
    for derivative, result in derivative_calls:
        with pytest.raises(ValueError, match="lattice-cutoff provenance"):
            derivative(system, result, params)


@pytest.mark.parametrize("method", ["dftb0", "scc_dftb"])
def test_gamma_finite_differences_reject_mixed_parameter_snapshots(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    """The displaced re-runs of an FD route must come from one snapshot.

    Only routes that re-run the SCC at displaced geometries can mix
    snapshots.  Gamma GFN2-xTB left this set on 2026-09-06 (issue #338): it
    runs the SCC once and differentiates that state analytically, so there is
    no second snapshot to disagree with the first.  That is asserted by
    test_gamma_gfn2_takes_one_snapshot_and_differentiates_it below.
    """
    from vibeqc.semiempirical import parameters as dftb_parameters

    calls = 0

    def fake_run(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            energy=-1.0,
            converged=True,
            parameter_sha256=("a" if calls == 1 else "b") * 64,
            smearing_temperature=0.0,
        )

    params = object()
    monkeypatch.setattr(dftb_parameters, "default_parameters", lambda: params)
    if method == "dftb0":
        monkeypatch.setattr(_se, "run_dftb0_gamma", fake_run)
    else:
        monkeypatch.setattr(_se, "run_scc_dftb_gamma", fake_run)

    with pytest.raises(RuntimeError, match="refusing to mix parameter snapshots"):
        evaluate_periodic_energy_gradient(
            method,
            _periodic_h2(),
            cutoff_bohr=14.0,
        )


def test_gamma_gfn2_takes_one_snapshot_and_differentiates_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #338: one SCC call, one parameter snapshot, analytic derivative.

    The snapshot-mixing failure mode is structurally absent from this route
    rather than guarded against: the gradient is taken from the state the
    single SCC call converged, with the same parameter object.
    """
    from vibeqc.semiempirical.methods import gfn2_params
    from vibeqc._vibeqc_core.semiempirical import xtb as _xtb

    system = _periodic_h2()
    params = object()
    sentinel = SimpleNamespace(
        energy=-1.0,
        converged=True,
        parameter_sha256="a" * 64,
        smearing_temperature=0.0,
    )
    scc_calls: list[tuple] = []
    gradient_calls: list[tuple] = []

    def fake_run(*args, **_kwargs):
        scc_calls.append(args)
        return sentinel

    def fake_gradient(*args):
        gradient_calls.append(args)
        return np.zeros((len(system.unit_cell), 3))

    monkeypatch.setattr(gfn2_params, "load_gfn2_params", lambda: params)
    monkeypatch.setattr(_xtb, "run_gfn2_xtb_gamma", fake_run)
    monkeypatch.setattr(_se, "compute_periodic_gfn2_gradient", fake_gradient)

    energy, gradient = evaluate_periodic_energy_gradient(
        "gfn2_xtb",
        system,
        cutoff_bohr=14.0,
    )

    assert energy == -1.0
    assert gradient.shape == (len(system.unit_cell), 3)
    assert len(scc_calls) == 1, "the analytic route must not re-run the SCC"
    assert len(gradient_calls) == 1
    # The derivative is handed the state that SCC call produced, and the very
    # same parameter object it was produced from.
    assert gradient_calls[0][1] is sentinel
    assert gradient_calls[0][2] is params
    assert scc_calls[0][1] is params


def test_dftb_repulsive_gradient_uses_snapshot_while_gil_is_released() -> None:
    params = _se.SemiempiricalParameters.dftb0_default()
    atoms = [Atom(1, [0.75 * index, 0.0, 0.0]) for index in range(4000)]
    molecule = Molecule(atoms, charge=0, multiplicity=1)
    expected = np.asarray(_se.dftb0_repulsive_gradient(molecule, params))

    started = threading.Event()
    finished = threading.Event()
    observed: list[np.ndarray] = []
    errors: list[BaseException] = []

    def calculate() -> None:
        try:
            started.set()
            observed.append(
                np.asarray(_se.dftb0_repulsive_gradient(molecule, params))
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)
        finally:
            finished.set()

    previous_switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(1.0)
    worker = threading.Thread(target=calculate)
    try:
        worker.start()
        assert started.wait(timeout=5.0)
        params.set_repulsive_pair_analytic(1, 1, 9.0, 0.25)
        mutated_during_native_execution = not finished.is_set()
        worker.join(timeout=30.0)
    finally:
        sys.setswitchinterval(previous_switch_interval)

    assert not worker.is_alive()
    assert not errors
    assert mutated_during_native_execution
    np.testing.assert_array_equal(observed[0], expected)


def _install_fake_kpoint_dftb_results(
    monkeypatch: pytest.MonkeyPatch,
    *,
    derivative_identity: str = DFTB_SCREENING_IDENTITY,
    derivative_sha256: str = DFTB_SCREENING_SHA256,
) -> None:
    energy = SimpleNamespace(
        energy=-0.75,
        free_energy=-0.76,
        parameter_identity=DFTB_SCREENING_IDENTITY,
        parameter_sha256=DFTB_SCREENING_SHA256,
    )
    derivatives = SimpleNamespace(
        gradient=np.zeros((2, 3)),
        stress=np.eye(3),
        energy_evaluations=14,
        workspace_bytes=512,
        differentiated_free_energy=False,
        memory_counters_complete=True,
        parameter_identity=derivative_identity,
        parameter_sha256=derivative_sha256,
    )
    monkeypatch.setattr(_se, "run_dftb0_kpoints", lambda *args: energy)
    monkeypatch.setattr(
        _se,
        "compute_dftb0_kpoints_fd_batch",
        lambda *args: derivatives,
    )


def test_periodic_dftb_derivative_result_propagates_exact_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_kpoint_dftb_results(monkeypatch)

    result = evaluate_periodic_kpoint_energy_gradient_stress(
        "dftb0",
        _periodic_h2(),
        kpoints=(1, 1, 1),
    )

    assert result.parameter_identity == DFTB_SCREENING_IDENTITY
    assert result.parameter_sha256 == DFTB_SCREENING_SHA256


@pytest.mark.parametrize(
    ("derivative_identity", "derivative_sha256"),
    [
        (f"custom:{'1' * 64}", DFTB_SCREENING_SHA256),
        (DFTB_SCREENING_IDENTITY, "1" * 64),
    ],
    ids=["identity", "sha256"],
)
def test_periodic_dftb_derivative_result_rejects_provenance_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    derivative_identity: str,
    derivative_sha256: str,
) -> None:
    _install_fake_kpoint_dftb_results(
        monkeypatch,
        derivative_identity=derivative_identity,
        derivative_sha256=derivative_sha256,
    )

    with pytest.raises(RuntimeError, match="different immutable parameter snapshots"):
        evaluate_periodic_kpoint_energy_gradient_stress(
            "dftb0",
            _periodic_h2(),
            kpoints=(1, 1, 1),
        )
