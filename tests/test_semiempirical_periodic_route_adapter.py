from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem


def _hydrogen_radical_cell() -> PeriodicSystem:
    return PeriodicSystem(
        1,
        np.diag([4.0, 20.0, 20.0]),
        [Atom(1, [0.0, 0.0, 0.0])],
        0,
        2,
    )


def _closed_shell_h2_cell() -> PeriodicSystem:
    return PeriodicSystem(
        1,
        np.diag([4.0, 20.0, 20.0]),
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [1.4, 0.0, 0.0])],
        0,
        1,
    )


@pytest.mark.parametrize(
    ("method", "backend_name"),
    [
        ("dftb0", "run_udftb0_gamma"),
        ("scc-dftb", "run_uscc_dftb_gamma"),
    ],
)
def test_periodic_energy_closure_selects_unrestricted_dftb_from_plan(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    backend_name: str,
):
    from vibeqc._vibeqc_core import semiempirical as native_se
    from vibeqc.semiempirical import parameters as parameter_module
    from vibeqc.semiempirical.periodic import make_periodic_energy_function

    params = object()
    calls = []

    def fake_backend(system, supplied_params, *args):
        calls.append((system.multiplicity, supplied_params, len(args)))
        return SimpleNamespace(energy=-0.5)

    monkeypatch.setattr(parameter_module, "default_parameters", lambda: params)
    monkeypatch.setattr(native_se, backend_name, fake_backend)

    system = _hydrogen_radical_cell()
    energy_fn = make_periodic_energy_function(method, system)

    assert energy_fn(system) == pytest.approx(-0.5)
    assert calls == [(2, params, int(method != "dftb0"))]


def test_periodic_pm6_model_rejects_open_shell_before_parameters(
    monkeypatch: pytest.MonkeyPatch,
):
    import vibeqc.semiempirical.methods.periodic_pm6 as pm6_module

    def fail_params(*_args, **_kwargs):
        raise AssertionError("PM6 parameters must follow route validation")

    monkeypatch.setattr(pm6_module, "_get_params", fail_params)

    with pytest.raises(NotImplementedError, match="no unrestricted periodic NDDO"):
        pm6_module.PeriodicPM6Model(_hydrogen_radical_cell())


def test_periodic_omx_model_rejects_open_shell_before_parameters(
    monkeypatch: pytest.MonkeyPatch,
):
    import vibeqc.semiempirical.methods.periodic_omx as omx_module

    def fail_params(*_args, **_kwargs):
        raise AssertionError("OMx parameters must follow route validation")

    monkeypatch.setattr(omx_module, "_get_omx_params", fail_params)

    with pytest.raises(NotImplementedError, match="no Bloch-periodic Gamma"):
        omx_module.PeriodicOMxModel(_hydrogen_radical_cell(), variant="om2")


def test_periodic_energy_closure_rejects_gated_omx_route():
    from vibeqc.semiempirical.periodic import make_periodic_energy_function

    with pytest.raises(NotImplementedError, match="no Bloch-periodic Gamma"):
        make_periodic_energy_function("om2", _closed_shell_h2_cell())


def test_periodic_pm6_cell_optimizer_rejects_open_shell_before_ase_setup():
    from vibeqc.semiempirical.periodic import optimize_pm6_cell

    with pytest.raises(NotImplementedError, match="no unrestricted periodic NDDO"):
        optimize_pm6_cell(_hydrogen_radical_cell())


def test_periodic_omx_cell_optimizer_is_gated_before_ase_setup():
    from vibeqc.semiempirical import PeriodicOMxModel, optimize_omx_cell

    with pytest.raises(NotImplementedError, match="no Bloch-periodic Gamma"):
        optimize_omx_cell(_closed_shell_h2_cell(), "om2")
    with pytest.raises(NotImplementedError, match="no Bloch-periodic Gamma"):
        PeriodicOMxModel(_closed_shell_h2_cell(), "om2")
