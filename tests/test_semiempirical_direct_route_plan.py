from __future__ import annotations

from types import SimpleNamespace

import pytest

from vibeqc import Atom, Molecule
from vibeqc.semiempirical import (
    DFTB0Model,
    SCCDFTBModel,
    UDFTB0Model,
    USCCDFTBModel,
    run_dftb0,
    run_scc_dftb,
)
from vibeqc.semiempirical.methods import gfn2 as gfn2_module
from vibeqc.semiempirical.methods import msindo as msindo_module
from vibeqc.semiempirical.methods import omx as omx_module
from vibeqc.semiempirical.methods import pm6 as pm6_module


def _oh_radical() -> Molecule:
    return Molecule(
        [Atom(8, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.8])],
        multiplicity=2,
    )


@pytest.mark.parametrize(
    ("direct", "message"),
    (
        (run_dftb0, "run_dftb0.* is restricted"),
        (run_scc_dftb, "run_scc_dftb.* is restricted"),
        (DFTB0Model, "DFTB0Model is restricted"),
        (SCCDFTBModel, "SCCDFTBModel is restricted"),
    ),
)
def test_restricted_dftb_wrappers_fail_before_parameter_loading(
    monkeypatch,
    direct,
    message,
):
    def fail_parameters():
        raise AssertionError("unsupported spin reached parameter loader")

    monkeypatch.setattr(
        "vibeqc.semiempirical.dftb0.default_parameters",
        fail_parameters,
    )

    with pytest.raises(NotImplementedError, match=message):
        direct(_oh_radical())


@pytest.mark.parametrize(
    ("model_type", "method_key"),
    ((UDFTB0Model, "dftb0"), (USCCDFTBModel, "scc_dftb")),
)
def test_unrestricted_dftb_models_store_canonical_route_plan(
    model_type,
    method_key,
):
    model = model_type(_oh_radical(), params=object())

    assert model._route_plan.method_key == method_key
    assert model._route_plan.spin == "unrestricted"


def test_gfn2_model_fails_before_parameter_loading(monkeypatch):
    def fail_parameters():
        raise AssertionError("unsupported spin reached parameter loader")

    monkeypatch.setattr(
        "vibeqc.semiempirical.methods.gfn2_params.load_gfn2_params",
        fail_parameters,
    )

    with pytest.raises(NotImplementedError, match="GFN2-xTB route is closed-shell"):
        gfn2_module.GFN2Model(_oh_radical())


def test_pm6_models_validate_planned_spin_before_parameters(monkeypatch):
    def fail_parameters(_elements):
        raise AssertionError("unsupported spin reached parameter loader")

    monkeypatch.setattr(
        "vibeqc.semiempirical.methods.pm6_params.load_pm6_params_auto",
        fail_parameters,
    )

    with pytest.raises(ValueError, match="PM6Model is restricted"):
        pm6_module.PM6Model(_oh_radical())


@pytest.mark.parametrize("unrestricted", [False, True], ids=["pm6", "upm6"])
@pytest.mark.parametrize("failed_energy", [0.0, -2.0], ids=["zero", "finite"])
def test_pm6_models_refuse_unconverged_energy_and_derivatives(
    monkeypatch, unrestricted, failed_energy,
):
    from vibeqc.semiempirical import SemiempiricalEnergyError

    molecule = Molecule(
        [Atom(1, [0., 0., 0.]), Atom(1, [0., 0., 1.4])],
        multiplicity=3 if unrestricted else 1,
    )
    result = SimpleNamespace(
        energy=failed_energy, converged=False, n_iter=7, n_alpha=2, n_beta=0,
        parameter_identity="failed-state", parameter_sha256="retained-identity",
    )
    calls = []

    def failed(mol, params, max_iter, conv_tol):
        calls.append((max_iter, conv_tol))
        return result

    def forbidden_gradient(*args, **kwargs):
        pytest.fail("A rejected state must not launch displaced SCFs")

    kernel = "upm6" if unrestricted else "pm6"
    monkeypatch.setattr(pm6_module._nddo, f"run_{kernel}", failed)
    monkeypatch.setattr(pm6_module._nddo,
                        f"compute_{kernel}_gradient_fd_from_result", forbidden_gradient)
    cls = pm6_module.UPM6Model if unrestricted else pm6_module.PM6Model
    model = cls(molecule, params=SimpleNamespace(method_name=lambda: "pm6"),
                max_iter=7, conv_tol=2e-8)
    with pytest.raises(SemiempiricalEnergyError, match="did not converge after 7"):
        model.energy()
    assert not model.converged and model.n_iter == 7
    assert model._last_result is result
    assert model.parameter_identity == "failed-state"
    assert model.parameter_sha256 == "retained-identity"
    with pytest.raises(SemiempiricalEnergyError, match="did not converge"):
        model.gradient()
    assert calls == [(7, 2e-8)]
    with pytest.raises(SemiempiricalEnergyError, match="did not converge"):
        model._energy_at(molecule)
    assert calls == [(7, 2e-8), (7, 2e-8)]


def test_open_shell_omx_model_uses_planned_unrestricted_kernel(monkeypatch):
    calls = []
    params = SimpleNamespace(method_name=lambda: "om2")

    def fake_run(molecule, loaded_params, max_iter, conv_tol):
        calls.append(
            (molecule.multiplicity, loaded_params, max_iter, conv_tol)
        )
        return SimpleNamespace(energy=-2.0, n_iter=4, converged=True)

    monkeypatch.setattr(omx_module._nddo, "run_uomx_v2", fake_run)
    model = omx_module.OMxModel(_oh_radical(), variant="om2", params=params)

    assert model.energy() == -2.0
    assert model._route_plan.spin == "unrestricted"
    assert calls == [(2, params, 100, 1e-7)]


def test_direct_msindo_plans_before_element_and_parameter_work(monkeypatch):
    seen = []

    def stop_after_plan(method, **kwargs):
        seen.append((method, kwargs))
        raise RuntimeError("planned")

    monkeypatch.setattr(
        "vibeqc.semiempirical.routes.SemiempiricalRoutePlan.from_request",
        stop_after_plan,
    )

    with pytest.raises(RuntimeError, match="planned"):
        msindo_module.run_msindo(
            [1, 1],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
            charge=1,
            multiplicity=2,
        )

    assert seen == [
        (
            "msindo",
            {
                "boundary": "molecule",
                "charge": 1,
                "multiplicity": 2,
                "nddo": False,
            },
        )
    ]
