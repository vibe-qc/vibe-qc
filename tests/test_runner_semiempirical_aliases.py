from __future__ import annotations

import json
import tomllib

import numpy as np
import pytest
import vibeqc as vq
import vibeqc.semiempirical.runner as semi_runner
import vibeqc.semiempirical.routes as semi_routes
from vibeqc import runner
from vibeqc.semiempirical import run_ccm as public_run_ccm
from vibeqc.semiempirical import run_seccm as public_run_seccm
from vibeqc.semiempirical import run_semiempirical as public_run_semiempirical
from vibeqc.semiempirical.runner import (
    MOLECULAR_SEMIEMPIRICAL_METHODS,
    SEMIEMPIRICAL_METHODS,
    SemiempiricalResult,
    normalise_semiempirical_method,
    run_ccm,
    run_seccm,
    run_semiempirical,
)


def _water() -> vq.Molecule:
    return vq.Molecule(
        [
            vq.Atom(8, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.0, 0.0, 1.8]),
            vq.Atom(1, [0.0, 1.8, 0.0]),
        ]
    )


def test_run_job_private_aliases_point_to_unified_runner():
    assert runner._SEMPR is SemiempiricalResult
    assert runner._run_ccm is run_ccm
    assert runner._run_semiempirical is run_semiempirical
    assert public_run_ccm is run_ccm
    assert public_run_seccm is run_seccm
    assert run_ccm is run_seccm
    assert public_run_semiempirical is run_semiempirical
    assert semi_runner.SEMIEMPIRICAL_METHOD_ALIASES is (
        semi_routes.SEMIEMPIRICAL_METHOD_ALIASES
    )
    assert semi_runner.normalise_semiempirical_method is (
        semi_routes.normalise_semiempirical_method
    )
    assert "ccm" in SEMIEMPIRICAL_METHODS
    assert "ccm" not in MOLECULAR_SEMIEMPIRICAL_METHODS
    assert {
        "dftb0",
        "scc_dftb",
        "gfn2_xtb",
        "pm6",
        "om1",
        "om2",
        "om3",
        "msindo",
    }.issubset(MOLECULAR_SEMIEMPIRICAL_METHODS)


def test_semiempirical_result_gradient_is_lazy_and_cached():
    calls = []

    def gradient():
        calls.append("gradient")
        return [[1.0, 2.0, 3.0]]

    result = SemiempiricalResult(-1.0, gradient, "toy")
    assert calls == []
    first = result.gradient()
    second = result.gradient()
    assert first == [[1.0, 2.0, 3.0]]
    assert second is first
    assert calls == ["gradient"]


def test_run_semiempirical_gfn2_gradient_is_lazy(monkeypatch):
    from vibeqc.semiempirical.methods import gfn2 as gfn2_module
    from vibeqc.semiempirical.methods import gfn2_params

    calls = []
    params = object()

    class FakeGFN2Model:
        converged = True
        n_iter = 4
        _last_result = type("NativeResult", (), {"charges": (0.0,)})()

        def __init__(self, molecule, loaded_params):
            calls.append(("init", molecule, loaded_params))

        def energy(self):
            calls.append(("energy",))
            return -3.0

        def gradient(self):
            calls.append(("gradient",))
            return np.array([[0.1, 0.2, 0.3]])

    monkeypatch.setattr(gfn2_params, "load_gfn2_params", lambda: params)
    monkeypatch.setattr(gfn2_module, "GFN2Model", FakeGFN2Model)

    mol = vq.Molecule([vq.Atom(2, [0.0, 0.0, 0.0])])
    result = run_semiempirical("gfn2", mol)

    assert result.energy == pytest.approx(-3.0)
    assert result.converged is True
    assert result.n_iter == 4
    assert result.mulliken_charges == (0.0,)
    assert [call[0] for call in calls] == ["init", "energy"]

    np.testing.assert_allclose(result.gradient(), [[0.1, 0.2, 0.3]])
    assert [call[0] for call in calls] == ["init", "energy", "gradient"]
    np.testing.assert_allclose(result.gradient(), [[0.1, 0.2, 0.3]])
    assert [call[0] for call in calls] == ["init", "energy", "gradient"]
    # The provenance path must not assume a .params attribute on
    # model-like stand-ins (GFN2-GRADIENT-LAZY): no provenance is recorded
    # rather than a crash.
    assert result.parameter_provenance is None


def test_gfn2_parameter_provenance_degrades_without_params():
    """GFN2-GRADIENT-LAZY regression: a model-like object without a
    params attribute yields no provenance instead of an AttributeError."""
    from vibeqc.semiempirical.runner import _gfn2_parameter_provenance

    class BareModel:
        pass

    assert _gfn2_parameter_provenance(BareModel()) is None


def test_hyphenated_semiempirical_aliases_skip_basis_requirement(
    monkeypatch,
    tmp_path,
):
    seen = []

    def fake_run_semiempirical(method, molecule, *, nddo=False):
        seen.append((method, molecule.n_electrons(), nddo))
        return runner._SEMPR(-1.0, None, method, converged=True, n_iter=1)

    monkeypatch.setattr(runner, "_run_semiempirical", fake_run_semiempirical)

    for method in ("gfn2", "gfn2-xtb", "scc-dftb"):
        result = vq.run_job(
            _water(),
            method=method,
            output=tmp_path / method,
            write_molden_file=False,
            write_xyz_file=False,
            write_population_file=False,
            citations=False,
            progress=False,
        )
        assert result.energy == -1.0

    assert [item[0] for item in seen] == ["gfn2_xtb", "gfn2_xtb", "scc_dftb"]


def test_run_job_semiempirical_nonconverged_marks_crashed(
    monkeypatch,
    tmp_path,
):
    def fake_run_semiempirical(method, molecule, *, nddo=False):
        assert method == "gfn2_xtb"
        assert nddo is False
        assert molecule.n_electrons() == 10
        return runner._SEMPR(-1.0, None, method, converged=False, n_iter=41)

    monkeypatch.setattr(runner, "_run_semiempirical", fake_run_semiempirical)
    stem = tmp_path / "gfn2_nonconverged"

    with pytest.raises(
        RuntimeError,
        match="GFN2_XTB semiempirical SCF did not converge",
    ):
        vq.run_job(
            _water(),
            method="gfn2",
            output=stem,
            write_molden_file=False,
            write_xyz_file=False,
            write_population_file=False,
            citations=False,
            progress=False,
        )

    manifest = tomllib.loads(stem.with_suffix(".system").read_text("utf-8"))
    assert manifest["outputs"]["status"] == "crashed"
    text = stem.with_suffix(".out").read_text("utf-8")
    assert "FATAL: GFN2_XTB semiempirical SCF did not converge" in text


def test_scc_dftb_default_plan_declares_only_native_population_sidecars(
    tmp_path,
) -> None:
    stem = tmp_path / "scc_auto_sidecars"

    result = vq.run_job(
        _water(),
        method="scc-dftb",
        output=stem,
        dry_run=True,
        citations=False,
    )

    assert result is None
    manifest = tomllib.loads(stem.with_suffix(".system").read_text("utf-8"))
    assert manifest["outputs"]["status"] == "dry_run"
    roles = [row["role"] for row in manifest["plan"]["files"]]
    assert "orbitals" not in roles
    assert roles.count("population") == 2


@pytest.mark.parametrize("method", ["pm6", "msindo"])
def test_semiempirical_without_native_populations_omits_sidecars(
    tmp_path,
    method,
) -> None:
    stem = tmp_path / f"{method}_auto_sidecars"

    result = vq.run_job(
        _water(),
        method=method,
        output=stem,
        dry_run=True,
        citations=False,
    )

    assert result is None
    manifest = tomllib.loads(stem.with_suffix(".system").read_text("utf-8"))
    roles = [row["role"] for row in manifest["plan"]["files"]]
    assert "orbitals" not in roles
    assert "population" not in roles


def test_dftb0_default_plan_declares_only_native_population_sidecars(
    tmp_path,
) -> None:
    stem = tmp_path / "dftb0_auto_sidecars"

    result = vq.run_job(
        _water(),
        method="dftb0",
        output=stem,
        dry_run=True,
        citations=False,
    )

    assert result is None
    manifest = tomllib.loads(stem.with_suffix(".system").read_text("utf-8"))
    roles = [row["role"] for row in manifest["plan"]["files"]]
    assert "orbitals" not in roles
    assert roles.count("population") == 2


def test_scc_dftb_explicit_molden_request_fails_before_engine(
    tmp_path,
    monkeypatch,
) -> None:
    def fail_engine(*_args, **_kwargs):
        raise AssertionError("semiempirical engine should not run")

    monkeypatch.setattr(runner, "_run_semiempirical", fail_engine)
    stem = tmp_path / "scc_explicit_molden"

    with pytest.raises(NotImplementedError, match=r"write_molden_file=True"):
        vq.run_job(
            _water(),
            method="scc-dftb",
            output=stem,
            citations=False,
            write_molden_file=True,
        )

    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("method", ["pm6", "om2", "om3", "msindo"])
def test_basis_free_explicit_population_request_fails_before_engine(
    tmp_path,
    monkeypatch,
    method,
) -> None:
    def fail_engine(*_args, **_kwargs):
        raise AssertionError("semiempirical engine should not run")

    monkeypatch.setattr(runner, "_run_semiempirical", fail_engine)
    stem = tmp_path / f"{method}_explicit_population"

    with pytest.raises(NotImplementedError, match=r"write_population_file=True"):
        vq.run_job(
            _water(),
            method=method,
            output=stem,
            citations=False,
            write_population_file=True,
        )

    assert not stem.with_suffix(".system").exists()


def test_dftb0_writes_method_native_mulliken_population(tmp_path) -> None:
    stem = tmp_path / "dftb0_native_population"

    result = vq.run_job(
        _water(),
        method="dftb0",
        output=stem,
        write_xyz_file=False,
        citations=True,
        output_qvf=False,
        crash_dump=False,
        progress=False,
        write_population_file=True,
    )

    payload = json.loads(stem.with_suffix(".population.json").read_text("utf-8"))
    charges = np.asarray([row["charge"] for row in payload["mulliken"]])
    np.testing.assert_allclose(charges, result.mulliken_charges, atol=1.0e-12)
    assert charges.shape == (3,)
    assert np.all(np.isfinite(charges))
    assert charges.sum() == pytest.approx(0.0, abs=1.0e-12)
    assert set(payload["errors"]) == {
        "loewdin",
        "hirshfeld",
        "mayer",
        "wiberg",
        "dipole",
    }
    assert all(
        message.startswith("unsupported: dftb0 native population")
        for message in payload["errors"].values()
    )
    # GitLab #197: NPA is unimplemented everywhere, not a failure of this
    # method's native population path, so it is reported once under
    # ``unavailable`` rather than sitting in ``errors`` next to sections
    # this method genuinely cannot do.
    assert "npa" not in payload["errors"]
    assert payload["npa"] == []
    assert "Natural Population Analysis" in payload["unavailable"]["npa"]
    bibtex = stem.with_suffix(".bibtex").read_text("utf-8")
    assert "mulliken_1955" in bibtex
    assert "lowdin_1950" not in bibtex


def test_scc_dftb_writes_method_native_mulliken_population(tmp_path) -> None:
    stem = tmp_path / "scc_native_population"

    result = vq.run_job(
        _water(),
        method="scc-dftb",
        output=stem,
        write_xyz_file=False,
        citations=True,
        output_qvf=False,
        crash_dump=False,
        progress=False,
    )

    payload = json.loads(stem.with_suffix(".population.json").read_text("utf-8"))
    charges = [row["charge"] for row in payload["mulliken"]]
    np.testing.assert_allclose(charges, result.mulliken_charges, atol=1.0e-12)
    assert sum(charges) == pytest.approx(0.0, abs=1.0e-10)
    assert set(payload["errors"]) == {
        "loewdin",
        "hirshfeld",
        "mayer",
        "wiberg",
        "dipole",
    }
    assert all(
        message.startswith("unsupported: scc_dftb native population")
        for message in payload["errors"].values()
    )
    # GitLab #197: NPA is unimplemented everywhere, not a failure of this
    # method's native population path, so it is reported once under
    # ``unavailable`` rather than sitting in ``errors`` next to sections
    # this method genuinely cannot do.
    assert "npa" not in payload["errors"]
    assert payload["npa"] == []
    assert "Natural Population Analysis" in payload["unavailable"]["npa"]
    manifest = tomllib.loads(stem.with_suffix(".system").read_text("utf-8"))
    assert manifest["outputs"]["status"] == "complete"
    population_rows = [
        row
        for row in manifest["outputs"]["files"]
        if ".population." in row["path"]
    ]
    assert len(population_rows) == 2
    assert all(row["written"] for row in population_rows)
    bibtex = stem.with_suffix(".bibtex").read_text("utf-8")
    assert "mulliken_1955" in bibtex
    assert "lowdin_1950" not in bibtex
    assert "mayer_bond_order_1983" not in bibtex
    assert "wiberg_bond_index_1968" not in bibtex
    assert "reed_weinhold_npa_1985" not in bibtex


def test_mace_explicit_wavefunction_sidecar_is_inapplicable(tmp_path) -> None:
    stem = tmp_path / "mace_explicit_molden"

    with pytest.raises(
        NotImplementedError,
        match="produces no electronic wavefunction",
    ):
        vq.run_job(
            _water(),
            method="mace",
            output=stem,
            write_molden_file=True,
            citations=False,
        )

    assert not stem.with_suffix(".system").exists()


def test_supported_rhf_default_plan_keeps_wavefunction_sidecars(tmp_path) -> None:
    stem = tmp_path / "rhf_auto_sidecars"

    result = vq.run_job(
        _water(),
        method="rhf",
        basis="sto-3g",
        output=stem,
        dry_run=True,
        citations=False,
    )

    assert result is None
    manifest = tomllib.loads(stem.with_suffix(".system").read_text("utf-8"))
    roles = [row["role"] for row in manifest["plan"]["files"]]
    assert roles.count("orbitals") == 1
    assert roles.count("population") == 2


def test_public_semiempirical_runner_normalizes_method_aliases(monkeypatch):
    seen = []

    def fake_run(plan, molecule):
        seen.append((plan.method_key, molecule.n_electrons(), plan.variant))
        return SemiempiricalResult(
            -2.0,
            None,
            plan.method_key,
            converged=True,
            n_iter=1,
        )

    monkeypatch.setattr(semi_runner, "_run_molecular_semiempirical", fake_run)

    result = run_semiempirical("  SCC-DFTB  ", _water())
    assert result.energy == -2.0
    assert result.route_plan.method_key == "scc_dftb"
    assert result.route_plan.boundary == "molecule"
    assert seen == [("scc_dftb", 10, "scc_dftb")]


def test_public_runner_builds_route_plan_before_molecular_dispatch(monkeypatch):
    calls = []
    real_from_request = semi_routes.SemiempiricalRoutePlan.from_request

    def tracked_from_request(method, **kwargs):
        calls.append(("plan", method, kwargs))
        return real_from_request(method, **kwargs)

    def fake_run(plan, molecule):
        calls.append(
            ("kernel", plan.method_key, plan.spin, molecule.n_electrons())
        )
        return SemiempiricalResult(-2.0, None, plan.method_key)

    monkeypatch.setattr(
        semi_runner.SemiempiricalRoutePlan,
        "from_request",
        staticmethod(tracked_from_request),
    )
    monkeypatch.setattr(semi_runner, "_run_molecular_semiempirical", fake_run)

    result = run_semiempirical("gfn2", _water())
    assert result.energy == -2.0
    assert calls == [
        (
            "plan",
            "gfn2",
            {
                "charge": 0,
                "multiplicity": 1,
                "nddo": False,
                "solvent": None,
                "ccm_options": None,
            },
        ),
        ("kernel", "gfn2_xtb", "closed_shell", 10),
    ]


@pytest.mark.parametrize(
    ("method", "model_name"),
    (("dftb0", "UDFTB0Model"), ("scc-dftb", "USCCDFTBModel")),
)
def test_public_open_shell_dftb_uses_planned_unrestricted_model(
    monkeypatch,
    method,
    model_name,
):
    calls = []

    class FakeModel:
        def __init__(self, molecule, *, params, **kwargs):
            calls.append((model_name, molecule.multiplicity, params))

        def energy(self):
            return -3.0

        def gradient(self):
            return np.zeros((2, 3))

    monkeypatch.setattr(vq.semiempirical, model_name, FakeModel)
    monkeypatch.setattr(
        "vibeqc.semiempirical.parameters.default_parameters",
        lambda: "params",
    )
    radical = vq.Molecule(
        [vq.Atom(8, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.8])],
        multiplicity=2,
    )

    result = run_semiempirical(method, radical)

    assert result.energy == -3.0
    assert calls == [(model_name, 2, "params")]


@pytest.mark.parametrize(
    ("method", "kwargs", "message"),
    (
        ("gfn2", {}, "GFN2-xTB route is closed-shell"),
        ("msindo", {"nddo": True}, "NDDO mode is closed-shell"),
    ),
)
def test_public_unvalidated_spin_route_fails_before_backend(
    monkeypatch,
    method,
    kwargs,
    message,
):
    def fail_backend(*_args, **_kwargs):
        raise AssertionError("unsupported spin route reached backend")

    monkeypatch.setattr(semi_runner, "_run_molecular_semiempirical", fail_backend)
    radical = vq.Molecule(
        [vq.Atom(8, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.8])],
        multiplicity=2,
    )

    with pytest.raises(NotImplementedError, match=message):
        run_semiempirical(method, radical, **kwargs)


@pytest.mark.parametrize(
    "alias",
    ("seccm", "se-ccm", "msindo-seccm", "msindo-ccm", "ccm"),
)
def test_public_seccm_aliases_use_legacy_dispatch_key(alias):
    assert normalise_semiempirical_method(alias) == "ccm"


def test_public_seccm_alias_dispatches_to_existing_backend(monkeypatch):
    options = object()
    seen = []

    def fake_run(molecule, ccm_options):
        seen.append((molecule.n_electrons(), ccm_options))
        return SemiempiricalResult(-4.0, None, "seccm")

    monkeypatch.setattr(semi_runner, "run_ccm", fake_run)

    result = run_semiempirical("se-ccm", _water(), ccm_options=options)
    assert result.energy == -4.0
    assert result.method == "seccm"
    assert seen == [(10, options)]


def test_run_job_seccm_alias_skips_basis_setup(monkeypatch, tmp_path):
    options = object()
    seen = []

    def fake_run(molecule, ccm_options):
        seen.append((molecule.n_electrons(), ccm_options))
        return SemiempiricalResult(-5.0, None, "seccm")

    monkeypatch.setattr(runner, "_run_ccm", fake_run)

    result = vq.run_job(
        _water(),
        method="seccm",
        ccm_options=options,
        output=tmp_path / "seccm",
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
        citations=False,
        progress=False,
    )
    assert result.energy == -5.0
    assert result.method == "seccm"
    assert seen == [(10, options)]


def test_run_job_seccm_citations_follow_concrete_route_plan(
    monkeypatch,
    tmp_path,
):
    class Options:
        translations = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0]]
        madelung = True

    options = Options()

    def fake_run(molecule, ccm_options):
        result = SemiempiricalResult(
            -5.0,
            None,
            "seccm",
            converged=True,
            n_iter=3,
        )
        result.route_plan = semi_routes.SemiempiricalRoutePlan.from_request(
            "seccm",
            charge=int(molecule.charge),
            multiplicity=int(molecule.multiplicity),
            ccm_options=ccm_options,
        ).with_seccm_runtime(
            periodic_dimension=2,
            electrostatics_family="madelung",
        )
        return result

    monkeypatch.setattr(runner, "_run_ccm", fake_run)
    stem = tmp_path / "seccm_citations"

    vq.run_job(
        _water(),
        method="seccm",
        ccm_options=options,
        output=stem,
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
        output_qvf=False,
        crash_dump=False,
        citations=True,
        progress=False,
    )

    bibtex = stem.with_suffix(".bibtex").read_text("utf-8")
    assert "ahlswede_jug_msindo_1_1999" in bibtex
    assert "ahlswede_jug_msindo_2_1999" in bibtex
    assert "bredow_geudtner_jug_ccm_2001" in bibtex
    assert "peintinger_bredow_ccm_2014" in bibtex
    assert "parry_2d_ewald_1975" in bibtex
    assert "de_leeuw_perram_2d_ewald_1979" in bibtex


def test_run_job_molecular_citations_follow_concrete_route_plan(
    monkeypatch,
    tmp_path,
):
    def fake_run(method, molecule, *, nddo=False):
        assert method == "dftb0"
        assert nddo is False
        result = SemiempiricalResult(-1.0, None, method, converged=True)
        result.route_plan = semi_routes.SemiempiricalRoutePlan.from_request(
            method,
            charge=int(molecule.charge),
            multiplicity=int(molecule.multiplicity),
        )
        return result

    monkeypatch.setattr(runner, "_run_semiempirical", fake_run)
    stem = tmp_path / "dftb0_citations"

    vq.run_job(
        _water(),
        method="dftb0",
        output=stem,
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
        output_qvf=False,
        crash_dump=False,
        citations=True,
        progress=False,
    )

    bibtex = stem.with_suffix(".bibtex").read_text("utf-8")
    assert "valeev_libint" in bibtex
    assert "pulay_diis_1980" not in bibtex
    assert "pulay_diis_1982" not in bibtex


def test_public_semiempirical_runner_rejects_unsupported_solvent():
    with pytest.raises(
        ValueError,
        match="Implicit solvation is not supported for method='scc_dftb'",
    ):
        run_semiempirical("scc-dftb", _water(), solvent="water")

    with pytest.raises(
        ValueError,
        match="Implicit solvation is not supported for SECCM",
    ):
        run_semiempirical("ccm", _water(), solvent="water")


def test_public_semiempirical_runner_rejects_unsupported_method_knobs():
    with pytest.raises(
        ValueError,
        match="nddo=True is only supported for method='msindo'",
    ):
        run_semiempirical("scc-dftb", _water(), nddo=True)

    with pytest.raises(
        ValueError,
        match="nddo=True is not implemented for SECCM",
    ):
        run_semiempirical("ccm", _water(), nddo=True)

    with pytest.raises(
        ValueError,
        match="ccm_options is only supported for the SECCM boundary",
    ):
        run_semiempirical("pm6", _water(), ccm_options=object())

    with pytest.raises(
        ValueError,
        match="nddo=True cannot be combined with solvent",
    ):
        run_semiempirical("msindo", _water(), nddo=True, solvent="water")


def test_public_seccm_open_shell_fails_before_backend(monkeypatch):
    def fail_backend(*_args, **_kwargs):
        raise AssertionError("unsupported open-shell SECCM reached backend")

    monkeypatch.setattr(semi_runner, "run_ccm", fail_backend)
    radical = vq.Molecule(
        [vq.Atom(8, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.8])],
        charge=0,
        multiplicity=2,
    )
    with pytest.raises(NotImplementedError, match="closed-shell only"):
        run_semiempirical("seccm", radical, ccm_options=object())


def test_public_pm6_runner_propagates_gradient_errors(monkeypatch):
    from vibeqc._vibeqc_core.semiempirical import nddo as _nddo

    def fail_gradient(*_args, **_kwargs):
        raise RuntimeError("pm6 gradient exploded")

    monkeypatch.setattr(
        _nddo, "compute_pm6_gradient_fd_from_result", fail_gradient
    )

    result = run_semiempirical("pm6", _water())
    with pytest.raises(RuntimeError, match="pm6 gradient exploded"):
        result.gradient()


def test_public_omx_runner_propagates_gradient_errors(monkeypatch):
    from vibeqc._vibeqc_core.semiempirical import nddo as _nddo

    def fail_gradient(*_args, **_kwargs):
        raise RuntimeError("omx gradient exploded")

    monkeypatch.setattr(
        _nddo,
        "compute_omx_v2_gradient_fd_from_result",
        fail_gradient,
    )

    result = run_semiempirical("om2", _water())
    with pytest.raises(RuntimeError, match="omx gradient exploded"):
        result.gradient()


def test_public_ccm_runner_rejects_malformed_options():
    class MissingMadelung:
        translations = [[5.0, 0.0, 0.0]]

    class FourVectors:
        translations = [
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
            [5.0, 5.0, 0.0],
        ]
        madelung = True

    class ShortVector:
        translations = [[5.0, 0.0]]
        madelung = True

    class NonNumericVector:
        translations = [["wide", 0.0, 0.0]]
        madelung = True

    with pytest.raises(
        ValueError,
        match="ccm_options.madelung",
    ):
        run_semiempirical("ccm", _water(), ccm_options=MissingMadelung())

    with pytest.raises(
        ValueError,
        match="1, 2, or 3 translation vectors",
    ):
        run_semiempirical("ccm", _water(), ccm_options=FourVectors())

    with pytest.raises(
        ValueError,
        match="three numeric components",
    ):
        run_semiempirical("ccm", _water(), ccm_options=ShortVector())

    with pytest.raises(
        ValueError,
        match="three numeric components",
    ):
        run_semiempirical("ccm", _water(), ccm_options=NonNumericVector())


def test_run_job_rejects_stray_semiempirical_method_knobs(tmp_path):
    with pytest.raises(
        ValueError,
        match="run_job: nddo=True is only valid with method='msindo'",
    ):
        vq.run_job(
            _water(),
            method="pm6",
            nddo=True,
            output=tmp_path / "pm6_nddo",
            progress=False,
        )

    with pytest.raises(
        ValueError,
        match="run_job: nddo=True is only valid with method='msindo'",
    ):
        vq.run_job(
            _water(),
            method="ccm",
            nddo=True,
            output=tmp_path / "ccm_nddo",
            progress=False,
        )

    with pytest.raises(
        ValueError,
        match="run_job: ccm_options= is only valid with method='seccm'",
    ):
        vq.run_job(
            _water(),
            method="scc-dftb",
            ccm_options=object(),
            output=tmp_path / "scc_ccm_options",
            progress=False,
        )


@pytest.mark.parametrize("optimizer_backend", ["native", "brent"])
def test_run_job_semiempirical_optimize_rejects_basis_backends(
    tmp_path,
    optimizer_backend,
):
    with pytest.raises(ValueError, match="requires optimizer_backend='ase'"):
        vq.run_job(
            _water(),
            method="pm6",
            optimize=True,
            optimizer_backend=optimizer_backend,
            output=tmp_path / f"pm6_{optimizer_backend}_opt",
            write_molden_file=False,
            write_xyz_file=False,
            write_population_file=False,
            citations=False,
            progress=False,
        )


def test_run_job_semiempirical_optimize_auto_without_ase_fails_cleanly(
    monkeypatch,
    tmp_path,
):
    def fake_resolve_optimizer_backend(requested):
        assert requested == "auto"
        return "native"

    monkeypatch.setattr(
        runner,
        "_resolve_optimizer_backend",
        fake_resolve_optimizer_backend,
    )

    with pytest.raises(ImportError, match="requires ASE"):
        vq.run_job(
            _water(),
            method="pm6",
            optimize=True,
            output=tmp_path / "pm6_auto_no_ase",
            write_molden_file=False,
            write_xyz_file=False,
            write_population_file=False,
            citations=False,
            progress=False,
        )


def test_run_job_semiempirical_hessian_skips_basis_path(monkeypatch, tmp_path):
    def fake_run_semiempirical(method, molecule, *, nddo=False):
        return runner._SEMPR(-1.0, None, method, converged=True, n_iter=1)

    def fail_hessian(*_args, **_kwargs):
        raise AssertionError("basis-free Hessian reached BasisSet path")

    import vibeqc.hessian as hessian_mod

    monkeypatch.setattr(runner, "_run_semiempirical", fake_run_semiempirical)
    monkeypatch.setattr(hessian_mod, "compute_hessian_fd", fail_hessian)

    out_stem = tmp_path / "scc_hessian"
    result = vq.run_job(
        _water(),
        method="scc-dftb",
        hessian=True,
        output=out_stem,
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
        citations=False,
        progress=False,
    )

    assert result.energy == -1.0
    text = out_stem.with_suffix(".out").read_text()
    assert "Vibrational Frequencies" in text
    assert "SKIPPED -- finite-difference Hessians" in text
    assert "BasisSet: no shells loaded" not in text
    assert "FAILED" not in text


def test_semiempirical_run_job_writes_basis_free_memory_estimate(
    monkeypatch,
    tmp_path,
):
    def fake_run_semiempirical(method, molecule, *, nddo=False):
        return runner._SEMPR(-1.0, None, method, converged=True, n_iter=1)

    monkeypatch.setattr(runner, "_run_semiempirical", fake_run_semiempirical)

    out_stem = tmp_path / "scc"
    result = vq.run_job(
        _water(),
        method="scc-dftb",
        output=out_stem,
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
        citations=False,
        progress=False,
    )

    assert result.energy == -1.0
    text = out_stem.with_suffix(".out").read_text()
    assert "vibe-qc estimates this calculation" in text
    assert "Semiempirical valence matrices" in text
    assert "converged in 1 iterations" in text
    assert "||[F,DS]||" not in text


# ---------------------------------------------------------------------------
# GitLab #160 (legacy BUG-024): a semiempirical route that returns an exactly
# zero total energy must refuse, not report.
#
# These assert on ``RuntimeError`` rather than on the concrete
# ``SemiempiricalEnergyError`` on purpose: at the parent commit the guard does
# not exist, and a test that imported the new symbol would fail with an
# ImportError, which proves only that a symbol is missing.  Failing with
# "DID NOT RAISE" is the behavioural failure -- a missing refusal.
# ---------------------------------------------------------------------------


class _NativePM6Stub:
    """The compute-study BUG-024 signature: 300 iterations, no exception, E as given."""

    def __init__(self, energy):
        self.energy = energy
        self.converged = False
        self.n_iter = 300
        self.density = None


def _drive_real_pm6_route(monkeypatch, energy):
    """Run the production ``_run_pm6`` over a native result reporting ``energy``.

    Only the native kernel is stubbed.  Plan construction, parameter loading,
    the Mermin recovery ladder, provenance projection and result assembly are
    the shipped code paths.  The recovery rungs are made unusable so the route
    reaches its return with the unconverged native result, exactly as the compute-study
    row did.
    """
    from vibeqc._vibeqc_core.semiempirical import nddo as _nddo

    def _unusable(*args, **kwargs):
        # Mirrors the C++ NaN/Inf guard's RuntimeError, which the ladder
        # already treats as "this rung did not help".
        raise RuntimeError("stub: recovery rung unavailable")

    for name in (
        "run_pm6_with_smearing",
        "run_pm6_with_density",
        "compute_pm6_gradient_fd_from_result",
    ):
        monkeypatch.setattr(_nddo, name, _unusable, raising=False)

    plan = semi_routes.SemiempiricalRoutePlan.from_request(
        "pm6", charge=0, multiplicity=1
    )

    def _fake_run_pm6(molecule, params, max_iter=200):
        return _NativePM6Stub(energy)

    return semi_runner._run_pm6(plan, _water(), _fake_run_pm6)


def test_pm6_route_refuses_a_silently_zero_total_energy(monkeypatch):
    """#160: the production PM6 route must not hand back E = 0.0 Ha."""
    with pytest.raises(RuntimeError) as excinfo:
        _drive_real_pm6_route(monkeypatch, 0.0)
    message = str(excinfo.value)
    assert "0.0" in message
    assert "#160" in message


def test_pm6_route_refuses_negative_zero_total_energy(monkeypatch):
    """-0.0 compares equal to 0.0 and is the same non-result."""
    with pytest.raises(RuntimeError):
        _drive_real_pm6_route(monkeypatch, -0.0)


def test_pm6_route_refuses_non_finite_total_energy(monkeypatch):
    """NaN/inf is the other value a total energy takes only on failure."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(RuntimeError):
            _drive_real_pm6_route(monkeypatch, bad)


def test_pm6_route_still_reports_a_real_energy(monkeypatch):
    """Negative control: the SAME route with the guard's condition unmet.

    -33.08480958728282 Ha is the converged PM6/norbornadiene energy measured
    on current main during the 2026-08-28 #160 re-test, i.e. the value the
    compute-study row should have carried.  It must pass through untouched, including
    ``converged=False``, which this guard deliberately does not police.
    """
    result = _drive_real_pm6_route(monkeypatch, -33.08480958728282)
    assert result.energy == -33.08480958728282
    assert result.converged is False
    # 300 from the native stub plus 100 per unusable recovery rung: the
    # ladder was entered and exhausted, so this control walks exactly the
    # code path the refusing cases take and differs only in the value.
    assert result.n_iter == 600


def test_pm6_route_accepts_a_legitimately_tiny_energy(monkeypatch):
    """The gate is bit-exact, not a tolerance band around zero.

    A tolerance would swallow real-but-small totals.  Only exact zero is a
    non-result, so the smallest representable non-zero double must pass.
    """
    tiny = 5e-324  # denormal min; the nearest non-zero double to 0.0
    result = _drive_real_pm6_route(monkeypatch, tiny)
    assert result.energy == tiny


def test_semiempirical_result_is_the_single_chokepoint():
    """Every route assembles its answer here, so the guard covers them all."""
    for method in ("pm6", "pm7", "om2", "msindo", "dftb0", "scc_dftb",
                   "gfn2_xtb", "seccm"):
        with pytest.raises(RuntimeError):
            SemiempiricalResult(0.0, None, method)
    # ... and a real energy through the same constructor still builds.
    assert SemiempiricalResult(-1.5, None, "pm6").energy == -1.5


def test_zero_energy_refusal_names_the_method_and_the_issue():
    with pytest.raises(RuntimeError) as excinfo:
        SemiempiricalResult(0.0, None, "gfn2_xtb")
    message = str(excinfo.value)
    assert "gfn2_xtb" in message
    assert "BUG-024" in message


def test_electron_free_system_keeps_its_legitimate_zero():
    """The one exact zero that is a real answer must not be refused.

    A bare proton has no electrons, so there is no electronic term, and a
    single centre has no core-core pair either: 0.0 Ha is that system's
    correct total.  Refusing it would make the guard itself the thing that
    reports a falsehood.  Passes at the parent too -- this is a control on
    the guard's reach, not fail-first evidence.
    """
    bare = vq.Molecule([vq.Atom(1, [0.0, 0.0, 0.0])], charge=1)
    assert bare.n_electrons() == 0
    result = public_run_semiempirical("pm6", bare)
    assert result.energy == 0.0

    # ... and the exemption is keyed on the electron count, not on the value:
    # the same zero with electrons present is still refused.
    with pytest.raises(RuntimeError):
        SemiempiricalResult(0.0, None, "pm6", n_electrons=10)
    # A non-finite energy is refused even for an electron-free system.
    with pytest.raises(RuntimeError):
        SemiempiricalResult(float("nan"), None, "pm6", n_electrons=0)
