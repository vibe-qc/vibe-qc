from __future__ import annotations

import json
import tomllib
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
from vibeqc import Atom, PeriodicSystem
from vibeqc import periodic_runner as pr
from vibeqc.output.formats.qvf import validate_qvf
from vibeqc.semiempirical.periodic import (
    BOUNDARY_PERIODIC_K,
    plan_periodic_semiempirical_route,
)


def _he_cell() -> PeriodicSystem:
    return PeriodicSystem(
        3,
        np.eye(3) * 5.0,
        [Atom(2, [0.0, 0.0, 0.0])],
        0,
        1,
    )


def _hli_chain() -> PeriodicSystem:
    return PeriodicSystem(
        1,
        np.diag([4.1, 30.0, 30.0]),
        [
            Atom(1, [0.17, 0.31, 0.0]),
            Atom(3, [1.39, -0.22, 0.0]),
        ],
        0,
        1,
    )


def _argon_fcc(scale: float = 1.0) -> PeriodicSystem:
    a = 5.256 * scale * 1.8897259885789233
    lattice = np.eye(3) * a
    fractional_atoms = (
        (0.0, 0.0, 0.0),
        (0.0, 0.5, 0.5),
        (0.5, 0.0, 0.5),
        (0.5, 0.5, 0.0),
    )
    atoms = [
        Atom(18, (np.asarray(frac) @ lattice).tolist())
        for frac in fractional_atoms
    ]
    return PeriodicSystem(3, lattice, atoms, 0, 1)


def test_run_periodic_job_accepts_scc_dftb_basis_free_route(tmp_path):
    stem = tmp_path / "he_scc"
    result = pr.run_periodic_job(
        _he_cell(),
        None,  # basis-free route must not inspect basis.name
        method="scc_dftb",
        output=stem,
        citations=False,
        record_hostname=False,
    )

    assert result.converged
    text = stem.with_suffix(".out").read_text(encoding="utf-8")
    assert "PERIODIC SCC_DFTB  basis=<basis-free>" in text
    assert "Gamma-point periodic semiempirical" in text
    assert "Supported: RHF, RKS, UHF, UKS" not in text
    assert "Molden orbitals and population analysis are inapplicable" in text
    qvf_path = stem.with_suffix(".qvf")
    assert qvf_path.exists()
    report = validate_qvf(qvf_path)
    assert report["valid"], report["errors"]
    with zipfile.ZipFile(qvf_path) as zf:
        qvf_manifest = json.loads(zf.read("manifest.json"))
        record = next(
            s
            for s in qvf_manifest["sections"]
            if s["kind"] == "run.record"
        )
        assert zf.read(record["members"]["log"]["path"]) == (
            stem.with_suffix(".out").read_bytes()
        )

    manifest = tomllib.loads(stem.with_suffix(".system").read_text(encoding="utf-8"))
    assert manifest["outputs"]["status"] == "complete"
    assert manifest["plan"]["method"] == "SCC_DFTB"
    assert manifest["plan"]["basis"] == "<basis-free>"
    roles = [row["role"] for row in manifest["plan"]["files"]]
    assert "orbitals" not in roles
    assert "population" not in roles
    guaranteed_paths = {
        row["path"] for row in manifest["plan"]["files"] if row["always"]
    }
    output_rows = {
        row["path"]: row for row in manifest["outputs"]["files"]
    }
    assert all(output_rows[path]["written"] for path in guaranteed_paths)


@pytest.mark.parametrize(
    "option",
    ["write_molden_file", "write_population_file"],
)
def test_periodic_semiempirical_explicit_sidecar_fails_before_engine(
    tmp_path,
    monkeypatch,
    option,
):
    def fail_engine(*_args, **_kwargs):
        raise AssertionError("periodic semiempirical engine should not run")

    monkeypatch.setattr(pr, "_run_periodic_semiempirical_engine", fail_engine)
    stem = tmp_path / f"scc_explicit_{option}"

    with pytest.raises(NotImplementedError, match=rf"{option}=True"):
        pr.run_periodic_job(
            _he_cell(),
            None,
            method="scc_dftb",
            output=stem,
            citations=False,
            record_hostname=False,
            **{option: True},
        )

    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize(
    ("method", "method_key", "smearing_hartree"),
    [
        ("dftb0", "dftb0", 0.0),
        ("scc_dftb", "scc_dftb", 0.0),
        ("gfn2_xtb", "gfn2_xtb", 0.001),
        ("pm6", "pm6", 0.0),
    ],
)
def test_run_periodic_job_accepts_gamma_se_methods_before_hf_gate(
    tmp_path, monkeypatch, method, method_key, smearing_hartree
):
    seen = {}

    def fake_engine(
        system,
        route_plan,
        *,
        max_iter,
        conv_tol,
        kpoints=None,
        smearing_temperature_hartree=None,
    ):
        seen["method_key"] = route_plan.method_key
        seen["boundary"] = route_plan.boundary
        seen["max_iter"] = max_iter
        seen["conv_tol"] = conv_tol
        seen["kpoints"] = kpoints
        seen["smearing_temperature_hartree"] = smearing_temperature_hartree
        return SimpleNamespace(
            energy=-0.25,
            converged=True,
            n_iter=3,
            n_basis=4,
        )

    monkeypatch.setattr(pr, "_run_periodic_semiempirical_engine", fake_engine)
    stem = tmp_path / f"he_{method_key}"
    result = pr.run_periodic_job(
        _he_cell(),
        None,
        method=method,
        output=stem,
        citations=False,
        output_qvf=False,
        record_hostname=False,
        write_xyz_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
    )

    assert result.energy == -0.25
    assert seen == {
        "method_key": method_key,
        "boundary": "periodic_gamma",
        "max_iter": 80,
        "conv_tol": 1e-7,
        "kpoints": None,
        "smearing_temperature_hartree": smearing_hartree,
    }
    text = stem.with_suffix(".out").read_text(encoding="utf-8")
    assert f"PERIODIC {method_key.upper()}  basis=<basis-free>" in text
    assert "Gamma-point periodic semiempirical" in text
    assert "Supported: RHF, RKS, UHF, UKS" not in text


def test_periodic_semiempirical_citations_follow_concrete_route_plan(
    tmp_path,
    monkeypatch,
):
    def fake_engine(*_args, **_kwargs):
        return SimpleNamespace(
            energy=-0.25,
            converged=True,
            n_iter=1,
            n_basis=4,
        )

    monkeypatch.setattr(pr, "_run_periodic_semiempirical_engine", fake_engine)
    stem = tmp_path / "periodic_dftb0_citations"
    pr.run_periodic_job(
        _he_cell(),
        None,
        method="dftb0",
        output=stem,
        citations=True,
        output_qvf=False,
        record_hostname=False,
        write_xyz_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
    )

    bibtex = stem.with_suffix(".bibtex").read_text(encoding="utf-8")
    assert "valeev_libint" in bibtex
    assert "togo_spglib_2024" in bibtex
    assert "mermin_finite_temperature_dft_1965" not in bibtex
    assert "pulay_diis_1980" not in bibtex
    assert "pulay_diis_1982" not in bibtex

    warm_stem = tmp_path / "periodic_gfn2_citations"
    pr.run_periodic_job(
        _he_cell(),
        None,
        method="gfn2_xtb",
        output=warm_stem,
        citations=True,
        output_qvf=False,
        record_hostname=False,
        write_xyz_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
    )

    warm_bibtex = warm_stem.with_suffix(".bibtex").read_text(
        encoding="utf-8"
    )
    assert "mermin_finite_temperature_dft_1965" in warm_bibtex


def test_periodic_parameter_identity_reaches_text_system_and_qvf(
    tmp_path, monkeypatch
):
    identity = "published:test-periodic-parameters-v1"
    sha256 = "3" * 64

    def fake_engine(*_args, **_kwargs):
        return SimpleNamespace(
            energy=-0.25,
            converged=True,
            n_iter=3,
            n_basis=4,
            parameter_identity=identity,
            parameter_sha256=sha256,
        )

    monkeypatch.setattr(pr, "_run_periodic_semiempirical_engine", fake_engine)
    stem = tmp_path / "identified_pm6"
    pr.run_periodic_job(
        _he_cell(),
        None,
        method="pm6",
        output=stem,
        citations=False,
        record_hostname=False,
        write_xyz_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
    )

    text = stem.with_suffix(".out").read_text(encoding="utf-8")
    assert f"parameter identity  = {identity}" in text
    assert f"parameter sha256    = {sha256}" in text
    system_manifest = tomllib.loads(
        stem.with_suffix(".system").read_text(encoding="utf-8")
    )
    assert system_manifest["run"]["parameter_identity"] == identity
    assert system_manifest["run"]["parameter_sha256"] == sha256
    with zipfile.ZipFile(stem.with_suffix(".qvf")) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        job = next(
            section
            for section in manifest["sections"]
            if section["kind"] == "job.spec"
        )
        spec = json.loads(zf.read(job["members"]["spec"]["path"]))
    assert spec["options"] == {
        "route_maturity": "mixed_native",
        "route_status": "periodic-pm6",
        "parameter_identity": identity,
        "parameter_sha256": sha256,
    }


def test_periodic_route_maturity_reaches_the_qvf_job_spec(tmp_path, monkeypatch):
    """An archived run records the maturity its route claimed (#150).

    The route plan is the one maturity vocabulary, and a periodic
    semiempirical run now carries that value into its own artifacts instead of
    leaving "experimental" to tribal knowledge. Rendering the matching line in
    the ``.out`` options block belongs to the output lane (#318).
    """

    def fake_engine(*_args, **_kwargs):
        return SimpleNamespace(
            energy=-0.25, converged=True, n_iter=3, n_basis=4
        )

    monkeypatch.setattr(pr, "_run_periodic_semiempirical_engine", fake_engine)
    stem = tmp_path / "gamma_gfn2"
    pr.run_periodic_job(
        _he_cell(),
        None,
        method="gfn2-xtb",
        output=stem,
        citations=False,
        record_hostname=False,
        write_xyz_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
    )

    system_manifest = tomllib.loads(
        stem.with_suffix(".system").read_text(encoding="utf-8")
    )
    assert system_manifest["run"]["route_maturity"] == "experimental"
    assert system_manifest["run"]["route_status"] == "gfn2-xtb"

    with zipfile.ZipFile(stem.with_suffix(".qvf")) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        job = next(
            section
            for section in manifest["sections"]
            if section["kind"] == "job.spec"
        )
        spec = json.loads(zf.read(job["members"]["spec"]["path"]))

    # No parameter identity on this fake result, so the maturity is recorded
    # on its own rather than riding along with the identity keys.
    assert spec["options"]["route_maturity"] == "experimental"
    assert spec["options"]["route_status"] == "gfn2-xtb"


@pytest.mark.parametrize("method", ["pm7", "om1", "om2", "om3"])
def test_run_periodic_job_gates_unimplemented_nddo_routes(tmp_path, method):
    stem = tmp_path / f"gated_{method}"
    with pytest.raises(NotImplementedError, match="no Bloch-periodic Gamma"):
        pr.run_periodic_job(
            _he_cell(),
            None,
            method=method,
            output=stem,
            citations=False,
            record_hostname=False,
        )
    assert not stem.with_suffix(".system").exists()


def test_run_periodic_job_pm6_fcc_argon_uses_bounded_native_branch(tmp_path):
    stem = tmp_path / "ar_pm6"
    result = pr.run_periodic_job(
        _argon_fcc(),
        None,
        method="pm6",
        output=stem,
        citations=False,
        output_qvf=False,
        record_hostname=False,
        write_xyz_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
    )

    assert result.converged
    # Four iterations, not the two of the collapsed-monopole model: the exact
    # zero-cell Ar-Ar sp-sp exchange block of issue #419 (44c7ec38a) makes
    # the first Fock matrix depend on the density, so the fixed point is no
    # longer reached in one update, and the exact heavy-heavy multipoles in
    # every cell lower the cell energy by 7.0e-5 Ha (-38.833302668977 ->
    # -38.833372664189). #419 re-pinned the lower-level driver test
    # (tests/test_semiempirical.py, TestPeriodicPM6) but not this public-route
    # pin, which went red on main (issue #641).
    assert result.n_iter == 4
    assert result.energy == pytest.approx(-38.833372664189, abs=2e-9)
    text = stem.with_suffix(".out").read_text(encoding="utf-8")
    assert "PERIODIC PM6  basis=<basis-free>" in text
    assert "iterations          = 4" in text
    # The public route delegates to the low-level Gamma driver with the same
    # default cutoff; pin the two against each other so the next driver
    # change cannot re-pin one and leave the other red (#641).
    from vibeqc.semiempirical.methods.periodic_pm6 import run_pm6_gamma

    reference = run_pm6_gamma(_argon_fcc())
    assert reference.converged
    assert result.n_iter == reference.n_iter
    assert result.energy == pytest.approx(reference.energy, abs=1e-12)


def test_periodic_gfn2_forwards_public_scc_budget_and_tolerance(monkeypatch):
    from vibeqc._vibeqc_core.semiempirical import xtb as _xtb
    from vibeqc.semiempirical.methods import gfn2_params

    seen = {}
    params = object()

    def fake_run(system, actual_params, opts):
        seen["system"] = system
        seen["params"] = actual_params
        seen["max_iter"] = opts.max_iter
        seen["conv_tol_charge"] = opts.conv_tol_charge
        seen["electronic_temperature"] = opts.electronic_temperature
        seen["electronic_temperature_explicit"] = (
            opts.electronic_temperature_explicit
        )
        return SimpleNamespace(energy=-0.25, converged=False, n_iter=7)

    monkeypatch.setattr(_xtb, "run_gfn2_xtb_gamma", fake_run)
    monkeypatch.setattr(gfn2_params, "load_gfn2_params", lambda: params)
    system = _he_cell()
    route_plan = pr._plan_periodic_semiempirical_method(
        "gfn2_xtb",
        system,
        kpoints=None,
    )

    # No smearing argument: the engine leaves the temperature unset so the
    # native default-on frontier smearing (0.001 Ha) applies.
    result = pr._run_periodic_semiempirical_engine(
        system,
        route_plan,
        max_iter=7,
        conv_tol=2.5e-8,
    )

    assert result.n_iter == 7
    assert seen["system"] is system
    assert seen["params"] is params
    assert seen["max_iter"] == 7
    assert seen["conv_tol_charge"] == 2.5e-8
    assert seen["electronic_temperature"] == 0.0
    assert seen["electronic_temperature_explicit"] is False

    # Explicit zero restores the exact Aufbau request.
    seen.clear()
    result = pr._run_periodic_semiempirical_engine(
        system,
        route_plan,
        max_iter=7,
        conv_tol=2.5e-8,
        smearing_temperature_hartree=0.0,
    )
    assert seen["electronic_temperature"] == 0.0
    assert seen["electronic_temperature_explicit"] is True

    # An explicit finite temperature is forwarded as-is.
    seen.clear()
    result = pr._run_periodic_semiempirical_engine(
        system,
        route_plan,
        max_iter=7,
        conv_tol=2.5e-8,
        smearing_temperature_hartree=0.01,
    )
    assert seen["electronic_temperature"] == pytest.approx(0.01)
    assert seen["electronic_temperature_explicit"] is True


def _gfn2_params_available() -> bool:
    try:
        from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

        load_gfn2_params()
    except Exception:
        return False
    return True


def _graphene_2c() -> PeriodicSystem:
    """Corrected article graphene cell (frontier fractionates at 0.001 Ha)."""
    angstrom_to_bohr = 1.8897259885789233
    lattice_rows = np.array(
        [
            [2.461, 0.0, 0.0],
            [-1.231, 2.131, 0.0],
            [0.0, 0.0, 20.0],
        ]
    )
    lattice_rows *= angstrom_to_bohr
    fractional = np.array(
        [
            [1.0 / 3.0, 2.0 / 3.0, 0.5],
            [2.0 / 3.0, 1.0 / 3.0, 0.5],
        ]
    )
    atoms = [
        Atom(6, position.tolist()) for position in fractional @ lattice_rows
    ]
    return PeriodicSystem(2, lattice_rows.T, atoms, 0, 1)


def test_run_periodic_job_gfn2_default_smearing_and_explicit_aufbau(tmp_path):
    """Public Gamma GFN2 route: default-on smearing, explicit T=0 Aufbau."""
    if not _gfn2_params_available():
        pytest.skip("GFN2-xTB parameters unavailable in this build")

    system = _graphene_2c()
    stem_default = tmp_path / "gr_gfn2_default"
    result = pr.run_periodic_job(
        system,
        None,
        method="gfn2_xtb",
        output=stem_default,
        citations=False,
        output_qvf=False,
        record_hostname=False,
        write_xyz_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        max_iter=1000,
    )
    assert result.converged
    assert float(result.smearing_temperature) == pytest.approx(0.001)
    assert float(result.entropy) > 0.0
    assert float(result.free_energy) == pytest.approx(
        float(result.energy)
        - float(result.smearing_temperature) * float(result.entropy),
        abs=1e-12,
        rel=0.0,
    )
    text = stem_default.with_suffix(".out").read_text(encoding="utf-8")
    assert "occupations         = fermi-dirac" in text
    assert "smearing_source     = auto" in text
    assert "Finite-temperature (smearing)" in text
    assert "free_energy (Ha)" in text

    stem_cold = tmp_path / "gr_gfn2_aufbau"
    result_cold = pr.run_periodic_job(
        system,
        None,
        method="gfn2_xtb",
        output=stem_cold,
        citations=False,
        output_qvf=False,
        record_hostname=False,
        write_xyz_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        smearing_temperature=0.0,
        max_iter=1000,
    )
    assert result_cold.converged
    assert float(result_cold.smearing_temperature) == 0.0
    assert float(result_cold.entropy) == 0.0
    assert float(result_cold.free_energy) == float(result_cold.energy)
    text_cold = stem_cold.with_suffix(".out").read_text(encoding="utf-8")
    assert "Finite-temperature (smearing)" not in text_cold

    stem_warm = tmp_path / "gr_gfn2_warm"
    result_warm = pr.run_periodic_job(
        system,
        None,
        method="gfn2_xtb",
        output=stem_warm,
        citations=False,
        output_qvf=False,
        record_hostname=False,
        write_xyz_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        smearing_temperature=0.01,
        max_iter=1000,
    )
    assert result_warm.converged
    assert float(result_warm.smearing_temperature) == pytest.approx(0.01)
    assert float(result_warm.free_energy) == pytest.approx(
        float(result_warm.energy)
        - float(result_warm.smearing_temperature) * float(result_warm.entropy),
        abs=1e-12,
        rel=0.0,
    )


def test_run_periodic_job_accepts_dftb0_kpoints(tmp_path):
    stem = tmp_path / "hli_dftb0_k"
    result = pr.run_periodic_job(
        _hli_chain(),
        None,
        method="dftb0",
        kpoints=(2, 1, 1),
        output=stem,
        citations=False,
        record_hostname=False,
        write_xyz_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
    )

    assert result.n_kpoints == 2
    assert np.isfinite(result.energy)
    text = stem.with_suffix(".out").read_text(encoding="utf-8")
    assert "PERIODIC DFTB0  basis=<basis-free>" in text
    assert "full k-point periodic semiempirical" in text
    assert "kpoints             = 2" in text
    assert "Band extrema (multi-k)" in text
    assert "Gamma-point periodic semiempirical" not in text

    manifest = tomllib.loads(stem.with_suffix(".system").read_text(encoding="utf-8"))
    assert manifest["outputs"]["status"] == "complete"
    assert manifest["plan"]["method"] == "DFTB0"
    assert manifest["plan"]["basis"] == "<basis-free>"


def test_run_periodic_job_scc_dftb_kpoints_converges(tmp_path):
    """The converging full-k SCC route still returns a result."""
    stem = tmp_path / "hli_scc_k"
    result = pr.run_periodic_job(
        _hli_chain(),
        None,
        method="scc_dftb",
        kpoints=(2, 1, 1),
        max_iter=300,
        output=stem,
        citations=False,
        record_hostname=False,
        write_xyz_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
    )

    assert bool(getattr(result, "converged", False))
    assert np.isfinite(result.energy)


def test_run_periodic_job_scc_dftb_kpoints_raises_on_nonconvergence(tmp_path):
    """A non-converged full-k SCC run retains fail-closed evidence (#342).

    The native driver returns a result object with ``converged=False`` so
    flag-reading consumers stay honest, but the public k-route must not
    hand that object to a status-screening caller as a successful run.  The
    public route writes its fatal diagnostic before raising and marks the
    manifest crashed; an engine-level raise would lose that evidence.
    """
    stem = tmp_path / "hli_scc_k_nonconv"
    with pytest.raises(
        RuntimeError,
        match=(
            "SCC_DFTB periodic semiempirical SCF did not converge after "
            "1 iterations; refusing to mark the calculation complete"
        ),
    ):
        pr.run_periodic_job(
            _hli_chain(),
            None,
            method="scc_dftb",
            kpoints=(2, 1, 1),
            max_iter=1,
            output=stem,
            citations=False,
            record_hostname=False,
            write_xyz_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
        )

    text = stem.with_suffix(".out").read_text(encoding="utf-8")
    assert "converged           = False" in text
    assert "iterations          = 1" in text
    assert (
        "FATAL: SCC_DFTB periodic semiempirical SCF did not converge after "
        "1 iterations; refusing to mark the calculation complete."
    ) in text

    manifest = tomllib.loads(stem.with_suffix(".system").read_text("utf-8"))
    assert manifest["outputs"]["status"] == "crashed"
    log_record = next(
        row
        for row in manifest["outputs"]["files"]
        if row["path"] == str(stem.with_suffix(".out"))
    )
    assert log_record["written"] is True
    assert log_record["bytes"] == len(text.encode("utf-8"))
    assert log_record["checksum_status"] == "sha256"


@pytest.mark.parametrize(
    "kpoints",
    [
        (1.2, 1, 1),
        (0, 1, 1),
        (np.nan, 1, 1),
        (1, 1),
        np.ones((3, 1)),
        ("2", 1, 1),
    ],
)
def test_periodic_dftb_runner_rejects_invalid_mesh_before_engine(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    kpoints,
) -> None:
    def fail_engine(*_args, **_kwargs):
        raise AssertionError("periodic semiempirical engine should not run")

    monkeypatch.setattr(pr, "_run_periodic_semiempirical_engine", fail_engine)
    stem = tmp_path / "invalid_dftb_mesh"
    with pytest.raises(ValueError, match="three finite positive integers"):
        pr.run_periodic_job(
            _hli_chain(),
            None,
            method="dftb0",
            kpoints=kpoints,
            output=stem,
            citations=False,
            record_hostname=False,
        )

    assert not stem.with_suffix(".system").exists()


def test_run_periodic_job_forwards_kpoint_smearing(tmp_path, monkeypatch):
    seen = {}

    def fake_engine(
        system,
        route_plan,
        *,
        max_iter,
        conv_tol,
        kpoints=None,
        smearing_temperature_hartree=0.0,
    ):
        seen["method_key"] = route_plan.method_key
        seen["boundary"] = route_plan.boundary
        seen["kpoints"] = kpoints
        seen["smearing_temperature_hartree"] = smearing_temperature_hartree
        return SimpleNamespace(
            energy=-0.30,
            free_energy=-0.31,
            converged=True,
            n_iter=2,
            n_basis=2,
            n_kpoints=2,
            smearing_temperature=0.01,
            entropy=1.0,
            fermi_level=-0.2,
            mo_energies=[np.array([-0.4, 0.2]), np.array([-0.3, 0.1])],
            occupations=[np.array([1.8, 0.2]), np.array([1.7, 0.3])],
            kpoints=[np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.0, 0.0])],
        )

    monkeypatch.setattr(pr, "_run_periodic_semiempirical_engine", fake_engine)
    stem = tmp_path / "hli_scc_k"
    result = pr.run_periodic_job(
        _hli_chain(),
        None,
        method="scc-dftb",
        kpoints=(2, 1, 1),
        smearing_temperature=0.01,
        output=stem,
        citations=False,
        record_hostname=False,
        write_xyz_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
    )

    assert result.free_energy == -0.31
    assert seen == {
        "method_key": "scc_dftb",
        "boundary": "periodic_k",
        "kpoints": (2, 1, 1),
        "smearing_temperature_hartree": 0.01,
    }
    text = stem.with_suffix(".out").read_text(encoding="utf-8")
    assert "full k-point periodic semiempirical" in text
    assert "smearing_temperature = 0.01" in text
    assert "Finite-temperature (smearing)" in text
    assert "free_energy (Ha)" in text


def test_smearing_summary_moves_energy_labels_with_policy() -> None:
    from vibeqc.output import DEFAULT_POLICY, active_policy, set_active_policy

    result = SimpleNamespace(
        method="rohf",
        energy=-0.30,
        free_energy=-0.31,
        smearing_temperature=0.01,
        entropy=1.0,
        fermi_level=-0.2,
    )
    saved = active_policy()
    try:
        set_active_policy(DEFAULT_POLICY.with_unit("energy", "eV"))
        text = pr._smearing_summary(result)
    finally:
        set_active_policy(saved)

    assert "kBT_smearing (eV)" in text
    assert "free_energy (eV)" in text
    assert "fermi_level (eV)" in text
    assert f"{-0.31 * 27.211386245988:20.10f}" in text
    lines = text.splitlines()
    assert len(lines[1]) == len(lines[2])


def test_smearing_summary_reports_both_open_shell_chemical_potentials() -> None:
    result = SimpleNamespace(
        energy=-0.30,
        free_energy=-0.31,
        smearing_temperature=0.01,
        entropy=1.0,
        fermi_level=-0.2,
        fermi_level_alpha=-0.25,
        fermi_level_beta=0.15,
    )

    text = pr._smearing_summary(result)

    assert "fermi_level_alpha (Ha)" in text
    assert "fermi_level_beta (Ha)" in text
    assert f"{-0.25:20.10f}" in text
    assert f"{0.15:20.10f}" in text


@pytest.mark.parametrize(
    ("present", "missing", "value"),
    [
        ("alpha", "beta", -0.25),
        ("beta", "alpha", 0.15),
    ],
)
def test_smearing_summary_reports_only_available_spin_chemical_potential(
    present,
    missing,
    value,
) -> None:
    result = SimpleNamespace(
        energy=-0.30,
        free_energy=-0.31,
        smearing_temperature=0.01,
        entropy=1.0,
        fermi_level=-0.2,
        **{f"fermi_level_{present}": value},
    )

    text = pr._smearing_summary(result)

    assert f"fermi_level_{present} (Ha)" in text
    assert f"fermi_level_{missing} (Ha)" not in text
    assert "fermi_level (Ha)" not in text
    assert f"{value:20.10f}" in text


def test_band_summary_sizes_rule_without_kpoint_annotations() -> None:
    result = SimpleNamespace(
        mo_energies=[np.array([-0.4, 0.2]), np.array([-0.3, 0.1])],
        occupations=[np.array([2.0, 0.0]), np.array([2.0, 0.0])],
        kpoints=[np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.0, 0.0])],
    )

    lines = pr._band_summary(result).splitlines()

    assert lines[0] == "  Band extrema (multi-k)"
    assert lines[2] == (
        "    VBM     =  -0.3000000000 Ha  (  -8.163 eV)"
        "  at k = [+0.5000, +0.0000, +0.0000]"
    )
    assert lines[3] == (
        "    CBM     =   0.1000000000 Ha  (   2.721 eV)"
        "  at k = [+0.5000, +0.0000, +0.0000]"
    )
    assert lines[4] == "    gap (direct) =   0.400000 Ha  (  10.885 eV)"
    assert lines[5] == (
        "    direct gap (min) =   0.400000 Ha  (  10.885 eV)"
        "  (k_idx = 1)"
    )

    def body_width(line: str) -> int:
        return len(line.split("  at k", 1)[0].split("  (k_idx", 1)[0])

    assert len(lines[1]) == max(body_width(line) for line in lines[2:])
    assert len(lines[2]) > len(lines[1])


def test_band_summary_uses_both_open_shell_occupation_channels() -> None:
    result = SimpleNamespace(
        mo_energies_alpha=[np.array([-0.8, 0.5]), np.array([-0.7, 0.4])],
        mo_energies_beta=[np.array([-0.6, 0.1]), np.array([-0.5, 0.2])],
        occupations_alpha=[np.array([1.0, 0.0]), np.array([1.0, 0.0])],
        occupations_beta=[np.array([1.0, 0.0]), np.array([1.0, 0.0])],
    )

    text = pr._band_summary(result)

    assert "VBM     =  -0.5000000000 Ha" in text
    assert "CBM     =   0.1000000000 Ha" in text
    assert "gap (indirect) =   0.600000 Ha" in text


def test_band_summary_fractional_frontier_does_not_infer_metallicity() -> None:
    result = SimpleNamespace(
        mo_energies=[np.array([-0.4, 0.1]), np.array([-0.3, 0.2])],
        occupations=[np.array([2.0, 0.5]), np.array([2.0, 0.5])],
    )

    text = pr._band_summary(result)

    assert "VBM" not in text
    assert "CBM" not in text
    assert "gap" not in text.lower()
    assert "occupation class = fractionally occupied / smeared" in text
    assert "zero-temperature band edges and metallicity not classified" in text


def test_band_summary_converged_true_still_reports_the_occupation_class() -> None:
    result = SimpleNamespace(
        mo_energies=[np.array([-0.4, 0.1]), np.array([-0.3, 0.2])],
        occupations=[np.array([2.0, 0.5]), np.array([2.0, 0.5])],
        converged=True,
    )

    text = pr._band_summary(result)

    assert "occupation class = fractionally occupied / smeared" in text
    assert "not converged" not in text


def test_band_summary_does_not_class_occupations_of_a_non_converged_scf() -> None:
    """A non-converged SCF has no occupation class to report.

    Regression for GitLab #106 / #85: the 2D slab V_ne(k) defect drove h-BN
    into non-convergence, and the fill attached to the last-iteration
    eigenvalues was printed as "fractionally occupied / smeared" -- a
    statement about the material, read off an unconverged Fock. That reading
    is what the loop recorded as "h-BN is band-overlapping".
    """
    result = SimpleNamespace(
        mo_energies=[np.array([-0.4, 0.1]), np.array([-0.3, 0.2])],
        occupations=[np.array([2.0, 0.5]), np.array([2.0, 0.5])],
        converged=False,
    )

    text = pr._band_summary(result)

    assert "occupation class" not in text
    assert "fractionally occupied" not in text
    assert "smeared" not in text
    assert "SCF not converged - band edges not classified" in text
    assert "VBM" not in text
    assert "CBM" not in text


def test_band_summary_reports_inverted_ordering_as_a_negative_gap() -> None:
    """An inverted band ordering prints its true NEGATIVE gap, never 0.

    Regression for the hBN sto-3g slab-GDF reproducer, which printed
    ``gap (indirect) = 0.000000 Ha`` while its CBM sat 3.270 eV *below*
    its VBM. Clamping the value at zero papers over the very defect the
    reader needs to see (CLAUDE.md section 7) and silently defeats any
    regression test written against this line. The eigenvalues below are
    the measured hBN extrema.
    """
    result = SimpleNamespace(
        mo_energies=[
            np.array([-0.2004980334, 0.0267097639]),
            np.array([-0.9, -0.3206728391]),
        ],
        occupations=[np.array([2.0, 0.0]), np.array([2.0, 0.0])],
    )

    text = pr._band_summary(result)

    assert "VBM     =  -0.2004980334 Ha" in text
    assert "CBM     =  -0.3206728391 Ha" in text
    # -0.1201748057 Ha = -3.270 eV, the true CBM - VBM.
    assert "gap (indirect) =  -0.120175 Ha  (  -3.270 eV)" in text
    assert "negative: band ordering inverted across the mesh" in text
    assert "gap (indirect) =   0.000000" not in text


def test_band_summary_does_not_clamp_a_negative_direct_gap() -> None:
    """A per-k inverted ordering is reported, not floored at zero."""
    result = SimpleNamespace(
        mo_energies=[np.array([-0.30, -0.50]), np.array([-0.20, -0.10])],
        occupations=[np.array([2.0, 0.0]), np.array([2.0, 0.0])],
    )

    text = pr._band_summary(result)

    # k_idx 0 has its "virtual" 0.2 Ha BELOW its occupied state.
    assert "direct gap (min) =  -0.200000 Ha" in text
    assert "direct gap (min) =   0.000000" not in text


def test_band_summary_keeps_a_positive_gap_unchanged() -> None:
    """Removing the clamp must not perturb ordinary gapped reporting."""
    result = SimpleNamespace(
        mo_energies=[np.array([-0.4, 0.2]), np.array([-0.3, 0.1])],
        occupations=[np.array([2.0, 0.0]), np.array([2.0, 0.0])],
    )

    text = pr._band_summary(result)

    assert "gap (direct) =   0.400000 Ha  (  10.885 eV)" in text
    assert "negative: band ordering inverted" not in text


def test_band_summary_never_infers_occupancy_from_energy_sign() -> None:
    result = SimpleNamespace(
        mo_energies=[np.array([-5.0, -4.0]), np.array([-5.1, -3.9])],
    )
    assert pr._band_summary(result) == ""


def test_band_summary_requires_all_open_shell_occupation_channels() -> None:
    result = SimpleNamespace(
        mo_energies_alpha=[np.array([-0.8, 0.5]), np.array([-0.7, 0.4])],
        mo_energies_beta=[np.array([-0.6, 0.1]), np.array([-0.5, 0.2])],
        occupations_alpha=[np.array([1.0, 0.0]), np.array([1.0, 0.0])],
    )
    assert pr._band_summary(result) == ""


def test_band_summary_prefers_restricted_open_occupations_over_spin_aliases() -> None:
    result = SimpleNamespace(
        method="rohf",
        mo_energies=[np.array([-0.8, -0.1, 0.5]), np.array([-0.7, 0.0, 0.4])],
        mo_energies_alpha=[np.array([-0.8, -0.1, 0.5]), np.array([-0.7, 0.0, 0.4])],
        mo_energies_beta=[np.array([-0.8, -0.1, 0.5]), np.array([-0.7, 0.0, 0.4])],
        occupations=[np.array([2.0, 1.0, 0.0]), np.array([2.0, 1.0, 0.0])],
    )
    text = pr._band_summary(result)
    assert "fractionally occupied / smeared" not in text
    assert "VBM     =   0.0000000000 Ha" in text
    assert "CBM     =   0.4000000000 Ha" in text

    mo_text = pr._mo_summary(result)
    assert mo_text.count(" HOCO") == 1
    assert mo_text.count(" LUCO") == 1


def test_band_summary_prefers_true_spin_channels_over_compatibility_fields() -> None:
    result = SimpleNamespace(
        method="uks",
        mo_energies=[np.array([-0.8, 0.5]), np.array([-0.7, 0.4])],
        occupations=[np.array([2.0, 0.0]), np.array([2.0, 0.0])],
        mo_energies_alpha=[np.array([-0.8, 0.5]), np.array([-0.7, 0.4])],
        mo_energies_beta=[np.array([-0.6, 0.1]), np.array([-0.5, 0.2])],
        occupations_alpha=[np.array([1.0, 0.0]), np.array([1.0, 0.0])],
        occupations_beta=[np.array([1.0, 0.0]), np.array([1.0, 0.0])],
    )
    text = pr._band_summary(result)
    assert "VBM     =  -0.5000000000 Ha" in text
    assert "CBM     =   0.1000000000 Ha" in text


def test_crystal_orbital_summary_uses_policy_and_ragged_markers() -> None:
    from vibeqc.output import DEFAULT_POLICY, active_policy, set_active_policy

    result = SimpleNamespace(
        mo_energies=np.array([-0.4, 0.2]),
        occupations=np.array([2.0, 0.0]),
    )
    saved = active_policy()
    try:
        set_active_policy(DEFAULT_POLICY.with_unit("energy", "eV"))
        lines = pr._mo_summary(result).splitlines()
    finally:
        set_active_policy(saved)

    assert lines[0] == "  Crystal orbital energies (eV) -- sorted low -> high"
    assert f"{-0.4 * 27.211386245988:18.10f}" in lines[2]
    assert lines[2].endswith(" HOCO")
    assert lines[3].endswith(" LUCO")
    assert len(lines[1]) == len(lines[0])
    assert len(lines[2]) < len(lines[1])


def test_crystal_orbital_summary_prints_both_spin_channels() -> None:
    result = SimpleNamespace(
        mo_energies_alpha=[np.array([-0.8, 0.5])],
        mo_energies_beta=[np.array([-0.6, 0.1])],
        occupations_alpha=[np.array([1.0, 0.0])],
        occupations_beta=[np.array([1.0, 0.0])],
    )
    text = pr._mo_summary(result)
    assert "Crystal orbital energies (alpha)" in text
    assert "Crystal orbital energies (beta)" in text


def test_crystal_orbital_summary_omits_zero_temperature_frontiers_when_smeared(
) -> None:
    result = SimpleNamespace(
        mo_energies=np.array([-0.4, 0.1, 0.3]),
        occupations=np.array([1.999, 0.75, 0.001]),
        smearing_temperature=0.01,
        fermi_level=0.05,
    )

    text = pr._mo_summary(result)

    assert "fractional occupations, frontier labels omitted" in text
    assert "occ=" in text
    assert " HOCO" not in text
    assert " LUCO" not in text


def test_open_shell_smeared_orbitals_use_spin_specific_chemical_potentials(
) -> None:
    energies = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    fractional = np.array([0.99, 0.9, 0.5, 0.1, 0.01])
    result = SimpleNamespace(
        mo_energies_alpha=[energies],
        mo_energies_beta=[energies],
        occupations_alpha=[fractional],
        occupations_beta=[fractional],
        fermi_level=-2.0,
        fermi_level_alpha=-1.5,
        fermi_level_beta=1.5,
    )

    text = pr._mo_summary(result, n_show=2)
    alpha, beta = text.split("  Crystal orbital energies (beta)", maxsplit=1)

    assert f"{-2.0:18.10f}" in alpha
    assert f"{-1.0:18.10f}" in alpha
    assert f"{1.0:18.10f}" in beta
    assert f"{2.0:18.10f}" in beta


def test_optimized_geometry_summary_sizes_rule_and_uses_policy() -> None:
    from vibeqc.output import DEFAULT_POLICY, active_policy, set_active_policy

    result = SimpleNamespace(energy=-0.5, n_iter=3, converged=True)
    saved = active_policy()
    try:
        set_active_policy(DEFAULT_POLICY.with_unit("energy", "eV"))
        lines = pr._optimized_geometry_summary(result).splitlines()
    finally:
        set_active_policy(saved)

    assert lines[0] == "  Optimized geometry"
    assert lines[2] == f"    E_final = {-0.5 * 27.211386245988:+.10f} eV"
    assert lines[3] == "    n_iter  = 3"
    assert lines[4] == "    converged = True"
    assert len(lines[1]) == max(len(line) for line in lines[2:])


def _fake_kpoint_result(energy: float = -0.30):
    return SimpleNamespace(
        energy=energy,
        free_energy=energy,
        converged=True,
        n_iter=2,
        n_basis=2,
        n_kpoints=2,
        smearing_temperature=0.0,
        entropy=0.0,
        fermi_level=-0.2,
        mo_energies=[np.array([-0.4, 0.2]), np.array([-0.3, 0.1])],
        occupations=[np.array([2.0, 0.0]), np.array([2.0, 0.0])],
        kpoints=[np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.0, 0.0])],
    )


def test_run_periodic_job_dftb_kpoints_optimize_uses_gradient_facade(
    tmp_path,
    monkeypatch,
):
    import vibeqc.semiempirical.periodic as periodic_module

    seen = {}

    def fake_engine(
        system,
        route_plan,
        *,
        max_iter,
        conv_tol,
        kpoints=None,
        smearing_temperature_hartree=0.0,
    ):
        seen.setdefault("engine_calls", []).append(
            (
                route_plan.method_key,
                route_plan.boundary,
                kpoints,
                smearing_temperature_hartree,
                max_iter,
                conv_tol,
            )
        )
        return _fake_kpoint_result()

    def fake_gradient(
        method,
        system,
        *,
        kpoints=None,
        cutoff_bohr=15.0,
        fd_step_bohr=0.001,
        max_iter=None,
        conv_tol_charge=None,
        smearing_temperature=None,
        _return_result=False,
    ):
        seen["gradient_call"] = {
            "method": method,
            "kpoints": kpoints,
            "cutoff_bohr": cutoff_bohr,
            "fd_step_bohr": fd_step_bohr,
            "max_iter": max_iter,
            "conv_tol_charge": conv_tol_charge,
            "smearing_temperature": smearing_temperature,
        }
        gradient = np.zeros((len(system.unit_cell), 3))
        if _return_result:
            return SimpleNamespace(energy=-0.40, gradient=gradient)
        return -0.40, gradient

    def fake_optimize(
        system,
        energy_fn,
        gradient_fn=None,
        *,
        derivatives_fn=None,
        gradient_tolerance_ha_bohr=1.0e-4,
        max_steps=100,
        route_plan=None,
    ):
        derivatives = derivatives_fn(system)
        seen["optimizer"] = {
            "energy": derivatives.energy,
            "gradient_shape": derivatives.gradient.shape,
            "gradient_fn": gradient_fn,
            "gradient_tolerance_ha_bohr": gradient_tolerance_ha_bohr,
            "max_steps": max_steps,
            "route": route_plan.status_route,
        }
        return periodic_module.PeriodicSemiempiricalOptimizationResult(
            system=system,
            energy=-0.50,
            gradient=np.zeros((len(system.unit_cell), 3)),
            n_iter=3,
            converged=True,
            route_plan=route_plan,
        )

    monkeypatch.setattr(pr, "_run_periodic_semiempirical_engine", fake_engine)
    monkeypatch.setattr(
        periodic_module,
        "evaluate_periodic_energy_gradient",
        fake_gradient,
    )
    monkeypatch.setattr(
        periodic_module,
        "optimize_periodic_positions",
        fake_optimize,
    )

    stem = tmp_path / "hli_dftb0_k_opt"
    result = pr.run_periodic_job(
        _hli_chain(),
        None,
        method="dftb0",
        kpoints=(2, 1, 1),
        optimize=True,
        optimize_max_iter=3,
        optimize_conv_tol_grad=2.0e-4,
        output=stem,
        citations=False,
        record_hostname=False,
        write_xyz_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
    )

    assert result.energy == -0.50
    assert seen["optimizer"] == {
        "energy": -0.40,
        "gradient_shape": (2, 3),
        "gradient_fn": None,
        "gradient_tolerance_ha_bohr": 2.0e-4,
        "max_steps": 3,
        "route": "periodic-dftb0-kpoint-gradient-fd",
    }
    assert seen["gradient_call"] == {
        "method": "dftb0",
        "kpoints": (2, 1, 1),
        "cutoff_bohr": 15.0,
        "fd_step_bohr": 0.001,
        "max_iter": 80,
        "conv_tol_charge": 1e-7,
        # #545: the derivative facade is asked for the run's resolved
        # temperature (0.0 here), the same one opt_energy evaluates at.
        "smearing_temperature": 0.0,
    }
    text = stem.with_suffix(".out").read_text(encoding="utf-8")
    assert "Geometry optimization" in text
    assert "cell                = fixed" in text
    assert "Optimized geometry" in text
    opt_lines = text.split("  Optimized geometry\n", 1)[1].splitlines()
    assert len(opt_lines[0]) == max(len(line) for line in opt_lines[1:4])
    manifest = tomllib.loads(stem.with_suffix(".system").read_text("utf-8"))
    assert manifest["outputs"]["status"] == "complete"


def test_run_periodic_job_dftb_kpoints_optimize_cell_uses_stress_facade(
    tmp_path,
    monkeypatch,
):
    import vibeqc.semiempirical.periodic as periodic_module

    seen = {}

    def fake_engine(
        system,
        route_plan,
        *,
        max_iter,
        conv_tol,
        kpoints=None,
        smearing_temperature_hartree=0.0,
    ):
        seen.setdefault("engine_calls", []).append(
            (
                route_plan.method_key,
                route_plan.boundary,
                kpoints,
                smearing_temperature_hartree,
                max_iter,
                conv_tol,
            )
        )
        return _fake_kpoint_result()

    def fake_derivatives(
        method,
        system,
        *,
        kpoints=None,
        cutoff_bohr=15.0,
        fd_step_bohr=0.001,
        strain_step=0.001,
        max_iter=None,
        conv_tol_charge=None,
        smearing_temperature=0.0,
        smearing_unit="hartree",
        smearing_method="fermi-dirac",
    ):
        seen["derivative_call"] = {
            "method": method,
            "kpoints": kpoints,
            "cutoff_bohr": cutoff_bohr,
            "fd_step_bohr": fd_step_bohr,
            "strain_step": strain_step,
            "max_iter": max_iter,
            "conv_tol_charge": conv_tol_charge,
            "smearing_temperature": smearing_temperature,
            "smearing_unit": smearing_unit,
            "smearing_method": smearing_method,
        }
        return SimpleNamespace(
            energy=-0.41,
            gradient=np.zeros((len(system.unit_cell), 3)),
            stress=np.zeros((3, 3)),
        )

    def fake_optimize_cell(
        system,
        energy_fn,
        gradient_fn=None,
        *,
        stress_fn=None,
        derivatives_fn=None,
        fmax=0.01,
        gradient_tolerance_ha_bohr=None,
        max_steps=100,
        isotropic=False,
        return_result=False,
        route_plan=None,
    ):
        derivatives = derivatives_fn(system)
        seen["optimizer"] = {
            "energy": derivatives.energy,
            "gradient_shape": derivatives.gradient.shape,
            "stress_shape": derivatives.stress.shape,
            "gradient_fn": gradient_fn,
            "stress_fn": stress_fn,
            "fmax": fmax,
            "gradient_tolerance_ha_bohr": gradient_tolerance_ha_bohr,
            "max_steps": max_steps,
            "isotropic": isotropic,
            "return_result": return_result,
            "route": route_plan.status_route,
        }
        return periodic_module.PeriodicSemiempiricalOptimizationResult(
            system=system,
            energy=-0.55,
            gradient=np.zeros((len(system.unit_cell), 3)),
            n_iter=4,
            converged=True,
            route_plan=route_plan,
        )

    monkeypatch.setattr(pr, "_run_periodic_semiempirical_engine", fake_engine)
    monkeypatch.setattr(
        periodic_module,
        "evaluate_periodic_kpoint_energy_gradient_stress",
        fake_derivatives,
    )
    monkeypatch.setattr(periodic_module, "optimize_cell", fake_optimize_cell)

    stem = tmp_path / "hli_dftb0_k_cell_opt"
    result = pr.run_periodic_job(
        _hli_chain(),
        None,
        method="dftb0",
        kpoints=(2, 1, 1),
        optimize=True,
        optimize_cell=True,
        optimize_max_iter=4,
        optimize_conv_tol_grad=2.0e-4,
        output=stem,
        citations=False,
        record_hostname=False,
        write_xyz_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
    )

    assert result.energy == -0.55
    assert seen["optimizer"] == {
        "energy": -0.41,
        "gradient_shape": (2, 3),
        "stress_shape": (3, 3),
        "gradient_fn": None,
        "stress_fn": None,
        "fmax": 0.01,
        "gradient_tolerance_ha_bohr": 2.0e-4,
        "max_steps": 4,
        "isotropic": False,
        "return_result": True,
        "route": "periodic-dftb0-kpoint-gradient-fd",
    }
    assert seen["derivative_call"] == {
        "method": "dftb0",
        "kpoints": (2, 1, 1),
        "cutoff_bohr": 15.0,
        "fd_step_bohr": 0.001,
        "strain_step": 0.001,
        "max_iter": 80,
        "conv_tol_charge": 1e-7,
        "smearing_temperature": 0.0,
        "smearing_unit": "hartree",
        "smearing_method": "fermi-dirac",
    }
    text = stem.with_suffix(".out").read_text(encoding="utf-8")
    assert "Geometry optimization" in text
    assert "cell                = variable" in text
    assert "Optimized geometry" in text
    manifest = tomllib.loads(stem.with_suffix(".system").read_text("utf-8"))
    assert manifest["outputs"]["status"] == "complete"


def test_periodic_dftb_kpoint_optimize_cell_requires_optimize(
    tmp_path,
    monkeypatch,
):
    def fail_engine(*_args, **_kwargs):
        raise AssertionError("periodic semiempirical engine should not run")

    monkeypatch.setattr(pr, "_run_periodic_semiempirical_engine", fail_engine)
    stem = tmp_path / "hli_dftb0_k_cell_without_opt"
    with pytest.raises(ValueError, match="requires optimize=True"):
        pr.run_periodic_job(
            _hli_chain(),
            None,
            method="dftb0",
            kpoints=(2, 1, 1),
            optimize_cell=True,
            output=stem,
            citations=False,
            record_hostname=False,
        )

    assert not stem.with_suffix(".system").exists()


def test_periodic_dftb_kpoint_optimize_threads_one_temperature_and_minimises_f(
    tmp_path,
    monkeypatch,
):
    """#545 (maintainer decision 2026-09-06): full-k DFTB optimization at
    finite temperature runs. Every optimizer callback evaluates at the SAME
    resolved temperature, the energy callback returns the Mermin free energy
    (not the internal energy), and the derivative facade is asked for the
    smeared surface, so BFGS never minimises E while consuming grad F.
    Pre-fix the runner raised "zero-temperature only" before the engine ran."""
    import vibeqc.semiempirical.periodic as periodic_module

    seen: dict = {"engine_T": [], "gradient_T": []}

    def fake_engine(
        system,
        route_plan,
        *,
        max_iter,
        conv_tol,
        kpoints=None,
        smearing_temperature_hartree=0.0,
    ):
        seen["engine_T"].append(smearing_temperature_hartree)
        return SimpleNamespace(
            energy=-0.30,
            free_energy=-0.31,
            converged=True,
            n_iter=2,
            n_basis=2,
            n_kpoints=2,
            smearing_temperature=smearing_temperature_hartree,
            entropy=1.0,
            fermi_level=-0.2,
            mo_energies=[np.array([-0.4, 0.2]), np.array([-0.3, 0.1])],
            occupations=[np.array([1.9, 0.1]), np.array([1.9, 0.1])],
            kpoints=[np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.0, 0.0])],
        )

    def fake_gradient(method, system, **kwargs):
        seen["gradient_T"].append(kwargs.get("smearing_temperature"))
        return periodic_module.PeriodicSemiempiricalDerivativeResult(
            energy=-0.30,
            gradient=np.zeros((len(system.unit_cell), 3)),
            stress=np.zeros((3, 3)),
            route_plan=plan_periodic_semiempirical_route(
                "dftb0", system, boundary=BOUNDARY_PERIODIC_K,
                properties=("energy", "gradient"),
            ),
            parameter_identity="test",
            parameter_sha256="0" * 64,
            free_energy=-0.31,
            differentiated_potential="free_energy",
            differentiated_free_energy=True,
            smearing_temperature=0.01,
        )

    def fake_optimize(system, energy_fn, gradient_fn=None, *, derivatives_fn=None, **kwargs):
        derivatives = derivatives_fn(system)
        seen["energy_fn"] = energy_fn(system)
        seen["paired_potential"] = periodic_module._minimised_potential(derivatives)
        return periodic_module.PeriodicSemiempiricalOptimizationResult(
            system=system,
            energy=-0.31,
            gradient=np.zeros((len(system.unit_cell), 3)),
            n_iter=3,
            converged=True,
            route_plan=kwargs.get("route_plan"),
            parameter_identity="test",
            parameter_sha256="0" * 64,
            **periodic_module._optimization_potential_fields(derivatives),
        )

    monkeypatch.setattr(pr, "_run_periodic_semiempirical_engine", fake_engine)
    monkeypatch.setattr(
        periodic_module, "evaluate_periodic_energy_gradient", fake_gradient
    )
    monkeypatch.setattr(
        periodic_module, "optimize_periodic_positions", fake_optimize
    )

    stem = tmp_path / "hli_dftb0_k_opt_smearing"
    result = pr.run_periodic_job(
        _hli_chain(),
        None,
        method="dftb0",
        kpoints=(2, 1, 1),
        optimize=True,
        smearing_temperature=0.01,
        output=stem,
        citations=False,
        record_hostname=False,
        write_xyz_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
    )

    # One temperature everywhere: the initial run, every opt_energy call and
    # the derivative facade.
    assert seen["engine_T"] and all(T == 0.01 for T in seen["engine_T"])
    assert seen["gradient_T"] == [0.01]
    # The energy callback returns F, and the potential the optimizer pairs
    # with grad F is F as well.
    assert seen["energy_fn"] == -0.31
    assert seen["paired_potential"] == -0.31
    assert result.energy == -0.31
    assert result.minimised_potential == "free_energy"
    assert result.internal_energy == -0.30
    assert result.free_energy == -0.31
    assert result.smearing_temperature == 0.01
    text = stem.with_suffix(".out").read_text(encoding="utf-8")
    assert "F_final" in text and "Mermin free energy, minimised" in text
    assert "E_final" in text and "-TS" in text
    assert stem.with_suffix(".system").exists()


def test_periodic_dftb_kpoint_finite_temperature_optimization_is_stationary_on_f(
    tmp_path,
):
    """Real engine, #545 closure criterion: full-k DFTB0 optimization of the
    HLi chain at kT = 0.05 Ha completes, reports the Mermin free energy as the
    minimised potential with E and -TS beside it, and at the final geometry
    the kernel gradient equals a central difference of F over a converging
    step sequence while it differs from the central difference of E, which
    is the negative control that grad F, not grad E, was consumed. Measured
    2026-09-06: |grad_kernel - FD(F)| 4.0e-9, 8.1e-10, 2.0e-10 at h = 4e-3,
    2e-3, 5e-4; |grad_kernel - FD(E)| 2.45e-6 at every step. Weinert and
    Davenport, Phys. Rev. B 45, 13709 (1992), Sec. III: with fractional
    occupations the force is -dF/dR and carries no occupation-number term."""
    from vibeqc.semiempirical.periodic import evaluate_periodic_energy_gradient

    T = 0.05
    stem = tmp_path / "hli_dftb0_k_opt_finite_t"
    result = pr.run_periodic_job(
        _hli_chain(),
        None,
        method="dftb0",
        kpoints=(2, 1, 1),
        optimize=True,
        smearing_temperature=T,
        optimize_max_iter=30,
        optimize_conv_tol_grad=1.0e-4,
        output=stem,
        citations=False,
        record_hostname=False,
        write_xyz_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        output_qvf=False,
    )
    assert result.converged
    assert result.minimised_potential == "free_energy"
    assert result.smearing_temperature == T
    assert result.energy == result.free_energy
    # The smearing engaged: -TS is a real, negative contribution.
    assert result.free_energy < result.internal_energy - 1.0e-3
    text = stem.with_suffix(".out").read_text(encoding="utf-8")
    assert "Mermin free energy, minimised" in text
    assert "(internal energy)" in text
    assert "kT = 0.05 Ha" in text

    final = result.system
    derivatives = evaluate_periodic_energy_gradient(
        "dftb0",
        final,
        kpoints=(2, 1, 1),
        cutoff_bohr=15.0,
        max_iter=80,
        conv_tol_charge=1.0e-7,
        smearing_temperature=T,
        _return_result=True,
    )
    assert derivatives.differentiated_potential == "free_energy"
    assert derivatives.smearing_temperature == T
    assert np.abs(derivatives.gradient).max() <= 1.0e-4 + 1.0e-8

    route_plan = plan_periodic_semiempirical_route(
        "dftb0", final, boundary=BOUNDARY_PERIODIC_K, properties=("energy",)
    )
    atoms = list(final.unit_cell)

    def displaced(atom_index, axis, h):
        xyz = [list(atom.xyz) for atom in atoms]
        xyz[atom_index][axis] += h
        return PeriodicSystem(
            final.dim,
            np.asarray(final.lattice, dtype=float),
            [Atom(atom.Z, xyz[i]) for i, atom in enumerate(atoms)],
            final.charge,
            final.multiplicity,
        )

    def run(system):
        return pr._run_periodic_semiempirical_engine(
            system,
            route_plan,
            max_iter=80,
            conv_tol=1.0e-7,
            kpoints=(2, 1, 1),
            smearing_temperature_hartree=T,
        )

    def central(h, field):
        return np.array(
            [
                [
                    (
                        getattr(run(displaced(a, c, h)), field)
                        - getattr(run(displaced(a, c, -h)), field)
                    )
                    / (2.0 * h)
                    for c in range(3)
                ]
                for a in range(len(atoms))
            ]
        )

    deviations_f = []
    for h in (4.0e-3, 2.0e-3, 5.0e-4):
        deviations_f.append(
            float(np.abs(derivatives.gradient - central(h, "free_energy")).max())
        )
        deviation_e = float(
            np.abs(derivatives.gradient - central(h, "energy")).max()
        )
        # grad F is what the kernel returns; grad E is measurably different.
        assert deviation_e > 1.0e-6, deviation_e
    assert max(deviations_f) < 1.0e-7, deviations_f


def test_periodic_dftb_kpoint_nonconverged_optimizer_marks_crashed(
    tmp_path,
    monkeypatch,
):
    import vibeqc.semiempirical.periodic as periodic_module

    def fake_engine(*_args, **_kwargs):
        return _fake_kpoint_result()

    def fake_gradient(method, system, **_kwargs):
        return -0.40, np.zeros((len(system.unit_cell), 3))

    def fake_optimize(system, *_args, route_plan=None, **_kwargs):
        return periodic_module.PeriodicSemiempiricalOptimizationResult(
            system=system,
            energy=-0.50,
            gradient=np.ones((len(system.unit_cell), 3)),
            n_iter=5,
            converged=False,
            route_plan=route_plan,
        )

    monkeypatch.setattr(pr, "_run_periodic_semiempirical_engine", fake_engine)
    monkeypatch.setattr(
        periodic_module,
        "evaluate_periodic_energy_gradient",
        fake_gradient,
    )
    monkeypatch.setattr(
        periodic_module,
        "optimize_periodic_positions",
        fake_optimize,
    )

    stem = tmp_path / "hli_dftb0_k_opt_red"
    with pytest.raises(RuntimeError, match="optimization did not converge"):
        pr.run_periodic_job(
            _hli_chain(),
            None,
            method="dftb0",
            kpoints=(2, 1, 1),
            optimize=True,
            output=stem,
            citations=False,
            record_hostname=False,
            write_xyz_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
        )

    manifest = tomllib.loads(stem.with_suffix(".system").read_text("utf-8"))
    assert manifest["outputs"]["status"] == "crashed"
    text = stem.with_suffix(".out").read_text("utf-8")
    assert "FATAL: full-k periodic semiempirical geometry optimization" in text


def test_periodic_semiempirical_non_dftb_kpoints_fail_before_engine(
    tmp_path,
    monkeypatch,
):
    def fail_engine(*_args, **_kwargs):
        raise AssertionError("periodic semiempirical engine should not run")

    monkeypatch.setattr(pr, "_run_periodic_semiempirical_engine", fail_engine)

    with pytest.raises(NotImplementedError, match="DFTB0 and SCC-DFTB"):
        pr.run_periodic_job(
            _he_cell(),
            None,
            method="gfn2",
            kpoints=(2, 1, 1),
            output=tmp_path / "he_gfn2_k",
            citations=False,
            record_hostname=False,
        )


def test_periodic_semiempirical_open_nddo_fails_before_engine(
    tmp_path,
    monkeypatch,
):
    def fail_engine(*_args, **_kwargs):
        raise AssertionError("periodic semiempirical engine should not run")

    monkeypatch.setattr(pr, "_run_periodic_semiempirical_engine", fail_engine)
    radical = PeriodicSystem(
        3,
        np.eye(3) * 5.0,
        [Atom(1, [0.0, 0.0, 0.0])],
        0,
        2,
    )

    with pytest.raises(NotImplementedError, match="no unrestricted periodic NDDO"):
        pr.run_periodic_job(
            radical,
            None,
            method="pm6",
            output=tmp_path / "h_pm6",
            citations=False,
            record_hostname=False,
        )


def test_gaussian_periodic_methods_still_require_basis(tmp_path):
    with pytest.raises(TypeError, match="requires a BasisSet"):
        pr.run_periodic_job(
            _he_cell(),
            None,
            method="RHF",
            output=tmp_path / "bad",
            citations=False,
            record_hostname=False,
        )


def test_periodic_output_writer_marks_engine_crash(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("synthetic periodic engine failure")

    monkeypatch.setattr(pr, "_run_periodic_semiempirical_engine", boom)
    stem = tmp_path / "crashed"
    with pytest.raises(RuntimeError, match="synthetic periodic engine failure"):
        pr.run_periodic_job(
            _he_cell(),
            None,
            method="pm6",
            output=stem,
            citations=False,
            record_hostname=False,
        )

    manifest = tomllib.loads(stem.with_suffix(".system").read_text("utf-8"))
    assert manifest["outputs"]["status"] == "crashed"


def test_periodic_semiempirical_nonconverged_marks_crashed(
    tmp_path,
    monkeypatch,
):
    def nonconverged_engine(*_args, **_kwargs):
        return SimpleNamespace(
            energy=-0.25,
            converged=False,
            n_iter=80,
            n_basis=4,
        )

    monkeypatch.setattr(pr, "_run_periodic_semiempirical_engine", nonconverged_engine)
    stem = tmp_path / "pm6_nonconverged"

    with pytest.raises(
        RuntimeError,
        match="PM6 periodic semiempirical SCF did not converge",
    ):
        pr.run_periodic_job(
            _he_cell(),
            None,
            method="pm6",
            output=stem,
            citations=False,
            record_hostname=False,
        )

    manifest = tomllib.loads(stem.with_suffix(".system").read_text("utf-8"))
    assert manifest["outputs"]["status"] == "crashed"
    text = stem.with_suffix(".out").read_text("utf-8")
    assert "FATAL: PM6 periodic semiempirical SCF did not converge" in text


def test_band_summary_gap_sign_follows_the_t_zero_occupation_contract() -> None:
    """#85 end to end: occupations in, printed indirect gap out.

    ``_band_summary`` classifies band edges from ``result.occupations``, so the
    sign of the gap it prints is decided entirely by how those occupations were
    assigned. Feed it both fills of the SAME two-k spectrum and pin the two
    reports.

    The per-k fill is what ``apply_smearing`` returned at T = 0 before #85: the
    lowest band occupied at each k independently. It occupies -0.20 at k0 while
    -0.50 at k1 stays empty, so the VBM lands 0.3 Ha ABOVE the CBM and the
    block prints a negative gap. The global fill -- one Fermi level over the
    mesh -- occupies both k1 states instead.
    """
    eps = [np.array([-0.20, 0.30, 0.90]), np.array([-0.60, -0.50, 0.80])]
    kpoints = [np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.0, 0.0])]

    per_k_fill = pr._band_summary(
        SimpleNamespace(
            mo_energies=eps,
            occupations=[np.array([2.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])],
            kpoints=kpoints,
        )
    )
    assert "gap (indirect) =  -0.300000 Ha" in per_k_fill
    assert "negative: band ordering inverted across the mesh" in per_k_fill

    global_fill = pr._band_summary(
        SimpleNamespace(
            mo_energies=eps,
            occupations=[np.array([0.0, 0.0, 0.0]), np.array([2.0, 2.0, 0.0])],
            kpoints=kpoints,
        )
    )
    assert "gap (indirect) =   0.300000 Ha" in global_fill
    assert "negative" not in global_fill


def test_band_summary_reports_the_gap_apply_smearing_actually_produces() -> None:
    """#85: the producer and the report, wired together.

    The two tests above pin each half. This one runs the real chain -- the
    shared T = 0 dispatcher every periodic backend calls, then the block that
    prints its band edges -- so a future change that reintroduces the per-k
    fill turns this red without anyone having to remember the connection.
    """
    from vibeqc import apply_smearing

    eps = [np.array([-0.20, 0.30, 0.90]), np.array([-0.60, -0.50, 0.80])]
    occ = apply_smearing(
        eps,
        weights=[0.5, 0.5],
        n_electrons_per_cell=2.0,
        n_occ_each=1,
        smearing=None,
    ).occupations_per_k

    text = pr._band_summary(
        SimpleNamespace(
            mo_energies=eps,
            occupations=list(occ),
            kpoints=[np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.0, 0.0])],
        )
    )

    assert "gap (indirect) =   0.300000 Ha" in text
    assert "negative: band ordering inverted across the mesh" not in text


def test_band_summary_absorbs_a_k_point_with_no_occupied_states() -> None:
    """#85: the VBM and CBM are independent maxima over the whole mesh.

    Under one global Fermi level a band-overlap mesh can leave a k point with
    every band above mu and therefore NO occupied state. Skipping that k
    outright hid its virtual states from the CBM search: the block reported
    the CBM as +0.80 Ha at the occupied k instead of -0.20 Ha at the empty
    one, a 1.300000-Ha "direct" gap in place of the true 0.300000-Ha indirect
    one. The direct gap is a per-k quantity and is still skipped there, since
    it is genuinely undefined without both halves.
    """
    text = pr._band_summary(
        SimpleNamespace(
            mo_energies=[
                np.array([-0.20, 0.30, 0.90]),
                np.array([-0.60, -0.50, 0.80]),
            ],
            occupations=[
                np.array([0.0, 0.0, 0.0]),
                np.array([2.0, 2.0, 0.0]),
            ],
            kpoints=[np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.0, 0.0])],
        )
    )

    assert "VBM     =  -0.5000000000 Ha" in text
    assert "CBM     =  -0.2000000000 Ha" in text
    assert "gap (indirect) =   0.300000 Ha" in text
    # Only k1 has both halves, so it alone sets the minimum direct gap.
    assert "direct gap (min) =   1.300000 Ha" in text
    assert "(k_idx = 1)" in text


def test_band_summary_says_n_a_when_no_k_point_has_both_halves() -> None:
    """The direct gap can be undefined on every k of the mesh.

    Two k points, one fully occupied and one fully empty. Previously every k
    was skipped, so the sentinel +1e300 would have been rendered as a number.
    """
    text = pr._band_summary(
        SimpleNamespace(
            mo_energies=[np.array([-0.60, -0.50]), np.array([0.30, 0.90])],
            occupations=[np.array([2.0, 2.0]), np.array([0.0, 0.0])],
            kpoints=[np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.0, 0.0])],
        )
    )

    assert "VBM     =  -0.5000000000 Ha" in text
    assert "CBM     =   0.3000000000 Ha" in text
    assert "gap (indirect) =   0.800000 Ha" in text
    assert "direct gap (min) = n/a" in text
    assert "no k-point has both an occupied and a virtual state" in text
    assert "1e+300" not in text and "1.000000e+300" not in text
