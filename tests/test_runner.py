"""run_job — classic-QC-program-style input/output workflow."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
from vibeqc import Atom, Molecule, run_job


def _h2o() -> Molecule:
    return Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.98]),
            Atom(1, [0.0, -1.43, -0.98]),
        ]
    )


def _n2_release_geometry() -> Molecule:
    bohr_per_angstrom = 1.889726124565062
    half_bond = 0.5 * 1.0976 * bohr_per_angstrom
    return Molecule([
        Atom(7, [0.0, 0.0, -half_bond]),
        Atom(7, [0.0, 0.0, half_bond]),
    ])


def test_molecular_geometry_summary_sizes_rule_to_coordinate_rows() -> None:
    from vibeqc.runner import _geom_summary

    lines = _geom_summary(_h2o()).splitlines()
    expected_first_row = (
        f"  {1:4d}  Z={8:3d}   "
        f"{0.0:14.8f}  {0.0:14.8f}  {0.0:14.8f}"
    )
    assert lines[0] == "  Atoms (bohr)"
    assert lines[2] == expected_first_row
    assert len(lines[1]) == len(lines[2])
    assert lines[-1].startswith("  charge=")


def test_run_job_writes_out_and_molden(tmp_path: Path) -> None:
    mol = _h2o()
    stem = tmp_path / "h2o"
    result = run_job(mol, basis="sto-3g", method="rhf", output=stem)

    out = stem.with_suffix(".out")
    molden = stem.with_suffix(".molden")
    assert out.is_file()
    assert molden.is_file()
    assert result.converged

    text = out.read_text()
    assert "vibe-qc" in text  # banner
    assert "Job: RHF" in text
    assert "Atoms (bohr)" in text
    assert "Molecular orbitals" in text
    # BUG 81 (43f1e433b) replaced the "HOMO-LUMO gap:" footer with the
    # explicit HOMO / LUMO / gap triple.
    assert "HOMO:" in text
    assert "LUMO:" in text
    assert "gap:" in text
    assert "h2o.molden" in text


def test_run_job_qvf_default_on_and_opt_out(tmp_path: Path) -> None:
    mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])

    default_stem = tmp_path / "h2_default_qvf"
    run_job(
        mol,
        basis="sto-3g",
        method="rhf",
        output=default_stem,
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
        citations=False,
        progress=False,
    )
    assert default_stem.with_suffix(".qvf").is_file()

    opt_out_stem = tmp_path / "h2_no_qvf"
    run_job(
        mol,
        basis="sto-3g",
        method="rhf",
        output=opt_out_stem,
        output_qvf=False,
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
        citations=False,
        progress=False,
    )
    assert not opt_out_stem.with_suffix(".qvf").exists()


def test_periodic_runner_qvf_default_is_on() -> None:
    from vibeqc import run_periodic_job

    param = inspect.signature(run_periodic_job).parameters["output_qvf"]
    assert param.default is True


def test_run_job_auto_method_rhf(tmp_path: Path) -> None:
    mol = _h2o()
    stem = tmp_path / "h2o_auto"
    result = run_job(mol, basis="sto-3g", output=stem)  # method="auto"
    assert result.converged
    assert "Job: RHF" in stem.with_suffix(".out").read_text()


def test_run_job_rks_writes_energy_components(tmp_path: Path) -> None:
    mol = _h2o()
    stem = tmp_path / "h2o_pbe"
    run_job(mol, basis="sto-3g", method="rks", functional="PBE", output=stem)

    text = stem.with_suffix(".out").read_text()
    assert "Job: RKS / PBE" in text
    assert "Energy components" in text
    assert "Exchange-correlation (XC)" in text


def test_run_job_geometry_optimisation_produces_traj(tmp_path: Path) -> None:
    # H2 stretched from its STO-3G equilibrium (1.346 bohr) — optimizer
    # should shorten the bond.
    mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.7])])
    stem = tmp_path / "h2_opt"
    run_job(
        mol,
        basis="sto-3g",
        method="rhf",
        output=stem,
        optimize=True,
        fmax=0.1,
        max_opt_steps=30,
    )

    traj = stem.with_suffix(".traj")
    assert traj.is_file()
    assert traj.stat().st_size > 0

    # ASE should be able to load it back as a frame sequence.
    from ase.io import read

    frames = read(str(traj), index=":")
    assert len(frames) >= 2  # at least start + optimized end

    # Optimized geometry should be closer to the STO-3G eq bond (~1.346 bohr =
    # 0.713 Å).
    from ase.units import Bohr

    optimized = frames[-1]
    r_final = abs(optimized.positions[1, 2] - optimized.positions[0, 2])
    assert r_final / Bohr == pytest.approx(1.346, abs=0.05)


def test_run_job_density_fit_reaches_the_ase_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue 31: ``run_job(density_fit=True, optimize=True)`` must hand the
    wired density_fit / aux_basis options to the optimizer's per-step SCFs.

    Since 2ce60be49 the kwargs are wired onto the SCF option structs before
    the dry-run short-circuit, which means the geometry-optimization
    sub-driver now runs the requested density-fitted route where it
    previously dropped the request and walked the four-index surface.
    """
    pytest.importorskip("ase")
    import vibeqc.ase as ase_mod

    captured = []
    original_init = ase_mod.VibeQC.__init__

    def _spy_init(self, **kwargs) -> None:
        captured.append(dict(kwargs))
        original_init(self, **kwargs)

    monkeypatch.setattr(ase_mod.VibeQC, "__init__", _spy_init)

    mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 2.0])])
    stem = tmp_path / "h2_df_opt"
    # A 1-step run at an unreachable fmax deterministically raises the
    # capped-optimization error after the optimizer constructed its
    # calculator and evaluated at least one DF SCF + gradient.
    with pytest.raises(
        RuntimeError, match="did not converge after 1 of 1 allowed steps"
    ):
        run_job(
            mol,
            basis="def2-svp",
            method="rhf",
            density_fit=True,
            aux_basis="def2-svp-jk",
            optimize=True,
            max_opt_steps=1,
            fmax=1e-9,
            output=stem,
            write_molden_file=False,
            write_xyz_file=False,
            write_population_file=False,
        )

    assert captured, "optimizer never constructed the ASE calculator"
    for kwargs in captured:
        opts = kwargs.get("rhf_options")
        assert opts is not None, "optimizer calculator got no rhf_options"
        assert opts.density_fit is True
        assert opts.aux_basis == "def2-svp-jk"


def test_run_job_geometry_optimisation_native_backend(tmp_path: Path) -> None:
    """H₂ (STO-3G) optimisation via native L-BFGS-B backend."""
    mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.7])])
    stem = tmp_path / "h2_native"
    result = run_job(
        mol,
        basis="sto-3g",
        method="rhf",
        output=stem,
        optimize=True,
        optimizer_backend="native",
        fmax=0.1,
        max_opt_steps=30,
    )
    assert result.converged
    # Read the final geometry from the XYZ file.
    from ase.io import read

    xyz_path = stem.with_suffix(".xyz")
    atoms = read(str(xyz_path))
    r_final = float(np.linalg.norm(atoms.positions[1] - atoms.positions[0]))
    from ase.units import Bohr

    assert r_final / Bohr == pytest.approx(1.346, abs=0.05)


def test_run_job_geomopt_reports_progress_and_units(tmp_path: Path) -> None:
    """geomopt-native run_job writes parseable progress with correct fmax units."""
    mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.7])])
    stem = tmp_path / "h2_geomopt_bfgs"

    run_job(
        mol,
        basis="sto-3g",
        method="rhf",
        output=stem,
        optimize=True,
        geom_opt="bfgs",
        fmax=0.05,
        max_opt_steps=2,
        output_qvf=False,
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
        citations=False,
        progress=False,
    )

    text = stem.with_suffix(".out").read_text()
    assert "Geometry optimization (geomopt/bfgs)" in text
    assert "gmax = 9.72e-04 Ha/bohr (0.05 eV/A)" in text
    assert "evaluating initial energy and gradient" in text
    assert "dE" in text
    assert "|step|" in text


def test_run_job_geomopt_not_cited_when_not_run(tmp_path: Path) -> None:
    """Setting geom_opt alone must not claim a native optimizer was used."""
    mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
    stem = tmp_path / "h2_geomopt_not_run"

    run_job(
        mol,
        basis="sto-3g",
        method="rhf",
        output=stem,
        geom_opt="fire",
        output_qvf=False,
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
        progress=False,
    )

    refs = stem.with_suffix(".references").read_text().lower()
    assert "fast inertial relaxation engine" not in refs
    assert "bitzek" not in refs


def test_run_job_geomopt_rejects_semiempirical_methods(tmp_path: Path) -> None:
    """run_job must not silently run ASE while claiming a native geomopt method."""
    mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])

    with pytest.raises(ValueError, match="Semi-empirical and MLIP"):
        run_job(
            mol,
            basis="sto-3g",
            method="msindo",
            output=tmp_path / "h2_msindo_fire",
            optimize=True,
            geom_opt="fire",
            output_qvf=False,
            write_molden_file=False,
            write_xyz_file=False,
            write_population_file=False,
            progress=False,
        )


def test_run_job_accepts_output_with_parent_dir(tmp_path: Path) -> None:
    """``output='runs/h2o'`` — parent directory is auto-created."""
    mol = _h2o()
    stem = tmp_path / "runs" / "h2o"
    run_job(mol, basis="sto-3g", method="rhf", output=stem)
    assert stem.with_suffix(".out").is_file()
    assert stem.with_suffix(".molden").is_file()


def test_run_job_can_disable_molden(tmp_path: Path) -> None:
    mol = _h2o()
    stem = tmp_path / "h2o_no_molden"
    run_job(
        mol,
        basis="sto-3g",
        method="rhf",
        output=stem,
        write_molden_file=False,
    )
    assert stem.with_suffix(".out").is_file()
    assert not stem.with_suffix(".molden").exists()


# ---------------------------------------------------------------------- #
# Dry-run peak-memory estimate (VIBEQC_DRY_RUN_ESTIMATE): producer half
# of memory-aware `vq submit auto`. Off by default; when set alongside
# VIBEQC_DRY_RUN the dry-run records estimate_memory(...).total_bytes as
# [memory].estimate_bytes (frozen contract read by vq's vibeqc_preflight).
# See vibe-queue/docs/STATUS.md ("Automatic estimate -- complete").
# ---------------------------------------------------------------------- #


def _implicit_rks_options(grid_level: str = "orca-defgrid3"):
    from vibeqc import RKSOptions
    from vibeqc.runner import _apply_grid_level

    options = RKSOptions()
    _apply_grid_level(options.grid, grid_level)
    return options


def test_dry_run_estimate_writes_memory_estimate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tomllib

    from vibeqc._vibeqc_core import BasisSet
    from vibeqc.memory import estimate_memory

    mol = _h2o()
    stem = tmp_path / "h2o_est"
    monkeypatch.setenv("VIBEQC_DRY_RUN", "1")
    monkeypatch.setenv("VIBEQC_DRY_RUN_ESTIMATE", "1")

    # The env var drives the short-circuit; run_job returns None without
    # touching the SCF, writing only the .system manifest.
    result = run_job(mol, basis="sto-3g", method="rks", functional="PBE", output=stem)
    assert result is None

    manifest = stem.with_suffix(".system")
    assert manifest.is_file()
    with manifest.open("rb") as fh:
        body = tomllib.load(fh)
    assert body["outputs"]["status"] == "dry_run"

    # The recorded estimate uses the same materialized DefGrid3 options as
    # the real RKS run, rather than estimating the legacy raw C++ default.
    scf_options = _implicit_rks_options()
    expected = estimate_memory(
        mol,
        BasisSet(mol, "sto-3g"),
        method="rks",
        options={"scf_options": scf_options, "functional": "PBE"},
    ).total_bytes
    assert body["memory"]["estimate_bytes"] == expected
    assert type(body["memory"]["estimate_bytes"]) is int
    assert body["memory"]["estimate_bytes"] > 0


def test_dry_run_estimate_passes_caspt2_active_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tomllib

    from vibeqc._vibeqc_core import BasisSet
    from vibeqc.memory import estimate_memory

    mol = _n2_release_geometry()
    stem = tmp_path / "n2_caspt2_est"
    monkeypatch.setenv("VIBEQC_DRY_RUN", "1")
    monkeypatch.setenv("VIBEQC_DRY_RUN_ESTIMATE", "1")

    result = run_job(
        mol,
        basis="cc-pvdz",
        method="caspt2",
        active_space=(6, 6),
        output=stem,
    )
    assert result is None

    with stem.with_suffix(".system").open("rb") as fh:
        body = tomllib.load(fh)

    expected = estimate_memory(
        mol,
        BasisSet(mol, "cc-pvdz"),
        method="caspt2",
        options={"active_space": (6, 6), "caspt2_options": None},
    ).total_bytes
    assert body["memory"]["estimate_bytes"] == expected
    assert body["memory"]["estimate_bytes"] > 48 * 1024**3


def test_dry_run_estimate_passes_caspt2_corr_gradient_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tomllib

    import vibeqc as vq
    from vibeqc._vibeqc_core import BasisSet
    from vibeqc.memory import estimate_memory

    mol = _n2_release_geometry()
    stem = tmp_path / "n2_caspt2_corr_grad_est"
    opts = vq.CASPT2Options(compute_corr_grad=True, use_zvector=True)
    monkeypatch.setenv("VIBEQC_DRY_RUN", "1")
    monkeypatch.setenv("VIBEQC_DRY_RUN_ESTIMATE", "1")

    result = run_job(
        mol,
        basis="cc-pvdz",
        method="caspt2",
        active_space=(6, 6),
        caspt2_options=opts,
        output=stem,
    )
    assert result is None

    with stem.with_suffix(".system").open("rb") as fh:
        body = tomllib.load(fh)

    expected = estimate_memory(
        mol,
        BasisSet(mol, "cc-pvdz"),
        method="caspt2",
        options={"active_space": (6, 6), "caspt2_options": opts},
    ).total_bytes
    assert body["memory"]["estimate_bytes"] == expected
    assert body["memory"]["estimate_bytes"] > 48 * 1024**3


def test_dry_run_estimate_passes_nevpt2_corr_gradient_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tomllib

    from vibeqc._vibeqc_core import BasisSet
    from vibeqc.memory import estimate_memory
    from vibeqc.solvers import NEVPT2Options

    mol = _n2_release_geometry()
    stem = tmp_path / "n2_nevpt2_corr_grad_est"
    opts = NEVPT2Options(compute_corr_grad=True, use_zvector=True)
    monkeypatch.setenv("VIBEQC_DRY_RUN", "1")
    monkeypatch.setenv("VIBEQC_DRY_RUN_ESTIMATE", "1")

    result = run_job(
        mol,
        basis="cc-pvdz",
        method="nevpt2",
        active_space=(6, 6),
        nevpt2_options=opts,
        output=stem,
    )
    assert result is None

    with stem.with_suffix(".system").open("rb") as fh:
        body = tomllib.load(fh)

    expected = estimate_memory(
        mol,
        BasisSet(mol, "cc-pvdz"),
        method="nevpt2",
        options={"active_space": (6, 6), "nevpt2_options": opts},
    ).total_bytes
    assert body["memory"]["estimate_bytes"] == expected
    assert body["memory"]["estimate_bytes"] > 0


def test_dry_run_estimate_passes_tddft_response_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tomllib

    from vibeqc._vibeqc_core import BasisSet
    from vibeqc.memory import estimate_memory

    mol = _h2o()
    stem = tmp_path / "h2o_tddft_est"
    monkeypatch.setenv("VIBEQC_DRY_RUN", "1")
    monkeypatch.setenv("VIBEQC_DRY_RUN_ESTIMATE", "1")

    result = run_job(
        mol,
        basis="6-31g*",
        method="rks",
        functional="PBE",
        tddft=True,
        tddft_type="casida",
        tddft_n_states=4,
        output=stem,
    )
    assert result is None

    with stem.with_suffix(".system").open("rb") as fh:
        body = tomllib.load(fh)

    basis_obj = BasisSet(mol, "6-31g*")
    scf_options = _implicit_rks_options()
    base = estimate_memory(
        mol,
        basis_obj,
        method="rks",
        options={"scf_options": scf_options, "functional": "PBE"},
    ).total_bytes
    expected = estimate_memory(
        mol,
        basis_obj,
        method="rks",
        options={
            "scf_options": scf_options,
            "tddft": True,
            "tddft_type": "casida",
            "tddft_n_states": 4,
            "functional": "PBE",
        },
    ).total_bytes

    assert body["memory"]["estimate_bytes"] == expected
    assert body["memory"]["estimate_bytes"] > base


def test_dry_run_without_estimate_flag_omits_estimate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tomllib

    mol = _h2o()
    stem = tmp_path / "h2o_noest"
    monkeypatch.setenv("VIBEQC_DRY_RUN", "1")
    monkeypatch.delenv("VIBEQC_DRY_RUN_ESTIMATE", raising=False)

    run_job(mol, basis="sto-3g", method="rks", functional="PBE", output=stem)
    manifest = stem.with_suffix(".system")
    with manifest.open("rb") as fh:
        body = tomllib.load(fh)
    assert body["outputs"]["status"] == "dry_run"
    # Default dry-run stays cheap: no estimate computed, key absent.
    assert "estimate_bytes" not in body["memory"]


def test_dry_run_estimate_writes_semiempirical_memory_estimate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tomllib

    from vibeqc.memory import estimate_semiempirical_memory

    mol = _h2o()
    stem = tmp_path / "h2o_se"
    monkeypatch.setenv("VIBEQC_DRY_RUN", "1")
    monkeypatch.setenv("VIBEQC_DRY_RUN_ESTIMATE", "1")

    result = run_job(mol, method="pm6", output=stem)
    assert result is None
    manifest = stem.with_suffix(".system")
    with manifest.open("rb") as fh:
        body = tomllib.load(fh)
    assert body["outputs"]["status"] == "dry_run"
    assert body["memory"]["estimate_bytes"] == estimate_semiempirical_memory(
        mol,
        method="pm6",
    ).total_bytes
    assert body["memory"]["estimate_bytes"] > 0


def test_run_job_tddft_nto_emits_qvf_sections(tmp_path: Path) -> None:
    """tddft=True + nto=True emits wf_nto_S{n}_{hole,electron} sections."""
    mol = Molecule(
        [Atom(1, [0.0, 0.0, -0.7]), Atom(1, [0.0, 0.0, 0.7])],
        charge=0,
        multiplicity=1,
    )
    result = run_job(
        mol,
        basis="sto-3g",
        method="rhf",
        output=tmp_path / "h2_nto",
        output_qvf=True,
        write_population_file=False,
        tddft=True,
        nto=True,
        citations=False,
    )
    qvf = tmp_path / "h2_nto.qvf"
    assert qvf.is_file()
    import json
    import zipfile

    with zipfile.ZipFile(qvf, "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))
    nto_ids = [s["id"] for s in manifest["sections"] if "nto" in s.get("id", "")]
    assert "wf_nto_S1_hole" in nto_ids
    assert "wf_nto_S1_electron" in nto_ids
