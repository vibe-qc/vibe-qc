"""``vibeqc.output.OutputPlan`` — declarative pre-flight contract.

Pins the contract documented in
``docs/design_output_module.md § OutputPlan and PlannedFile``:

  1. ``OutputPlan.from_run_job_kwargs`` always declares the four
     always-on artefacts (``.out``, ``.system``, ``.bibtex``,
     ``.references``) and the three default-on artefacts (``.molden``,
     ``.xyz``, ``.qvf``).
  2. Opt-out kwargs (``write_molden_file=False``, ``citations=False``,
     ``write_xyz=False``) suppress only their matching artefact.
  3. Opt-in kwargs (``perf_log=True``, ``structured_log=True``,
     ``optimize=True``) add their matching artefact at the canonical
     sibling path.
  4. ``crash_dump=True`` declares a conditional ``.dump`` artefact
     (``always=False``) so vq can warn the user it *may* appear.
  5. ``OutputPlan.to_toml_section()`` and ``to_jsonable()`` round-trip
     through stdlib ``tomllib`` / ``json`` without information loss.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from vibeqc.output import OutputPlan, PlannedFile


# ---------------------------------------------------------------------------
# Factory: the canonical kwarg → plan mapping
# ---------------------------------------------------------------------------

def _roles(plan: OutputPlan) -> set[str]:
    return {f.role for f in plan.files}


def _paths(plan: OutputPlan) -> set[str]:
    return {str(f.path) for f in plan.files}


def test_default_plan_declares_always_on_artefacts(tmp_path: Path) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o",
        method="rhf",
        basis="6-31g*",
        functional=None,
    )
    roles = _roles(plan)

    # Always-on regardless of kwargs.
    assert "log" in roles
    assert "manifest" in roles
    # Default-on opt-out artefacts.
    assert "orbitals" in roles
    assert "geometry" in roles
    assert "citations" in roles
    assert "qvf" in roles


def test_plan_files_use_canonical_sibling_paths(tmp_path: Path) -> None:
    stem = tmp_path / "h2o"
    plan = OutputPlan.from_run_job_kwargs(
        output=stem, method="rhf", basis="sto-3g", functional=None,
    )
    paths = _paths(plan)
    assert str(stem.with_suffix(".out")) in paths
    assert str(stem.with_suffix(".system")) in paths
    assert str(stem.with_suffix(".molden")) in paths
    assert str(stem.with_suffix(".xyz")) in paths
    assert str(stem.with_suffix(".qvf")) in paths
    assert str(stem.with_suffix(".bibtex")) in paths
    assert str(stem.with_suffix(".references")) in paths


def test_disabling_molden_drops_orbitals_artefact(tmp_path: Path) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o", method="rhf", basis="sto-3g",
        functional=None, write_molden_file=False,
    )
    assert "orbitals" not in _roles(plan)
    # Other always-on artefacts still present.
    assert "log" in _roles(plan)
    assert "manifest" in _roles(plan)


def test_disabling_xyz_drops_geometry_artefact(tmp_path: Path) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o", method="rhf", basis="sto-3g",
        functional=None, write_xyz=False,
    )
    assert "geometry" not in _roles(plan)


def test_disabling_citations_drops_both_citation_artefacts(
    tmp_path: Path,
) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o", method="rhf", basis="sto-3g",
        functional=None, citations=False,
    )
    cit_files = plan.files_by_role("citations")
    assert cit_files == ()


def test_optimize_declares_trajectory(tmp_path: Path) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o", method="rhf", basis="sto-3g",
        functional=None, optimize=True,
    )
    traj_files = plan.files_by_role("trajectory")
    assert len(traj_files) == 1
    assert traj_files[0].path == (tmp_path / "h2o").with_suffix(".traj")
    # Only the ASE backend writes this sibling; other optimizers retain their
    # history for QVF without manufacturing an ASE trajectory.
    assert traj_files[0].always is False


def test_perf_log_kwarg_true_declares_perf_sibling(tmp_path: Path) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o", method="rhf", basis="sto-3g",
        functional=None, perf_log=True,
    )
    perf_files = plan.files_by_role("perf")
    assert len(perf_files) == 1
    assert perf_files[0].path == (tmp_path / "h2o").with_suffix(".perf")


def test_perf_log_kwarg_path_uses_that_path(tmp_path: Path) -> None:
    explicit = tmp_path / "elsewhere" / "custom.perf"
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o", method="rhf", basis="sto-3g",
        functional=None, perf_log=explicit,
    )
    perf_files = plan.files_by_role("perf")
    assert len(perf_files) == 1
    assert perf_files[0].path == explicit


def test_structured_log_kwarg_true_declares_jsonl_sibling(
    tmp_path: Path,
) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o", method="rhf", basis="sto-3g",
        functional=None, structured_log=True,
    )
    files = plan.files_by_role("structured")
    assert len(files) == 1
    assert files[0].path == (tmp_path / "h2o").with_suffix(".scf.jsonl")


def test_crash_dump_is_conditional_always_false(tmp_path: Path) -> None:
    """The .dump file only appears on failure; the plan declares it so
    vq can advertise *may produce* without lying about guarantees."""
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o", method="rhf", basis="sto-3g",
        functional=None, crash_dump=True,
    )
    crash_files = plan.files_by_role("crash")
    assert len(crash_files) == 1
    assert crash_files[0].always is False
    assert crash_files[0].path == (tmp_path / "h2o").with_suffix(".dump")


def test_crash_dump_false_drops_artefact(tmp_path: Path) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o", method="rhf", basis="sto-3g",
        functional=None, crash_dump=False,
    )
    assert plan.files_by_role("crash") == ()


def test_output_qvf_false_drops_qvf_artefact(tmp_path: Path) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o", method="rhf", basis="sto-3g",
        functional=None, output_qvf=False,
    )
    assert plan.files_by_role("qvf") == ()


def test_qvf_plan_role_matches_dispatch_key(tmp_path: Path) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o",
        method="rhf",
        basis="sto-3g",
        functional=None,
    )
    qvf_files = plan.files_by_role("qvf")
    assert len(qvf_files) == 1
    assert qvf_files[0].format == "qvf"


def test_integer_mo_label_declares_canonical_mo_path(tmp_path: Path) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o",
        method="rhf",
        basis="sto-3g",
        functional=None,
        cube_mo_labels=(5,),
    )
    mo_files = plan.files_by_role("orbital_vol")
    assert [f.path.name for f in mo_files] == ["h2o.mo_5.cube"]


@pytest.mark.parametrize(
    ("variant", "formats"),
    [
        ("standard", ["text", "json"]),
        ("bipole", ["bipole-text", "bipole-json"]),
        (
            "aiccm2026dev-b",
            ["aiccm2026dev-b-text", "aiccm2026dev-b-json"],
        ),
    ],
)
def test_periodic_population_variant_is_declared(
    tmp_path: Path,
    variant: str,
    formats: list[str],
) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "cell",
        method="rhf",
        basis="sto-3g",
        functional=None,
        job_kind="periodic_scf",
        population_variant=variant,
    )
    assert [row.format for row in plan.files_by_role("population")] == formats


def test_unknown_population_variant_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="population_variant"):
        OutputPlan.from_run_job_kwargs(
            output=tmp_path / "cell",
            method="rhf",
            basis="sto-3g",
            functional=None,
            population_variant="unknown",
        )


# ---------------------------------------------------------------------------
# Serialisation: TOML / JSON
# ---------------------------------------------------------------------------

def test_to_toml_section_round_trips_via_tomllib(tmp_path: Path) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o",
        method="RKS", basis="def2-svp", functional="PBE",
        optimize=True, perf_log=True, structured_log=True,
    )
    section = plan.to_toml_section()
    # Build a minimal TOML doc that places this section under [plan].
    body_lines = ["[plan]"]
    for k, v in section.items():
        if isinstance(v, list):
            continue
        if isinstance(v, bool):
            body_lines.append(f"{k} = {'true' if v else 'false'}")
        elif isinstance(v, str):
            esc = v.replace("\\", "\\\\").replace('"', '\\"')
            body_lines.append(f'{k} = "{esc}"')
        else:
            body_lines.append(f"{k} = {v}")
    body_lines.append("")
    for item in section["files"]:
        body_lines.append("[[plan.files]]")
        for k, v in item.items():
            if isinstance(v, bool):
                body_lines.append(f"{k} = {'true' if v else 'false'}")
            else:
                esc = str(v).replace("\\", "\\\\").replace('"', '\\"')
                body_lines.append(f'{k} = "{esc}"')
        body_lines.append("")
    doc = "\n".join(body_lines)

    parsed = tomllib.loads(doc)
    assert parsed["plan"]["method"] == "RKS"
    assert parsed["plan"]["basis"] == "def2-svp"
    assert parsed["plan"]["functional"] == "PBE"
    assert len(parsed["plan"]["files"]) == len(section["files"])


def test_to_jsonable_round_trips_via_json(tmp_path: Path) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o",
        method="rhf", basis="sto-3g", functional=None,
    )
    payload = json.dumps(plan.to_jsonable())
    reconstructed = json.loads(payload)
    assert reconstructed["method"] == "RHF"
    assert reconstructed["basis"] == "sto-3g"
    assert reconstructed["functional"] is None
    assert len(reconstructed["files"]) == len(plan.files)


# ---------------------------------------------------------------------------
# Inspection helpers
# ---------------------------------------------------------------------------

def test_guaranteed_files_excludes_conditional_dump(tmp_path: Path) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o", method="rhf", basis="sto-3g",
        functional=None, crash_dump=True,
    )
    guaranteed = plan.guaranteed_files()
    crash_in_guaranteed = any(f.role == "crash" for f in guaranteed)
    assert not crash_in_guaranteed


def test_options_digest_is_stable_across_calls(tmp_path: Path) -> None:
    a = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o", method="rhf", basis="sto-3g",
        functional=None,
    )
    b = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o-elsewhere", method="rhf",
        basis="sto-3g", functional=None,
    )
    # Same method+basis+functional → same digest. The stem doesn't
    # change the identity of the *job* for vq's "did this run change?"
    # purposes.
    assert a.options_digest == b.options_digest


def test_options_digest_changes_when_functional_changes(
    tmp_path: Path,
) -> None:
    pbe = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o", method="rks", basis="sto-3g",
        functional="PBE",
    )
    b3lyp = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o", method="rks", basis="sto-3g",
        functional="B3LYP",
    )
    assert pbe.options_digest != b3lyp.options_digest


def test_planned_file_is_frozen() -> None:
    """PlannedFile is a dataclass(frozen=True) — mutation must raise."""
    f = PlannedFile(
        role="log", path=Path("x.out"), format="text",
        always=True, description="",
    )
    with pytest.raises(Exception):
        f.role = "manifest"  # type: ignore[misc]
